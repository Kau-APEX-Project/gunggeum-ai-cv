from pathlib import Path
import json,random,hashlib,time,tempfile
import numpy as np
import onnx,onnxruntime as ort
from PIL import Image
from onnx import helper,numpy_helper,TensorProto
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType,CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
ROOT=Path(__file__).resolve().parent;B=ROOT/'assets';O=ROOT/'generated/quantization';O.mkdir(parents=True,exist_ok=True);(O/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str(O/'tmp')
src=ROOT/'generated/onnx/model_fp32.onnx'
paths={'fp32':src,'weight_fp16':ROOT/'generated/onnx/model_weight_fp16.onnx','full_fp16':O/'model_full_fp16.onnx','weight_int8':O/'model_weight_int8.onnx','static_int8':O/'model_static_int8.onnx'}
doc=json.loads((B/'manifest.json').read_text(encoding='utf-8'));train=[r for r in doc['rows'] if r['split']=='train'];val=[r for r in doc['rows'] if r['split']=='validation']
def tensor(r):
 with Image.open(B/r['file']) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)/255
 return np.ascontiguousarray(((a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)[None])
model=convert_float_to_float16(onnx.load(str(src)),keep_io_types=True)
available={x.name for x in model.graph.input}|{x.name for x in model.graph.initializer};pending=list(model.graph.node);ordered=[]
while pending:
 ready=[n for n in pending if all(not x or x in available for x in n.input)]
 if not ready:raise ValueError('Unresolved graph')
 for n in ready:ordered.append(n);available.update(n.output);pending.remove(n)
del model.graph.node[:];model.graph.node.extend(ordered);onnx.save(model,str(paths['full_fp16']))
# Reuse exact EXP005 per-output-channel Conv/Gemm storage quantization.
dst=paths['weight_int8']
model=onnx.load(str(src));initializers={t.name:t for t in model.graph.initializer};new_nodes=[];removed=set();extra=[]
for node in model.graph.node:
 if node.op_type in ('Conv','Gemm') and node.input[1] in initializers:
  old=node.input[1];w=numpy_helper.to_array(initializers[old]);axis=0
  if node.op_type=='Gemm':
   attrs={x.name:helper.get_attribute_value(x) for x in node.attribute};axis=0 if attrs.get('transB',0) else 1
  reduction=tuple(i for i in range(w.ndim) if i!=axis)
  scale=np.maximum(np.max(np.abs(w),axis=reduction)/127,np.finfo(np.float32).tiny).astype(np.float32)
  shape=[1]*w.ndim;shape[axis]=len(scale)
  q=np.clip(np.rint(w/scale.reshape(shape)),-127,127).astype(np.int8)
  qn=old+'_int8';sn=old+'_scale';zn=old+'_zero';dn=old+'_dequant'
  extra.extend([numpy_helper.from_array(q,qn),numpy_helper.from_array(scale,sn),numpy_helper.from_array(np.zeros_like(scale,dtype=np.int8),zn)])
  new_nodes.append(helper.make_node('DequantizeLinear',[qn,sn,zn],[dn],axis=axis,name=old+'_weight_dequant'))
  node.input[1]=dn;removed.add(old)
 new_nodes.append(node)
kept=[t for t in model.graph.initializer if t.name not in removed]
del model.graph.initializer[:];model.graph.initializer.extend(kept+extra)
del model.graph.node[:];model.graph.node.extend(new_nodes)
onnx.save(model,str(dst))
onnx.checker.check_model(str(dst));print('Quantization complete',dst.stat().st_size,flush=True)

rng=random.Random(20261030);cal=[]
for domain,limit in [('combined',4),('field',2)]:
 for label in range(60):
  pool=[r for r in train if r['domain']==domain and r['label']==label];rng.shuffle(pool);cal+=pool[:limit]
assert not ({r['source_sha256'] for r in cal}&{r['source_sha256'] for r in val})
(O/'calibration_manifest.json').write_text(json.dumps({'selection':'up to4 Combined train per class plus up to2 field train per class; no augmentation','seed':20261030,'n':len(cal),'images':cal},ensure_ascii=False,indent=2),encoding='utf-8')
class Reader(CalibrationDataReader):
 def __init__(self):self.it=iter(cal);self.n=0
 def get_next(self):
  r=next(self.it,None)
  if r is None:return None
  self.n+=1
  if self.n%60==0:print('Calibration',self.n,flush=True)
  return {'image':tensor(r)}
pre=O/'preprocessed_fp32.onnx';quant_pre_process(str(src),str(pre),skip_optimization=True,skip_symbolic_shape=True)
quantize_static(str(pre),str(paths['static_int8']),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,calibrate_method=CalibrationMethod.MinMax,op_types_to_quantize=['Conv','Gemm','MatMul'],extra_options={'CalibMaxIntermediateOutputs':8})
opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
sessions={}
for k,f in paths.items():onnx.checker.check_model(str(f));sessions[k]=ort.InferenceSession(str(f),sess_options=opt,providers=['CPUExecutionProvider'])
rows=[]
for i,r in enumerate(val):
 x=tensor(r);row={'file':r['file'],'label':r['label'],'domain':r['domain']}
 for k,s in sessions.items():
  y=s.run(None,{'image':x})[0][0];assert np.isfinite(y).all();row[k]=np.argsort(-y,kind='stable')[:3].tolist()
 rows.append(row)
 if (i+1)%100==0:print('Evaluated',i+1,flush=True)
metrics={}
for domain in ['combined','field','all']:
 rr=[r for r in rows if domain=='all' or r['domain']==domain];metrics[domain]={'n':len(rr)}
 for k in paths:
  cm=np.zeros((60,60),dtype=int)
  for r in rr:cm[r['label'],r[k][0]]+=1
  den=cm.sum(0)+cm.sum(1);f=np.divide(2*cm.diagonal(),den,out=np.zeros(60),where=den>0)
  metrics[domain][k]={'correct_top1_top2_top3':[sum(r['label'] in r[k][:n] for r in rr) for n in (1,2,3)],'macro_f1_present':float(f[cm.sum(1)>0].mean())}
xs=[tensor(r) for r in val[::max(1,len(val)//12)][:12]];ms={k:[] for k in paths}
for s in sessions.values():
 for x in xs[:3]:s.run(None,{'image':x})
order=list(paths)
for repeat in range(3):
 for x in xs:
  rng.shuffle(order)
  for k in order:
   t=time.perf_counter();sessions[k].run(None,{'image':x});ms[k].append((time.perf_counter()-t)*1000)
report={'scope':'CPU only, warm inference excludes download/load/decode; browser/mobile untested; field150 outcome-selected development data','model':'field_expand_epoch12','calibration_n':len(cal),'versions':{'onnx':onnx.__version__,'onnxruntime':ort.__version__},'models':{k:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for k,f in paths.items()},'metrics':metrics,'latency':{k:{'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} for k,v in ms.items()},'predictions':rows}
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=False),flush=True)
