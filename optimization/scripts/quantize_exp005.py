from pathlib import Path
import json,hashlib,random,time,sys
import numpy as np
import onnx,onnxruntime as ort
from PIL import Image,ImageOps
from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType,CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
B=Path('outputs/gunggeum_camera').resolve();O=B/'quantization_exp005';O.mkdir(exist_ok=True)
import tempfile
(O/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str(O/'tmp')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
classes=read(B/'onnx_exp005/classes.json'); names=[c['name_ko'] for c in classes]
man=read(B/'splits_v1/manifest.json');root=(B/'splits_v1'/man['dataset_root']).resolve()
ext=read(B/'class60_exp005_gpu/external_manifest.json')
train=[];val=[]
for r in man['images']:
 if r['split'] in ('train','validation'):
  item={'path':str(root/r['image']),'label':names.index(r['class'].split('_',1)[1]),'split':r['split'],'group':r['group'],'source':'original'}
  (train if r['split']=='train' else val).append(item)
for r in ext['images']:
 if r['split'] in ('train','validation'):
  item={'path':str(B/'class60_exp005_gpu'/r['file']),'label':names.index(r['name_ko']),'split':r['split'],'group':r['group'],'source':'added'}
  (train if r['split']=='train' else val).append(item)
assert len(val)==568 and len(train)==2554
rng=random.Random(20261030);cal=[]
for label in range(60):
 pool=[r for r in train if r['label']==label and r['source']=='original'];rng.shuffle(pool);cal+=pool[:4]
for label in range(60):
 pool=[r for r in train if r['label']==label and r['source']=='added'];rng.shuffle(pool);cal+=pool[:2]
assert not ({r['path'] for r in cal}&{r['path'] for r in val})
val_hashes={digest(Path(r['path'])) for r in val}
for r in cal:
 r['sha256']=digest(Path(r['path']));assert r['sha256'] not in val_hashes
(O/'calibration_manifest.json').write_text(json.dumps({'seed':20261030,'selection':'up to4 original train/class plus up to2 added train/class, no augmentation','images':cal},ensure_ascii=False,indent=2),encoding='utf-8')
def tensor(path):
 with Image.open(path) as im:
  im=ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(320,320),method=Image.Resampling.BICUBIC,color=(124,116,104))
  a=np.asarray(im,dtype=np.float32)/np.float32(255)
 return np.ascontiguousarray(((a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)[None])
class Reader(CalibrationDataReader):
 def __init__(self):self.i=iter(cal);self.n=0
 def get_next(self):
  r=next(self.i,None)
  if r is None:return None
  self.n+=1
  if self.n%40==0:print('Calibration',self.n,'/',len(cal),flush=True)
  return {'image':tensor(r['path'])}
src=B/'onnx_exp005/efficientnet_b0_exp005_fp32.onnx';pre=O/'preprocessed_fp32.onnx';dst=O/'efficientnet_b0_exp005_int8_qdq.onnx'
print('Calibration count',len(cal),flush=True)
quant_pre_process(str(src),str(pre),skip_optimization=True,skip_symbolic_shape=True)
quantize_static(str(pre),str(dst),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,calibrate_method=CalibrationMethod.MinMax,op_types_to_quantize=['Conv','Gemm','MatMul'],extra_options={'CalibMaxIntermediateOutputs':8})
onnx.checker.check_model(str(dst));print('Quantization complete',dst.stat().st_size,flush=True)
opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
sessions={k:ort.InferenceSession(str(f),sess_options=opt,providers=['CPUExecutionProvider']) for k,f in [('fp32',src),('int8',dst)]}
rows=[]
for i,r in enumerate(val):
 x=tensor(r['path']);row={'file':r['path'],'truth':names[r['label']],'label':r['label']}
 for k,s in sessions.items():
  logits=s.run(None,{'image':x})[0][0]; assert np.isfinite(logits).all();row[k]=np.argsort(-logits,kind='stable')[:3].tolist()
 rows.append(row)
 if (i+1)%100==0: print('Evaluated',i+1,'/568',flush=True)
metrics={}
for k in sessions:
 cm=np.zeros((60,60),dtype=int)
 for r in rows:cm[r['label'],r[k][0]]+=1
 tp=cm.diagonal();den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(60,dtype=float),where=den!=0)
 metrics[k]={'n':len(rows),'correct_top1_top2_top3':[sum(r['label'] in r[k][:n] for r in rows) for n in [1,2,3]],'macro_f1':float(f1.mean()),'confusion_matrix':cm.tolist()}
# Fixed 12 preprocessed validation tensors, warmed sequential alternating runs. Model-only latency.
xs=[tensor(r['path']) for r in val[::max(1,len(val)//12)][:12]]
for s in sessions.values():
 for x in xs[:3]:s.run(None,{'image':x})
ms={k:[] for k in sessions}
for repeat in range(3):
 for x in xs:
  for k in (['fp32','int8'] if repeat%2==0 else ['int8','fp32']):
   t=time.perf_counter();sessions[k].run(None,{'image':x});ms[k].append((time.perf_counter()-t)*1000)
report={'scope':'Combined development validation; same Pillow preprocessing; CPU only, NOT browser/mobile or independent test','calibration_n':len(cal),'calibration_train_only':True,'model_version':'exp005_epoch16','versions':{'onnx':onnx.__version__,'onnxruntime':ort.__version__},'settings':{'format':'QDQ','activation':'int8','weight':'int8 per-channel','calibration':'MinMax','operators':['Conv','Gemm','MatMul'],'threads':4},'models':{k:{'bytes':f.stat().st_size,'sha256':digest(f)} for k,f in [('fp32',src),('int8',dst)]},'metrics':metrics,'latency':{k:{'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'scope':'warm model.run only excludes image decode/preprocess/load'} for k,v in ms.items()},'top1_agreement':sum(r['fp32'][0]==r['int8'][0] for r in rows),'predictions':rows}
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('predictions','metrics')},ensure_ascii=False),flush=True)
for k,v in metrics.items(): print(k,{a:b for a,b in v.items() if a!='confusion_matrix'},flush=True)

