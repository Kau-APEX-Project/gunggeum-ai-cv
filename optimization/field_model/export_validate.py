from pathlib import Path
import json,hashlib,time
import numpy as np
import torch,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from torchvision.models import efficientnet_b0
from torch import nn
from PIL import Image,ImageOps
B=Path(__file__).resolve().parent/'assets';O=Path(__file__).resolve().parent/'generated/onnx';O.mkdir(parents=True,exist_ok=True)
cp=B/'best.pt'
c=torch.load(cp,map_location='cpu',weights_only=True);assert c['epoch']==12 and c['size']==320
torch.set_num_threads(4);model=efficientnet_b0(weights=None);model.classifier[1]=nn.Linear(1280,60);model.load_state_dict(c['model']);model.eval()
src=O/'model_fp32.onnx';dst=O/'model_weight_fp16.onnx'
with torch.inference_mode():torch.onnx.export(model,torch.zeros(1,3,320,320),str(src),input_names=['image'],output_names=['logits'],opset_version=17,do_constant_folding=True,dynamo=False)
onnx.checker.check_model(str(src))
m=onnx.load(str(src)); casts=[]; converted=[]
for init in m.graph.initializer:
    if init.data_type!=TensorProto.FLOAT: continue
    a=numpy_helper.to_array(init)
    if a.size<64 or not np.isfinite(a).all() or np.abs(a).max()>65504: continue
    name=init.name; stored=name+'__stored_fp16'
    init.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=stored))
    casts.append(helper.make_node('Cast',[stored],[name],to=TensorProto.FLOAT,name=name+'__restore_fp32'))
    converted.append(name)
nodes=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(casts+nodes)
onnx.checker.check_model(m);onnx.save(m,str(dst))

(O/'classes.json').write_text(json.dumps(c['classes'],ensure_ascii=False,indent=2),encoding='utf-8')
opt=ort.SessionOptions();opt.intra_op_num_threads=4
sessions={k:ort.InferenceSession(str(f),sess_options=opt,providers=['CPUExecutionProvider']) for k,f in [('onnx_fp32',src),('weight_fp16',dst)]}
doc=json.loads((B/'manifest.json').read_text(encoding='utf-8'));items=[r for r in doc['rows'] if r['split']=='validation'];results=[]
def tensor(im):
 a=np.asarray(im,dtype=np.float32)/255
 return np.ascontiguousarray(((a-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)[None])
for i,r in enumerate(items):
 with Image.open(B/r['file']) as im:x=tensor(im.convert('RGB'))
 with torch.inference_mode():ref=model(torch.from_numpy(x)).numpy()[0]
 row={'file':r['file'],'domain':r['domain'],'label':r['label'],'pytorch':np.argsort(-ref,kind='stable')[:3].tolist()}
 for k,s in sessions.items():
  y=s.run(None,{'image':x})[0][0];assert np.isfinite(y).all();row[k]=np.argsort(-y,kind='stable')[:3].tolist();row[k+'_max_logit_error']=float(np.max(np.abs(y-ref)))
 results.append(row)
 if (i+1)%100==0:print('Checked',i+1,'/',len(items),flush=True)
metrics={}
for domain in ['combined','field','all']:
 rr=[r for r in results if domain=='all' or r['domain']==domain];metrics[domain]={'n':len(rr)}
 for k in ['pytorch',*sessions]:
  cm=np.zeros((60,60),dtype=int)
  for r in rr:cm[r['label'],r[k][0]]+=1
  den=cm.sum(0)+cm.sum(1);f=np.divide(2*cm.diagonal(),den,out=np.zeros(60),where=den>0);present=cm.sum(1)>0
  metrics[domain][k]={'correct_top1_top2_top3':[sum(r['label'] in r[k][:n] for r in rr) for n in (1,2,3)],'macro_f1_present':float(f[present].mean())}
demo=[]
for p in sorted((B/'field_demo_20261109').glob('[0-9]*.jpg')):
 with Image.open(p) as im:x=tensor(ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(320,320),method=Image.Resampling.BICUBIC,color=(124,116,104)))
 y=sessions['weight_fp16'].run(None,{'image':x})[0][0];pr=np.exp(y-y.max());pr/=pr.sum();idx=np.argsort(-pr)[:3]
 demo.append({'file':p.name,'candidates':[{'name':c['classes'][int(j)]['name_ko'],'score':float(pr[j])} for j in idx]})
report={'epoch':12,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'method':'same as EXP005: FP16 initializer storage >=64 elements, restored to FP32 with Cast; no QAT/calibration','models':{k:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for k,f in [('onnx_fp32',src),('weight_fp16',dst)]},'metrics':metrics,'agreement':{k:{'top1':sum(r[k][0]==r['pytorch'][0] for r in results),'top3_order':sum(r[k]==r['pytorch'] for r in results),'max_logit_error':max(r[k+'_max_logit_error'] for r in results)} for k in sessions},'demo':demo,'predictions':results,'scope':'CPU verification only, browser/mobile not yet tested. Existing service model unchanged. Field150 is outcome-selected development data.'}
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=False),flush=True)
