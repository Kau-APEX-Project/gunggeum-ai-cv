"""Matched FP32 / FX-QAT field adaptation. No held-out test images in package."""
from pathlib import Path
import argparse,copy,hashlib,json,random,zipfile
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.models import efficientnet_b0
from torchvision import transforms as T
from PIL import Image
from torch.ao.quantization import get_default_qat_qconfig_mapping,disable_observer
from torch.ao.quantization.quantize_fx import prepare_qat_fx,convert_fx
from torch.ao.nn.intrinsic.qat import freeze_bn_stats

ROOT=Path(__file__).resolve().parent
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
class Photos(Dataset):
 def __init__(self,rows,train=False):
  self.rows=rows
  ops=[]
  if train:ops=[T.RandomAffine(4,translate=(.025,.025),scale=(.95,1.05),interpolation=T.InterpolationMode.BILINEAR,fill=(124,116,104)),T.ColorJitter(brightness=.5,contrast=.15,saturation=.1)]
  self.tf=T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(ROOT/r['file']) as im:x=self.tf(im.convert('RGB'))
  return x,r['label']
def evaluate(model,loader,device):
 model.eval();truth=[];top=[];scores=[]
 with torch.inference_mode():
  for x,y in loader:
   z=model(x.to(device)).softmax(1);v,i=z.topk(3,1);truth+=y.tolist();top+=i.cpu().tolist();scores+=v.cpu().tolist()
 cm=np.zeros((60,60),dtype=int)
 for y,p in zip(truth,top):cm[y,p[0]]+=1
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(60),where=den>0);present=np.flatnonzero(cm.sum(1))
 return {'n':len(truth),'top1':float(np.trace(cm)/len(truth)),'top2':sum(y in p[:2] for y,p in zip(truth,top))/len(truth),'top3':sum(y in p for y,p in zip(truth,top))/len(truth),'macro_f1_present':float(f[present].mean()),'present_class_count':len(present),'truth':truth,'top3_indices':top,'top3_scores':scores,'confusion_matrix':cm.tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fp32','qat'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20);p.add_argument('--workers',type=int,default=2);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
 torch.set_num_threads(4);torch.backends.quantized.engine='fbgemm'
 seed=20261109;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 doc=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));rows=doc['rows'];groups={};hashes={}
 for r in rows:
  assert r['split'] in ('train','validation')
  for dic,key in ((groups,r['group']),(hashes,r['source_sha256'])):
   assert key not in dic or dic[key]==r['split'];dic[key]=r['split']
  assert hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256'],r['file']
 c=torch.load(ROOT/'start.pt',map_location='cpu',weights_only=True);assert c['classes']==doc['classes'] and c['epoch']==16 and c['size']==320
 model=efficientnet_b0(weights=None);model.classifier[1]=nn.Linear(1280,60);model.load_state_dict(c['model'])
 if a.mode=='qat':model=prepare_qat_fx(model.train(),get_default_qat_qconfig_mapping('fbgemm'),(torch.zeros(1,3,320,320),))
 model.to(a.device)
 tr=[r for r in rows if r['split']=='train'];targets=np.array([r['label'] for r in tr]);counts=np.bincount(targets,minlength=60);assert (counts>0).all()
 sample=np.minimum(np.sqrt(counts.max()/counts),2);weights=1/np.sqrt(counts);weights/=weights.mean()
 ds=Photos(tr,True);sampler=WeightedRandomSampler(torch.tensor(sample[targets],dtype=torch.double),len(ds),replacement=True,generator=torch.Generator().manual_seed(seed))
 train=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,num_workers=0 if a.smoke else a.workers,generator=torch.Generator().manual_seed(seed))
 val={s:DataLoader(Photos([r for r in rows if r['split']=='validation' and r['domain']==s]),batch_size=a.batch_size,num_workers=0) for s in ('combined','field')}
 lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=a.device))
 def train_mode():
  model.train();model.apply(freeze_bn_stats)
  for m in model.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
 def assess(m=model,device=a.device):
  if a.mode=='qat':m.apply(disable_observer)
  results={k:evaluate(m,v,device) for k,v in val.items()};results['selection_score']=(results['field']['macro_f1_present']+results['combined']['macro_f1_present'])/2
  return results
 if a.smoke:
  train_mode();x,y=next(iter(train));loss=lossfn(model(x.to(a.device)),y.to(a.device));loss.backward()
  if a.mode=='qat':
   model.eval();model.apply(disable_observer);q=convert_fx(copy.deepcopy(model).cpu());assert q(x[:1]).shape==(1,60)
  print(json.dumps({'smoke':'passed','mode':a.mode,'train':len(tr),'loss':loss.item()}));return
 out=ROOT/'runs'/a.mode;out.mkdir(parents=True,exist_ok=False)
 config={'mode':a.mode,'seed':seed,'epochs_max':a.epochs,'batch_size':a.batch_size,'train_images':len(tr),'field_train_images':sum(r['domain']=='field' for r in tr),'input_size':320,'sampling':'per-image min(sqrt(max_class_count/n),2)','loss':'CrossEntropy normalized inverse sqrt class weights','feature_lr':1e-5,'classifier_lr':1e-4,'weight_decay':1e-4,'selection':'mean of combined and field present-class Macro F1; tie: field Top1','patience':5,'bn':'running statistics frozen','augmentation':'affine degrees4 translate.025 scale.95-1.05; brightness.5 contrast.15 saturation.1','qat':'FX fbgemm default W8/A8 where supported; float fallback for unsupported operators; observer warmup train only then frozen' if a.mode=='qat' else None,'checkpoint_sha256':hashlib.sha256((ROOT/'start.pt').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'torch':str(torch.__version__)}
 save(out/'config.json',config)
 if a.mode=='qat':
  # Initialize quantization ranges on training data, never on validation/test.
  model.eval()
  with torch.no_grad():
   for i,(x,y) in enumerate(train):
    model(x.to(a.device))
    if i>=15:break
  model.apply(disable_observer)
 baseline=assess();save(out/'baseline.json',baseline)
 best=(baseline['selection_score'],baseline['field']['top1']);stale=0;history=[]
 def checkpoint(epoch,result):
  torch.save({'model':model.state_dict(),'classes':c['classes'],'size':320,'epoch':epoch,'mode':a.mode,'validation':result,'config':config},out/'best.pt');save(out/'selected.json',{'epoch':epoch,'metrics':result})
 checkpoint(0,baseline)
 features=[v for n,v in model.named_parameters() if not n.startswith('classifier')];head=[v for n,v in model.named_parameters() if n.startswith('classifier')]
 opt=torch.optim.AdamW([{'params':features,'lr':1e-5},{'params':head,'lr':1e-4}],weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=2,factor=.5)
 for epoch in range(1,a.epochs+1):
  train_mode();total=0
  for x,y in train:
   opt.zero_grad(set_to_none=True);loss=lossfn(model(x.to(a.device)),y.to(a.device));loss.backward();opt.step();total+=loss.item()*len(y)
  result=assess();key=(result['selection_score'],result['field']['top1']);history.append({'epoch':epoch,'loss':total/len(tr),**result});save(out/'history.json',history)
  if key>best:best=key;stale=0;checkpoint(epoch,result)
  else:stale+=1
  scheduler.step(key[0]);print(f"{a.mode} epoch {epoch}: combined={result['combined']['top1']:.4f} field={result['field']['top1']:.4f} field_f1={result['field']['macro_f1_present']:.4f}",flush=True)
  if stale>=5:break
 selected=torch.load(out/'best.pt',map_location=a.device,weights_only=True);model.load_state_dict(selected['model']);model.eval();result=assess();save(out/'best_evaluation.json',result)
 if a.mode=='qat':
  quantized=convert_fx(copy.deepcopy(model).cpu());converted={k:evaluate(quantized,v,'cpu') for k,v in val.items()}
  converted['note']='Actual converted CPU INT8 with float fallback; browser ONNX/WebGPU compatibility not established.'
  save(out/'converted_int8_evaluation.json',converted);torch.save(quantized.state_dict(),out/'converted_int8_state.pt')
  torch.jit.save(torch.jit.trace(quantized,torch.zeros(1,3,320,320)),str(out/'converted_int8.ts'))
  print('Actual converted INT8 field Top1:',converted['field']['top1'],flush=True)
 print('Finished',a.mode,'selected epoch',selected['epoch'],flush=True)
if __name__=='__main__':main()
