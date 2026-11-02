from pathlib import Path
import argparse,json,hashlib,random,time,zipfile
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from PIL import Image,ImageOps
from data import PalaceDataset
from model_utils import build_model,transform,metrics
class External(Dataset):
 def __init__(self,rows,mapping,train=False):self.rows=rows;self.mapping=mapping;self.tf=transform(224,train)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['file']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,self.mapping[r['name_ko']]
def evaluate(model,loader,device,n):
 ys=[];ps=[];top3=0
 with torch.inference_mode():
  for x,y in loader:
   z=model(x.to(device));ys+=y.tolist();ps+=z.argmax(1).cpu().tolist();top3+=(z.topk(3,1).indices.cpu()==y[:,None]).any(1).sum().item()
 m=metrics(ys,ps,n);present=sorted(set(ys));cm=np.array(m['confusion_matrix_true_rows']);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(n,dtype=float),where=den>0)
 m.update(macro_f1_present=float(f[present].mean()),top3=top3/len(ys),truth=ys,predictions=ps)
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='../class60_exp001_gpu');p.add_argument('--checkpoint');p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 base=Path(a.base).resolve();cp=Path(a.checkpoint) if a.checkpoint else base/'runs/effb0_60_baseline/best.pt'
 random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 doc=json.loads(Path('external_manifest.json').read_text(encoding='utf-8'));rs=doc['images'];groups={}
 for r in rs:
  assert hashlib.sha256(Path(r['file']).read_bytes()).hexdigest()==r['sha256']
  assert r['group'] not in groups or groups[r['group']]==r['split'];groups[r['group']]=r['split']
 ds=PalaceDataset('train',base/'splits_v1/manifest.json',transform=transform(224,True));val=PalaceDataset('validation',base/'splits_v1/manifest.json',transform=transform(224))
 c=torch.load(cp,map_location='cpu',weights_only=True);assert c['classes']==ds.class_mapping;assert c['manifest_sha256']==ds.manifest_sha256;assert c['size']==224
 mapping={r['name_ko']:r['index'] for r in c['classes']};et=External([r for r in rs if r['split']=='train'],mapping,True);ev=External([r for r in rs if r['split']=='validation'],mapping)
 train=ConcatDataset([ds,et]);loader=DataLoader(train,batch_size=32,shuffle=True,num_workers=0 if a.smoke else 2,generator=torch.Generator().manual_seed(20261023));vl=DataLoader(val,batch_size=32);el=DataLoader(ev,batch_size=32)
 model=build_model(60,False);model.load_state_dict(c['model']);model.to(a.device)
 counts=np.bincount([ds.class_to_index[r['class']] for r in ds.samples]+[mapping[r['name_ko']] for r in et.rows],minlength=60);weights=1/np.sqrt(counts);weights/=weights.mean();lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=a.device))
 if a.smoke:
  model.train();x,y=next(iter(loader));loss=lossfn(model(x.to(a.device)),y.to(a.device));loss.backward();print(json.dumps({'smoke':'forward_backward_passed','train':len(train),'external_train':len(et),'external_validation':len(ev),'internal_validation':len(val),'loss':loss.item()}));return
 out=Path('runs/exp002');out.mkdir(parents=True,exist_ok=False)
 config={'seed':20261023,'start_checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'external_manifest_sha256':hashlib.sha256(Path('external_manifest.json').read_bytes()).hexdigest(),'classes':c['classes'],'train_counts':counts.tolist(),'class_weights':weights.tolist(),'epochs_max':25,'patience':7,'optimizer':'AdamW','feature_lr':1e-5,'classifier_lr':1e-4,'weight_decay':1e-4,'augmentation':'existing affine degrees4 translate.025 scale.95-1.05; brightness.5 contrast.15 saturation.1','batchnorm':'running statistics frozen; affine parameters trainable','selection':'mean of internal 60-class MacroF1 and external present-class MacroF1','torch':str(torch.__version__)}
 (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
 def assess():
  model.eval();iv=evaluate(model,vl,a.device,60);ex=evaluate(model,el,a.device,60);return {'internal_validation':iv,'external_validation':ex,'score':(iv['macro_f1']+ex['macro_f1_present'])/2}
 baseline=assess();(out/'baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8');best=baseline['score'];stale=0;history=[]
 def checkpoint(epoch,result):torch.save({'model':model.state_dict(),'classes':c['classes'],'size':224,'epoch':epoch,'validation':result,'config':config},out/'best.pt')
 checkpoint(0,baseline)
 opt=torch.optim.AdamW([{'params':model.features.parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=1e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=2,factor=.5)
 for epoch in range(1,26):
  model.train()
  for mod in model.modules():
   if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.eval()
  total=0
  for x,y in loader:
   opt.zero_grad(set_to_none=True);loss=lossfn(model(x.to(a.device)),y.to(a.device));loss.backward();opt.step();total+=loss.item()*len(y)
  result=assess();row={'epoch':epoch,'train_loss':total/len(train),**result};history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
  if result['score']>best+1e-8:best=result['score'];stale=0;checkpoint(epoch,result)
  else:stale+=1
  sched.step(result['score']);print(f"epoch {epoch}: internal_acc={result['internal_validation']['accuracy']:.4f} external_acc={result['external_validation']['accuracy']:.4f} external_f1={result['external_validation']['macro_f1_present']:.4f}",flush=True)
  if stale>=7:break
 with zipfile.ZipFile('class60_exp002_results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in out.rglob('*'):
   if f.is_file():z.write(f,f.as_posix())
  z.write('external_manifest.json')
 print('Return: '+str(Path('class60_exp002_results.zip').resolve()))
if __name__=='__main__':main()
