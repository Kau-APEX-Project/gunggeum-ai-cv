"""EXP005 inference. Scores are uncalibrated; service IDs remain provisional."""
from pathlib import Path
import argparse,json,time
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from torchvision import transforms as T
from PIL import Image,ImageOps
DEFAULT_CHECKPOINT=Path(__file__).resolve().parents[1]/'models/exp005_epoch16/best.pt'
class Predictor:
 def __init__(self,checkpoint=DEFAULT_CHECKPOINT,threads=4):
  torch.set_num_threads(threads);self.path=Path(checkpoint)
  c=torch.load(self.path,map_location='cpu',weights_only=True)
  self.classes=c['classes'];self.size=c['size'];self.epoch=c['epoch']
  if len(self.classes)!=60 or self.size!=320 or self.epoch!=16:raise ValueError('Expected EXP005 epoch16, 60 classes, size320')
  if [x['index'] for x in self.classes]!=list(range(60)):raise ValueError('Invalid class order')
  self.model=efficientnet_b0(weights=None);self.model.classifier[1]=nn.Linear(1280,60);self.model.load_state_dict(c['model']);self.model.eval()
  self.normalize=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def predict(self,path):
  start=time.perf_counter()
  with Image.open(path) as image:
   image=ImageOps.exif_transpose(image).convert('RGB');image=ImageOps.pad(image,(self.size,self.size),method=Image.Resampling.BICUBIC,color=(124,116,104));x=self.normalize(image).unsqueeze(0)
  prep=time.perf_counter()
  with torch.inference_mode():scores=self.model(x).softmax(1)[0];values,indices=scores.topk(3)
  end=time.perf_counter();candidates=[]
  for rank,(i,v) in enumerate(zip(indices.tolist(),values.tolist()),1):
   r=self.classes[i];candidates.append({'rank':rank,'class_index':i,'name_ko':r['name_ko'],'building_id':r['service_id'],'id_status':r.get('id_status','provisional'),'score':v})
  return {'model_version':'exp005_epoch16_fp32','input_size':self.size,'candidates':candidates,'score_type':'uncalibrated_softmax','timing_ms':{'decode_preprocess':(prep-start)*1000,'forward':(end-prep)*1000,'decode_through_forward':(end-start)*1000}}
def main():
 p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT));p.add_argument('--out');a=p.parse_args();result=Predictor(a.checkpoint).predict(a.image);text=json.dumps(result,ensure_ascii=False,indent=2)
 if a.out:Path(a.out).write_text(text,encoding='utf-8')
 print(text)
if __name__=='__main__':main()

