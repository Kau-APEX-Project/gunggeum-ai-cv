from pathlib import Path
import sys,json,hashlib,time
import torch
from PIL import Image,ImageOps
B=Path('outputs/gunggeum_camera');sys.path.insert(0,str((B/'class60_exp001_gpu/code').resolve()))
from finetune import build_model,transform
from data import PalaceDataset
O=B/'exp005_topk_2026-10-24';O.mkdir(exist_ok=True);cp=B/'gpu_results/class60_exp005_2026-10-24/runs/exp005/best.pt';c=torch.load(cp,map_location='cpu',weights_only=True)
torch.set_num_threads(4);m=build_model(60,False);m.load_state_dict(c['model']);m.eval();tf=transform(320);ds=PalaceDataset('validation',B/'splits_v1/manifest.json',transform=tf);assert ds.class_mapping==c['classes'];names={r['index']:r['name_ko'] for r in c['classes']};idx={v:k for k,v in names.items()}
rows=[{'path':ds.root/r['image'],'label':ds.class_to_index[r['class']],'source':'public','file':r['image']} for r in ds.samples]
manifest=B/'class60_exp005_gpu/external_manifest.json'
for r in json.loads(manifest.read_text(encoding='utf-8'))['images']:
 if r['split']=='validation':
  path=B/'class60_exp005_gpu'/r['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];rows.append({'path':path,'label':idx[r['name_ko']],'source':'external','file':r['source_file']})
preds=[]
with torch.inference_mode():
 for start in range(0,len(rows),16):
  batch=rows[start:start+16];xs=[]
  for r in batch:
   with Image.open(r['path']) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
  vals,inds=m(torch.stack(xs)).softmax(1).topk(3,1)
  for r,v,ii in zip(batch,vals.tolist(),inds.tolist()):preds.append({'source':r['source'],'file':r['file'],'truth':names[r['label']],'top3':[names[i] for i in ii],'scores':v,'correct_k':[r['label'] in ii[:k] for k in [1,2,3]]})
  if start%80==0:print(len(preds),flush=True)
report={'epoch':c['epoch'],'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'size':320,'status':'development_validation','metrics':{},'predictions':preds}
for source in ['public','external','mixed']:
 rr=[r for r in preds if source=='mixed' or r['source']==source];correct=[sum(r['correct_k'][k] for r in rr) for k in range(3)];report['metrics'][source]={'n':len(rr),'correct_top1_top2_top3':correct,'accuracy_top1_top2_top3':[v/len(rr) for v in correct]}
assert report['metrics']['mixed']['correct_top1_top2_top3'][0]==527
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report['metrics'],ensure_ascii=False))
lines=['# EXP005 Top k 평가','','동일 epoch16, 입력320, 혼합 개발검증568장. 재학습 없음.','|자료|장수|Top1|Top2|Top3|','|---|---:|---:|---:|---:|']
for src,r in report['metrics'].items():lines.append('|'+src+'|'+str(r['n'])+'|'+'|'.join(f'{v:.2%} ({n}/{r["n"]})' for v,n in zip(r['accuracy_top1_top2_top3'],r['correct_top1_top2_top3']))+'|')
lines+=['','Top2/3는 후보 목록에 정답이 포함된 비율이며 사용자가 정답을 선택하는 비율은 아님. GPS결합은 당장 필수로 구현하지 않고 후보3개 반환 및 사용자 선택 흐름을 우선하는 방안. 어느 것도 아님/재촬영 경로 필요. 점수를 보정된 확률로 표시하지 않음.']
(O/'평가결과.md').write_text('\n'.join(lines),encoding='utf-8')
