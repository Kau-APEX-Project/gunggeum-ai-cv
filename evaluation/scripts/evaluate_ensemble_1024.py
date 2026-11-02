from pathlib import Path
import sys,json,hashlib
import torch
from PIL import Image,ImageOps
B=Path('outputs/gunggeum_camera');sys.path.insert(0,str((B/'class60_exp001_gpu/code').resolve()))
from finetune import build_model,transform
O=B/'ensemble_2026-10-24';O.mkdir(exist_ok=True);torch.set_num_threads(4)
models=[];checks=[];manifests=[]
for exp in ['003','004']:
 r=B/f'gpu_results/class60_exp{exp}_2026-10-24';cp=r/f'runs/exp{exp}/best.pt';c=torch.load(cp,map_location='cpu',weights_only=True)
 manifests.append((r/'external_manifest.json').read_bytes());checks.append({'experiment':exp,'epoch':c['epoch'],'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()})
 if models:assert classes==c['classes']
 classes=c['classes'];m=build_model(60,False);m.load_state_dict(c['model']);m.eval();models.append(m)
assert manifests[0]==manifests[1]
rows=[r for r in json.loads(manifests[0])['images'] if r['split']=='validation'];names={r['index']:r['name_ko'] for r in classes};idx={v:k for k,v in names.items()};tf=transform(320);preds=[]
with torch.inference_mode():
 for start in range(0,len(rows),8):
  batch=rows[start:start+8];xs=[]
  for r in batch:
   p=B/'class60_exp004_gpu'/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
   with Image.open(p) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
  x=torch.stack(xs);probs=[m(x).softmax(1) for m in models];avg=(probs[0]+probs[1])*.5
  for j,r in enumerate(batch):preds.append({'file':r['source_file'],'truth':r['name_ko'],'top3':[names[i] for i in avg[j].topk(3).indices.tolist()],'individual_top1':[names[p[j].argmax().item()] for p in probs]})
correct=sum(p['truth']==p['top3'][0] for p in preds);top3=sum(p['truth'] in p['top3'] for p in preds)
out={'method':'equal probability average EXP003 EXP004 at 320','status':'development_validation_not_independent_test','checkpoints':checks,'n':len(preds),'correct':correct,'top1':correct/len(preds),'top3_correct':top3,'top3':top3/len(preds),'predictions':preds}
(O/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='predictions'},ensure_ascii=False,indent=2))
(O/'結果.md').write_text(f'# 앙상블 결과\n\n동일57장, EXP003·EXP004 예측확률50:50, 입력320. Top1 {correct}/57 ({correct/57:.2%}), Top3 {top3}/57 ({top3/57:.2%}). 개발 검증이며 최종 시험 아님. 원본 삭제·평가 대상 변경 없음.\n',encoding='utf-8')
