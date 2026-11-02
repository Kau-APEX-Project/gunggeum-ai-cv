from pathlib import Path
import sys,json,hashlib,time
import torch
from PIL import Image,ImageOps
B=Path('outputs/gunggeum_camera');sys.path.insert(0,str((B/'class60_exp001_gpu/code').resolve()))
from finetune import build_model,transform,metrics
R=B/'gpu_results/class60_exp002_2026-10-24';O=B/'resolution_extended_2026-10-24';O.mkdir(exist_ok=True)
mp=R/'external_manifest.json';rows=[r for r in json.loads(mp.read_text(encoding='utf-8'))['images'] if r['split']=='validation'];cp=R/'runs/exp002/best.pt';c=torch.load(cp,map_location='cpu',weights_only=True);names={r['index']:r['name_ko'] for r in c['classes']};idx={v:k for k,v in names.items()}
torch.set_num_threads(4);model=build_model(60,False);model.load_state_dict(c['model']);model.eval();output={'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'epoch':c['epoch'],'status':'development_validation_not_final_test','device':'cpu','sizes':{}}
for size in [224,320,384,448]:
 tf=transform(size);ys=[];ps=[];pred=[];t=time.time()
 with torch.inference_mode():
  for start in range(0,len(rows),8):
   batch=rows[start:start+8];xs=[]
   for r in batch:
    p=B/'class60_exp002_gpu'/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    with Image.open(p) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
   z=model(torch.stack(xs));top=z.topk(3,1).indices.tolist()
   for r,ii in zip(batch,top):ys.append(idx[r['name_ko']]);ps.append(ii[0]);pred.append({'file':r['source_file'],'truth':r['name_ko'],'top3':[names[i] for i in ii]})
 m=metrics(ys,ps,60);m['top3']=sum(r['truth'] in r['top3'] for r in pred)/len(pred);m['n']=len(rows);m['predictions']=pred;m['seconds']=time.time()-t
 # Do not report 60-class macro as external coverage is only 14.
 m.pop('macro_f1');output['sizes'][str(size)]=m
 print(size,m['accuracy'],m['top3'],flush=True)
(O/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
a=output['sizes']['224'];b=output['sizes']['320'];history=json.loads((R/'runs/exp002/history.json').read_text());ref=next(r for r in history if r['epoch']==c['epoch'])['external_validation'];assert a['predictions'] and abs(a['accuracy']-ref['accuracy'])<1e-8
lines=['# 입력 해상도 비교 — 2026-10-24','','동일 모델(epoch4), 동일 외부 개발 검증57장. 모델은224로 학습됨. 이미지 삭제 및 추가학습 없음.','', '|입력|Top1|Top3|','|---|---:|---:|']
for size,m in output['sizes'].items():lines.append(f"|{size}|{m['accuracy']:.2%}|{m['top3']:.2%}|")
lines+=['','입력 크기 선택도 검증자료를 사용한 개발 과정임. 독립된 최종 시험 성능이 아님. 오답을 선택적으로 제외한 점수를 개선 성능으로 보고하지 않음. 기존 동일57장 기준 외부 Top1은54.39%에서70.18%로15.79%p 개선됨. 내부 검증은91.59%에서91.00%임.']
(O/'비교결과.md').write_text('\n'.join(lines),encoding='utf-8')

