from pathlib import Path
import os
import sys,json,hashlib,time
import torch
from PIL import Image,ImageOps
B=Path('outputs/gunggeum_camera');sys.path.insert(0,str((B/'class60_exp001_gpu/code').resolve()))
from finetune import build_model,transform
O=B/'external_baseline_2026-10-23';O.mkdir(exist_ok=True)
source=B/'external_roles_2026-10-23/role_manifest_draft.json';m=json.loads(source.read_text(encoding='utf-8'));rows=[r for r in m['images'] if r['proposed_role']=='dev_candidate' and not r['exact_duplicate_suppressed']]
root=Path(os.environ.get('EXTERNAL_DATA_ROOT', 'external_dataset'))  # 외부 평가 이미지 폴더;cp=B/'gpu_results/class60_exp001_2026-10-17/runs/effb0_60_baseline/best.pt';c=torch.load(cp,map_location='cpu',weights_only=True)
classes=c['classes'];idx={r['name_ko']:r['index'] for r in classes};names={r['index']:r['name_ko'] for r in classes};assert len(classes)==60
assert all(r['class'] in idx for r in rows)
torch.set_num_threads(4);model=build_model(60,False);model.load_state_dict(c['model']);model.eval();tf=transform(c['size']);results=[];start=time.time()
with torch.inference_mode():
 for pos in range(0,len(rows),16):
  batch=rows[pos:pos+16];xs=[]
  for r in batch:
   path=root/r['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
   with Image.open(path) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
  probs=model(torch.stack(xs)).softmax(1);vals,inds=probs.topk(3,dim=1)
  for r,v,ii in zip(batch,vals.tolist(),inds.tolist()):results.append({'file':r['file'],'group_id':r['group_id'],'label':r['class'],'label_status':'user_folder_label_unverified','top3':[names[i] for i in ii],'scores':v,'top1_correct':ii[0]==idx[r['class']],'top3_correct':idx[r['class']] in ii})
  print(f'{len(results)}/{len(rows)}',flush=True)
def summarize(rs):
 per={};labels=sorted({r['label'] for r in rs});fs=[]
 for name in labels:
  subset=[r for r in rs if r['label']==name];tp=sum(r['top1_correct'] for r in subset);fp=sum(r['top3'][0]==name and r['label']!=name for r in rs);fn=len(subset)-tp
  fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0);per[name]={'n':len(subset),'correct':tp,'top3_correct':sum(r['top3_correct'] for r in subset)}
 return {'n':len(rs),'top1':sum(r['top1_correct'] for r in rs)/len(rs),'top3':sum(r['top3_correct'] for r in rs)/len(rs),'macro_f1_over_present_gt_classes':sum(fs)/len(fs),'per_class':per}
reps={}
for r in results:reps.setdefault(r['group_id'],r)
report={'status':'provisional_folder_labels_not_final_test','model_classes':60,'evaluated_gt_classes':len({r['label'] for r in results}),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'role_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'device':'cpu','elapsed_seconds':time.time()-start,'image_level':summarize(results),'one_representative_per_candidate_group':summarize(list(reps.values())),'predictions':results}
(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report['image_level'],ensure_ascii=False,indent=2))
