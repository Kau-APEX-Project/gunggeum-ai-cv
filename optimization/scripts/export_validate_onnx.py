from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch,onnx,onnxruntime as ort
from PIL import Image,ImageOps
B=Path('outputs/gunggeum_camera').resolve();sys.path.insert(0,str(B/'inference'))
from predict import Predictor
p=Predictor();O=B/'onnx_exp005';O.mkdir(exist_ok=True)
f=O/'efficientnet_b0_exp005_fp32.onnx'
with torch.inference_mode():torch.onnx.export(p.model,torch.zeros(1,3,320,320),str(f),input_names=['image'],output_names=['logits'],opset_version=17,do_constant_folding=True,dynamo=False)
onnx.checker.check_model(str(f))
(O/'classes.json').write_text(json.dumps(p.classes,ensure_ascii=False,indent=2),encoding='utf-8')
spec={'input_name':'image','shape':[1,3,320,320],'dtype':'float32','layout':'NCHW','color':'RGB','exif_transpose':True,'resize':'Pillow ImageOps.pad BICUBIC centered, aspect preserving','fill_rgb':[124,116,104],'scale':1/255,'mean':[.485,.456,.406],'std':[.229,.224,.225],'output':'60 logits; softmax then descending top3','ids':'provisional; actual card mapping pending','browser_preprocessing_parity':'not yet verified'}
(O/'preprocessing.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
ref=json.loads((B/'exp005_topk_2026-10-24/results.json').read_text(encoding='utf-8'))
ext=json.loads((B/'class60_exp005_gpu/external_manifest.json').read_text(encoding='utf-8'))
lookup={r['source_file']:B/'class60_exp005_gpu'/r['file'] for r in ext['images'] if r['split']=='validation'}
sys.path.insert(0,str(B/'class60_exp001_gpu/code'))
from data import PalaceDataset
ds=PalaceDataset('validation',B/'splits_v1/manifest.json',transform=None)
opt=ort.SessionOptions();opt.intra_op_num_threads=4
s=ort.InferenceSession(str(f),sess_options=opt,providers=['CPUExecutionProvider'])
rows=[];mx=0.;close=True
for j,r in enumerate(ref['predictions']):
 path=ds.root/r['file'] if r['source']=='public' else lookup[r['file']]
 with Image.open(path) as im:x=p.normalize(ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(320,320),method=Image.Resampling.BICUBIC,color=(124,116,104))).unsqueeze(0)
 with torch.inference_mode():a=p.model(x).numpy()
 b=s.run(None,{'image':x.numpy()})[0];delta=float(np.max(np.abs(a-b)));mx=max(mx,delta);close=close and bool(np.allclose(a,b,atol=1e-4,rtol=1e-4))
 aa=np.argsort(-a[0])[:3];bb=np.argsort(-b[0])[:3];names=[p.classes[int(i)]['name_ko'] for i in bb]
 rows.append({'source':r['source'],'file':r['file'],'truth':r['truth'],'top3':names,'same_top1':bool(aa[0]==bb[0]),'same_top3_order':bool(np.array_equal(aa,bb)),'same_saved_reference':names==r['top3'],'max_abs_logit_error':delta})
 if (j+1)%100==0:print(j+1,flush=True)
metrics={}
for src in ['public','external','mixed']:
 rr=[r for r in rows if src=='mixed' or r['source']==src];metrics[src]={'n':len(rr),'correct_top1_top2_top3':[sum(r['truth'] in r['top3'][:k] for r in rr) for k in (1,2,3)]}
report={'checkpoint_sha256':hashlib.sha256(p.path.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'model_bytes':f.stat().st_size,'epoch':p.epoch,'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__},'allclose_1e4':close,'max_abs_logit_error':mx,'top1_agree':sum(r['same_top1'] for r in rows),'top3_order_agree':sum(r['same_top3_order'] for r in rows),'saved_reference_agree':sum(r['same_saved_reference'] for r in rows),'metrics':metrics,'predictions':rows,'scope':'CPU numerical parity on existing development validation; NOT browser or independent test'}
(O/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=False))
