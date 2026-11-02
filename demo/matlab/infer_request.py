from pathlib import Path
import sys,json,argparse,time,hashlib
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'inference'))
from predict import Predictor,DEFAULT_CHECKPOINT
p=argparse.ArgumentParser();p.add_argument('--request',required=True);args=p.parse_args()
request=Path(args.request);r=json.loads(request.read_text(encoding='utf-8-sig'))
start=time.perf_counter()
result=Predictor().predict(r['image'])
preview=request.with_suffix('.png')
with Image.open(r['image']) as im:
 im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1400,1400));im.save(preview)
result.update({'image_path':r['image'],'preview_path':str(preview),'device':'cpu','checkpoint_sha256':hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'actual_inference':True})
request.with_suffix('.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('Inference complete')
