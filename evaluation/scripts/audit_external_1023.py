from pathlib import Path
import os
import json,hashlib,itertools
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
B=Path('outputs/gunggeum_camera'); R=Path(os.environ.get('EXTERNAL_DATA_ROOT', 'external_dataset'))  # 외부 평가 이미지 폴더; O=B/'external_audit_2026-10-23';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
old=read(B/'external_user_2026-10-12/manifest.json')
oldhash={r['sha256'] for r in old}; oldpixel={r['pixel_sha256'] for r in old}
exp=B/'gpu_results/exp004_2026-10-13/prepared'
trainpaths={r['image'] for r in read(exp/'manifest.json')['images'] if r['split']=='train'}
trainhash={r['sha256'] for r in read(exp/'dataset_clean/manifest.json')['images'] if r['image'] in trainpaths}
public={r['sha256'] for r in read(B/'dataset_clean/manifest.json')['images']}
rows=[]; thumbs={}
def bits(a):return int(''.join('1' if v else '0' for v in a.flatten()),2)
def dct(a,axis):
 n=a.shape[axis];m=2*np.cos(np.pi*np.arange(n)[:,None]*(2*np.arange(n)[None,:]+1)/(2*n))
 return m@a if axis==0 else a@m.T
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.avif','.bmp','.tiff','.tif'}:continue
 rel=p.relative_to(R).as_posix();folder=rel.split('/')[0];r={'id':len(rows),'file':rel,'class':folder.removesuffix('train'),'folder':folder,'label_status':'user_folder_label_not_independently_verified'}
 try:
  r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  with Image.open(p) as im:
   im=ImageOps.exif_transpose(im).convert('RGB');im.load();r['size']=list(im.size)
   r['pixel_sha256']=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   a=np.asarray(im.resize((9,8)).convert('L'));r['dhash']=bits(a[:,1:]>a[:,:-1])
   a=np.asarray(im.resize((32,32)).convert('L'),dtype=float);v=dct(dct(a,axis=0),axis=1)[:8,:8];r['phash']=bits(v>np.median(v))
   thumbs[r['id']]=ImageOps.contain(im,(235,145))
  r['previous_eval']=r['sha256'] in oldhash or r['pixel_sha256'] in oldpixel
  r['previous_train']=r['sha256'] in trainhash
  r['public_exact']=r['sha256'] in public;r['status']='ok'
 except Exception as e:r.update(status='error',error=str(e))
 rows.append(r)
dups=defaultdict(list)
for r in rows:
 if r['status']=='ok':dups[r['pixel_sha256']].append(r['id'])
pairs=[]
for a,b in itertools.combinations([r for r in rows if r['status']=='ok'],2):
 ph=(a['phash']^b['phash']).bit_count();dh=(a['dhash']^b['dhash']).bit_count()
 if a['pixel_sha256']!=b['pixel_sha256'] and (ph<=8 or dh<=6):pairs.append({'a':a['id'],'b':b['id'],'phash_distance':ph,'dhash_distance':dh})
ref=read(B/'similarity_review/hashes.json');near_public=[]
for a in rows:
 if a['status']!='ok':continue
 for b in ref:
  ph=(a['phash']^b['phash']).bit_count();dh=(a['dhash']^b['dhash']).bit_count()
  if ph<=8 and dh<=10:near_public.append({'external_id':a['id'],'public_file':b['path'],'phash_distance':ph,'dhash_distance':dh})
summary={'total':len(rows),'unique_pixels':len(dups),'folders':dict(Counter(r['folder'] for r in rows)),'errors':[r for r in rows if r['status']!='ok'],'previous_eval_files':sum(r.get('previous_eval',False) for r in rows),'previous_train_files':sum(r.get('previous_train',False) for r in rows),'exact_duplicate_groups':[v for v in dups.values() if len(v)>1],'near_duplicate_candidate_pairs':len(pairs),'near_public_candidate_pairs':len(near_public)}
report={'summary':summary,'rows':rows,'near_duplicate_candidates':pairs,'near_public_candidates':near_public,'limitations':'Perceptual hash candidates are not confirmed duplicates and miss crops. History comparison uses file SHA256 plus pixels for old evaluation. No image inferred to be an untouched test merely from absence of history. No source files modified.'}
(O/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
font_path=os.environ.get('FONT_PATH')
try:
 font=ImageFont.truetype(font_path,14) if font_path else ImageFont.load_default()
except OSError:
 font=ImageFont.load_default()
for start in range(0,len(rows),40):
 sheet=Image.new('RGB',(1200,1440),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[start:start+40]):
  x=(j%5)*240;y=(j//5)*180
  if r['id'] in thumbs:sheet.paste(thumbs[r['id']],(x,y))
  draw.text((x+3,y+146),f"{r['id']:03d} {r['folder']}",font=font,fill='black')
 sheet.save(O/f'review_{start//40+1}.jpg')
print(json.dumps(summary,ensure_ascii=False,indent=2))

