"""Local perceptual-hash screening, not automatic duplicate/group decisions."""
import csv
import html
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dataset_clean'
OUT=ROOT/'similarity_review'

def bits(values):
    result=0
    for value in values: result=(result<<1)|int(value)
    return result

def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'thumbs').mkdir(exist_ok=True)
    paths=sorted(DATA.rglob('*.jpg'))
    n=32
    k=np.arange(8)[:,None]; x=np.arange(n)[None,:]
    matrix=np.cos(np.pi*(2*x+1)*k/(2*n))
    matrix[0]*=1/np.sqrt(2)
    records=[]
    for i,path in enumerate(paths):
        with Image.open(path) as im:
            grey=im.convert('L')
            arr=np.asarray(grey.resize((32,32),Image.Resampling.LANCZOS),dtype=float)
            low=(matrix@arr@matrix.T).flatten()[1:]
            ph=bits(low>np.median(low))
            small=np.asarray(grey.resize((9,8),Image.Resampling.LANCZOS))
            dh=bits((small[:,1:]>small[:,:-1]).flatten())
            thumb=ImageOps.contain(im.convert('RGB'),(300,180))
            thumb.save(OUT/'thumbs'/f'{i:04d}.jpg',quality=85)
        records.append({'id':i,'path':path.relative_to(DATA).as_posix(),
                        'class':path.relative_to(DATA).parts[0],'phash':ph,'dhash':dh})
        if (i+1)%500==0: print('Hashed',i+1,flush=True)
    pairs={}; nearest={}
    for i,a in enumerate(records):
        best=None
        for j in range(i+1,len(records)):
            b=records[j]; pd=(a['phash']^b['phash']).bit_count()
            if a['class']==b['class']:
                dd=(a['dhash']^b['dhash']).bit_count()
                key=(pd,dd)
                for origin,other in [(i,j),(j,i)]:
                    if origin not in nearest or key<nearest[origin][0]: nearest[origin]=(key,other)
                if pd<=10 and dd<=14:
                    pairs[(i,j)]={'p_distance':pd,'d_distance':dd,'kind':'same_class_threshold'}
            elif pd<=6:
                dd=(a['dhash']^b['dhash']).bit_count()
                if dd<=10: pairs[(i,j)]={'p_distance':pd,'d_distance':dd,'kind':'cross_class_threshold'}
    # A nearest neighbour is only a review candidate, even if it is dissimilar.
    for i,(key,j) in nearest.items():
        pair=tuple(sorted((i,j)))
        if pair not in pairs: pairs[pair]={'p_distance':key[0],'d_distance':key[1],'kind':'nearest_only'}
    rows=[]
    for (i,j),v in pairs.items():
        rows.append({'a_id':i,'b_id':j,'a':records[i]['path'],'b':records[j]['path'],**v,'review':'pending'})
    rows.sort(key=lambda r:(r['p_distance'],r['d_distance'],r['a']))
    with (OUT/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'hashes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'images':len(records),'candidates':len(rows),'by_kind':dict(Counter(r['kind'] for r in rows)),
             'method':'63-bit DCT pHash and 64-bit dHash, plus within-class nearest candidates',
             'warning':'Heuristic distances are not probabilities. Thresholds are screening defaults, not calibrated. Similar architecture can be a false positive. Crops/view changes can be missed. No grouping/deletion/split performed.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>경복궁 유사 사진 후보</title><style>body{font-family:system-ui;margin:24px;background:#f4f5f7}article{background:white;padding:12px;margin:12px 0}img{width:300px;height:180px;object-fit:contain}code{display:block} .pair{display:flex;gap:20px}select{padding:6px}</style>',
           '<h1>유사 사진 후보 — 자동 확정 아님</h1><p>인터넷 사진 없음. 현재 데이터 3,438장만 비교. 같은 전각이어도 별도 구도일 수 있습니다. nearest_only는 가장 가까운 사진을 보여줄 뿐 유사하다는 판정이 아닙니다.</p>',
           '<label>필터 <select onchange="document.querySelectorAll(\'article\').forEach(e=>e.hidden=this.value!==\'all\'&&e.dataset.kind!==this.value)"><option value="all">전체</option><option>same_class_threshold</option><option>cross_class_threshold</option><option>nearest_only</option></select></label>']
    for idx,r in enumerate(rows):
        parts.append(f'<article data-kind="{r["kind"]}"><b>후보 {idx+1} · {r["kind"]} · pHash {r["p_distance"]}, dHash {r["d_distance"]}</b><div class="pair">')
        for side in ['a','b']:
            parts.append(f'<div><img loading="lazy" src="thumbs/{r[side+"_id"]:04d}.jpg"><code>{html.escape(r[side])}</code></div>')
        parts.append('</div></article>')
    (OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    # Overview of 12 highest-ranking candidates for direct visual inspection.
    sheet=Image.new('RGB',(640,12*215),'white'); draw=ImageDraw.Draw(sheet)
    for idx,r in enumerate(rows[:12]):
        y=idx*215
        draw.text((10,y+2),f'Pair {idx+1}: {r["a_id"]} / {r["b_id"]}   p={r["p_distance"]} d={r["d_distance"]}',fill='black')
        for col,side in enumerate(['a','b']):
            with Image.open(OUT/'thumbs'/f'{r[side+"_id"]:04d}.jpg') as im: sheet.paste(im,(col*320+10,y+25))
    sheet.save(OUT/'top12.jpg')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
