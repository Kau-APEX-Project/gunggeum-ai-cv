"""Build review-only similarity components. These are not verified sessions."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent/'similarity_review'

def main():
    records=json.loads((ROOT/'hashes.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((ROOT/'candidates.csv').open(encoding='utf-8-sig')))
    parent=list(range(len(records)))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    edges=[]; flagged=set()
    for r in rows:
        a,b=int(r['a_id']),int(r['b_id'])
        if r['kind']=='cross_class_threshold': flagged.update([a,b])
        if r['kind']=='same_class_threshold' and int(r['p_distance'])<=6 and int(r['d_distance'])<=8:
            x,y=find(a),find(b)
            if x!=y:parent[y]=x
            edges.append((a,b))
    buckets=defaultdict(list)
    for r in records:buckets[find(r['id'])].append(r)
    groups=[]
    for members in sorted(buckets.values(),key=lambda g:min(r['id'] for r in g)):
        pd=dd=0
        for i,a in enumerate(members):
            for b in members[i+1:]:
                pd=max(pd,(a['phash']^b['phash']).bit_count())
                dd=max(dd,(a['dhash']^b['dhash']).bit_count())
        groups.append({'group_id':f'candidate_{min(r["id"] for r in members):04d}',
                       'class':members[0]['class'],'size':len(members),
                       'members':[r['path'] for r in members],'image_ids':[r['id'] for r in members],
                       'max_pair_phash':pd,'max_pair_dhash':dd,
                       'chain_review':pd>6 or dd>8,
                       'label_review':any(r['id'] in flagged for r in members),
                       'status':'unreviewed_not_split_ready'})
    def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    save('group_draft.json',{'thresholds':{'phash':6,'dhash':8},'groups':groups,
                           'warning':'Singletons are not proven independent. Weak candidates and shooting context still need review.'})
    lookup={r['id']:g['group_id'] for g in groups for r in records if r['id'] in g['image_ids']}
    boundary=[r for r in rows if r['kind']=='same_class_threshold' and lookup[int(r['a_id'])]!=lookup[int(r['b_id'])]]
    save('boundary_pairs.json',boundary)
    summary={'images':len(records),'candidate_groups':len(groups),
             'multi_image_groups':sum(g['size']>1 for g in groups),
             'singleton_groups':sum(g['size']==1 for g in groups),
             'largest_group':max(g['size'] for g in groups),
             'chain_review_groups':sum(g['chain_review'] for g in groups),
             'remaining_boundary_pairs':len(boundary),'label_review_images':len(flagged),
             'status':'draft_not_split_ready'}
    save('group_summary.json',summary)
    selected=sorted([g for g in groups if g['size']>1],key=lambda g:(-g['size'],g['group_id']))[:6]
    height=sum(30+((g['size']+3)//4)*165 for g in selected)
    sheet=Image.new('RGB',(960,height),'white');draw=ImageDraw.Draw(sheet);y=0
    for g in selected:
        draw.text((8,y+4),f'{g["group_id"]}  n={g["size"]}  max p={g["max_pair_phash"]} d={g["max_pair_dhash"]}',fill='black');y+=30
        for idx,i in enumerate(g['image_ids']):
            with Image.open(ROOT/'thumbs'/f'{i:04d}.jpg') as im:
                im.thumbnail((230,140));sheet.paste(im,((idx%4)*240,y+(idx//4)*165))
            draw.text(((idx%4)*240,y+(idx//4)*165+142),str(i),fill='black')
        y+=((g['size']+3)//4)*165
    sheet.save(ROOT/'largest_groups.jpg')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
