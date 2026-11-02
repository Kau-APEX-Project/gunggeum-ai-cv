"""Reproducible group-aware split draft; not an independent field test."""
import csv
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT=Path(__file__).resolve().parent
REVIEW=ROOT/'similarity_review'
OUT=ROOT/'splits_draft'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-nearest',action='store_true')
    args=parser.parse_args()
    out=ROOT/'splits_v1' if args.include_nearest else OUT
    records=json.loads((REVIEW/'hashes.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((REVIEW/'candidates.csv').open(encoding='utf-8-sig')))
    parent=list(range(len(records)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    quarantine=set()
    for r in rows:
        a,b=int(r['a_id']),int(r['b_id'])
        if r['kind']=='cross_class_threshold':quarantine.update([a,b])
        if r['kind']=='same_class_threshold' or (args.include_nearest and r['kind']=='nearest_only'):
            x,y=find(a),find(b)
            if x!=y:parent[y]=x
    groups=defaultdict(list)
    for r in records:
        if r['id'] not in quarantine:groups[find(r['id'])].append(r)
    classes=defaultdict(list)
    for gid,members in groups.items():classes[members[0]['class']].append((gid,members))
    ratios=[.7,.15,.15]; names=['train','validation','test']; assignment={}; stats=[]
    rng=random.Random(20261010)
    for cls,gs in sorted(classes.items()):
        if len(gs)<3:raise ValueError('Not enough groups: '+cls)
        total=sum(len(m) for _,m in gs);targets=[total*r for r in ratios]
        best=None
        for trial in range(300):
            order=list(gs);rng.shuffle(order)
            counts=[0,0,0];ng=[0,0,0];chosen={}
            for idx,(gid,members) in enumerate(order):
                empty=[s for s in range(3) if ng[s]==0]
                if len(order)-idx==len(empty):allowed=empty
                else:allowed=list(range(3))
                selected=max(allowed,key=lambda s:targets[s]-counts[s])
                counts[selected]+=len(members);ng[selected]+=1;chosen[gid]=selected
            score=sum(((counts[s]-targets[s])/total)**2 for s in range(3))
            if best is None or score<best[0]:best=(score,chosen,counts,ng)
        assignment.update(best[1])
        stats.append({'class':cls,'images':total,'groups':len(gs),'largest_group':max(len(m) for _,m in gs),
                      **{names[s]:best[2][s] for s in range(3)}})
    output=[]
    for gid,members in groups.items():
        for r in members:output.append({'image_id':r['id'],'image':r['path'],
                                       'label':str(Path(r['path']).with_suffix('.json')).replace('\\','/'),
                                       'class':r['class'],'group':f'group_{gid:04d}','split':names[assignment[gid]]})
    output.sort(key=lambda r:r['image_id'])
    lookup={r['image_id']:r for r in output}
    leaked=[]; unchecked=[]
    for r in rows:
        a,b=int(r['a_id']),int(r['b_id'])
        if a not in lookup or b not in lookup:continue
        if lookup[a]['split']!=lookup[b]['split']:
            if r['kind']=='same_class_threshold':leaked.append(r)
            elif r['kind']=='nearest_only':unchecked.append(r)
    assert not leaked
    assert len(output)+len(quarantine)==len(records)
    assert len({r['image'] for r in output})==len(output)
    if args.include_nearest:assert not unchecked
    out.mkdir(exist_ok=True)
    def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    save('manifest.json',{'status':'internal_benchmark_v1' if args.include_nearest else 'draft_pending_visual_boundary_review','dataset_root':'../dataset_clean',
                         'grouping_policy':'all same-class candidates including nearest-only; conservative overgrouping' if args.include_nearest else 'threshold pairs only',
                         'candidate_sha256':hashlib.sha256((REVIEW/'candidates.csv').read_bytes()).hexdigest(),
                         'dataset_manifest_sha256':hashlib.sha256((ROOT/'dataset_clean/manifest.json').read_bytes()).hexdigest(),
                         'seed':20261010,'target_ratios':dict(zip(names,ratios)),'images':output})
    save('label_review_exclusions.json',[records[i] for i in sorted(quarantine)])
    save('nearest_cross_split_review.json',unchecked)
    with (out/'class_counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    summary={'status':'internal_benchmark_v1' if args.include_nearest else 'draft_not_training_ready','images':len(output),'classes':len(classes),'groups':len(groups),
             'split_counts':dict(Counter(r['split'] for r in output)),
             'label_review_excluded_images':len(quarantine),'known_threshold_pairs_crossing_splits':len(leaked),
             'nearest_only_pairs_crossing_splits':len(unchecked),
             'limitations':'0 known pair leakage is only for the selected hash thresholds. It does not prove session independence or eliminate missed near duplicates. Review boundaries and large components before use.'}
    save('summary.json',summary)
    if args.include_nearest:
        save('groups.json',[{'group':f'group_{gid:04d}','class':members[0]['class'],
                            'split':names[assignment[gid]],'members':[r['path'] for r in members]}
                           for gid,members in sorted(groups.items())])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
