from pathlib import Path
import zipfile,json
import sys
# 사용법: python analyze_exp60_002.py <실험 결과 ZIP 경로>
if len(sys.argv) < 2:
    raise SystemExit('실험 결과 ZIP 경로를 인자로 전달하세요.')
p=Path(sys.argv[1]);o=Path('outputs/gunggeum_camera/gpu_results/class60_exp002_2026-10-24');o.mkdir(exist_ok=True)
with zipfile.ZipFile(p) as z:
 for n in z.namelist():
  assert (o/n).resolve().is_relative_to(o.resolve())
 z.extractall(o)
b=json.loads((o/'runs/exp002/baseline.json').read_text());h=json.loads((o/'runs/exp002/history.json').read_text());best=max([{'epoch':0,**b}]+h,key=lambda x:x['score'])
print('BEST',best['epoch']);
for name,r in [('before',b),('after',best)]:
 for k in ['internal_validation','external_validation']:
  m=r[k];print(name,k,{x:m[x] for x in ['accuracy','macro_f1_present','top3']})
config=json.loads((o/'runs/exp002/config.json').read_text(encoding='utf-8'));names={r['index']:r['name_ko'] for r in config['classes']}
from collections import Counter
for name,r in [('before',b),('after',best)]:
 m=r['external_validation'];sup=Counter(m['truth']);print(name,'perclass',[(names[i],sum(y==i and p==i for y,p in zip(m['truth'],m['predictions'])),n) for i,n in sup.items()]);print('confusions',[(names[y],names[p],n) for (y,p),n in Counter(zip(m['truth'],m['predictions'])).most_common() if y!=p])
