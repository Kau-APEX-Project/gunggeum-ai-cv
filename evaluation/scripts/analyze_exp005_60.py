from pathlib import Path
import json,zipfile
from collections import Counter
b=Path('outputs/gunggeum_camera');o=b/'gpu_results/class60_exp005_2026-10-24';o.mkdir(exist_ok=True)
import sys
# 사용법: python analyze_exp005_60.py <실험 결과 ZIP 경로>
if len(sys.argv) < 2:
    raise SystemExit('실험 결과 ZIP 경로를 인자로 전달하세요.')
p=Path(sys.argv[1])
with zipfile.ZipFile(p) as z:
 for n in z.namelist():assert (o/n).resolve().is_relative_to(o.resolve())
 z.extractall(o)
h=json.loads((o/'runs/exp005/history.json').read_text());base=json.loads((o/'runs/exp005/baseline.json').read_text());best=max([{'epoch':0,**base}]+h,key=lambda r:r['score']);cfg=json.loads((o/'runs/exp005/config.json').read_text(encoding='utf-8'));names={r['index']:r['name_ko'] for r in cfg['classes']}
print('selected',best['epoch'])
for k in ['internal_validation','external_validation']:
 m=best[k];print(k,{x:m[x] for x in ['accuracy','macro_f1_present','top3']})
m=best['external_validation'];print('confusions',[(names[y],names[p],n) for (y,p),n in Counter(zip(m['truth'],m['predictions'])).most_common() if y!=p]);print('perclass',[(names[i],sum(y==i and p==i for y,p in zip(m['truth'],m['predictions'])),n) for i,n in Counter(m['truth']).items()])
(o/'selected_summary.json').write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding='utf-8')

