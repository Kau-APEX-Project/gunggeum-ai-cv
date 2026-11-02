import json
from pathlib import Path
from collections import Counter
p=Path('outputs/gunggeum_camera/external_baseline_2026-10-23');r=json.loads((p/'results.json').read_text(encoding='utf-8'))
for name,v in sorted(r['image_level']['per_class'].items(),key=lambda x:x[1]['correct']/x[1]['n']):
 c=Counter(x['top3'][0] for x in r['predictions'] if x['label']==name and not x['top1_correct'])
 print(name, f"{v['correct']}/{v['n']}",dict(c))
q=Path('outputs/gunggeum_camera/external_roles_2026-10-23/user_label_policy_20261023.json')
q.write_text(json.dumps({'label_policy':'accept_user_folder_labels_as_ground_truth','authorized_by':'user instruction 2026-10-23','full_manual_review_required':False,'review_strategy':'only low Top1 classes and confusion pairs on request','historical_results':'preserve existing 180-image provisional evaluation; do not silently change its sample set','note':'Label acceptance does not override duplicate grouping, previous training history, or train/evaluation separation. Previous hold records require role reassessment, not mandatory user label review.'},ensure_ascii=False,indent=2),encoding='utf-8')
