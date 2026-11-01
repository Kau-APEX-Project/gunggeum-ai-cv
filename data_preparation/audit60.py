import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

base = Path(__file__).resolve().parent
out = base / 'class60_preparation'
out.mkdir(exist_ok=True)
mp = base / 'splits_v1/manifest.json'
m = json.loads(mp.read_text(encoding='utf-8'))
root = (mp.parent / m['dataset_root']).resolve()
original = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
rows = m['images']
issues = []
group_splits = defaultdict(set)
byclass = defaultdict(list)
seen = set()
for row in rows:
    if row['image'] in seen: issues.append(['duplicate_path',row['image']])
    seen.add(row['image'])
    group_splits[row['group']].add(row['split'])
    if not (root/row['image']).is_file(): issues.append(['missing_image',row['image']])
    try:
        label = json.loads((root/row['label']).read_text(encoding='utf-8-sig'))
        name = label['Annotations_Info']['Building_name_kor']
        if name != row['class'].split('_',1)[1]: issues.append(['name_mismatch',row['image'],name])
        byclass[row['class']].append(dict(row, date=label['Image_Info'].get('Image_Date','UNKNOWN')))
    except Exception as e:
        issues.append(['label_error',row['label'],str(e)])
for g,s in group_splits.items():
    if len(s)>1: issues.append(['group_cross_split',g,sorted(s)])
if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()!=m['dataset_manifest_sha256']:
    issues.append(['dataset_manifest_changed'])
stats=[]
for c,rs in sorted(byclass.items()):
    train=[r for r in rs if r['split']=='train']
    counts=Counter(r['split'] for r in rs)
    for split in ('train','validation','test'):
        if not counts[split]: issues.append(['empty_class_split',c,split])
    dates=Counter(r['date'] for r in train)
    reasons=[]
    if len(train)<20: reasons.append('학습 사진 20장 미만')
    if len(dates)==1: reasons.append('학습 촬영일 1개')
    if min(counts['validation'],counts['test'])<5: reasons.append('내부 검증 또는 시험 5장 미만')
    stats.append({'class':c,'total':len(rs),**dict(counts),'train_dates':dict(dates),
                  'train_groups':len(set(r['group'] for r in train)),
                  'priority_score':(3 if len(train)<20 else 0)+(1 if len(dates)==1 else 0)+(1 if min(counts['validation'],counts['test'])<5 else 0),
                  'reasons':reasons})
ranked=sorted(stats,key=lambda s:(-s['priority_score'],s['train'],s['class']))
excluded=[r['image'] for r in original['images'] if r['image'] not in seen]
report={'classes':len(stats),'images':len(rows),'counts':dict(Counter(r['split'] for r in rows)),
        'groups':len(group_splits),'issues':issues,'excluded':excluded,'class_stats':stats,
        'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
        'limits':'Existence, metadata, recorded group separation checked. No fresh image decoding, byte hash verification, visual label review or new duplicate search.'}
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 60개 전각 점검 및 수집 우선순위','',
       f'사진 {len(rows)}장, 클래스 {len(stats)}개, 그룹 {len(group_splits)}개. 구조 점검 오류 {len(issues)}건.',
       '기존 분할을 변경하지 않음. 이미지 존재·JSON 명칭·그룹 분할을 점검했으며 전체 이미지 재디코딩/육안 검수를 완료했다는 의미는 아님.',
       '우선순위는 사진 수·촬영일 편중·내부 평가 표본 수로 정한 수집 순서이며 실제 모델 오분류 순위가 아님.', '',
       '| 순서 | 전각 | 전체 | 학습 | 검증 | 시험 | 학습 촬영일 수 | 학습 그룹 수 | 수집 이유 |',
       '|---:|---|---:|---:|---:|---:|---:|---:|---|']
for i,s in enumerate(ranked,1):
    lines.append(f'| {i} | {s["class"]} | {s["total"]} | {s["train"]} | {s["validation"]} | {s["test"]} | {len(s["train_dates"])} | {s["train_groups"]} | {", ".join(s["reasons"]) or "외부 촬영 조건 확보"} |')
lines += ['', '## 기존 제외 사진', '정시합/곤녕합 라벨 범위가 불명확하여 기존 분할에서 보류한 3장. 이번에 임의 재편입하지 않음.']+[f'- {p}' for p in excluded]
lines += ['', '## 이번 수집 작업',
          '우선 상위 5개 전각에서 서로 다른 장면을 5장씩 찾아본다. 같은 앨범 연사로 장수만 채우지 않는다. 정답과 이용허락이 명확한 사진을 우선한다.',
          '이미지는 수집대기/전각명 폴더에 넣고 출처기록.txt에 파일명, 원본 페이지 URL, 작가, 라이선스, 촬영일 또는 앨범을 남긴다. 모르는 정보는 미상이라고 적는다.',
          '수집 후 전체 중복과 출처를 확인하여 학습/개발검증/최종시험을 배정한다. 사용자가 임의로 무작위 분할할 필요는 없다.',
          '찾기 힘든 전각은 찾기 어려움으로 기록하고, 잘못된 전각 사진으로 대체하지 않는다.',
          '서비스 ID는 별도 확정이 필요하며 이번 점검 완료에 포함하지 않는다.']
(out/'점검표_수집순서.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['classes','images','counts','groups','issues','excluded']},ensure_ascii=False,indent=2))
print(json.dumps(ranked[:5],ensure_ascii=False,indent=2))
