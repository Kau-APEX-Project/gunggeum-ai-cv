from pathlib import Path
import json
from collections import Counter,defaultdict
B=Path('outputs/gunggeum_camera');O=B/'external_roles_2026-10-23';O.mkdir(exist_ok=True)
a=json.loads((B/'external_audit_2026-10-23/audit.json').read_text(encoding='utf-8'));rows=a['rows'];p=list(range(len(rows)))
def find(x):
 while p[x]!=x:p[x]=p[p[x]];x=p[x]
 return x
def union(x,y):p[find(y)]=find(x)
for g in a['summary']['exact_duplicate_groups']:
 for x in g[1:]:union(g[0],x)
for pair in a['near_duplicate_candidates']:union(pair['a'],pair['b'])
old=json.loads((B/'external_audit_2026-10-22/audit.json').read_text(encoding='utf-8'))['rows']
flags={old[i]['file']:reason for i,reason in [(120,'target_identity_unclear'),(50,'aerial_promotional_image'),(160,'signboard_detail'),(161,'signboard_detail')]}
groups=defaultdict(list)
for r in rows:groups[find(r['id'])].append(r)
out=[]
for gi,items in enumerate(groups.values()):
 hist_train=any(r['previous_train'] for r in items);hist_dev=any(r['previous_eval'] for r in items)
 train_folder=any(r['folder'].endswith('train') for r in items)
 review=[flags[r['file']] for r in items if r['file'] in flags]
 if hist_train and hist_dev:role='hold';reason='historical_train_dev_overlap'
 elif review:role='hold';reason=';'.join(sorted(set(review)))
 elif hist_dev and train_folder:role='hold';reason='dev_similar_to_train_folder_file_check_history'
 elif hist_train:role='train_existing';reason='previous_training_hash_match'
 elif train_folder:role='hold';reason='train_folder_without_confirmed_training_history'
 else:role='dev_candidate';reason='historical_dev_group' if hist_dev else 'new_limited_external_group'
 # Keep all independent candidate views, suppress only exact pixel copies.
 seen=set()
 for r in sorted(items,key=lambda r:(not r['previous_eval'],r['file'])):
  duplicate=r['pixel_sha256'] in seen;seen.add(r['pixel_sha256'])
  out.append({**r,'group_id':f'ext_{gi:04d}','proposed_role':role,'reason':reason,'exact_duplicate_suppressed':duplicate,'label_review':'pending','enabled_for_training_or_scored_evaluation':False})
assert all(len({r['proposed_role'] for r in out if r['group_id']==g})==1 for g in {r['group_id'] for r in out})
assert sum(r['exact_duplicate_suppressed'] for r in out)==2
counts=Counter(r['proposed_role'] for r in out if not r['exact_duplicate_suppressed'])
per=defaultdict(Counter)
for r in out:
 if not r['exact_duplicate_suppressed']:per[r['class']][r['proposed_role']]+=1
payload={'status':'draft_pending_label_review','source_audit':'external_audit_2026-10-23/audit.json','final_test_count':0,'counts_unique_pixels':dict(counts),'per_class':{k:dict(v) for k,v in per.items()},'images':out,'policy':'Conservative near-candidate connected components; no random repartition of public data. No untouched final test inferred from new file history. All execution disabled until label review.'}
(O/'role_manifest_draft.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 외부 데이터 역할 분리 초안 — 2026-10-23','','원본 및 기존 공유마당 분할은 변경하지 않음. 이미지 복사·이동·삭제·학습·성능 평가는 수행하지 않음.','','## 배정 결과',f'- 원본 {len(out)}장, 완전 중복 2장 제외 {sum(counts.values())}장.',f'- 역할별 장수: {dict(counts)}','- 최종 시험 자료: 0장. 새 파일이라는 이유만으로 미사용 최종 시험으로 간주하지 않음.','- 모든 라벨은 검수 대기이며 실행은 비활성화함. 폴더명을 정답으로 확정하지 않음.','','|전각|기존 학습|개발 검증 후보|보류|','|---|---:|---:|---:|']
for c,v in sorted(per.items()):lines.append(f"|{c}|{v['train_existing']}|{v['dev_candidate']}|{v['hold']}|")
lines+=['','## 다음 작업','- 보류 사진의 정체와 사용 이력 확인. 현판·항공 사진은 일반 외관 사진과 별도 조건으로 처리할지 결정.','- 소량 신규 전각은 우선 개발 검증 후보로 배정함. 신규 학습 보강 효과를 보려면 별도 독립 사진이 필요하며 검증 후보를 무작정 학습으로 옮기지 않음.','- 정답 검수 후 개발 검증 자료로 60개 기준 모델 평가. 같은 사진 그룹은 역할을 넘나들지 않음.','- 과거 학습 사진은 기존 60개 모델 학습에 포함되었다는 뜻이 아니라, 5개 모델 실험에서 사용된 이력임.','- 유사 후보를 보수적으로 묶었으므로 별개 사진도 같은 그룹일 수 있음. 크롭 중복 등 누락 가능성도 있음.','- 전체 14개 전각만 외부 자료가 있어 60개 전각 전체의 외부 성능으로 표현하지 않음.']
(O/'역할분리_초안.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'counts':dict(counts),'per_class':{k:dict(v) for k,v in per.items()}},ensure_ascii=False,indent=2))
