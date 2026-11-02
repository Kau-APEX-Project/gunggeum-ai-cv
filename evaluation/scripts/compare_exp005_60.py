from pathlib import Path
import json,numpy as np
B=Path('outputs/gunggeum_camera/gpu_results');out=[];cms=[]
for e in ['003','005']:
 p=B/f'class60_exp{e}_2026-10-24';r=json.loads((p/'selected_summary.json').read_text(encoding='utf-8'));cfg=json.loads((p/f'runs/exp{e}/config.json').read_text(encoding='utf-8'));small=np.array(cfg['train_counts'])<=37
 cm=sum(np.array(r[k]['confusion_matrix_true_rows']) for k in ['internal_validation','external_validation']);tp=np.diag(cm);sup=cm.sum(1);den=sup+cm.sum(0);f=np.divide(2*tp,den,out=np.zeros(60),where=den>0);rec=tp/sup
 result={'experiment':e,'epoch':r['epoch'],'correct':int(tp.sum()),'n':int(cm.sum()),'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f.mean()),'small_class_count':int(small.sum()),'small_macro_recall':float(rec[small].mean()),'small_correct':int(tp[small].sum()),'small_n':int(sup[small].sum())};out.append(result);cms.append((tp,sup,rec));print(result)
changes=[]
for i,c in enumerate(cfg['classes']):
 if small[i] and cms[0][0][i]!=cms[1][0][i]:changes.append({'class':c['name_ko'],'before':int(cms[0][0][i]),'after':int(cms[1][0][i]),'n':int(cms[1][1][i])})
p=B/'class60_exp005_2026-10-24';(p/'comparison.json').write_text(json.dumps({'metrics':out,'small_class_changes':changes},ensure_ascii=False,indent=2),encoding='utf-8');print(changes)
lines=['# EXP005 적은 전각 샘플링 결과','','선택 epoch16, 총23epoch 실행. 동일 혼합검증568장 비교. 최종시험 아님.','', '|실험|정답|혼합 정확도|혼합60개 Macro F1|소량29개 평균 재현율|','|---|---:|---:|---:|---:|']
for r in out:lines.append(f"|EXP{r['experiment']}|{r['correct']}/{r['n']}|{r['accuracy']:.2%}|{r['macro_f1']:.2%}|{r['small_macro_recall']:.2%}|")
lines+=['','소량 전각은 공유마당+외부 학습장수37장 이하29개. 재현율은 혼합검증에서 클래스별로 계산한 뒤 동일 비중 평균. 검증 표본수가 작은 클래스의 변동에 유의.','EXP005 내부94.52%, 외부77.19%. EXP003 내부92.56%, 외부78.95%. 혼합 성능과 소량전각 지표는 개선되었지만 외부 성능은 개선되지 않음. 출처별 성능 저하도 보존. 단일 실행 결과이며 일반적 개선을 보장하지 않음.','기존 모델 선택 기준으로는 EXP003의 내부/외부 MacroF1 평균이 더 높음. 혼합 지표 중심으로 EXP005를 후보로 삼을 수 있으나 기준 변경을 명시하고 EXP003도 유지할 것.','','|소량 전각|EXP003 정답|EXP005 정답|검증 장수|','|---|---:|---:|---:|']
for r in changes:lines.append(f"|{r['class']}|{r['before']}|{r['after']}|{r['n']}|")
(p/'결과비교.md').write_text('\n'.join(lines),encoding='utf-8')
