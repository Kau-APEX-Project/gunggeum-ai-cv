# 파일 안내

처음 확인할 때는 README → development → results → optimization 순서로 읽으면 됩니다.

|위치|담긴 내용|확인할 때|
|---|---|---|
|data_preparation/|손상 검사, 대체, ID 점검, 유사 그룹·분할 코드와 목록|데이터 구성 확인|
|experiments/class5/|5개 전각 실험 설정·학습 이력·평가·저장 소스|최초 모델과 보강 과정 확인|
|experiments/class60/|60개 전각 EXP001~005 기록|확대·개선 과정 확인|
|experiments/sampling_caps/|추출 가중치 상한 비교|모델 선택 근거 확인|
|experiments/training_sources/|반환 기록에 없는 학습 소스|학습 구현 확인|
|evaluation/|Top-k, 해상도, 앙상블 결과와 분석 코드|지표·대안 비교|
|optimization/|ONNX·FP16·INT8 코드와 결과|경량화 방법 확인|
|web/|브라우저·모바일 코드, 카드 매핑, 실행 결과|프론트 연결|
|demo/matlab/|MATLAB 시연 코드|발표 시연|

큰 결과 JSON은 상세 확인용입니다. 먼저 docs/results.md의 표를 확인하세요. 사진별 오분류와 미채택 실험도 원본 결과에 포함되어 있습니다. 원본 파일 위치와 해시는 패키지 외부의 SOURCE_MAP.json에 보존했습니다.
