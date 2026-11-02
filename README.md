# 宮今 카메라 전각 인식

사진에서 경복궁 전각 후보 3개를 반환하고, 사용자가 선택한 전각의 설명 카드로 연결합니다. 담당: 곽도혁.

## 현재 상태
- 대상: 60개 전각, EfficientNet-B0.
- 사용 모델: EXP005 16회차. 가중치 FP16 저장·FP32 계산, 약 8.22MB.
- 개발 검증 568장: **Top-1 92.78% · Top-2 96.83% · Top-3 97.89%**.
- 노트북과 아이폰에서 WebGPU·WASM 실행 확인. 실제 서비스 통합과 독립 현장 시험은 진행 예정입니다.

## 문서
|문서|내용|
|---|---|
|[개발 과정](docs/development.md)|데이터 준비부터 5개·60개 전각 실험까지|
|[실험 결과](docs/results.md)|모델별 비교와 현재 모델 선택 근거|
|[경량화와 모바일](docs/optimization.md)|FP16·INT8 비교 및 휴대폰 실행 결과|
|[실행과 연동](docs/integration.md)|외부 자산, 실행 방법, 전각 ID 연결|
|[파일 안내](docs/file_map.md)|폴더별 내용과 읽는 순서|

코드는 data_preparation, experiments, evaluation, optimization, web, demo에 구분했습니다. experiments는 당시 실험 코드이며 web은 팀 전달용 점검 코드입니다.

## 협업
Issue별 브랜치에서 작업하고 PR 승인 후 Squash merge합니다. 카드 ID 변경은 FE·BE와 함께 검토합니다. 모델·사진·데이터셋은 팀 Drive/S3에서 별도로 관리합니다. 모델 공유 위치는 아직 등록하지 않았습니다.
