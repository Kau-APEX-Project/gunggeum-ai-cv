# 현장 보강 모델 ONNX 변환·경량화 코드

현장 보강 epoch12 모델을 ONNX로 변환하고 FP16·INT8 형식으로 경량화하는 코드입니다. 기존 실험의 성능표와 사진별 예측 기록은 포함하지 않습니다. QAT 추가 학습 코드는 아닙니다.

## 파일

| 파일 | 역할 |
|---|---|
| export_validate.py | PyTorch 모델을 ONNX FP32·가중치 FP16으로 변환하고 동일 입력의 예측 비교 |
| compare_quantization.py | 내부 연산 FP16·가중치 INT8·정적 INT8 변환, 보정 및 평가 |
| requirements.txt | 필요한 라이브러리 |
| .gitignore | 모델·사진·실행 결과의 커밋 제외 |

## 별도 준비 파일

모델과 데이터는 팀 Drive/S3에서 별도로 전달합니다. 공유 주소는 아직 등록하지 않았으므로 담당자에게 요청합니다.

```text
optimization/field_model/
  assets/
    best.pt
    manifest.json
    images/
  export_validate.py
  compare_quantization.py
```

- best.pt: 현장 보강 epoch12의 PyTorch 체크포인트. EXP005 시작 가중치가 아닙니다.
- manifest.json 및 images/: field_expand_gpu 데이터 패키지의 대응 파일. manifest에 기록된 경로·라벨·분할을 유지합니다.
- images는 미리 준비한 RGB 320×320 이미지입니다. 임의의 원본 사진을 복사하는 것으로 대체하지 않습니다.

## 실행

별도 Python 환경에서 아래 순서대로 실행합니다. CPU에서도 실행할 수 있습니다.

```bash
cd optimization/field_model
python -m pip install -r requirements.txt
python export_validate.py
python compare_quantization.py
```

첫 스크립트는 generated/onnx에 FP32·가중치 FP16 모델을 생성합니다. 두 번째는 이를 입력으로 generated/quantization에 나머지 모델과 보정·평가 기록을 생성합니다. 재실행하면 같은 이름의 생성 파일을 덮어쓰므로 필요한 결과는 별도로 보관합니다. 생성물은 Git에서 제외합니다.

## 변환 방식

- 가중치 FP16: 큰 가중치를 FP16으로 저장하고 계산 시 FP32로 복원합니다.
- 내부 연산 FP16: 내부 그래프를 FP16으로 변환하고 입출력은 FP32로 유지합니다. 실제 연산 정밀도는 실행 엔진에 따라 달라질 수 있습니다.
- 가중치 INT8: Conv/Gemm 가중치를 채널별 INT8로 저장하고 복원합니다. 활성값은 FP32입니다.
- 정적 INT8: 학습 이미지로 MinMax 보정 후 Conv/Gemm/MatMul에 QDQ를 적용합니다. 기존 Combined 학습은 클래스당 최대 4장, 현장 학습은 최대 2장을 선택하며 검증 자료와 원본 해시 중복을 검사합니다.

이 폴더는 모델 변환용입니다. 서비스에서 사진을 받아 후보를 표시하는 프론트 추론 모듈은 별도 연동 작업으로 관리합니다. 실제 ONNX 모델은 S3/팀 Drive에 보관합니다.

## 코드 확인 범위

등록용으로 경로를 이 폴더 기준으로 변경하고 외부 코드 조각을 파일 안에 포함했습니다. Python 문법 검사를 통과했습니다. 이 정리본으로 전체 변환·평가를 다시 실행하지는 않았습니다.
