# MATLAB 시연 실행

EXP005 epoch 16 PyTorch 모델을 Python CPU로 추론하고 MATLAB에서 표시합니다. 웹 ONNX/FP16 추론과는 별도입니다.

저장소 루트의 PowerShell에서:

```powershell
py -3.11 -m venv demo/.venv-inference
& ./demo/.venv-inference/Scripts/python.exe -m pip install -r demo/matlab/requirements.txt
```

공유받은 EXP005 epoch 16 best.pt를 demo/models/exp005_epoch16/best.pt에 배치하세요. 모델은 Git에 올리지 않습니다. 배포 경로는 팀 공유 Drive/S3에서 확인하세요.
MATLAB에서 demo/matlab/camera_demo.m을 실행해 사진을 선택합니다.

Python 단독 검사:
request.json에 실제 사진 경로를 작성합니다.
```json
{"image":"C:/photos/palace.jpg"}
```
```powershell
& ./demo/.venv-inference/Scripts/python.exe demo/matlab/infer_request.py --request request.json
```
요청 파일 옆에 .result.json과 미리보기 .png가 생성됩니다. checkpoint 필드로 다른 위치의 동일 가중치를 지정할 수도 있습니다.

Softmax는 정답률이 아닙니다. 체크포인트의 building_id는 당시 임시 ID이므로 서비스 연결에 사용하지 않습니다.
Python 실행 검사는 MATLAB 화면 자체 검증을 대체하지 않습니다.
