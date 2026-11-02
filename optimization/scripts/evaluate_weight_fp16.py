from pathlib import Path
import json, hashlib
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

B=Path(__file__).resolve().parents[1]/'outputs/gunggeum_camera'
O=B/'weight_fp16_exp005'; O.mkdir(exist_ok=True)
src=B/'onnx_exp005/efficientnet_b0_exp005_fp32.onnx'
dst=O/'efficientnet_b0_exp005_weight_fp16.onnx'
m=onnx.load(str(src)); casts=[]; converted=[]
for init in m.graph.initializer:
    if init.data_type!=TensorProto.FLOAT: continue
    a=numpy_helper.to_array(init)
    if a.size<64 or not np.isfinite(a).all() or np.abs(a).max()>65504: continue
    name=init.name; stored=name+'__stored_fp16'
    init.CopyFrom(numpy_helper.from_array(a.astype(np.float16),name=stored))
    casts.append(helper.make_node('Cast',[stored],[name],to=TensorProto.FLOAT,name=name+'__restore_fp32'))
    converted.append(name)
nodes=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(casts+nodes)
onnx.checker.check_model(m);onnx.save(m,str(dst))
opt=ort.SessionOptions();opt.intra_op_num_threads=4
s=ort.InferenceSession(str(dst),sess_options=opt,providers=['CPUExecutionProvider'])
manifest=json.loads((B/'browser_inference/full_validation/manifest.json').read_text(encoding='utf-8'))
cm=np.zeros((60,60),dtype=int); correct=[0,0,0]; rows=[]
for i,r in enumerate(manifest['items']):
    x=np.fromfile(B/'browser_inference'/r['tensor'],dtype='<f4').reshape(1,3,320,320)
    y=s.run(None,{'image':x})[0].reshape(-1);assert np.isfinite(y).all()
    top=np.argsort(-y,kind='stable')[:3].tolist();cm[r['label'],top[0]]+=1
    for k in range(3):correct[k]+=int(r['label'] in top[:k+1])
    rows.append({'id':r['id'],'label':r['label'],'truth':r['truth'],'top3':top,'reference_top3':r['reference_top3']})
    if (i+1)%100==0:print(i+1,flush=True)
den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(60),where=den!=0)
report={'scope':'568 Combined development validation, CPU only; browser not tested','method':'FP16 storage for float initializers >=64 elements, Cast to FP32 before original computation; no retraining/calibration, not integer quantization','bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'converted_initializers':len(converted),'correct_top1_top2_top3':correct,'accuracy_percent':[n/568*100 for n in correct],'macro_f1':float(f1.mean()),'top1_reference_agreement':sum(r['top3'][0]==r['reference_top3'][0] for r in rows),'predictions':rows}
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=False),flush=True)
