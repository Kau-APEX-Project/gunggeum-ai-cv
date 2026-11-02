// Browser integration draft. Inject the team's onnxruntime-web instance.
export class RecognitionError extends Error {
  constructor(code, message, cause) { super(message, {cause}); this.name='RecognitionError'; this.code=code; }
}
export function validateMapping(mapping, modelVersion) {
  if(mapping.model_version!==modelVersion || !Array.isArray(mapping.classes) || mapping.classes.length!==60 ||
    mapping.classes.some((c,i)=>c.class_index!==i || typeof c.name_ko!=='string' ||
      (c.card_id!==null && (c.mapping_status!=='confirmed' || !/^b-\d+$/.test(c.card_id)))))
    throw new RecognitionError('INVALID_MAPPING','모델 버전·60개 클래스 순서·확정 카드 ID를 확인하세요.');
}
export function formatResult(logits, mapping) {
  if(logits.length!==mapping.classes.length || Array.from(logits).some(v=>!Number.isFinite(v)))
    throw new RecognitionError('INVALID_OUTPUT','모델 출력이 클래스 목록과 맞지 않습니다.');
  const max=Math.max(...logits), exp=Array.from(logits,v=>Math.exp(v-max)), sum=exp.reduce((a,b)=>a+b,0);
  const indices=Array.from(logits,(_,i)=>i).sort((a,b)=>logits[b]-logits[a] || a-b).slice(0,3);
  return {schema_version:'0.1',model_version:mapping.model_version,mapping_version:mapping.mapping_version,status:'ok',
    candidates:indices.map((i,r)=>({rank:r+1,class_index:i,name_ko:mapping.classes[i].name_ko,
      score:exp[i]/sum,card_id:mapping.classes[i].card_id,mapping_status:mapping.classes[i].mapping_status}))};
}
async function preprocess(blob) {
  let bitmap;
  try { bitmap=await createImageBitmap(blob,{imageOrientation:'from-image'}); }
  catch(e) { throw new RecognitionError('IMAGE_DECODE_FAILED','사진을 읽을 수 없습니다.',e); }
  try {
    const canvas=document.createElement('canvas');canvas.width=canvas.height=320;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});
    if(!ctx) throw new Error('Canvas 2D unavailable');
    ctx.fillStyle='rgb(124,116,104)';ctx.fillRect(0,0,320,320);
    const ratio=Math.min(320/bitmap.width,320/bitmap.height),w=Math.round(bitmap.width*ratio),h=Math.round(bitmap.height*ratio);
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
    ctx.drawImage(bitmap,Math.round((320-w)/2),Math.round((320-h)/2),w,h);
    const rgba=ctx.getImageData(0,0,320,320).data,x=new Float32Array(3*320*320),mean=[.485,.456,.406],std=[.229,.224,.225];
    for(let i=0;i<320*320;i++)for(let c=0;c<3;c++) x[c*320*320+i]=(rgba[i*4+c]/255-mean[c])/std[c];
    return x;
  } finally { bitmap.close(); }
}
export function createRecognizer({ort,modelUrl,mappingUrl,wasmPaths,modelVersion='exp005_epoch16',maxBytes=20*1024*1024}) {
  let session=null,mapping=null,busy=false;
  return {
    async load() {
      if(busy)throw new RecognitionError('BUSY','이전 작업이 진행 중입니다.');
      if(session)return;
      busy=true;
      try {
        const response=await fetch(mappingUrl);
        if(!response.ok)throw new Error('Mapping HTTP '+response.status);
        const next=await response.json();validateMapping(next,modelVersion);
        ort.env.wasm.numThreads=1;
        if(wasmPaths)ort.env.wasm.wasmPaths=wasmPaths;
        session=await ort.InferenceSession.create(modelUrl,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
        mapping=next;
      }catch(e){throw new RecognitionError('MODEL_LOAD_FAILED','모델 또는 클래스 목록을 준비하지 못했습니다.',e);}
      finally{busy=false;}
    },
    async predict(imageBlob) {
      if(busy)throw new RecognitionError('BUSY','이전 작업이 진행 중입니다.');
      if(!session)throw new RecognitionError('NOT_READY','load()를 먼저 완료하세요.');
      if(!(imageBlob instanceof Blob)||!imageBlob.size||imageBlob.size>maxBytes)
        throw new RecognitionError('INVALID_IMAGE','비어 있지 않은 20MB 이하 이미지 Blob/File이 필요합니다.');
      busy=true;let input,outputs;
      try {
        const x=await preprocess(imageBlob);
        input=new ort.Tensor('float32',x,[1,3,320,320]);
        outputs=await session.run({image:input});
        if(!outputs.logits)throw new Error('Missing logits output');
        return formatResult(outputs.logits.data,mapping);
      }catch(e){if(e instanceof RecognitionError)throw e;throw new RecognitionError('INFERENCE_FAILED','사진 인식 실행에 실패했습니다.',e);}
      finally{input?.dispose?.();if(outputs)for(const tensor of Object.values(outputs))tensor.dispose?.();busy=false;}
    },
    async dispose() {
      if(busy)throw new RecognitionError('BUSY','작업이 끝난 뒤 해제하세요.');
      busy=true;
      try{if(session)await session.release();}finally{session=null;mapping=null;busy=false;}
    }
  };
}
