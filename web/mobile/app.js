import * as ort from '../browser/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs';
import {preprocess} from './preprocess.mjs';
import {formatResult,validateMapping} from '../integration/recognizer.mjs';
ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('../browser/node_modules/onnxruntime-web/dist/',location.href).href;
const $=id=>document.getElementById(id);
let session,mapping,provider,modelBytes,preview,report;
const modes=()=> $('mode').value==='wasm'?['wasm']:['webgpu','wasm'];
function lock(on){$('mode').disabled=$('load').disabled=on;$('photo').disabled=on||!session;}
async function fetchOK(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r;}
async function create(ep){if(session){await session.release();session=null;}if(ep==='webgpu'&&!await navigator.gpu?.requestAdapter())throw Error('WebGPU adapter unavailable');session=await ort.InferenceSession.create(modelBytes,{executionProviders:[ep],graphOptimizationLevel:'all'});provider=ep;}
async function infer(x){let y;const input=new ort.Tensor('float32',x,[1,3,320,320]);try{const t=performance.now();y=await session.run({image:input});const result=formatResult(y.logits.data,mapping);return {...result,inference_ms:performance.now()-t};}finally{input.dispose();if(y)Object.values(y).forEach(o=>o.dispose());}}
$('load').onclick=async()=>{const preparationStart=performance.now();lock(true);$('save').disabled=true;report={created:new Date().toISOString(),reported_device:'not supplied',reported_os:'not supplied',user_agent:navigator.userAgent,secure_context:isSecureContext,requested_mode:$('mode').value,model:'exp005_weight_fp16_fp32_compute',ort_version:'1.30.0',model_sha256:'48f165745903ccb7dfa791411f1475874183f868020a4a09ecefe482d81a46c4',events:[],photos:[]};try{
 $('status').textContent='모델 다운로드 중';const t=performance.now();
 modelBytes=await(await fetchOK('../models/efficientnet_b0_exp005_weight_fp16.onnx')).arrayBuffer();report.download_ms=performance.now()-t;report.model_bytes=modelBytes.byteLength;
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',modelBytes)),b=>b.toString(16).padStart(2,'0')).join('');if(digest!==report.model_sha256)throw Error('Model hash mismatch');
 mapping=await(await fetchOK('../integration/class_mapping.json')).json();validateMapping(mapping,'exp005_epoch16');
 for(const ep of modes()){try{const t=performance.now();await create(ep);report.session_create_ms=performance.now()-t;break;}catch(e){report.events.push({stage:'load',provider:ep,error:String(e)});if(ep==='wasm')throw e;}}
 report.preparation_total_ms=performance.now()-preparationStart;report.provider=provider;report.status='ready';$('status').textContent=`준비 완료: ${provider}. 사진을 선택하세요.`;
 }catch(e){report.status='failed';report.error=String(e);if(session){await session.release();session=null;}$('status').textContent=String(e);}finally{lock(false);$('save').disabled=false;$('result').textContent=JSON.stringify(report,null,2);}};
$('photo').onchange=async()=>{const f=$('photo').files[0];if(!f)return;lock(true);try{
 if(f.size>20*1024*1024||!f.size)throw Error('20MB 이하 사진을 선택하세요.');
 $('status').textContent='사진 처리 중';const total=performance.now(),t=performance.now(),x=await preprocess(f),preprocessing_ms=performance.now()-t;
 let first;
 try{first=await infer(x);}catch(e){if(provider!=='webgpu'||report.requested_mode!=='auto')throw e;report.events.push({stage:'inference',provider,error:String(e)});await create('wasm');report.provider=provider;first=await infer(x);}
 const end_to_end_first_ms=performance.now()-total;
 const timings=[];for(let i=0;i<3;i++)timings.push((await infer(x)).inference_ms);
 report.photos.push({bytes:f.size,mime:f.type,provider,preprocessing_ms,first_inference_ms:first.inference_ms,end_to_end_first_ms,repeat_inference_ms:timings,candidates:first.candidates});report.status='completed';
 if(preview)URL.revokeObjectURL(preview);preview=URL.createObjectURL(f);$('preview').src=preview;$('preview').hidden=false;$('candidates').replaceChildren();
 for(const c of first.candidates){const li=document.createElement('li');li.textContent=`${c.name_ko} ${(c.score*100).toFixed(2)}% (${c.card_id??'미연결'})`;$('candidates').append(li);}
 $('status').textContent=`완료: ${provider}. 결과를 다운로드하세요.`;
 }catch(e){report.events.push({stage:'photo',error:String(e)});report.status='photo_failed';$('status').textContent=String(e);}finally{lock(false);$('result').textContent=JSON.stringify(report,null,2);}};
$('save').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download=`camera_mobile_${report.requested_mode}_${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};

