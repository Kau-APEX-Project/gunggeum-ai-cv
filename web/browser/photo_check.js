import * as ort from './node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs';
import {preprocess} from './photo_preprocess.mjs';
import {formatResult,validateMapping} from '../integration/recognizer.mjs';
const $=id=>document.getElementById(id);
ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('./node_modules/onnxruntime-web/dist/',location.href).href;
const urls={fp32:'../models/efficientnet_b0_exp005_fp32.onnx',weight_fp16:'../models/efficientnet_b0_exp005_weight_fp16.onnx'};
let sessions={},mapping,provider,report,previewUrl;
const top=a=>a.map((_,i)=>i).sort((i,j)=>a[j]-a[i]).slice(0,3);
async function fetchOK(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: ${r.status}`);return r;}
function busy(on){$('load').disabled=on||!!sessions.fp32;$('test').disabled=true;$('photo').disabled=on||!sessions.fp32;}
async function release(){for(const s of Object.values(sessions))await s.release();sessions={};}
async function infer(model,x){const input=new ort.Tensor('float32',x,[1,3,320,320]);let y;try{const t=performance.now();y=await sessions[model].run({image:input});const logits=Array.from(y.logits.data),ms=performance.now()-t;return {logits,inference_ms:ms,...formatResult(logits,mapping)};}finally{input.dispose();if(y)Object.values(y).forEach(t=>t.dispose());}}
function base(){return {created:new Date().toISOString(),user_agent:navigator.userAgent,provider,model_version:'exp005_epoch16',mapping_version:mapping.mapping_version,scope:'Browser Canvas image preprocessing; comparison, not full validation or independent test',rows:[]};}
async function pair(blob){if(!blob.size||blob.size>20*1024*1024)throw Error('20MB 이하의 이미지가 필요합니다.');const t=performance.now();const x=await preprocess(blob);const preprocessing_ms=performance.now()-t;return {preprocessing_ms,fp32:await infer('fp32',x),weight_fp16:await infer('weight_fp16',x)};}
$('load').onclick=async()=>{busy(true);try{
 mapping=await(await fetchOK('../integration/class_mapping.json')).json();validateMapping(mapping,'exp005_epoch16');
 let fallback_reason=null;
 for(const ep of ['webgpu','wasm']){
  try{
   if(ep==='webgpu'&&!await navigator.gpu?.requestAdapter())throw Error('WebGPU unavailable');
   $('status').textContent=`모델 준비 중: ${ep}`;
   for(const [name,url] of Object.entries(urls))sessions[name]=await ort.InferenceSession.create(url,{executionProviders:[ep],graphOptimizationLevel:'all'});
   provider=ep;break;
  }catch(e){await release();if(ep==='wasm')throw e;fallback_reason=String(e);}
 }
 $('status').textContent=`준비 완료: ${provider}${fallback_reason?' (WASM 대체: '+fallback_reason+')':''}`;
 }catch(e){$('status').textContent=String(e);}finally{busy(false);}};
$('test').onclick=async()=>{busy(true);$('save').disabled=true;report=base();try{
 const refs=await(await fetchOK('fixtures/reference.json')).json();
 for(const [i,r] of refs.entries()){
  $('status').textContent=`사진 파일 비교 ${i+1}/${refs.length}`;
  const p=await pair(await(await fetchOK(r.image)).blob());
  report.rows.push({image:r.image,truth:r.truth,pillow_reference_top3:top(r.logits),...p});
 }
 report.summary={n:report.rows.length,canvas_models_top1_agree:report.rows.filter(r=>top(r.fp32.logits)[0]===top(r.weight_fp16.logits)[0]).length,canvas_models_top3_order_agree:report.rows.filter(r=>JSON.stringify(top(r.fp32.logits))===JSON.stringify(top(r.weight_fp16.logits))).length,weight_fp16_canvas_vs_pillow_top1_agree:report.rows.filter(r=>top(r.weight_fp16.logits)[0]===r.pillow_reference_top3[0]).length};
 report.status='completed';$('result').textContent=JSON.stringify(report.summary,null,2);$('status').textContent=`완료 (${provider}). 결과를 다운로드하세요.`;
 }catch(e){report.status='failed';report.error=String(e);$('status').textContent=String(e);}finally{busy(false);$('save').disabled=false;}};
$('photo').onchange=async()=>{const f=$('photo').files[0];if(!f)return;busy(true);$('save').disabled=true;report=base();$('predictions').replaceChildren();try{
 $('status').textContent='사진 처리 중';const p=await pair(f);report.rows=[{filename:f.name,...p}];report.status='completed';
 if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(f);$('preview').src=previewUrl;$('preview').hidden=false;
 const table=document.createElement('table');for(const cells of [['순위','가중치 FP16 예측','Softmax','카드 ID'],...p.weight_fp16.candidates.map(c=>[c.rank,c.name_ko,(c.score*100).toFixed(2)+'%',c.card_id??'미연결'])]){const tr=table.insertRow();for(const value of cells)tr.insertCell().textContent=String(value);}$('predictions').append(table);
 $('result').textContent=JSON.stringify({provider,preprocessing_ms:p.preprocessing_ms,weight_fp16_inference_ms:p.weight_fp16.inference_ms,fp32_candidates:p.fp32.candidates},null,2);
 $('status').textContent='사진 인식 완료. 정답 라벨이 없으므로 정확도는 계산하지 않습니다.';
 }catch(e){report.status='failed';report.error=String(e);$('status').textContent=String(e);}finally{busy(false);$('save').disabled=false;}};
$('save').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='camera_photo_check_'+Date.now()+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
