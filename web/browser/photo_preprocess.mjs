class RecognitionError extends Error { constructor(code,message,cause){ super(message);this.code=code;this.cause=cause;} }
export async function preprocess(blob) {
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

