"""Copy the prepared manifest into a portable dataset; validate every pair."""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    entries = json.loads((ROOT/'prepared/dataset_manifest.json').read_text(encoding='utf-8'))['images']
    dest = ROOT/'dataset_clean'
    if dest.exists():
        raise FileExistsError('Output already exists; refusing to overwrite: '+str(dest))
    rels=[e['relative_path'] for e in entries]
    assert len(entries)==3438 and len(set(rels))==3438
    for rel in rels:
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts: raise ValueError('Invalid path')
    dest.mkdir()
    records=[]; issues=[]; counts=Counter()
    for n,e in enumerate(entries,1):
        rel=Path(e['relative_path']); output=dest/rel
        output.parent.mkdir(parents=True,exist_ok=True)
        try:
            source=Path(e['image_path']); label=Path(e['label_path'])
            shutil.copyfile(source,output)
            shutil.copyfile(label,output.with_suffix('.json'))
            before=digest(source); after=digest(output)
            if before!=after: raise ValueError('Image copy hash mismatch')
            if digest(label)!=digest(output.with_suffix('.json')): raise ValueError('Label copy hash mismatch')
            meta=json.loads(output.with_suffix('.json').read_text(encoding='utf-8-sig'))
            if meta['File_Info']['File_ID']!=output.stem: raise ValueError('File ID mismatch')
            if meta['Annotations_Info']['Building_name_kor']!=rel.parts[0].split('_',1)[1]:
                raise ValueError('Building name mismatch')
            with Image.open(output) as im: im.verify()
            with Image.open(output) as im:
                im.load()
                if im.size!=(1920,1080) or im.mode!='RGB': raise ValueError('Unexpected image shape/mode')
                info=meta['Image_Info']
                if im.size!=(int(info['Image_Width']),int(info['Image_Length'])): raise ValueError('JSON size mismatch')
            counts[rel.parts[0]]+=1
            records.append({'image':rel.as_posix(),'label':rel.with_suffix('.json').as_posix(),
                            'source':e['source'],'sha256':after})
        except Exception as exc:
            issues.append({'file':rel.as_posix(),'error':str(exc)})
        if n%250==0: print(f'Copied and checked {n}/{len(entries)}',flush=True)
    summary={'expected_images':len(entries),'verified_images':len(records),'class_count':len(counts),
             'source_counts':dict(Counter(r['source'] for r in records)),
             'counts_by_folder':dict(counts),'issues':issues,
             'status':'pass' if not issues and len(records)==len(entries) else 'failed',
             'scope':'File copy integrity, full decoding, size/mode, label names and IDs. Not visual label accuracy or near-duplicate/split validation.'}
    (dest/'validation_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps({'status':'unsplit','paths':'relative to dataset_clean','images':records},ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(ROOT/'prepared/replacement_manifest.json',dest/'replacement_manifest.json')
    shutil.copyfile(ROOT/'reports/building_mapping.csv',dest/'building_mapping.csv')
    (dest/'README.md').write_text('# 경복궁 통합 이미지 데이터\n\n이미지 3438장: 정상 FHD 3360장 + QHD에서 변환한 대체본 78장. 원본은 변경하지 않았습니다.\n\nmanifest.json의 경로는 이 폴더 기준 상대 경로입니다. replacement_manifest.json의 원본 경로는 출처 기록이며 실행에 필요하지 않습니다. 전체 dataset_clean 폴더를 옮겨 사용하세요. building_mapping.csv의 서비스 ID는 임시입니다.\n\nvalidation_report.json에서 검증 결과를 확인하세요. 아직 Train/Validation/Test 분할 전이며 유사 이미지 검토가 필요합니다.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='counts_by_folder'},ensure_ascii=False,indent=2))
    if issues: raise SystemExit(1)

if __name__=='__main__': main()
