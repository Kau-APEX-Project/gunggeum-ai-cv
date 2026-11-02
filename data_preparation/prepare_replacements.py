"""Create a non-destructive replacement overlay from matched QHD sources."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from PIL import Image

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    source=a.source.resolve(); out=a.out.resolve()
    if out==source or source in out.parents:
        raise ValueError('Output must be outside source directory')
    out.mkdir(parents=True,exist_ok=True)
    audit=read(a.audit); records=[]; failures=[]
    for error in audit['image_errors']:
        rel=Path(error['file'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Invalid relative path')
        original=source/'FHD_경복궁'/rel
        qhd=source/rel.parent/rel.name.replace('_fhd_','_qhd_')
        try:
            old=read(original.with_suffix('.json')); new=read(qhd.with_suffix('.json'))
            if new['File_Info']['File_ID'].replace('_qhd_','_fhd_')!=old['File_Info']['File_ID']:
                raise ValueError('File ID mismatch')
            for section in ['Annotations_Info','Caption_info','Camera_Intrinsic_Parameter']:
                if old.get(section)!=new.get(section): raise ValueError('Metadata mismatch: '+section)
            for field in ['Image_Date','Image_Pointofview','Image_Device','Camera_lens']:
                if old['Image_Info'].get(field)!=new['Image_Info'].get(field): raise ValueError('Metadata mismatch: '+field)
            size=(int(old['Image_Info']['Image_Width']),int(old['Image_Info']['Image_Length']))
            with Image.open(qhd) as im:
                im.load()
                actual=im.size
                if actual!=(int(new['Image_Info']['Image_Width']),int(new['Image_Info']['Image_Length'])):
                    raise ValueError('QHD dimensions mismatch')
                if actual[0]*size[1]!=actual[1]*size[0]: raise ValueError('Aspect ratio mismatch')
                converted=im.convert('RGB').resize(size,Image.Resampling.LANCZOS)
            dest=out/'replacements'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            converted.save(dest,quality=95,subsampling=0)
            with Image.open(dest) as im:
                im.load()
                if im.size!=size: raise ValueError('Output dimension mismatch')
            record={'relative_path':rel.as_posix(),'source_qhd':str(qhd),
                    'source_sha256':sha(qhd),'original_bad_sha256':sha(original),
                    'output_sha256':sha(dest),'source_size':list(actual),'output_size':list(size),
                    'matching':'file ID, labels, captions, date, viewpoint, camera metadata; not visual identity proof',
                    'conversion':'RGB, LANCZOS resize, JPEG quality=95 subsampling=0'}
            label=copy.deepcopy(old)
            label['Derived_Data_Provenance']=record
            save(dest.with_suffix('.json'),label)
            records.append(record)
        except Exception as e:
            failures.append({'file':rel.as_posix(),'error':str(e)})
    save(out/'replacement_manifest.json',{'records':records,'failures':failures})
    # Effective dataset references originals except for verified replacements.
    replaced={r['relative_path'] for r in records}
    excluded={x['file'] for x in failures}
    dataset=[]
    root=source/'FHD_경복궁'
    for file in sorted(root.rglob('*.jpg')):
        rel=file.relative_to(root).as_posix()
        if rel in excluded: continue
        selected=out/'replacements'/rel if rel in replaced else file
        dataset.append({'relative_path':rel,'image_path':str(selected),
                        'label_path':str(selected.with_suffix('.json')),
                        'source':'qhd_replacement' if rel in replaced else 'original_fhd'})
    save(out/'dataset_manifest.json',{'status':'unsplit','images':dataset})
    print(json.dumps({'requested':len(audit['image_errors']),'replaced':len(records),
                      'failures':failures,'effective_images':len(dataset)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
