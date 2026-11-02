"""Model-independent PIL dataset; pass the chosen model's transform at runtime."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1]/'splits_v1/manifest.json'

class PalaceDataset:
    def __init__(self, split, manifest=DEFAULT_MANIFEST, dataset_root=None, transform=None):
        if split not in {'train','validation','test'}:
            raise ValueError('split must be train, validation, or test')
        self.manifest_path=Path(manifest).resolve()
        doc=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.root=(Path(dataset_root) if dataset_root is not None else self.manifest_path.parent/doc['dataset_root']).resolve()
        self.transform=transform
        entries=doc['images']
        self.classes=sorted({r['class'] for r in entries})
        self.class_to_index={c:i for i,c in enumerate(self.classes)}
        self.manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()!=doc['dataset_manifest_sha256']:
            raise ValueError('Dataset manifest does not match frozen split')
        with (self.root/'building_mapping.csv').open(encoding='utf-8-sig',newline='') as f:
            mapping=list(csv.DictReader(f))
        lookup={r['dataset_folder']:r for r in mapping}
        if len(lookup)!=len(mapping) or any(c not in lookup for c in self.classes):
            raise ValueError('Missing or duplicate class mapping')
        self.class_mapping=[{'index':i,'class':c,'name_ko':lookup[c]['name_ko'],
                             'service_id':lookup[c]['provisional_service_id'],'id_status':'provisional'}
                            for i,c in enumerate(self.classes)]
        seen=set();groups={}
        for r in entries:
            if r['split'] not in {'train','validation','test'}:raise ValueError('Unknown split')
            if r['image'] in seen:raise ValueError('Repeated image in split manifest')
            seen.add(r['image'])
            if r['group'] in groups and groups[r['group']]!=r['split']:raise ValueError('Group crosses splits')
            groups[r['group']]=r['split']
            for key in ['image','label']:
                path=(self.root/r[key]).resolve()
                if not path.is_relative_to(self.root):raise ValueError('Path escapes dataset root')
                if not path.is_file():raise FileNotFoundError(path)
            if Path(r['image']).parts[0]!=r['class']:raise ValueError('Folder/class mismatch')
        self.samples=[r for r in entries if r['split']==split]
        if not self.samples:raise ValueError('Empty split')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        row=self.samples[index]
        label=json.loads((self.root/row['label']).read_text(encoding='utf-8-sig'))
        target=self.class_to_index[row['class']]
        if label['Annotations_Info']['Building_name_kor']!=self.class_mapping[target]['name_ko']:
            raise ValueError('JSON label does not match class: '+row['label'])
        with Image.open(self.root/row['image']) as im:
            image=ImageOps.exif_transpose(im).convert('RGB')
        if self.transform is not None:image=self.transform(image)
        return image,target

    def export_class_mapping(self,path):
        Path(path).write_text(json.dumps({'split_manifest_sha256':self.manifest_sha256,
                                         'classes':self.class_mapping},ensure_ascii=False,indent=2),encoding='utf-8')
