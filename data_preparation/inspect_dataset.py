"""Read-only inventory of extracted JPG/JSON data; Python standard library only."""
import argparse
import json
from pathlib import Path
from collections import Counter


def inspect(root):
    if not root.is_dir():
        raise ValueError('Dataset directory does not exist: ' + str(root))
    files = sorted(p for p in root.rglob('*') if p.is_file())
    images = [p for p in files if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    labels = [p for p in files if p.suffix.lower() == '.json']
    image_keys = {p.with_suffix('') for p in images}
    label_keys = {p.with_suffix('') for p in labels}
    counts = Counter(p.relative_to(root).parts[0] for p in images)
    names, dates, views = Counter(), Counter(), Counter()
    issues = []
    for p in labels:
        rel = p.relative_to(root)
        try:
            data = json.loads(p.read_text(encoding='utf-8-sig'))
            name = data.get('Annotations_Info', {}).get('Building_name_kor')
            if not isinstance(name, str) or not name.strip():
                issues.append({'file': str(rel), 'issue': 'missing_building_name'})
                continue
            names[name] += 1
            folder_name = rel.parts[0].split('_', 1)[-1]
            if name.strip() != folder_name:
                issues.append({'file': str(rel), 'issue': 'folder_label_mismatch', 'name': name})
            if data.get('File_Info', {}).get('File_ID') != p.stem:
                issues.append({'file': str(rel), 'issue': 'file_id_mismatch'})
            info = data.get('Image_Info', {})
            dates[str(info.get('Image_Date'))] += 1
            views[str(info.get('Image_Pointofview'))] += 1
        except (ValueError, OSError, AttributeError, TypeError) as e:
            issues.append({'file': str(rel), 'issue': 'json_read_or_schema_error', 'detail': str(e)})
    return {
        'root': str(root), 'image_count': len(images), 'json_count': len(labels),
        'images_by_folder': dict(sorted(counts.items())),
        'labels_by_building': dict(sorted(names.items())),
        'dates': dict(sorted(dates.items())), 'views': dict(sorted(views.items())),
        'images_without_json': [str(p.relative_to(root)) for p in images if p.with_suffix('') not in label_keys],
        'json_without_image': [str(p.relative_to(root)) for p in labels if p.with_suffix('') not in image_keys],
        'issues': issues,
        'limits': 'Does not decode images, detect duplicate photos, establish shooting sessions, or verify archive completeness.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
