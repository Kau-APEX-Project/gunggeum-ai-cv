"""Inspect images without modifying originals. Requires Pillow."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def inspect(root):
    if not root.is_dir():
        raise ValueError('Missing data directory: ' + str(root))
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError('No images found')
    sizes, modes = Counter(), Counter()
    hashes = defaultdict(list)
    dates = defaultdict(Counter)
    errors, metadata_errors, dimension_mismatches = [], [], []
    for i, path in enumerate(paths, 1):
        rel = path.relative_to(root).as_posix()
        try:
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[digest.hexdigest()].append(rel)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                sizes[f'{width}x{height}'] += 1
                modes[im.mode] += 1
        except Exception as e:
            errors.append({'file': rel, 'error': str(e)})
            continue
        try:
            # Known dataset uses .json alongside each image.
            data = json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
            info = data['Image_Info']
            date = str(info.get('Image_Date') or 'unknown')
            dates[path.relative_to(root).parts[0]][date] += 1
            expected = (int(info['Image_Width']), int(info['Image_Length']))
            if expected != (width, height):
                dimension_mismatches.append({'file': rel, 'actual': [width, height], 'json': list(expected)})
        except Exception as e:
            metadata_errors.append({'file': rel, 'error': str(e)})
        if i % 250 == 0:
            print(f'Checked {i}/{len(paths)}', flush=True)
    groups = [v for v in hashes.values() if len(v) > 1]
    return {'image_count': len(paths), 'decoded_count': sum(sizes.values()),
            'dimensions': dict(sizes), 'color_modes': dict(modes),
            'image_errors': errors, 'metadata_errors': metadata_errors,
            'dimension_mismatches': dimension_mismatches,
            'exact_duplicate_groups': groups,
            'cross_folder_duplicate_groups': [g for g in groups if len({p.split('/')[0] for p in g}) > 1],
            'dates_by_folder': {k: dict(sorted(v.items())) for k, v in sorted(dates.items())},
            'limits': ['SHA256 identifies identical file bytes only; not recompressed or visually similar images.',
                       'A date is not a verified shooting session. Do not use random splits before similarity review.',
                       'No originals were changed or deleted.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    result = inspect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {k: result[k] for k in ['image_count', 'decoded_count', 'dimensions', 'color_modes']}
    summary.update({k: len(result[k]) for k in ['image_errors', 'metadata_errors', 'dimension_mismatches', 'exact_duplicate_groups', 'cross_folder_duplicate_groups']})
    summary['folders_with_one_date'] = sum(len(v) == 1 for v in result['dates_by_folder'].values())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Report:', args.out)
