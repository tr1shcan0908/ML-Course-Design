"""Create a reproducible, exact-duplicate-free FER development split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def inspect(item):
    root, path, label, split = item
    with Image.open(path) as image:
        image.load()
        if image.size != (48, 48) or image.mode != 'L':
            raise ValueError(f'Unexpected image format: {path}: {image.size}/{image.mode}')
        pixels = np.array(image, dtype=np.uint8)
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    return dict(path=path.relative_to(root).as_posix(), label=label, original_split=split,
                pixel_sha256=digest, constant=bool(pixels.min() == pixels.max())), pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/data'))
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--val-fraction', type=float, default=.2)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'split.json').exists():
        raise FileExistsError('A locked split already exists; use a new output directory to regenerate.')
    jobs = []
    for label, name in enumerate(CLASSES):
        jobs.extend((root, p, label, 'train') for p in sorted((root/'train'/name).glob('*.jpg')))
    jobs.extend((root, p, -1, 'test') for p in sorted((root/'test').glob('*.jpg')))
    images = np.lib.format.open_memmap(out/'images.npy', mode='w+', dtype=np.uint8,
                                     shape=(len(jobs), 48, 48))
    rows = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, (row, pixels) in enumerate(pool.map(inspect, jobs)):
            row['index'] = i
            rows.append(row)
            images[i] = pixels
            if (i+1) % 5000 == 0:
                print(f'Scanned {i+1}/{len(jobs)} images ({time.time()-start:.1f}s)', flush=True)
    images.flush()
    groups = defaultdict(list)
    for row in rows:
        if row['original_split'] == 'train':
            groups[row['pixel_sha256']].append(row)
    clean, excluded = [], []
    conflict_groups = 0
    for digest, members in sorted(groups.items()):
        labels = {r['label'] for r in members}
        if len(labels) > 1:
            conflict_groups += 1
            reason = 'conflicting_labels_for_identical_pixels'
        elif members[0]['constant']:
            reason = 'constant_image'
        else:
            clean.append(members[0]['index'])
            excluded.extend(dict(index=r['index'], path=r['path'], reason='same_label_duplicate',
                                 representative=members[0]['path']) for r in members[1:])
            continue
        excluded.extend(dict(index=r['index'], path=r['path'], reason=reason) for r in members)
    train, val = train_test_split(clean, test_size=args.val_fraction, random_state=args.seed,
                                 stratify=[rows[i]['label'] for i in clean])
    train, val = sorted(train), sorted(val)
    train_hashes = {rows[i]['pixel_sha256'] for i in train}
    val_hashes = {rows[i]['pixel_sha256'] for i in val}
    assert not train_hashes & val_hashes
    assert len(train_hashes) == len(train) and len(val_hashes) == len(val)
    test = [r['index'] for r in rows if r['original_split'] == 'test']
    manifest = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows)
    (out/'manifest.jsonl').write_bytes(manifest.encode('utf-8'))
    split = dict(version=1, seed=args.seed, classes=CLASSES, train=train, val=val, test=test,
                 manifest_sha256=hashlib.sha256(manifest.encode('utf-8')).hexdigest(),
                 selection_policy='one representative per nonconstant, label-consistent pixel hash; stratified 80/20',
                 group_key='sha256 of decoded 48x48 grayscale pixels')
    (out/'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'excluded.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding='utf-8')
    all_train_hashes = set(groups)
    summary = dict(original_train=sum(r['original_split']=='train' for r in rows),
                   original_test=len(test), clean_unique=len(clean), train=len(train), val=len(val),
                   train_class_counts={CLASSES[k]:v for k,v in sorted(Counter(rows[i]['label'] for i in train).items())},
                   val_class_counts={CLASSES[k]:v for k,v in sorted(Counter(rows[i]['label'] for i in val).items())},
                   exclusions=dict(Counter(r['reason'] for r in excluded)),
                   conflicting_pixel_groups=conflict_groups, train_val_pixel_overlap=0,
                   test_images_matching_original_train=sum(rows[i]['pixel_sha256'] in all_train_hashes for i in test),
                   manifest_sha256=split['manifest_sha256'], seconds=round(time.time()-start,2))
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
