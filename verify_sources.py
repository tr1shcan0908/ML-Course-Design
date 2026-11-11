"""Verify archived source files and final submission CSVs using only Python stdlib."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    count = 0
    for task in ('task1', 'task2', 'task3', 'task4'):
        manifest = json.loads((ROOT / task / 'provenance/EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            path = ROOT / task / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected['sha256']:
                raise ValueError('Fingerprint mismatch: ' + task + '/' + name)
            count += 1
    index = json.loads((ROOT / 'submission_index.json').read_text(encoding='utf-8'))
    for entry in index['tasks']:
        task = entry['directory']
        name = 'submission_v9_calibrated.csv' if task == 'task4' else 'outputs/round2_fov/submission.csv' if task == 'task3' else 'reference/submission.csv'
        if hashlib.sha256((ROOT / task / name).read_bytes()).hexdigest() != entry['submission_sha256']:
            raise ValueError('Final CSV mismatch: ' + task)
    print(json.dumps(dict(verified_files=count, final_submission_csvs=4, verified=True)))


if __name__ == '__main__':
    main()
