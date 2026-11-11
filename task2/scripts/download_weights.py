"""Download only the public pretrained weights cited by the selected projects."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

SOURCES = {
    'emotieff_b0': ('https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b0_8_best_vgaf.pt',
                    'enet_b0_8_best_vgaf.pt', 'https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684'),
    'emotieff_b2': ('https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b2_7.pt',
                    'enet_b2_7.pt', 'https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684'),
    'convnext_small22k': ('https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth',
                         'convnext_small_22k_224.pth', 'https://github.com/facebookresearch/ConvNeXt'),
    'ir50': ('https://drive.usercontent.google.com/download?id=1FV8kUSeVbZ815iWt-YIYiQrCDChrhO2G&export=download&authuser=0&confirm=t',
             'ir50_msceleb.pth', 'https://github.com/birdwcp/QCS'),
    'convnext22k': ('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth',
                    'convnext_tiny_22k_224.pth', 'https://github.com/yelboudouri/EmoNeXt'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=list(SOURCES), required=True)
    args = parser.parse_args()
    url, filename, provenance = SOURCES[args.name]
    directory = Path(__file__).resolve().parents[1] / 'artifacts' / 'weights'
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    if dest.exists():
        print(f'Already exists: {dest.name} ({dest.stat().st_size} bytes)', flush=True)
        return
    temp = dest.with_suffix('.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'FER-research-weight-download/1.0'})
    digest = hashlib.sha256()
    total = 0
    started = time.time()
    with urllib.request.urlopen(request, timeout=45) as response:
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise RuntimeError('Source returned an HTML page instead of a checkpoint.')
        with temp.open('wb') as file:
            while chunk := response.read(4*1024*1024):
                file.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                print(f'{filename}: {total/2**20:.1f} MiB', flush=True)
    if total < 10*1024*1024:
        raise RuntimeError(f'Checkpoint unexpectedly small: {total}; partial file retained for diagnosis.')
    temp.replace(dest)
    metadata = dict(source=url, provenance=provenance, filename=filename, bytes=total,
                    sha256=digest.hexdigest(), seconds=round(time.time()-started,2))
    dest.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
