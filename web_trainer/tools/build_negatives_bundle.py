#!/usr/bin/env python3
"""
Build a compact binary bundle of negative spectrograms for in-browser training.

Reads from the locally downloaded RaggedMmap negative datasets (produced by
train.py's download_negative_datasets step) and packs a random sample from each
category into a browser-friendly binary file.

Bundle binary format (all fields little-endian):
    Magic:         4 bytes  "MWWN"
    Version:       uint8    = 1
    NumFrames:     uint16   frames per sample (default 160)
    NumFeatures:   uint8    filterbanks (40)
    NumCategories: uint8
    Per category:
      NameLen:    uint8
      Name:       NameLen bytes (UTF-8)
      NumSamples: uint32
      Data:       NumSamples × NumFrames × NumFeatures × uint16 LE
                  (value × 0.0390625 → float32 spectrogram value, matches data.py)

Usage:
    cd /home/work/wakeword_relay
    pip install mmap-ninja          # if not already installed
    python3 web_trainer/tools/build_negatives_bundle.py \\
        --input_dir /vms2/work_tmp/download/negative_datasets \\
        --output web_trainer/www/negatives.bin \\
        --samples_per_category 1000 \\
        --num_frames 160

The output file is git-ignored (listed in .gitignore). Copy it to www/ so the
local HTTP server can serve it at /negatives.bin, or host it remotely.
"""

import argparse
import json
import random
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path


CATEGORIES = {
    'speech': [
        'speech/training/voices_lav_clo_training_mmap',
        'speech/training/voices_lav_mid_training_mmap',
        'speech/training/voices_lav_far_training_mmap',
    ],
    'no_speech': [
        'no_speech/training/fma_medium_mmap',
        'no_speech/training/fsd50k_no_speech_mmap',
        'no_speech/training/wham_train_mmap',
    ],
    'dinner_party': [
        'dinner_party/training/chime6_train_u01_ch1_mmap',
        'dinner_party/training/chime6_train_u02_ch1_mmap',
        'dinner_party/training/fma_medium_mmap',
    ],
}

NUM_FEATURES = 40  # filterbanks — fixed by MicroFrontend config


def main():
    p = argparse.ArgumentParser(description='Build a browser-ready negatives bundle')
    p.add_argument('--input_dir', default='/vms2/work_tmp/download/negative_datasets',
                   help='Path to the downloaded negative_datasets directory')
    p.add_argument('--output', default='negatives.bin',
                   help='Output path for the .bin bundle')
    p.add_argument('--samples_per_category', type=int, default=1000,
                   help='Number of spectrograms to sample per category (default: 1000)')
    p.add_argument('--num_frames', type=int, default=160,
                   help='Frames per sample (must be >= 157 for MixedNet; default: 160)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--upload', action='store_true',
                   help='Upload the bundle to a GitHub Release after building')
    p.add_argument('--repo', default='',
                   help='GitHub repo (owner/name) for upload, e.g. adji/wakeword_relay')
    p.add_argument('--tag', default='bundles',
                   help='Release tag to create or reuse (default: bundles)')
    p.add_argument('--token', default='',
                   help='GitHub personal access token (or set GITHUB_TOKEN env var)')
    args = p.parse_args()

    if args.upload and not args.token:
        import os
        args.token = os.environ.get('GITHUB_TOKEN', '')
    if args.upload and not args.token:
        print('ERROR: --upload requires --token or GITHUB_TOKEN env var', file=sys.stderr)
        sys.exit(1)
    if args.upload and not args.repo:
        print('ERROR: --upload requires --repo owner/name', file=sys.stderr)
        sys.exit(1)

    try:
        from mmap_ninja.ragged import RaggedMmap
    except ImportError:
        print('ERROR: mmap_ninja not installed. Run: pip install mmap-ninja', file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    n = args.samples_per_category
    f = args.num_frames

    if f < 157:
        print(f'WARNING: --num_frames={f} < 157 (MixedNet minimum input). Increasing to 157.')
        f = 157

    # Collect existing mmap paths per category
    resolved = {}
    for cat_name, rel_paths in CATEGORIES.items():
        paths = [input_dir / rp for rp in rel_paths if (input_dir / rp).exists()]
        if paths:
            resolved[cat_name] = paths
        else:
            print(f'WARNING: no data found for category "{cat_name}", skipping')

    if not resolved:
        print('ERROR: no categories found. Check --input_dir.', file=sys.stderr)
        sys.exit(1)

    out = open(args.output, 'wb')
    out.write(b'MWWN')                                   # magic
    out.write(struct.pack('<B', 1))                       # version
    out.write(struct.pack('<H', f))                       # num_frames
    out.write(struct.pack('<B', NUM_FEATURES))            # num_features
    out.write(struct.pack('<B', len(resolved)))           # num_categories

    for cat_name, mmap_paths in resolved.items():
        print(f'[{cat_name}] loading mmaps…', flush=True)
        all_items = []   # list of (mmap_obj, item_index)
        for path in mmap_paths:
            ds = RaggedMmap(path)
            for i in range(len(ds)):
                all_items.append((ds, i))

        sampled = random.sample(all_items, min(n, len(all_items)))
        actual_n = len(sampled)

        name_bytes = cat_name.encode('utf-8')
        out.write(struct.pack('<B', len(name_bytes)))
        out.write(name_bytes)
        out.write(struct.pack('<I', actual_n))

        print(f'[{cat_name}] writing {actual_n} × ({f}, {NUM_FEATURES}) uint16…', flush=True)
        for ds, idx in sampled:
            spec = ds[idx]          # shape (500, 40) uint16
            frame_count = spec.shape[0]
            if frame_count < f:
                # Pad with zeros if (unlikely) the spectrogram is shorter than f frames
                import numpy as np
                pad = np.zeros((f - frame_count, NUM_FEATURES), dtype=np.uint16)
                spec = np.concatenate([spec, pad], axis=0)
            out.write(spec[:f, :].astype('<u2').tobytes())

    out.close()
    size = Path(args.output).stat().st_size
    print(f'\nBundle written: {args.output}')
    print(f'  Size: {size / 1024 / 1024:.1f} MB')
    print(f'  Frames per sample: {f}, features: {NUM_FEATURES}')

    if args.upload:
        _github_upload(Path(args.output), args.repo, args.tag, args.token)


def _github_api(method, url, token, body=None, content_type='application/json'):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github+json')
    req.add_header('X-GitHub-Api-Version', '2022-11-28')
    if data:
        req.add_header('Content-Type', content_type)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'GitHub API {method} {url} → {e.code}: {e.read().decode()}') from e


def _github_upload(path: Path, repo: str, tag: str, token: str):
    api = f'https://api.github.com/repos/{repo}'
    print(f'\nUploading to GitHub release {repo}@{tag}…')

    # Find or create the release
    try:
        release = _github_api('GET', f'{api}/releases/tags/{tag}', token)
        print(f'  Found existing release: {release["html_url"]}')
    except RuntimeError:
        release = _github_api('POST', f'{api}/releases', token, {
            'tag_name': tag,
            'name': f'Training bundles ({tag})',
            'body': 'Pre-built negatives and noise bundles for the web trainer.',
            'prerelease': True,
        })
        print(f'  Created release: {release["html_url"]}')

    # Delete existing asset with the same name (so re-runs work cleanly)
    for asset in release.get('assets', []):
        if asset['name'] == path.name:
            print(f'  Replacing existing asset: {asset["name"]}')
            _github_api('DELETE', f'{api}/releases/assets/{asset["id"]}', token)
            break

    # Upload
    upload_url = release['upload_url'].split('{')[0]  # strip {?name,label} template
    data = path.read_bytes()
    req = urllib.request.Request(
        f'{upload_url}?name={path.name}',
        data=data, method='POST')
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github+json')
    req.add_header('Content-Type', 'application/octet-stream')
    req.add_header('Content-Length', str(len(data)))
    print(f'  Uploading {len(data) / 1024 / 1024:.1f} MB…', end=' ', flush=True)
    with urllib.request.urlopen(req) as resp:
        asset = json.loads(resp.read())
    print('done.')
    print(f'\n  Download URL: {asset["browser_download_url"]}')
    print('  Paste this URL into the Train tab → Negatives → "Load from URL"')


if __name__ == '__main__':
    main()
