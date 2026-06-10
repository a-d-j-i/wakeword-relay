#!/usr/bin/env python3
"""
Build a compact binary bundle of background audio clips for in-browser augmentation.

Reads 16 kHz mono WAV files from audioset_16k/ and/or fma_16k/ directories,
extracts fixed-length clips, and packs them into a browser-friendly binary file.

Bundle binary format (all fields little-endian):
    Magic:       4 bytes  "MWWB"
    Version:     uint8    = 1
    SampleRate:  uint16   LE (always 16000)
    ClipSamples: uint32   LE — samples per clip (e.g. 64000 for 4.0 s)
    NumClips:    uint32   LE
    Data:        NumClips × ClipSamples × int16 LE
                 (value / 32768.0 → float32 audio sample)

Usage:
    cd /home/work/wakeword_relay
    python3 web_trainer/tools/build_noise_bundle.py \\
        --input_dirs /vms2/work_tmp/download/audioset_16k \\
                     /vms2/work_tmp/download/fma_16k \\
        --output web_trainer/www/noise.bin \\
        --num_clips 500 \\
        --clip_seconds 4.0

Requires: numpy (already in train/requirements.txt)
"""

import argparse
import random
import struct
import sys
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
MAX_INT16 = 32768.0


def read_wav_mono16k(path):
    """Read a WAV file and return a float32 mono array resampled to 16 kHz."""
    try:
        with wave.open(str(path), 'rb') as wf:
            sr    = wf.getframerate()
            nch   = wf.getnchannels()
            sw    = wf.getsampwidth()
            nf    = wf.getnframes()
            raw   = wf.readframes(nf)
    except Exception:
        return None

    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / MAX_INT16
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None

    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)

    if sr != SAMPLE_RATE:
        ratio   = sr / SAMPLE_RATE
        new_len = int(len(samples) / ratio)
        if new_len < 1:
            return None
        idx = np.arange(new_len) * ratio
        i0  = idx.astype(int)
        i1  = np.minimum(i0 + 1, len(samples) - 1)
        frac = idx - i0
        samples = samples[i0] * (1 - frac) + samples[i1] * frac

    return samples.astype(np.float32)


def extract_clips(samples, clip_len):
    """Extract one random non-silent clip from a samples array."""
    if len(samples) < clip_len:
        return []
    start = random.randint(0, len(samples) - clip_len)
    clip  = samples[start:start + clip_len]
    if np.abs(clip).max() < 0.001:
        return []
    return [clip]


def main():
    p = argparse.ArgumentParser(description='Build a browser-ready background noise bundle')
    p.add_argument('--input_dirs', nargs='+',
                   default=['/vms2/work_tmp/download/audioset_16k',
                            '/vms2/work_tmp/download/fma_16k'],
                   help='Directories containing 16 kHz mono WAV files')
    p.add_argument('--output', default='noise.bin',
                   help='Output path for the .bin bundle')
    p.add_argument('--num_clips', type=int, default=500,
                   help='Number of clips to include (default: 500)')
    p.add_argument('--clip_seconds', type=float, default=4.0,
                   help='Duration of each clip in seconds (default: 4.0)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clip_len = int(args.clip_seconds * SAMPLE_RATE)

    wav_files = []
    for d in args.input_dirs:
        dp = Path(d)
        if not dp.exists():
            print(f'WARNING: directory not found: {d}', file=sys.stderr)
            continue
        found = list(dp.rglob('*.wav'))
        print(f'{d}: {len(found)} WAV files')
        wav_files.extend(found)

    if not wav_files:
        print('ERROR: no WAV files found. Check --input_dirs.', file=sys.stderr)
        sys.exit(1)

    print(f'Total: {len(wav_files)} files. Extracting {args.num_clips} clips…')
    random.shuffle(wav_files)

    clips = []
    for i, path in enumerate(wav_files):
        if len(clips) >= args.num_clips:
            break
        samples = read_wav_mono16k(path)
        if samples is not None:
            clips.extend(extract_clips(samples, clip_len))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(wav_files)} files, {len(clips)} clips…', flush=True)

    if not clips:
        print('ERROR: no valid clips extracted.', file=sys.stderr)
        sys.exit(1)

    random.shuffle(clips)
    clips    = clips[:args.num_clips]
    num_clips = len(clips)

    print(f'\nWriting {num_clips} clips × {clip_len} samples ({args.clip_seconds:.1f} s @ {SAMPLE_RATE} Hz)…')

    out_path = Path(args.output)
    with open(out_path, 'wb') as f:
        f.write(b'MWWB')
        f.write(struct.pack('<B', 1))
        f.write(struct.pack('<H', SAMPLE_RATE))
        f.write(struct.pack('<I', clip_len))
        f.write(struct.pack('<I', num_clips))
        for clip in clips:
            int16 = (np.clip(clip, -1.0, 1.0) * MAX_INT16).astype(np.int16)
            f.write(int16.tobytes())

    size = out_path.stat().st_size
    print(f'\nBundle written: {out_path}')
    print(f'  Size:  {size / 1024 / 1024:.1f} MB')
    print(f'  Clips: {num_clips} × {clip_len} samples = {args.clip_seconds:.1f} s each')


if __name__ == '__main__':
    main()
