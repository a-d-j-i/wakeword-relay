#!/usr/bin/env python3
"""Fetch real speech as 16 kHz mono WAVs — real-speech NEGATIVES for the wake-word
pipeline (distillation teacher, DISTILLATION.md B4; and optionally the student).

Why: the teacher's default negatives are FMA music + AudioSet ambient, with NO
speech, so it fires on any speech in the target language (observed for Spanish: it
reacts to "mi mamá me mima"). Feeding real non-wake speech as negatives teaches it
that speech != the wake word. Real public corpora are allowed by the "just Piper +
public corpora" rule.

LANGUAGE-VERSATILE: pass --lang <code> and the script pulls from a per-language list
of corpora known to work here (see LANG_SOURCES). Add a language by extending that
dict, or override entirely with one or more --source dataset:config on the CLI.

Corpus constraints (learned the hard way):
  * `datasets` 5.x REMOVED loading-script support -> any dataset that ships a .py
    loader FAILS (mozilla-foundation/common_voice_*, fsicoli mirror,
    facebook/multilingual_librispeech). Only parquet/tar-native repos work.
  * We avoid gated repos so no Hugging Face login / terms acceptance is needed.
  * `datasets` 5.x decodes audio via `torchcodec`, which this venv intentionally
    does NOT install. So we read the RAW audio bytes (Audio(decode=False)) and
    decode them ourselves with soundfile (safe here: this script never imports
    TensorFlow, so the libsndfile/TF-malloc conflict that bans soundfile elsewhere
    does not apply). soundfile handles WAV/FLAC/OGG/MP3.

Verified-good sources: google/fleurs (per-lang configs like es_419, fr_fr, de_de,
en_us; ~a few k clips/lang) and facebook/voxpopuli (plain ISO configs es/fr/de/…;
much larger, European-Parliament speech).

Usage (run in train/venv):
    python train/tools/fetch_speech.py --lang es --out /vms2/work_tmp/download/es_speech_16k --n 8000
    python train/tools/fetch_speech.py --source google/fleurs:fr_fr --source facebook/voxpopuli:fr \
        --out /vms2/work_tmp/download/fr_speech_16k --n 8000

Re-runnable: numbering continues from existing files, so a second run (or a second
source) accumulates rather than overwriting; stops once the dir holds --n clips.
run.sh auto-includes <download>/<lang>_speech_16k in the teacher's --negatives_dirs.
"""

import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.io import wavfile

SR = 16000

# Per-language corpora, tried in order. Each entry is (hf_dataset_id, config).
# Only parquet/tar-native + ungated datasets belong here (see module docstring).
# FLEURS configs are <iso>_<REGION> (es_419, fr_fr, de_de, en_us, pt_br, it_it …);
# VoxPopuli configs are the plain ISO code and cover EU-Parliament languages.
LANG_SOURCES = {
    "es": [("google/fleurs", "es_419"), ("facebook/voxpopuli", "es")],
    "en": [("google/fleurs", "en_us"), ("facebook/voxpopuli", "en")],
    "fr": [("google/fleurs", "fr_fr"), ("facebook/voxpopuli", "fr")],
    "de": [("google/fleurs", "de_de"), ("facebook/voxpopuli", "de")],
    "it": [("google/fleurs", "it_it"), ("facebook/voxpopuli", "it")],
    "pt": [("google/fleurs", "pt_br"), ("facebook/voxpopuli", "pt")],
    "pl": [("google/fleurs", "pl_pl"), ("facebook/voxpopuli", "pl")],
    "nl": [("google/fleurs", "nl_nl"), ("facebook/voxpopuli", "nl")],
}


def hf_cache_dir() -> Path:
    """Where `datasets` stashes downloaded shards. Same logic on Linux/macOS/Windows:
    HF_HOME (or HF_DATASETS_CACHE) if set, else ~/.cache/huggingface. Point HF_HOME
    at any drive to relocate — e.g. on Windows: set HF_HOME=D:\\hf_cache."""
    if v := os.environ.get("HF_DATASETS_CACHE"):
        return Path(v)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _dirsize(root: Path) -> int:
    total = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


class Heartbeat:
    """Background liveness ping: WAVs written + how fast the HF cache is growing, so
    long "0 wavs" stretches (a shard downloading) still show MB/s and don't look
    frozen. Watching the cache dir works the same on Windows as on Linux.
    """

    def __init__(self, out: Path, every: float = 5.0):
        self.out = out
        self.every = every
        self.cache = hf_cache_dir()
        self.stage = "connecting / downloading first shard"
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _disk(self):
        wavs = list(self.out.glob("*.wav"))
        mb = sum(w.stat().st_size for w in wavs) / 1e6 if wavs else 0.0
        return len(wavs), mb

    def _run(self):
        prev = _dirsize(self.cache)
        while not self._stop.wait(self.every):
            el = int(time.monotonic() - self._t0)
            n, mb = self._disk()
            cur = _dirsize(self.cache)
            rate = (cur - prev) / self.every / 1e6  # MB/s into the HF cache
            prev = cur
            dl = f", cache +{rate:.0f} MB/s" if rate > 0.5 else ""
            print(f"  … alive {el}s | {self.stage} | {n} wavs on disk ({mb:.0f} MB{dl})",
                  flush=True)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def decode_to_16k_mono(raw: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes -> mono float32 [-1,1] @ 16 kHz.

    soundfile (libsndfile) handles WAV/FLAC/OGG/MP3 without torchcodec. Falls back
    to scipy for plain WAV in the unlikely event soundfile is missing.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except ImportError:
        sr, data = wavfile.read(io.BytesIO(raw))
        data = data.astype(np.float32)
        if data.dtype == np.int16 or np.issubdtype(data.dtype, np.integer):
            data = data / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != SR:
        data = scipy.signal.resample(data, round(len(data) * SR / sr)).astype(np.float32)
    return data


def parse_sources(args) -> list[tuple[str, str]]:
    if args.source:
        out = []
        for s in args.source:
            if ":" not in s:
                raise SystemExit(f"--source must be dataset:config, got {s!r}")
            ds, cfg = s.split(":", 1)
            out.append((ds, cfg))
        return out
    if args.lang not in LANG_SOURCES:
        raise SystemExit(
            f"No built-in sources for --lang {args.lang!r}. Known: "
            f"{', '.join(sorted(LANG_SOURCES))}. Add it to LANG_SOURCES or pass "
            f"--source dataset:config (e.g. --source google/fleurs:{args.lang}_xx)."
        )
    return LANG_SOURCES[args.lang]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for 16 kHz WAVs")
    ap.add_argument("--n", type=int, default=8000, help="total clips to reach in --out")
    ap.add_argument("--lang", default="es",
                    help=f"language code; picks built-in sources ({', '.join(sorted(LANG_SOURCES))})")
    ap.add_argument("--source", action="append", default=None,
                    help="override sources: dataset:config (repeatable), e.g. google/fleurs:es_419")
    ap.add_argument("--split", default="train", help="dataset split to stream")
    ap.add_argument("--heartbeat", type=float, default=5.0,
                    help="seconds between liveness pings (0 to disable)")
    args = ap.parse_args()

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' package is required: pip install datasets\n"
            "(kept out of train/requirements.txt so the default pipeline stays lean)."
        )

    sources = parse_sources(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    idx = len(list(out.glob("*.wav")))   # resume: continue numbering across runs
    if idx >= args.n:
        print(f"{out} already has {idx} >= {args.n} clips; nothing to do.")
        return

    print(f"Target {args.n} clips in {out} (have {idx}). Sources: "
          f"{', '.join(f'{d}:{c}' for d, c in sources)}", flush=True)
    print(f"HF download cache: {hf_cache_dir()}  (set HF_HOME to relocate)", flush=True)

    hb = Heartbeat(out, every=args.heartbeat) if args.heartbeat > 0 else None
    ctx = hb if hb is not None else _nullcontext()
    with ctx:
        for ds_id, cfg in sources:
            if idx >= args.n:
                break
            if hb is not None:
                hb.stage = f"streaming {ds_id}:{cfg}"
            print(f"[{ds_id}:{cfg}] streaming split={args.split} …", flush=True)
            try:
                ds = load_dataset(ds_id, cfg, split=args.split, streaming=True)
                # decode=False -> raw bytes, so we never trigger torchcodec.
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception as e:
                print(f"[{ds_id}:{cfg}] SKIP ({type(e).__name__}: {str(e)[:100]})",
                      flush=True)
                continue
            got = 0
            for ex in ds:
                if idx >= args.n:
                    break
                a = ex.get("audio") or {}
                raw = a.get("bytes")
                if raw is None:
                    continue
                try:
                    wav = decode_to_16k_mono(raw)
                except Exception as e:
                    print(f"[{ds_id}:{cfg}] decode error: {type(e).__name__}", flush=True)
                    continue
                pcm = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
                wavfile.write(out / f"speech_{idx:06d}.wav", SR, pcm)
                idx += 1
                got += 1
                if idx % 100 == 0:
                    print(f"  {idx}/{args.n}", flush=True)
            print(f"[{ds_id}:{cfg}] added {got} clips (total {idx}).", flush=True)

    print(f"Done: {idx} clips in {out}.")
    if idx == 0:
        print("No clips written — check the sources are parquet-native and streamable "
              "on datasets 5.x, and that 'audio' carries bytes.")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
    # datasets/torch spawn C-extension background threads that can raise a benign
    # "PyGILState_Release ... finalizing" at interpreter shutdown. The work is done
    # and files are flushed, so exit hard to skip that noisy finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
