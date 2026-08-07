#!/usr/bin/env python3
"""Generate SPEECH negatives for the distillation teacher with Piper TTS.

The teacher's default negatives (FMA music + AudioSet ambient) contain no speech,
so it fires on any Spanish phrase (observed: "mi mamá me mima"). This synthesizes
non-wake Spanish phrases — including hard negatives (partials / confusables of the
wake word) — as 16 kHz WAVs, using the SAME vendored piper_sample_generator and the
SAME voices as the positives. Because positives and negatives share the TTS engine,
the teacher head can't cheat on synthesis artifacts and must learn the phrase itself.

Stays within the "just Piper + public corpora" rule; no gated download. Use these
alone, or alongside tools/fetch_commonvoice_es.py for real-speech generalization.

Usage (train/venv):
    python tools/gen_piper_negatives.py \
        --out /vms2/work_tmp/download/piper_negatives_16k \
        --voices_dir /vms2/work_tmp/download/piper_voices \
        --per_phrase 30

run.sh auto-includes <download>/piper_negatives_16k in the teacher's negatives when
present. IMPORTANT: keep the actual wake phrases OUT of --phrases (partials like
"chispa" / "buenas" are fine and are good hard negatives).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"

# Default Spanish negatives: everyday phrases + partials/confusables of the current
# wake words ("chispa mágica" / "buenas noches"). Edit or pass --phrases_file.
DEFAULT_PHRASES = [
    "mi mamá me mima",
    "hola qué tal",
    "buenos días",
    "buenas tardes",
    "cómo estás",
    "muchas gracias",
    "hasta luego",
    "por favor",
    "qué hora es",
    "no entiendo nada",
    "enciende la luz",
    "apaga la tele",
    "chispa",            # partial of "chispa mágica"
    "mágica",
    "chistes mágicos",
    "una chispa pequeña",
    "buenas",            # partial of "buenas noches"
    "noches",
    "buenas noticias",
    "las buenas costumbres",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for 16 kHz negative WAVs")
    ap.add_argument("--voices_dir", required=True,
                    help="dir of downloaded Piper voices (*.onnx), e.g. <download>/piper_voices")
    ap.add_argument("--per_phrase", type=int, default=30,
                    help="clips per (voice, phrase) (default: 30)")
    ap.add_argument("--phrases_file", default=None,
                    help="optional text file, one negative phrase per line")
    args = ap.parse_args()

    phrases = DEFAULT_PHRASES
    if args.phrases_file:
        phrases = [ln.strip() for ln in Path(args.phrases_file).read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")]

    voices = sorted(Path(args.voices_dir).glob("*.onnx"))
    if not voices:
        raise SystemExit(f"No *.onnx voices in {args.voices_dir} — run training once "
                         "so voices download, or point --voices_dir at them.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAIN_DIR) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    idx = len(list(out.glob("*.wav")))   # continue numbering if re-run
    for voice in voices:
        vshort = voice.stem
        for phrase in phrases:
            tmp = out / "_tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            subprocess.run([
                sys.executable, "-m", "piper_sample_generator",
                phrase,
                "--model", str(voice),
                "--max-samples", str(args.per_phrase),
                "--batch-size", "50",
                "--output-dir", str(tmp),
            ], check=True, env=env)
            for wav in sorted(tmp.glob("*.wav"), key=lambda p: int(p.stem)):
                wav.rename(out / f"neg_{idx:06d}.wav")
                idx += 1
            shutil.rmtree(tmp)
        print(f"[neg] {vshort}: total so far {idx}")

    print(f"Wrote {idx} negative WAVs to {out}")


if __name__ == "__main__":
    main()
