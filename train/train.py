#!/usr/bin/env python3
"""
Micro wake word training pipeline based on basic_training_notebook.ipynb.

Usage:
    python train.py --phrase "hey jarvis"
    python train.py --phrase "okay computer" --phonetic "oh kay kompyooter" --samples 2000 --steps 20000

Notes:
  - Run from a venv with: pip install -r requirements.txt
  - Downloaded datasets land in --downloads_dir (default: same as --data_dir).
    Point multiple training runs at the same --downloads_dir to avoid re-downloading.
  - Generated/phrase-specific files land in --data_dir (default: ./data).
    Safe to delete between runs; downloads are unaffected.
  - Output: <data_dir>/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
  - Phonetic spellings often produce better TTS samples (e.g. "khum_puter" for "computer")
"""

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; suppress CUDA init errors

import numpy as np
import scipy.io.wavfile
import yaml
from tqdm import tqdm


# Directory containing this script — vendored microwakeword/ lives here.
_TRAIN_DIR = Path(__file__).parent.resolve()

NEGATIVE_DATASETS_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_DATASETS = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]


@dataclass
class Paths:
    data_dir: Path
    downloads_dir: Path
    piper_model_path: Path


def parse_args():
    p = argparse.ArgumentParser(description="microWakeWord training pipeline")
    p.add_argument("--phrase", required=True,
                   help="Wake word phrase to train (e.g. 'hey jarvis')")
    p.add_argument("--phonetic", default=None,
                   help="Phonetic spelling for TTS (e.g. 'hay jar vis'). Defaults to --phrase.")
    p.add_argument("--samples", type=int, default=1000,
                   help="Number of positive TTS samples to generate (default: 1000)")
    p.add_argument("--steps", type=int, default=10000,
                   help="Training steps (default: 10000; increase for better accuracy)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--neg_class_weight", type=int, default=20,
                   help="Penalty weight for false positives (default: 20; increase to reduce false activations)")
    p.add_argument("--data_dir", default="data",
                   help="Root directory for generated/phrase-specific data (default: ./data)")
    p.add_argument("--downloads_dir", default=None,
                   help="Root directory for downloaded datasets and repos (default: same as --data_dir). "
                        "Shared across training runs to avoid re-downloading.")
    p.add_argument("--output_dir", default=None,
                   help="Training output directory (default: <data_dir>/trained_models/wakeword)")
    p.add_argument("--piper_model", default=None,
                   help="Path to a Piper TTS voice model (.onnx or .pt). "
                        "If omitted, the default English .pt model is downloaded automatically. "
                        "Download any voice from https://huggingface.co/rhasspy/piper-voices")
    p.add_argument("--regen_features", action="store_true",
                   help="Delete and regenerate augmented features even if they exist")
    p.add_argument("--regen_samples", action="store_true",
                   help="Delete and regenerate TTS positive samples even if they exist")
    return p.parse_args()


_DEFAULT_PIPER_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    "en_US-libritts_r-medium.pt"
)


def resolve_piper_model(piper_model_arg: str | None, downloads_dir: Path) -> Path:
    """Return the path to the Piper model to use, downloading it if necessary.

    - None → download the default English .pt model from piper-sample-generator releases.
    - Existing file path → use it as-is.
    - Voice name (e.g. "es_AR-daniela-high") → download via `python -m piper.download_voices`.
    """
    if piper_model_arg is None:
        model_path = downloads_dir / "piper_models" / "en_US-libritts_r-medium.pt"
        if not model_path.exists():
            print("[setup] Downloading default Piper model (~75MB)...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_DEFAULT_PIPER_MODEL_URL, str(model_path))
        return model_path

    candidate = Path(piper_model_arg)
    if candidate.exists():
        return candidate

    # Treat as a voice name and download via piper.download_voices.
    voice_dir = downloads_dir / "piper_voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voice_dir / f"{piper_model_arg}.onnx"
    if not onnx_path.exists():
        print(f"[setup] Downloading Piper voice '{piper_model_arg}'...")
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices",
             piper_model_arg, "--download-dir", str(voice_dir)],
            check=True,
        )
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Expected {onnx_path} after download — check that '{piper_model_arg}' is a valid voice name. "
            "Run `python -m piper.download_voices` with no arguments to list available voices."
        )
    return onnx_path


def setup_piper(paths: Paths) -> None:
    pass  # piper_sample_generator and piper_train are vendored into train/


# ---------------------------------------------------------------------------
# Positive sample generation
# ---------------------------------------------------------------------------

def generate_positive_samples(paths: Paths, phonetic: str, n_samples: int, regen: bool) -> Path:
    """Generate TTS wake word samples using piper-sample-generator."""
    import shutil

    out_dir = paths.data_dir / "generated_samples"

    if regen and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(exist_ok=True)

    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= n_samples:
        print(f"[samples] {len(existing)} samples already exist, skipping generation.")
        return out_dir

    print(f"[samples] Generating {n_samples} samples for '{phonetic}'...")
    env = os.environ.copy()
    # piper_sample_generator and piper_train are vendored in _TRAIN_DIR; add it to
    # PYTHONPATH so the subprocess finds both packages without any editable install.
    env["PYTHONPATH"] = str(_TRAIN_DIR) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run([
        sys.executable, "-m", "piper_sample_generator",
        phonetic,
        "--model", str(paths.piper_model_path),
        "--max-samples", str(n_samples),
        "--batch-size", "100",
        "--output-dir", str(out_dir),
    ], check=True, env=env)
    return out_dir


# ---------------------------------------------------------------------------
# Dataset downloads
# ---------------------------------------------------------------------------

def download_mit_rirs(paths: Paths) -> Path:
    import datasets as hf
    out_dir = paths.downloads_dir / "mit_rirs"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir(exist_ok=True)
    print("[data] Downloading MIT Environmental Impulse Responses...")
    # streaming=True + no cast_column: avoids torchcodec backend, matches notebook approach
    ds = hf.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    for idx, row in enumerate(tqdm(ds)):
        scipy.io.wavfile.write(
            out_dir / f"rir_{idx:05d}.wav", 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    return out_dir


_AUDIOSET_N_CLIPS = 500


def download_audioset(paths: Paths) -> Path:
    import datasets as hf

    out_dir = paths.downloads_dir / "audioset_16k"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir(exist_ok=True)

    # Dataset was converted from .tar to Parquet in late 2025; stream to avoid
    # pulling all 25GB — we only need a few hundred background clips.
    print(f"[data] Downloading AudioSet ({_AUDIOSET_N_CLIPS} clips, streaming)...")
    ds = hf.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True, trust_remote_code=True
    )
    for idx, row in enumerate(tqdm(ds, total=_AUDIOSET_N_CLIPS)):
        if idx >= _AUDIOSET_N_CLIPS:
            break
        try:
            audio = row["audio"]
            arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            if sr != 16000:
                import torchaudio
                import torch
                waveform = torch.tensor(arr).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                arr = waveform.squeeze().numpy()
            scipy.io.wavfile.write(
                out_dir / f"audioset_{idx:05d}.wav", 16000,
                (arr * 32767).astype(np.int16),
            )
        except Exception:
            continue
    return out_dir


def download_fma(paths: Paths) -> Path:
    import torchaudio

    out_dir = paths.downloads_dir / "fma_16k"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir

    fma_dir = paths.downloads_dir / "fma"
    fma_dir.mkdir(exist_ok=True)
    zip_path = fma_dir / "fma_xs.zip"
    if not zip_path.exists():
        print("[data] Downloading FMA xsmall dataset (~7GB)...")
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
            str(zip_path),
        )
    print("[data] Extracting FMA...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(fma_dir)

    out_dir.mkdir(exist_ok=True)
    print("[data] Converting FMA to 16kHz WAV...")
    mp3_files = list((fma_dir / "fma_small").glob("**/*.mp3"))
    resampler_cache = {}
    for p in tqdm(mp3_files):
        name = Path(p).name.replace(".mp3", ".wav")
        try:
            waveform, sr = torchaudio.load(str(p))
        except Exception:
            continue
        if sr != 16000:
            if sr not in resampler_cache:
                resampler_cache[sr] = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler_cache[sr](waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        scipy.io.wavfile.write(
            out_dir / name, 16000,
            (waveform.squeeze().numpy() * 32767).astype(np.int16),
        )
    return out_dir


def download_negative_datasets(paths: Paths) -> Path:
    out_dir = paths.downloads_dir / "negative_datasets"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir(exist_ok=True)
    print("[data] Downloading pre-generated negative spectrogram datasets...")
    for fname in NEGATIVE_DATASETS:
        zip_path = out_dir / fname
        urllib.request.urlretrieve(NEGATIVE_DATASETS_ROOT + fname, str(zip_path))
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(out_dir)], check=True)
        zip_path.unlink()
    return out_dir


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def _features_complete(out_dir: Path) -> bool:
    """Return True only if all three split mmaps exist and contain data."""
    for split in ["training", "validation", "testing"]:
        mmap_dir = out_dir / split / "wakeword_mmap"
        if not mmap_dir.exists() or not any(mmap_dir.iterdir()):
            return False
    return True


def generate_augmented_features(paths: Paths, regen: bool) -> Path:
    """Build augmented spectrogram RaggedMmap datasets for train/val/test splits."""
    import shutil

    out_dir = paths.data_dir / "generated_augmented_features"
    if out_dir.exists():
        if _features_complete(out_dir) and not regen:
            print("[features] Augmented features already exist, skipping. Use --regen_features to rebuild.")
            return out_dir
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    print("[features] Generating augmented spectrograms...")

    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    # clips.py is patched to load audio with scipy.io.wavfile (not soundfile/HuggingFace),
    # avoiding the TF/libsndfile memory allocator conflict.
    clips = Clips(
        input_directory=str(paths.data_dir / "generated_samples"),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=[str(paths.downloads_dir / "mit_rirs")],
        background_paths=[str(paths.downloads_dir / "fma_16k"), str(paths.downloads_dir / "audioset_16k")],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    # slide_frames=10 simulates streaming inference during non-streaming training;
    # testing uses slide_frames=1 (true streaming evaluation)
    split_configs = [
        ("training",   "train",      2, 10),
        ("validation", "validation", 1, 10),
        ("testing",    "test",       1,  1),
    ]
    for split_dir_name, split_name, repetition, slide_frames in split_configs:
        split_dir = out_dir / split_dir_name
        split_dir.mkdir()
        print(f"[features] Generating {split_dir_name} spectrograms...")
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=slide_frames, step_ms=10
        )
        RaggedMmap.from_generator(
            out_dir=str(split_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )
    return out_dir


# ---------------------------------------------------------------------------
# Training config + execution
# ---------------------------------------------------------------------------

def write_training_config(
    paths: Paths, output_dir: str, steps: int, batch_size: int, neg_class_weight: int
) -> Path:
    config = {
        "window_step_ms": 10,
        "train_dir": output_dir,
        "features": [
            {
                "features_dir": str(paths.data_dir / "generated_augmented_features"),
                "sampling_weight": 2.0, "penalty_weight": 1.0,
                "truth": True, "truncation_strategy": "truncate_start", "type": "mmap",
            },
            {
                "features_dir": str(paths.downloads_dir / "negative_datasets/speech"),
                "sampling_weight": 10.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {
                "features_dir": str(paths.downloads_dir / "negative_datasets/dinner_party"),
                "sampling_weight": 10.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {
                "features_dir": str(paths.downloads_dir / "negative_datasets/no_speech"),
                "sampling_weight": 5.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {   # used only for validation/testing
                "features_dir": str(paths.downloads_dir / "negative_datasets/dinner_party_eval"),
                "sampling_weight": 0.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "split", "type": "mmap",
            },
        ],
        "training_steps": [steps],
        "positive_class_weight": [1],
        "negative_class_weight": [neg_class_weight],
        "learning_rates": [0.001],
        "batch_size": batch_size,
        "time_mask_max_size": [0], "time_mask_count": [0],
        "freq_mask_max_size": [0], "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    config_path = paths.data_dir / "training_parameters.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def run_training(config_path: Path):
    """Invoke microwakeword.model_train_eval with default MixedNet architecture."""
    import re
    import time

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAIN_DIR) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1, 1, 1, 1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]

    total_steps = yaml.safe_load(config_path.read_text()).get("training_steps", [10000])
    total_steps = sum(total_steps)

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, env=env)

    step_re = re.compile(r"Step #(\d+):")
    first_step = None
    first_time = None

    for line in proc.stderr:
        print(line, end="", flush=True)
        match = step_re.search(line)
        if match:
            step = int(match.group(1))
            now = time.time()
            if first_step is None:
                first_step, first_time = step, now
            elif step > first_step:
                elapsed = now - first_time
                rate = (step - first_step) / elapsed
                remaining = (total_steps - step) / rate
                h, r = divmod(int(remaining), 3600)
                mins, s = divmod(r, 60)
                pct = 100 * step / total_steps
                print(
                    f"\n{'='*60}\n"
                    f"  PROGRESS:  step {step} / {total_steps}  ({pct:.0f}%)\n"
                    f"  ETA:       {h}h {mins:02d}m {s:02d}s remaining\n"
                    f"{'='*60}\n",
                    flush=True,
                )

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    phonetic = args.phonetic or args.phrase

    data_dir = Path(args.data_dir)
    downloads_dir = Path(args.downloads_dir) if args.downloads_dir else data_dir
    downloads_dir.mkdir(parents=True, exist_ok=True)
    piper_model_path = resolve_piper_model(args.piper_model, downloads_dir)
    paths = Paths(data_dir=data_dir, downloads_dir=downloads_dir, piper_model_path=piper_model_path)

    if args.output_dir is None:
        args.output_dir = str(paths.data_dir / "trained_models/wakeword")

    paths.data_dir.mkdir(exist_ok=True)

    setup_piper(paths)
    generate_positive_samples(paths, phonetic, args.samples, regen=args.regen_samples)

    download_mit_rirs(paths)
    download_audioset(paths)
    download_fma(paths)
    download_negative_datasets(paths)

    generate_augmented_features(paths, regen=args.regen_features)

    config_path = write_training_config(
        paths, args.output_dir, args.steps, args.batch_size, args.neg_class_weight
    )
    run_training(config_path)

    tflite_path = (
        Path(args.output_dir)
        / "tflite_stream_state_internal_quant"
        / "stream_state_internal_quant.tflite"
    )
    print(f"\nDone. Model: {tflite_path}")
    print("Write a model manifest JSON to deploy in ESPHome: https://esphome.io/components/micro_wake_word")
    print("See manifest examples: https://github.com/esphome/micro-wake-word-models/tree/main/models/v2")
