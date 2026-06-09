#!/usr/bin/env python3
"""
Micro wake word training pipeline based on basic_training_notebook.ipynb.

Usage:
    python train.py --phrase "hey jarvis"
    python train.py --phrase "okay computer" --phonetic "oh kay kompyooter" --samples 2000 --steps 20000

Notes:
  - Run from a venv with: pip install -r requirements.txt
  - All downloaded/generated data lands in ./data/ (safe to delete to start fresh)
  - Output: data/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
  - Phonetic spellings often produce better TTS samples (e.g. "khum_puter" for "computer")
"""

import argparse
import os
import subprocess
import sys
import urllib.request  # used for PIPER_MODEL_URL download only
import zipfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; suppress CUDA init errors

import numpy as np
import scipy.io.wavfile
import yaml
from tqdm import tqdm


DATA_DIR = Path("data")
MWW_REPO_DIR = DATA_DIR / "microWakeWord"
PIPER_REPO_DIR = DATA_DIR / "piper-sample-generator"

PIPER_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    "en_US-libritts_r-medium.pt"
)
PIPER_MODEL_PATH = PIPER_REPO_DIR / "models" / "en_US-libritts_r-medium.pt"
NEGATIVE_DATASETS_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_DATASETS = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]


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
                   help="Root directory for all downloads and generated data (default: ./data)")
    p.add_argument("--output_dir", default=None,
                   help="Training output directory (default: <data_dir>/trained_models/wakeword)")
    p.add_argument("--regen_features", action="store_true",
                   help="Delete and regenerate augmented features even if they exist")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_microwakeword():
    """Clone microWakeWord, apply compatibility patches, and prepend to sys.path.

    The wheel installed by requirements.txt omits microwakeword/audio/ because
    find_packages() requires __init__.py and audio/ has none. Prepending the
    clone root to sys.path makes audio.* importable as a namespace package, and
    also ensures run_training()'s subprocess sees it via PYTHONPATH.
    """
    clone_dir = MWW_REPO_DIR
    if not clone_dir.exists():
        print("[setup] Cloning microWakeWord...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/kahrendt/microWakeWord", str(clone_dir)],
            check=True,
        )
    _patch_microwakeword(clone_dir)
    if str(clone_dir) not in sys.path:
        sys.path.insert(0, str(clone_dir))


def _patch_microwakeword(clone_dir: Path):
    """Apply idempotent compatibility patches to the microWakeWord clone.

    These fix upstream code for the library versions we install:
    - pymicro_features.MicroFrontend segfaults on some systems; use TF path.
    - model.evaluate() now returns numpy arrays directly (no .numpy() needed).
    """
    patches = [
        (
            clone_dir / "microwakeword/audio/audio_utils.py",
            "use_c: bool = True",
            "use_c: bool = False",
        ),
        (
            clone_dir / "microwakeword/audio/augmentation.py",
            'audiomentations.AddColorNoise(\n                    p=augmentation_probabilities.get("AddColorNoise", 0.0),\n                    min_snr_db=color_min_snr_db,\n                    max_snr_db=color_max_snr_db,\n                ),',
            '# AddColorNoise removed: not available in audiomentations>=0.36',
        ),
        (
            clone_dir / "microwakeword/train.py",
            "np.trapz(",
            "np.trapezoid(",
        ),
        (
            clone_dir / "microwakeword/test.py",
            "np.trapz(",
            "np.trapezoid(",
        ),
        (
            clone_dir / "microwakeword/train.py",
            'test_set_fp = result["fp"].numpy()',
            'test_set_fp = result["fp"]',
        ),
        (
            clone_dir / "microwakeword/train.py",
            'all_true_positives = ambient_predictions["tp"].numpy()',
            'all_true_positives = ambient_predictions["tp"]',
        ),
        (
            clone_dir / "microwakeword/train.py",
            'ambient_false_positives = ambient_predictions["fp"].numpy() - test_set_fp',
            'ambient_false_positives = ambient_predictions["fp"] - test_set_fp',
        ),
        (
            clone_dir / "microwakeword/train.py",
            'all_false_negatives = ambient_predictions["fn"].numpy()',
            'all_false_negatives = ambient_predictions["fn"]',
        ),
        (
            clone_dir / "microwakeword/audio/clips.py",
            '        # Load all filtered clips\n'
            '        audio_dataset = datasets.Dataset.from_dict(\n'
            '            {"audio": [str(i) for i in filtered_paths]}\n'
            '        ).cast_column("audio", datasets.Audio())\n'
            '\n'
            '        # Convert all clips to 16 kHz sampling rate when accessed\n'
            '        audio_dataset = audio_dataset.cast_column(\n'
            '            "audio", datasets.Audio(sampling_rate=16000)\n'
            '        )\n'
            '\n'
            '        if random_split_seed is not None:\n'
            '            train_testvalid = audio_dataset.train_test_split(\n'
            '                test_size=2 * split_count, seed=random_split_seed\n'
            '            )\n'
            '            test_valid = train_testvalid["test"].train_test_split(test_size=0.5)\n'
            '            split_dataset = datasets.DatasetDict(\n'
            '                {\n'
            '                    "train": train_testvalid["train"],\n'
            '                    "test": test_valid["test"],\n'
            '                    "validation": test_valid["train"],\n'
            '                }\n'
            '            )\n'
            '            self.split_clips = split_dataset\n'
            '\n'
            '        self.clips = audio_dataset',
            '        # Load all clips eagerly with scipy to avoid TF/libsndfile memory allocator conflict.\n'
            '        # soundfile/libsndfile called after TF initialises its custom malloc hooks causes\n'
            '        # a segfault; scipy.io.wavfile is pure Python/numpy and has no such conflict.\n'
            '        import scipy.io.wavfile as _wf\n'
            '        import scipy.signal as _ss\n'
            '\n'
            '        def _load_16k(path):\n'
            '            sr, data = _wf.read(str(path))\n'
            '            if data.dtype == np.int16:\n'
            '                data = data.astype(np.float32) / 32768.0\n'
            '            else:\n'
            '                data = data.astype(np.float32)\n'
            '            if sr != 16000:\n'
            '                data = _ss.resample(data, int(round(len(data) * 16000 / sr)))\n'
            '            return {"audio": {"array": data}}\n'
            '\n'
            '        all_entries = [_load_16k(p) for p in filtered_paths]\n'
            '\n'
            '        if random_split_seed is not None:\n'
            '            rng = random.Random(random_split_seed)\n'
            '            idx = list(range(len(all_entries)))\n'
            '            rng.shuffle(idx)\n'
            '            n_split = max(2, int(round(len(idx) * 2 * split_count)))\n'
            '            n_half = n_split // 2\n'
            '            self.split_clips = {\n'
            '                "train": [all_entries[i] for i in idx[n_split:]],\n'
            '                "validation": [all_entries[i] for i in idx[:n_half]],\n'
            '                "test": [all_entries[i] for i in idx[n_half:n_split]],\n'
            '            }\n'
            '\n'
            '        self.clips = all_entries',
        ),
    ]
    for file_path, old, new in patches:
        text = file_path.read_text()
        if old in text:
            file_path.write_text(text.replace(old, new))
            print(f"[patch] {file_path.relative_to(clone_dir)}")


def download_piper_model() -> Path:
    """Clone piper-sample-generator (editable install) and download model weights.

    The repo's pyproject.toml only bundles piper_sample_generator, not the
    sibling piper_train package that __main__.py requires. An editable install
    adds the repo root to sys.path via a .pth file, making both packages visible
    to all subprocesses without needing PYTHONPATH tweaks.
    """
    repo_dir = PIPER_REPO_DIR
    if not repo_dir.exists():
        print("[setup] Cloning piper-sample-generator...")
        subprocess.run(
            ["git", "clone", "https://github.com/rhasspy/piper-sample-generator",
             str(repo_dir)],
            check=True,
        )
        print("[setup] Installing piper-sample-generator (editable)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(repo_dir)],
            check=True,
        )

    if not PIPER_MODEL_PATH.exists():
        print("[setup] Downloading Piper TTS model weights (~75MB)...")
        PIPER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PIPER_MODEL_URL, str(PIPER_MODEL_PATH))

    return PIPER_MODEL_PATH


# ---------------------------------------------------------------------------
# Positive sample generation
# ---------------------------------------------------------------------------

def generate_positive_samples(phonetic: str, n_samples: int, model_path: Path) -> Path:
    """Generate TTS wake word samples using piper-sample-generator."""
    out_dir = DATA_DIR / "generated_samples"
    out_dir.mkdir(exist_ok=True)

    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= n_samples:
        print(f"[samples] {len(existing)} samples already exist, skipping generation.")
        return out_dir

    print(f"[samples] Generating {n_samples} samples for '{phonetic}'...")
    import os
    env = os.environ.copy()
    # piper_train is a sibling package in the repo not exposed by the editable install's
    # custom finder; adding the repo root to PYTHONPATH makes it importable in subprocesses.
    env["PYTHONPATH"] = str(PIPER_REPO_DIR.resolve()) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run([
        sys.executable, "-m", "piper_sample_generator",
        phonetic,
        "--model", str(model_path),
        "--max-samples", str(n_samples),
        "--batch-size", "100",
        "--output-dir", str(out_dir),
    ], check=True, env=env)
    return out_dir


# ---------------------------------------------------------------------------
# Dataset downloads
# ---------------------------------------------------------------------------

def download_mit_rirs() -> Path:
    import datasets as hf
    out_dir = DATA_DIR / "mit_rirs"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir()
    print("[data] Downloading MIT Environmental Impulse Responses...")
    ds = hf.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train"
    )
    ds = ds.cast_column("audio", hf.Audio(sampling_rate=16000))
    for idx, row in enumerate(tqdm(ds)):
        scipy.io.wavfile.write(
            out_dir / f"rir_{idx:05d}.wav", 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    return out_dir


def download_audioset() -> Path:
    import datasets as hf
    out_dir = DATA_DIR / "audioset_16k"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir()
    print("[data] Downloading AudioSet (one balanced-train shard, ~500 clips)...")
    ds = hf.load_dataset(
        "agkphysics/AudioSet",
        data_files={"train": "data/bal_train/00.parquet"},
        split="train",
    )
    ds = ds.cast_column("audio", hf.Audio(sampling_rate=16000))
    for row in tqdm(ds):
        scipy.io.wavfile.write(
            out_dir / (row["video_id"] + ".wav"), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    return out_dir


def download_fma() -> Path:
    import datasets as hf
    out_dir = DATA_DIR / "fma_16k"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir

    fma_dir = DATA_DIR / "fma"
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

    out_dir.mkdir()
    print("[data] Converting FMA to 16kHz WAV...")
    mp3_files = list((fma_dir / "fma_small").glob("**/*.mp3"))
    ds = hf.Dataset.from_dict({"audio": [str(p) for p in mp3_files]})
    ds = ds.cast_column("audio", hf.Audio(sampling_rate=16000))
    for row in tqdm(ds):
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            out_dir / name, 16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    return out_dir


def download_negative_datasets() -> Path:
    out_dir = DATA_DIR / "negative_datasets"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir()
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


def generate_augmented_features(samples_dir: Path, regen: bool) -> Path:
    """Build augmented spectrogram RaggedMmap datasets for train/val/test splits."""
    import shutil

    out_dir = DATA_DIR / "generated_augmented_features"
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
        input_directory=str(samples_dir),
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
        impulse_paths=[str(DATA_DIR / "mit_rirs")],
        background_paths=[str(DATA_DIR / "fma_16k"), str(DATA_DIR / "audioset_16k")],
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

def write_training_config(output_dir: str, steps: int, batch_size: int, neg_class_weight: int) -> Path:
    config = {
        "window_step_ms": 10,
        "train_dir": output_dir,
        "features": [
            {
                "features_dir": str(DATA_DIR / "generated_augmented_features"),
                "sampling_weight": 2.0, "penalty_weight": 1.0,
                "truth": True, "truncation_strategy": "truncate_start", "type": "mmap",
            },
            {
                "features_dir": str(DATA_DIR / "negative_datasets/speech"),
                "sampling_weight": 10.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {
                "features_dir": str(DATA_DIR / "negative_datasets/dinner_party"),
                "sampling_weight": 10.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {
                "features_dir": str(DATA_DIR / "negative_datasets/no_speech"),
                "sampling_weight": 5.0, "penalty_weight": 1.0,
                "truth": False, "truncation_strategy": "random", "type": "mmap",
            },
            {   # used only for validation/testing
                "features_dir": str(DATA_DIR / "negative_datasets/dinner_party_eval"),
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
    config_path = DATA_DIR / "training_parameters.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def run_training(config_path: Path):
    """Invoke microwakeword.model_train_eval with default MixedNet architecture."""
    import re
    import time

    env = os.environ.copy()
    mww_clone = str(MWW_REPO_DIR.resolve())
    env["PYTHONPATH"] = mww_clone + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

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
    last_step = 0

    for line in proc.stderr:
        print(line, end="", flush=True)
        m = step_re.search(line)
        if m:
            step = int(m.group(1))
            now = time.time()
            if first_step is None:
                first_step, first_time = step, now
            elif step > first_step:
                elapsed = now - first_time
                rate = (step - first_step) / elapsed          # steps/sec
                remaining = (total_steps - step) / rate
                h, r = divmod(int(remaining), 3600)
                m2, s = divmod(r, 60)
                pct = 100 * step / total_steps
                print(
                    f"\n{'='*60}\n"
                    f"  PROGRESS:  step {step} / {total_steps}  ({pct:.0f}%)\n"
                    f"  ETA:       {h}h {m2:02d}m {s:02d}s remaining\n"
                    f"{'='*60}\n",
                    flush=True,
                )
            last_step = step

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    phonetic = args.phonetic or args.phrase

    # Update module-level path constants from --data_dir so all functions see them.
    DATA_DIR = Path(args.data_dir)
    MWW_REPO_DIR = DATA_DIR / "microWakeWord"
    PIPER_REPO_DIR = DATA_DIR / "piper-sample-generator"
    PIPER_MODEL_PATH = PIPER_REPO_DIR / "models" / "en_US-libritts_r-medium.pt"
    if args.output_dir is None:
        args.output_dir = str(DATA_DIR / "trained_models/wakeword")

    DATA_DIR.mkdir(exist_ok=True)
    setup_microwakeword()

    model_path = download_piper_model()
    samples_dir = generate_positive_samples(phonetic, args.samples, model_path)

    download_mit_rirs()
    download_audioset()
    download_fma()
    download_negative_datasets()

    generate_augmented_features(samples_dir, regen=args.regen_features)

    config_path = write_training_config(
        args.output_dir, args.steps, args.batch_size, args.neg_class_weight
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
