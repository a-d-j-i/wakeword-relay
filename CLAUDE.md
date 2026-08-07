# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two-folder project:
- **`train/`** — CLI pipeline for training INT8-quantized TFLite wake word models (based on [microWakeWord](https://github.com/kahrendt/microWakeWord))
- **`firmware/`** — ESPHome YAML + model manifest for ESP32-S3 deployment

## Environment Setup

**Two separate venvs, one per subproject** — they must NOT be merged: ESPHome
pins `click==8.3.3` while the training stack's `huggingface_hub` requires
`click>=8.4.0`, so a single shared venv cannot satisfy both. Each subproject has
its own `requirements.txt`; there is no root one.

### Training / distillation (`train/requirements.txt`)
```bash
cd train
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
This venv also runs `web_trainer/tools/quantize_from_browser.py` (needs Keras +
the vendored microwakeword). The web trainer's JS/WASM deps live in
`web_trainer/package.json`, not here.

### Firmware / tooling (`firmware/requirements.txt`) — ESPHome + Ruff
```bash
python3 -m venv firmware/venv
source firmware/venv/bin/activate
pip install -r firmware/requirements.txt
```

Python 3.11 tested; 3.10–3.11 should work.

## Development Tools

We use **Ruff** for linting and formatting. It replaces Black, isort, and Flake8.

```bash
# Format Python code
ruff format .

# Check for linting errors and fix them automatically
ruff check --fix .
```

## Running `train/train.py`

```bash
cd train && source venv/bin/activate

# Basic training (default: "hey jarvis", 1000 samples, 10000 steps)
python train.py --phrase "hey jarvis"

# hey lumus — current production model params
python train.py --phrase "hey lumus" --phonetic "hay loo mus" \
  --samples 3000 --steps 30000 --neg_class_weight 25

# Share downloads across runs to avoid re-downloading 10GB
python train.py --phrase "hey lumus" --phonetic "hay loo mus" \
  --samples 3000 --steps 30000 --neg_class_weight 25 \
  --downloads_dir /path/to/shared/downloads --data_dir /path/to/run_output

# Tune false-positive rate (higher = fewer false activations, lower recall)
python train.py --phrase "hey jarvis" --neg_class_weight 30

# Train a non-English phrase — pass a Piper voice name to auto-download it
python train.py --phrase "activa" \
  --piper_model es_AR-daniela-high \
  --samples 3000 --steps 30000 --neg_class_weight 25
```

`--piper_model` accepts either a path to an existing `.onnx`/`.pt` file, or a Piper voice name (e.g. `es_AR-daniela-high`). Voice names are downloaded automatically via `python -m piper.download_voices` and cached in `<downloads_dir>/piper_voices/`. Run `python -m piper.download_voices` with no arguments to list all available voices. Omit `--piper_model` to use the default English model.

**First run downloads ~10GB** of datasets into `--downloads_dir` (defaults to `--data_dir`, default `train/data/`). Point multiple runs at the same `--downloads_dir` to avoid re-downloading. Use `--regen_features` to force rebuilding augmented spectrograms, `--regen_samples` to regenerate TTS positive samples.

**Output**: `<data_dir>/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

After training, **only copy the `.tflite` to `firmware/models/`** — no other files need changing. The manifest JSONs are decoupled from the training phrase:

| File | `wake_word` | Purpose |
|------|-------------|---------|
| `firmware/models/turn_on.json` | `"turn_on"` | triggers `switch.turn_on` |
| `firmware/models/turn_off.json` | `"turn_off"` | triggers `switch.turn_off` |

The `wake_word` field in the JSON is what ESPHome matches on — it has nothing to do with `--phrase` used during training. Swap in a new `.tflite` with any phrase and the firmware works unchanged. Tune `probability_cutoff` in each JSON to trade false-accepts vs misses.

See the [ESPHome docs](https://esphome.io/components/micro_wake_word).

## Firmware (`firmware/`)

```bash
source firmware/venv/bin/activate             # firmware venv (see Environment Setup)
esphome run firmware/wakeword-relay.yaml      # compile + flash over USB
esphome logs firmware/wakeword-relay.yaml     # watch detections
```

The old English `hey_lumus.*` and `lumus.*` models are kept in `firmware/models/` for reference.

## Architecture: `train.py` Pipeline

| Stage | Function | Description |
|-------|----------|-------------|
| 1 | `setup_piper()` | No-op — `piper_sample_generator` and `piper_train` are vendored into `train/` |
| 2 | `generate_positive_samples()` | Runs vendored `piper_sample_generator` to produce 1000+ TTS WAV clips of the wake phrase |
| 3 | `download_mit_rirs()` / `download_audioset()` / `download_fma()` | Downloads room impulse responses and background audio for augmentation |
| 4 | `download_negative_datasets()` | Downloads pre-generated negative spectrograms (speech, dinner party, silence) from HuggingFace |
| 5 | `generate_augmented_features()` | Applies 8 augmentation types (EQ, pitch shift, background noise, RIR convolution, etc.) and saves as `RaggedMmap` spectrogram datasets for train/val/test splits |
| 6 | `write_training_config()` | Writes `training_parameters.yaml` with dataset sampling/penalty weights |
| 7 | `run_training()` | Calls `python -m microwakeword.model_train_eval` — trains a MixedNet (`mixednet`) model and quantizes to streaming INT8 TFLite |

**Key design choices:**
- MixedNet architecture (depthwise separable + mixed convolutions) is the standard microWakeWord architecture for ESP32-S3
- `slide_frames=10` in training/validation simulates streaming inference; `slide_frames=1` in testing uses true streaming evaluation
- MIT RIRs download uses `streaming=True` with no `cast_column` to avoid pulling in `torchcodec` as an audio backend (not in requirements)
- Negative dataset sampling weights (speech: 10×, dinner party: 10×, no speech: 5×) are critical hyperparameters — adjust if the model has too many false activations
- Best model weights are selected by maximizing `average_viable_recall` once a target threshold is met

## Vendored packages in `train/`

- **`microwakeword/`** — microWakeWord training framework (patches below)
- **`piper_sample_generator/`** — TTS sample generator (v3.2.0, from PyPI wheel); includes bundled impulse WAVs in `impulses/`. Patched to decorate: after each WAV write, `_ensure_16k()` normalizes to 16 kHz mono via `train/audio_io.load_wav_16k` (Piper voices synthesize at their native rate, e.g. 22050 Hz for `-medium`; no-op when already 16 kHz)
- **`piper_train/vits/commons.py`** — VITS utilities required by `piper_sample_generator.__main__`; not distributed in the PyPI wheel, fetched from the git repo

## Patches in `train/microwakeword/`

The vendored source has these changes applied directly (no runtime patching):

| File | Change | Reason |
|------|--------|--------|
| `audio/audio_utils.py` | `use_c=False`; `pymicro_features` import made lazy | MicroFrontend segfaults on some Linux; avoids hard dep |
| `audio/augmentation.py` | Remove `AddColorNoise` | Dropped in audiomentations 0.36 |
| `microwakeword/train.py`, `test.py` | `np.trapz` → `np.trapezoid` | Deprecated in NumPy 2.0 |
| `microwakeword/train.py` | Remove `.numpy()` on `evaluate()` results | TF 2.16+ returns plain NumPy |
| `audio/clips.py` | Decorate: delegate WAV loading to our own `train/audio_io.load_wav_16k` (mono, float32, resampled to 16 kHz) instead of soundfile | libsndfile segfaults after TF malloc hooks; single shared loader keeps student and teacher from diverging on sample-rate handling |
| `audio/clips.py` | Replace `audio_metadata.load()` with `torchaudio.info()` for non-WAV duration | Removes `audio-metadata` dep (PyPI version hard-pins `attrs`, causing conflicts) |
| `audio/clips.py` | Remove dead `import datasets` | Leftover from original HuggingFace audio loading path; no longer used after scipy patch |
| `audio/__init__.py`, `layers/__init__.py` | Added (were missing) | Required for proper package import; wheel silently dropped these subdirs |
| `audio/spectrograms.py` | Optional `clip_scorer` callback on `SpectrogramGeneration`; records one score per yielded spectrogram into `clip_scores` | Distillation B2 — score each augmented waveform with the teacher, aligned to sample order |
| `microwakeword/data.py` | `get_data(..., return_soft=True)` and per-sample soft targets; loads `teacher_scores.npy` beside each mmap when present (falls back to hard label) | Distillation B3 — plumb teacher scores through the feature store; no-op without the file |
| `microwakeword/train.py` | Blend `target = (1−λ)·hard + λ·soft` when `config["distill_weight"] > 0` (λ=0 is byte-identical to before) | Distillation B3 — soft-label loss; BCE is affine in its target so no custom loss needed |

## Distillation (optional, off by default)

`train.py --distill` trains the student against soft labels from an
openWakeWord-style teacher; see [`DISTILLATION.md`](DISTILLATION.md). Without
`--distill` the pipeline is unchanged. Everything runs in the **same
`train/venv`** — the teacher only needs `onnx` + `onnxruntime` (added to
`train/requirements.txt`; onnxruntime is required by `--distill` scoring
regardless), and `torch` is already a training dep. New pieces:

- **`train/train_teacher.py`** — trains the teacher head on Piper positives vs local negatives, exports 3 ONNX files (backbone auto-downloaded from openWakeWord's release).
- **`train/teacher_infer.py`** — shared `Teacher` ONNX runtime (melspec → embedding → head), used by both the scorer and the browser emulator.
- **`train.py` flags** `--distill --teacher_model <dir> --distill_weight λ` (default λ=0.5): scores augmented clips during `generate_augmented_features()` into `<split>/teacher_scores.npy`, and writes `distill_weight` into `training_parameters.yaml`. Export/quantization/manifest are unchanged.

## Speech negatives (teacher + student)

The stock HuggingFace negative **spectrograms** (`negative_datasets/{speech,dinner_party,no_speech}`) are all **English** speech (LibriSpeech/VOiCES/CHiME-6/DiPCo/FMA/FSD50K/WHAM). So a non-English model never learns to reject ordinary target-language speech and false-fires on it. Two tools generate real target-language speech negatives:

- **`tools/gen_piper_negatives.py`** — synthesizes non-wake / confusable phrases (e.g. `"chispa"`, `"buenas"`, `"mi mamá me mima"`) with the **same** Piper voices as the positives → `$DL/piper_negatives_16k`. No download; hard phonetic negatives.
- **`tools/fetch_speech.py`** — streams real speech to 16 kHz WAVs → `$DL/<lang>_speech_16k`. `--lang es` picks an ungated, parquet-native source set (FLEURS + VoxPopuli) from a built-in `LANG_SOURCES` registry (`es/en/fr/de/it/pt/pl/nl`); override with `--source dataset:config`. **Common Voice / MLS are unusable** — they ship `datasets`-loading scripts, which `datasets` 5.x dropped. Decodes raw bytes with `soundfile` to avoid `torchcodec` (not installed).

These feed **two** consumers:
- **Teacher** — `run.sh` passes the speech dirs to `train_teacher.py --negatives_dirs` (env `TEACHER_NEG_DIRS` to override). Fixes the "teacher fires on any speech" problem.
- **Student** — **`train.py --student_neg_audio_dirs DIR... --student_neg_weight W`** (default 5.0): `generate_negative_features()` spectrogram-izes the raw WAVs (same augmenter as positives) into `generated_negative_features/{training,validation,testing}/negatives_mmap/`, appended to `training_parameters.yaml` as a `truth:False` source. `data.py` needs no change (it globs any `**/*_mmap/`). `run.sh` wires this via `STUDENT_NEG=1` (default) / `STUDENT_NEG_WEIGHT`.

**Audio vs. spectrogram sources — the asymmetry:** a negative source can be raw **audio** (we derive spectrograms) or a **pre-computed spectrogram mmap**. They are interchangeable **only for the student**, whose input *is* the microfrontend spectrogram. The **teacher requires audio** — it runs its own mel front-end (32 mel bins → embedding → head) over the waveform and can neither consume the 40-channel microfrontend features nor the audio-less bundles. So **audio is a superset**: usable by teacher + student and teacher-scorable for soft labels; spectrogram-only sources are student-only, hard-label-only. (This is why the English mmap bundles stay hard negatives, and why the target-architecture keeps audio as the canonical input with `type:mmap` as an optional accelerator.)

## Key Hyperparameters to Tune

- `--steps`: 10000 is a starting point; 20000–50000 often needed for production quality
- `--neg_class_weight`: default 20; increase to reduce false positives (at cost of recall)
- `--samples`: default 1000; more helps up to a point (~2000–5000)
- `--phonetic`: phonetic spellings (e.g. `"khum_puter"` for `"computer"`) can improve TTS sample quality

## Data Layout (after first run)

Everything lands under `data/` which is git-ignored.

```
data/
  piper_models/                  # Downloaded Piper TTS model weights (.pt or .onnx)
  piper_voices/                  # Voices downloaded via --piper_model <voice-name>
  generated_samples/             # TTS-generated positive WAV clips
  mit_rirs/                      # Room impulse responses (WAV, 16kHz)
  audioset/                      # Raw AudioSet download
  audioset_16k/                  # Background ambient sounds (WAV, 16kHz)
  fma/                           # Raw FMA download
  fma_16k/                       # Background music (WAV, 16kHz)
  negative_datasets/             # Pre-generated negative spectrograms (RaggedMmap)
    speech/  dinner_party/  dinner_party_eval/  no_speech/
  generated_augmented_features/  # Augmented positive spectrograms (RaggedMmap)
    training/  validation/  testing/
  trained_models/wakeword/       # Checkpoints + TFLite output
  training_parameters.yaml       # Written by train.py before training
```

## Target Hardware

- **MCU**: ESP32-S3 with Vector Extension (required for INT8 inference acceleration)
- **RAM**: 8MB PSRAM recommended
- **Mic**: I2S digital (e.g., INMP441)
- **Stack**: ESPHome v2024.3.0+ integrated into Home Assistant
