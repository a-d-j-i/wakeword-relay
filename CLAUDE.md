# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two-folder project:
- **`train/`** — CLI pipeline for training INT8-quantized TFLite wake word models (based on [microWakeWord](https://github.com/kahrendt/microWakeWord))
- **`firmware/`** — ESPHome YAML + model manifest for ESP32-S3 deployment

## Environment Setup

```bash
cd train
python3.10 -m venv .venv-mww
source .venv-mww/bin/activate
pip install -r requirements.txt
```

Python 3.10 recommended; 3.9–3.11 should work.

## Running `train/train.py`

```bash
cd train && source .venv-mww/bin/activate

# Basic training (default: "hey jarvis", 1000 samples, 10000 steps)
python train.py --phrase "hey jarvis"

# Custom wake word with phonetic spelling (often produces better TTS quality)
python train.py --phrase "okay computer" --phonetic "oh kay kompyooter" --samples 2000 --steps 20000

# Tune false-positive rate (higher = fewer false activations, lower recall)
python train.py --phrase "hey jarvis" --neg_class_weight 30
```

**First run downloads ~10GB** of datasets into `--data_dir` (default `train/data/`). Subsequent runs reuse them. Use `--regen_features` to force rebuilding augmented spectrograms.

**Output**: `<data_dir>/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

After training, copy the `.tflite` to `firmware/models/` and update `firmware/models/<phrase>.json`. See the [ESPHome docs](https://esphome.io/components/micro_wake_word).

## Firmware (`firmware/`)

```bash
pip install esphome
esphome run firmware/wakeword-relay.yaml      # compile + flash over USB
esphome logs firmware/wakeword-relay.yaml     # watch detections
```

Model manifest is at `firmware/models/lumus.json`. Tune `probability_cutoff` there to trade false-accepts vs misses. Current ROC operating points (from training):
- cutoff 0.98 → 6% miss rate, ~0 false accepts/hour
- cutoff 0.44 → 2% miss rate, ~0.2 false accepts/hour

## Architecture: `train.py` Pipeline

| Stage | Function | Description |
|-------|----------|-------------|
| 1 | `setup_piper_generator()` | Clones `piper-sample-generator` repo + downloads LibriTTS model weights |
| 2 | `generate_positive_samples()` | Runs `piper-sample-generator` to produce 1000+ TTS WAV clips of the wake phrase |
| 3 | `download_mit_rirs()` / `download_audioset()` / `download_fma()` | Downloads room impulse responses and background audio for augmentation |
| 4 | `download_negative_datasets()` | Downloads pre-generated negative spectrograms (speech, dinner party, silence) from HuggingFace |
| 5 | `generate_augmented_features()` | Applies 8 augmentation types (EQ, pitch shift, background noise, RIR convolution, etc.) and saves as `RaggedMmap` spectrogram datasets for train/val/test splits |
| 6 | `write_training_config()` | Writes `training_parameters.yaml` with dataset sampling/penalty weights |
| 7 | `run_training()` | Calls `python -m microwakeword.model_train_eval` — trains a MixedNet (`mixednet`) model and quantizes to streaming INT8 TFLite |

**Key design choices:**
- MixedNet architecture (depthwise separable + mixed convolutions) is the standard microWakeWord architecture for ESP32-S3
- `slide_frames=10` in training/validation simulates streaming inference; `slide_frames=1` in testing uses true streaming evaluation
- Negative dataset sampling weights (speech: 10×, dinner party: 10×, no speech: 5×) are critical hyperparameters — adjust if the model has too many false activations
- Best model weights are selected by maximizing `average_viable_recall` once a target threshold is met

## Key Hyperparameters to Tune

- `--steps`: 10000 is a starting point; 20000–50000 often needed for production quality
- `--neg_class_weight`: default 20; increase to reduce false positives (at cost of recall)
- `--samples`: default 1000; more helps up to a point (~2000–5000)
- `--phonetic`: phonetic spellings (e.g. `"khum_puter"` for `"computer"`) can improve TTS sample quality

## Data Layout (after first run)

Everything lands under `data/` which is git-ignored.

```
data/
  piper-sample-generator/        # Cloned automatically
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
