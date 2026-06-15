# wakeword-relay

Offline wake-word detection on an ESP32-S3 that toggles a relay. The model runs entirely on-device — no Wi-Fi, no Home
Assistant, no cloud at runtime.

- **`train/`** — CLI pipeline to train a custom INT8 TFLite wake word model
- **`firmware/`** — ESPHome YAML + model manifest for ESP32-S3 deployment
- **[Browser tester](https://a-d-j-i.github.io/wakeword-relay/)** — load a `.tflite` and test against WAV files or your
  live mic

## Hardware

- **MCU**: ESP32-S3 N16R8 (16MB flash, 8MB octal PSRAM, Vector Extension for INT8 inference)
- **Mic**: I2S digital — INMP441, ICS-43434, or SPH0645. Tie `SEL` → GND for the left channel.
- **Relay**: any GPIO; default is GPIO5 (change in `wakeword-relay.yaml`)

### Wiring

| Signal        | Mic pin | ESP32-S3 GPIO |
|---------------|---------|---------------|
| BCLK / SCK    | SCK     | GPIO1         |
| LRCLK / WS    | WS      | GPIO2         |
| Data in       | SD      | GPIO3         |
| Relay control | —       | GPIO5         |

## Flash the firmware

```bash
pip install esphome
esphome run firmware/wakeword-relay.yaml   # compile + flash over USB
esphome logs firmware/wakeword-relay.yaml  # watch for "Detected 'lumus'"
```

The firmware ships with a custom-trained `lumus` model in `firmware/models/`. To switch to a prebuilt model (
`okay_nabu`, `hey_jarvis`, `hey_mycr!
oft`), edit the `models:` block in `wakeword-relay.yaml`.

## Train a custom wake word

**Note: Training runs on CPU only.** This project's script is intentionally set to CPU to avoid CUDA versioning issues.

```bash
cd train
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Basic run — 1000 samples, 10 000 steps
python train.py --phrase "hey jarvis"

# With phonetic spelling for better TTS quality
python train.py --phrase "okay computer" --phonetic "oh kay kompyooter" --samples 2000 --steps 20000

# Separate downloads from generated files (recommended for repeated runs)
python train.py --phrase "hey jarvis" \
  --downloads_dir /path/to/shared/downloads \
  --data_dir /path/to/hey_jarvis_run
```

**First run downloads ~10 GB** of datasets into `--downloads_dir` (defaults to `--data_dir`). Output:

```
<data_dir>/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

Copy the `.tflite` to `firmware/models/`, update the manifest JSON, and reflash.

### Key training hyperparameters

| Flag                 | Default            | Effect                                                          |
|----------------------|--------------------|-----------------------------------------------------------------|
| `--steps`            | 10000              | More steps → better accuracy (try 20 000–50 000 for production) |
| `--samples`          | 1000               | More positive TTS samples → better robustness (up to ~5000)     |
| `--neg_class_weight` | 20                 | Higher → fewer false activations, lower recall                  |
| `--phonetic`         | same as `--phrase` | Phonetic spelling for TTS (e.g. `"oh kay"`)                     |
| `--data_dir`         | `./data`           | Output directory for generated files (samples, features, model) |
| `--downloads_dir`    | same as `--data_dir` | Directory for downloaded datasets — share across runs to avoid re-downloading 10 GB |

## Tune the model after deployment

Edit `firmware/models/lumus.json` and adjust `probability_cutoff`:

- **Raise** (e.g. `0.7` → `0.9`) to reduce false triggers
- **Lower** (e.g. `0.7` → `0.5`) if it misses you too often

Reflash after every change. Current ROC operating points from the trained `lumus` model:

| Cutoff | Miss rate | False accepts/hour |
|--------|-----------|--------------------|
| 0.98   | 6%        | ~0                 |
| 0.44   | 2%        | ~0.2               |

## Browser tester

Serve locally (required — WASM won't load from `file://`):

```bash
cd docs && python3 -m http.server 8080
# Open http://localhost:8080
```

Or visit **https://a-d-j-i.github.io/wakeword-relay/**

Load a `.tflite`, then:

- **Test WAV file** — per-frame scores, peak, and detected/not
- **Scan directory** — batch inference over a folder of WAV files
- **Live mic** — streaming inference identical to the ESP32 cadence

The tester reimplements the ESP32 TFLM audio pipeline in WebAssembly: Hann window → 512-pt FFT → 40 mel bins (125–7500
Hz) → PCAN normalization → INT8 inference.

## Credits

The training pipeline is built on top of **[microWakeWord](https://github.com/kahrendt/microWakeWord)**
by [Kevin Ahrendt (@kahrendt)](https://github.com/kahrendt), licensed under
the [Apache 2.0 License](https://github.com/kahrendt/microWakeWord/blob/main/LICENSE). The model architecture, training
framework, and dataset pipeline all originate from that project. `train.py` automates the steps from
the [basic training notebook](https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb)
with the compatibility fixes and improvements listed below.

Positive sample generation uses **[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)** by
Rhasspy, with the LibriTTS-R TTS model. Background augmentation data comes from the MIT Environmental Impulse Responses
dataset, AudioSet, and the Free Music Archive (FMA).

### Compatibility patches

Applied directly to the vendored source in `train/microwakeword/` to fix breakage with current library versions:

| File                    | Change                                                  | Reason                                                              |
|-------------------------|---------------------------------------------------------|---------------------------------------------------------------------|
| `audio/audio_utils.py`  | `use_c=False`                                           | `pymicro-features` MicroFrontend segfaults on some Linux systems    |
| `audio/augmentation.py` | Remove `AddColorNoise`                                  | Removed from `audiomentations >= 0.36`                              |
| `train.py`, `test.py`   | `np.trapz` → `np.trapezoid`                             | Deprecated in NumPy 2.0                                             |
| `train.py`              | Remove `.numpy()` on `model.evaluate()` results         | Returns plain NumPy arrays in TF 2.16+                              |
| `audio/clips.py`        | Load WAV with `scipy.io.wavfile` instead of `soundfile` | `libsndfile` segfaults after TF initialises its custom malloc hooks |
| `audio/clips.py`        | Replace `audio_metadata.load()` with `torchaudio.info()` for non-WAV duration | Removes `audio-metadata` dep — PyPI version hard-pins `attrs`, causing conflicts |
| `audio/clips.py`        | Remove dead `import datasets`                           | Leftover from original HuggingFace audio path; unused after scipy patch        |
| `audio/__init__.py`, `layers/__init__.py` | Added (were missing)                   | Required for proper package import; wheel silently dropped these subdirs        |

### Improvements over the microWakeWord basic training notebook

| Area             | Notebook                                            | `train.py`                                                                    |
|------------------|-----------------------------------------------------|-------------------------------------------------------------------------------|
| Interface        | Jupyter cells, hard-coded paths                     | `argparse` CLI — `--phrase`, `--phonetic`, `--samples`, `--steps`, etc.       |
| Data layout      | Single directory for everything                     | `--downloads_dir` / `--data_dir` split — reuse 10 GB of downloads across runs |
| AudioSet loading | `cast_column` → crashes with `torchcodec` installed | `torchaudio.load` directly                                                    |
| FMA loading      | Same crash                                          | `torchaudio.load` directly                                                    |
| MIT RIRs loading | `streaming=True`, no `cast_column`                  | Matches notebook                                                              |
| Progress         | No ETA                                              | Step-level ETA during training                                                |
| Resumability     | Restarts from scratch on re-run                     | Each stage skips if output already exists                                     |
| SpecAugment      | Disabled (`time/freq_mask = [0]`)                   | Enabled (`time_mask 5×2`, `freq_mask 5×2`) — masks spectrogram bands to regularize / generalize past the few TTS voices |