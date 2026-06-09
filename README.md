# wakeword-relay

Offline wake-word detection on an ESP32-S3 that toggles a relay. The model runs entirely on-device — no Wi-Fi, no Home Assistant, no cloud at runtime.

- **`train/`** — CLI pipeline to train a custom INT8 TFLite wake word model
- **`firmware/`** — ESPHome YAML + model manifest for ESP32-S3 deployment
- **[Browser tester](https://a-d-j-i.github.io/wakeword-relay/)** — load a `.tflite` and test against WAV files or your live mic

## Hardware

- **MCU**: ESP32-S3 N16R8 (16MB flash, 8MB octal PSRAM, Vector Extension for INT8 inference)
- **Mic**: I2S digital — INMP441, ICS-43434, or SPH0645. Tie `SEL` → GND for the left channel.
- **Relay**: any GPIO; default is GPIO5 (change in `wakeword-relay.yaml`)

### Wiring

| Signal | Mic pin | ESP32-S3 GPIO |
|--------|---------|---------------|
| BCLK / SCK | SCK | GPIO1 |
| LRCLK / WS | WS | GPIO2 |
| Data in | SD | GPIO3 |
| Relay control | — | GPIO5 |

## Flash the firmware

```bash
pip install esphome
esphome run firmware/wakeword-relay.yaml   # compile + flash over USB
esphome logs firmware/wakeword-relay.yaml  # watch for "Detected 'lumus'"
```

The firmware ships with a custom-trained `lumus` model in `firmware/models/`. To switch to a prebuilt model (`okay_nabu`, `hey_jarvis`, `hey_mycroft`), edit the `models:` block in `wakeword-relay.yaml`.

## Train a custom wake word

```bash
cd train
python3.10 -m venv .venv-mww && source .venv-mww/bin/activate
pip install -r requirements.txt

# Basic run — 1000 samples, 10 000 steps
python train.py --phrase "hey jarvis"

# With phonetic spelling for better TTS quality
python train.py --phrase "okay computer" --phonetic "oh kay kompyooter" --samples 2000 --steps 20000
```

**First run downloads ~10 GB** of datasets into `train/data/`. Output:

```
train/data/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

Copy the `.tflite` to `firmware/models/`, update the manifest JSON, and reflash.

See [`firmware/train.md`](firmware/train.md) for detailed guides: Google Colab, local CUDA GPU, and CPU-only.

### Key training hyperparameters

| Flag | Default | Effect |
|------|---------|--------|
| `--steps` | 10000 | More steps → better accuracy (try 20 000–50 000 for production) |
| `--samples` | 1000 | More positive TTS samples → better robustness (up to ~5000) |
| `--neg_class_weight` | 20 | Higher → fewer false activations, lower recall |
| `--phonetic` | same as `--phrase` | Phonetic spelling for TTS (e.g. `"oh kay"`) |

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

The tester reimplements the ESP32 TFLM audio pipeline in WebAssembly: Hann window → 512-pt FFT → 40 mel bins (125–7500 Hz) → PCAN normalization → INT8 inference.