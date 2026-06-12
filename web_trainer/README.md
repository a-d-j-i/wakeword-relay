# web_trainer

Browser-based wake-word training pipeline. Train an INT8-quantized TFLite model using only a browser — no Python, no
GPU, no cloud. All computation runs locally.

## What it does

Replicates the `train/train.py` pipeline entirely in-browser:

1. **Piper TTS** — synthesise dozens of varied pronunciations of your wake phrase (VITS ONNX model + espeak-ng
   phonemizer, both running as WASM/Web Workers)
2. **Augmentation** — pitch shift, 7-band EQ, room reverb (bundled impulse responses), background noise mixing
3. **MixedNet training** — MicroFrontend spectrogram → MixedNet forward+backward+Adam, all in C++ compiled to WASM
4. **TFLite export** — streaming INT8 TFLite ready to flash to ESP32-S3 (no Python needed)

---

## Building

Requires [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) for the WASM target.

```bash
# WASM build (outputs to www/)
source /home/work/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm --target web_trainer -j4

# Native build (for C++ unit tests)
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Debug
cmake --build build-native --target web_trainer
```

The WASM build copies `web_trainer.js` and `web_trainer.wasm` into `www/` automatically.

### Running the app locally

Any static file server works:

```bash
cd www
python3 -m http.server 8080
# open http://localhost:8080
```

---

## Using the app

The app has four tabs. Work through them in order for a full training run.

### Frontend tab

Drag any WAV file to visualise its 40-channel log-mel filterbank spectrogram — the same features the model trains on.
Useful for sanity-checking audio quality.

### TTS tab

Generates positive training samples via Piper TTS.

**Step 1 — Get a voice model**

Download a `.onnx` + `.onnx.json` pair
from [rhasspy/piper-voices on HuggingFace](https://huggingface.co/rhasspy/piper-voices):

- English: `en/en_US/amy/medium/` — `en_US-amy-medium.onnx` + `.onnx.json`
- Spanish (Argentine): `es/es_AR/daniela/high/` — `es_AR-daniela-high.onnx` + `.onnx.json`

**Step 2 — Load model**

Select the `.onnx` and `.onnx.json` files and click **Load**. A Blob URL is created in-browser; no data leaves your
machine.

**Step 3 — Generate**

Type your wake phrase, set sample count (50–200 recommended for training), click **Generate**. Up to 4 Web Workers
synthesise in parallel — one ORT session per worker, phoneme IDs computed once on the main thread. Each sample uses
randomised `noise_scale`, `length_scale`, `noise_w` for diversity.

Results appear with audio players and spectrograms.

### Augment tab

Preview the augmentation pipeline on a single WAV file. Adjust pitch, EQ, reverb (choose a specific room IR or random),
and noise SNR. The "before" and "after" spectrograms show the effect.

This is the same augmentation applied during training — use it to tune parameters before committing to a long run.

### Train tab

Full training loop. Requires TTS model (loaded in TTS tab) and negative samples (see below).

**Prepare TTS samples** — generates N positive samples from the TTS model (same as TTS tab but stored for training). 50
samples is a reasonable default.

**Start Training** — runs the MixedNet training loop:

- Hyperparameters: steps (1000–10 000), neg-per-step (5), learning rate (0.001), architecture (full-window recommended)
- Loss curve updates every 10 steps
- Stop at any time; weights are preserved

**Export weights** — downloads `wakeword_weights.bin` (float32, `MWWW` magic header). Can be loaded in a future session
to continue training.

**Export TFLite** — downloads `wakeword.tflite` (streaming INT8, ready for ESP32-S3). Internally runs 500 calibration
passes for activation quantization.

---

## Data bundles

### Negative samples (required)

Pre-computed negative spectrograms (speech, dinner party, silence) in MWWN format.

**Build from `train.py` output:**

```bash
python3 tools/build_negatives_bundle.py \
  --input_dir /path/to/train/data/negative_datasets \
  --output www/negatives.mwwn
```

In the Train tab, click **use bundled** (loads `negatives.mwwn` from the same server) or paste a URL. Cached in
IndexedDB — survives page reloads.

### Background noise (optional, ~100 MB)

Ambient audio clips (AudioSet + FMA music) in MWWB format.

```bash
python3 tools/build_noise_bundle.py \
  --input_dirs /path/to/audioset_16k /path/to/fma_16k \
  --output www/noise.mwwb
```

Cached in OPFS — survives reloads. Without it, training still works (just no noise augmentation).

---

## Architecture

### C++ / WASM (`src/`)

| File                   | Purpose                                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------------------|
| `main.cpp`             | Emscripten entry point; exports `frontend_create/process/destroy`, `mixednet_*`, `adam_*`, `train_step`     |
| `mixednet.cpp/.h`      | MixedNet: depthwise separable + mixed convolutions, forward + backward pass                                 |
| `nn.cpp/.h`            | Primitive layers: conv2d, batchnorm, dense, relu; Adam optimizer                                            |
| `tflite_export.cpp/.h` | Streaming INT8 TFLite export: calibration, BN folding, per-channel INT8 quant, FlatBuffer template patching |
| `phonemize.cpp`        | Thin C wrapper around the espeak-ng WASM module                                                             |

### JavaScript (`www/js/`)

| File              | Purpose                                                                                                                   |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| `piper.js`        | Piper TTS: `initPhonemizer()` (espeak-ng), `textToIpa()`, `ipaToIds()`, `synthesise()`, `generateSamples()` (worker pool) |
| `piper_worker.js` | Web Worker: ORT VITS inference (no espeak-ng — receives pre-computed phoneme IDs)                                         |
| `augment.js`      | Audio augmentation: pitch shift, EQ, reverb, noise mixing; WAV load/encode                                                |
| `spectrogram.js`  | WASM `audioToSpectrogram()` wrapper + canvas `drawSpectrogram()`                                                          |
| `trainer.js`      | WASM training wrappers: `audioToFloat32Spec()`, `trainCreate/Destroy/Step/GetParams`, `trainExportTFLite()`               |
| `negatives.js`    | Negative bundle: MWWN parse, IndexedDB cache, `negGetSample()`                                                            |
| `noise.js`        | Noise bundle: MWWB parse, OPFS cache, `noiseGetSample()`                                                                  |
| `app.js`          | UI glue: tab switching, all event handlers, training loop                                                                 |

### TTS parallelism

`generateSamples` spawns a pool of up to 4 Web Workers. The main thread:

1. Computes IPA + phoneme IDs once (espeak-ng already loaded)
2. Creates a Blob URL for the `.onnx` model file

Each worker:

1. Fetches the model from the Blob URL and creates its own ORT session
2. Receives phoneme IDs as a transferred `ArrayBuffer` + random `noise_scale/length_scale/noise_w`
3. Returns the WAV as a transferred `ArrayBuffer`

Workers are terminated when all samples are done.

### MixedNet architecture

Two variants (selectable in Train tab):

| Variant                | Params | Min frames | Output layer                   |
|------------------------|--------|------------|--------------------------------|
| Full window (pooled=0) | 25 825 | 204        | flatten(17×64) → Dense(1088→1) |
| Pooled (pooled=1)      | 24 801 | 157        | global avg pool → Dense(64→1)  |

Full window is the production default (matches `train.py` output).

### TFLite export

Uses a template-and-patch approach:

- `www/tflite_template.bin` — a `train.py`-produced TFLite (buenas_noches, 62 KB) used as graph topology template
- `src/tflite_export.cpp` runs 500 calibration passes, folds batchnorm into conv weights, quantizes weights to INT8
  per-channel and activations asymmetrically, then patches weight buffers + quant params into the FlatBuffer template
  via `UnPackModel()` → `Model::Pack()`
- Output is a valid streaming TFLite: input `[1,3,40]` int8, output `[1,1]` uint8

---

## Tests

```bash
cd /home/work/wakeword_relay/web_trainer
npx playwright test           # all specs
npx playwright test trainer   # trainer.spec.js only (13 tests)
```

The playwright config starts an HTTP server automatically.

---

## Deploying to ESP32-S3

1. Click **Export TFLite** in the Train tab — saves `wakeword.tflite`
2. Copy to `firmware/models/turn_on.tflite` (or `turn_off.tflite`)
3. Flash: `esphome run firmware/wakeword-relay.yaml`

The `wake_word` field in the JSON manifest (`"turn_on"` / `"turn_off"`) is what ESPHome matches on — it is completely
decoupled from the phrase you trained on.
