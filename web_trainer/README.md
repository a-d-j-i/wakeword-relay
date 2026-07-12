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
# WASM build (outputs to ../docs/trainer/ + ../docs/common/)
source /home/work/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm --target web_trainer -j4

# Native build (for C++ unit tests)
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Debug
cmake --build build-native --target web_trainer
```

The WASM build copies `web_trainer.js` and `web_trainer.wasm` into `../docs/trainer/` automatically.

**Build directories** — all are generated outputs (git-ignored):

| Directory | Purpose | Needed? |
|-----------|---------|---------|
| `build-wasm/` | Emscripten WASM build → `../docs/trainer/` | Yes — primary build |
| `build-native/` | Native x86 build → C++ unit tests | Yes, if running native tests |
| `build/` | Stale WASM build (old directory name) | Delete — superseded by `build-wasm/` |
| `build_native/` | Stale native build (old naming) | Delete — superseded by `build-native/` |

To build `piper_phonemize` only (fast, just relinking — needed after adding languages or changing its CMake flags):

```bash
source /home/work/emsdk/emsdk_env.sh
cmake --build build-wasm --target piper_phonemize
```

### Running the app locally

Any static file server works:

```bash
cd ../docs
python3 -m http.server 8080
# open http://localhost:8080/trainer/
```

---

## Using the app

The app has seven tabs. Work through them roughly in order for a full training run.

### Secure context

For full functionality, serve via **HTTPS or localhost**. On plain HTTP (e.g. `http://192.168.x.x`), a yellow warning
banner lists what is unavailable:

| Feature | Requires | Impact |
|---------|----------|--------|
| WebGPU  | secure context | Piper ORT uses WASM fallback (slower synthesis) |
| OPFS    | secure context | Background noise bundle not cached (must re-download each session) |
| AudioWorklet (live mic in Test tab) | secure context | Live mic inference unavailable |

Everything else — training, TTS, augmentation, WAV inference — works on plain HTTP.

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
- French: `fr/fr_FR/siwis/medium/` — `fr_FR-siwis-medium.onnx` + `.onnx.json`
- German: `de/de_DE/thorsten/medium/` — `de_DE-thorsten-medium.onnx` + `.onnx.json`
- More at [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

The loaded model is cached in IndexedDB and auto-loads on the next visit. Click **Clear cached model** to remove it.

**Step 2 — Load model**

Select the `.onnx` and `.onnx.json` files and click **Load**. A Blob URL is created in-browser; no data leaves your
machine.

**Step 3 — Choose language**

Select the espeak-ng language that matches your voice model. English (`en`) and Spanish (`es`) are pre-loaded in the
WASM binary. All other languages are fetched on first use from `espeak-ng-data/` (~50–200 KB per language, cached by
the browser). The language affects:

- Pronunciation rules (letter-to-sound, phoneme set, stress)
- Negative sample generation phrases (the negatives "Generate in-browser" section uses phrases in the selected language)

Use the same language everywhere (TTS, Preview, Train, Negatives).

**Step 4 — Generate**

Type your wake phrase, set sample count (50–200 recommended for training), click **Generate**. Up to 4 Web Workers
synthesise in parallel. Each sample uses randomised `noise_scale`, `length_scale`, `noise_w` for diversity.

### Augment tab

Preview the augmentation pipeline on a single WAV file. Adjust pitch, EQ, reverb (choose a specific room IR or random),
and noise SNR. The "before" and "after" spectrograms show the effect.

This is the same augmentation applied during training — use it to tune parameters before committing to a long run.

### Preview tab

Run the full TTS + augmentation pipeline end-to-end, generating N augmented samples from your phrase. Each result shows
the audio player and spectrogram so you can hear what the training data actually sounds like.

### Record tab

Record your own voice samples via the microphone and add them to the training pool. Useful for supplementing TTS with
real voice data, especially for phrases that TTS pronounces poorly.

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
passes for activation quantization. The exported model is also saved to IndexedDB so the **Test tab** auto-loads it on
next visit.

### Test tab

Test a trained model against live microphone input or uploaded WAV files.

- **Model loading** — auto-loads from IndexedDB after a Train tab export; or pick any `.tflite` file manually
- **Live mic** — AudioWorklet → WASM MicroFrontend → stride-T streaming inference → score bar + rolling spectrogram +
  detection log with beep
- **WAV inference** — upload any WAV; shows peak score, detection result, and a per-frame timeline
- **Threshold slider** — adjust the detection threshold (default 0.5)
- **ESP32 exact quantization** — use the firmware's fixed `256/666` quantization formula instead of reading it from the
  model FlatBuffer (matches on-device behaviour exactly)

---

## Data bundles

### Negative samples (required)

Pre-computed negative spectrograms (speech, dinner-party, silence) in the custom `MWWN` binary format. The file is
**37 MB and not committed to git** — it must be generated once and placed (or hosted) separately.

**Option A — Build and upload to GitHub Releases** (recommended):

```bash
# 1. Run train.py at least once to download the HuggingFace negative datasets (~10 GB)
cd train && source venv/bin/activate
python train.py --phrase "hey lumus" --steps 100   # minimal run just to trigger downloads

# 2. Build the bundle and upload directly to a GitHub Release
python3 web_trainer/tools/build_negatives_bundle.py \
  --input_dir train/data/negative_datasets \
  --output negatives.mwwn \
  --upload \
  --repo your-user/wakeword_relay \
  --tag bundles \
  --token ghp_yourPersonalAccessToken
# → prints a download URL, e.g.:
#   https://github.com/your-user/wakeword_relay/releases/download/bundles/negatives.mwwn
```

The token needs `Contents: write` scope. Pass it via `--token` or the `GITHUB_TOKEN` env var. Re-running with
the same `--tag` replaces the existing asset cleanly.

**Option B — Build locally, load by URL**

Build without `--upload` and host `negatives.mwwn` anywhere (your own server, S3, etc). Paste the URL into
the Train tab's "Load negatives from URL" field. The browser caches it in IndexedDB after the first download.

**Option C — Generate in-browser** (no bundle needed)

The Train tab → Negatives card → "Generate in-browser" uses the loaded Piper model to synthesise diverse speech
samples directly. Smaller and less diverse than the HuggingFace set, but enough for a first training run.

### Language data

The espeak-ng phonemizer supports 100+ languages. English and Spanish are pre-bundled in `piper_phonemize.data`
(always available, no network request). All other languages are lazy-loaded from `../docs/trainer/espeak-ng-data/` on first use:

- `../docs/trainer/espeak-ng-data/<lang>_dict` — compiled pronunciation dictionary (~50–200 KB)
- `../docs/trainer/espeak-ng-data/lang/<family>/<lang>` — language config (few KB)

Once fetched, the browser HTTP cache keeps them across sessions. The language map is in `../docs/trainer/js/lang_loader.js`. The `../docs/trainer/espeak-ng-data/` files are committed to the repo (12 MB) so
the app works on a fresh clone without any setup step.

**Rebuilding language data** (after upgrading espeak-ng or adding a new language):

```bash
# 1. Update system package
sudo apt install espeak-ng-data

# 2. Copy to vendor/ (updates the WASM preload for en+es, and shared phoneme tables)
cp /usr/lib/x86_64-linux-gnu/espeak-ng-data/*_dict  vendor/espeak-ng/data/
cp -r /usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/. vendor/espeak-ng/data/lang/

# 3. Copy to ../docs/trainer/ (static files for lazy loading)
cp /usr/lib/x86_64-linux-gnu/espeak-ng-data/*_dict  ../docs/trainer/espeak-ng-data/
cp -r /usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/. ../docs/trainer/espeak-ng-data/lang/

# 4. Rebuild piper_phonemize only (fast — just relink, no C recompilation)
source /home/work/emsdk/emsdk_env.sh
cmake --build build-wasm --target piper_phonemize

# 5. If new language codes appeared, add them to _LANG_PATHS in ../docs/trainer/js/lang_loader.js
#    (generate the mapping with:)
find /usr/lib/x86_64-linux-gnu/espeak-ng-data/lang -type f | sort | while read f; do
    family=$(basename $(dirname $f)); code=$(basename $f)
    echo "\"$code\": \"$family/$code\""
done
```

Note: `eu`, `ko`, `qu` live directly in `lang/` with no subfamily — their path is just the code itself (e.g.
`"eu": "eu"`), not `"eu": "lang/eu"`.

### Background noise (optional, ~100 MB)

Ambient audio clips (AudioSet + FMA music) in MWWB format.

```bash
python3 tools/build_noise_bundle.py \
  --input_dirs /path/to/audioset_16k /path/to/fma_16k \
  --output ../docs/trainer/noise.mwwb
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

### JavaScript (`../docs/trainer/js/`)

| File               | Purpose                                                                                                                    |
|--------------------|----------------------------------------------------------------------------------------------------------------------------|
| `piper.js`         | Piper TTS: `initPhonemizer()` (espeak-ng), `textToIpa()`, `ipaToIds()`, `synthesise()`, `generateSamples()` (worker pool) |
| `piper_worker.js`  | Web Worker: ORT VITS inference (no espeak-ng — receives pre-computed phoneme IDs)                                          |
| `piper_cache.js`   | IndexedDB cache for the loaded Piper `.onnx` + `.onnx.json` (persists across sessions)                                    |
| `lang_loader.js`   | Lazy-loads espeak-ng language data into WASM FS on demand; `loadLanguage(code)` fetches `*_dict` + lang config             |
| `augment.js`       | Audio augmentation: pitch shift, EQ, reverb (24-bit WAV IRs), noise mixing; WAV load/encode                               |
| `spectrogram.js`   | WASM `audioToSpectrogram()` wrapper + canvas `drawSpectrogram()`                                                           |
| `trainer.js`       | WASM training wrappers: `audioToFloat32Spec()`, `trainCreate/Destroy/Step/GetParams`, `trainExportTFLite()`                |
| `negatives.js`     | Negative bundle: MWWN parse, IndexedDB cache, `negGetSample()`                                                             |
| `noise.js`         | Noise bundle: MWWB parse, OPFS cache, `noiseGetSample()`                                                                   |
| `tflite_cache.js`  | IndexedDB cache for the last exported TFLite model — shared between Train and Test tabs                                    |
| `test_tab.js`      | Test tab: FlatBuffer quant parsing, stride-T inference, live mic AudioWorklet, WAV inference with timeline                 |
| `train_worker.js`  | Web Worker: full MixedNet training loop + TFLite export (runs off main thread)                                             |
| `app.js`           | UI glue: tab switching, all event handlers, training loop orchestration, secure-context warning                            |

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

- `../docs/trainer/tflite_template.bin` — a `train.py`-produced TFLite (buenas_noches, 62 KB) used as graph topology template
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
