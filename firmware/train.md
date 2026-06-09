# Training a custom wake word

## How custom wake words work

The microWakeWord training framework (https://github.com/kahrendt/microWakeWord)
generates thousands of spoken variants of your phrase using Piper TTS (many
synthetic voices), then augments them with background noise, music, and
room-reverb impulse responses. That augmented dataset trains a small streaming
model that gets quantized to int8 and exported as:

- a `.tflite` model file, plus
- a `.json` manifest (metadata: probability cutoff, sliding-window size, etc.)

Those two files are what ESPHome consumes (see `./my_models/` in
`wakeword-relay.yaml`).

## The big picture

1. **Pick a good phrase.** This matters more than any hyperparameter:
   - 3–4 syllables ("computer wake up", "okay sentinel") — too short = false
     triggers, too long = harder to detect reliably.
   - Distinct, uncommon word combos — avoid everyday words you'd say in normal
     conversation, or the relay will flip randomly.
   - Phonetically rich beats flat. Hard consonants and varied vowels are easier
     for the model to lock onto than a string of soft syllables.
2. **Train it** (the heavy part — see below). Expect 1–3 hours on a free Colab
   GPU, and plan to iterate; first attempts are usually too trigger-happy or
   too deaf.
3. **Get back** `your_phrase.tflite` + `your_phrase.json`.
4. **Point ESPHome at it** (deployment — one-line swap, see the end).

---

## Step-by-step: training in Google Colab (recommended)

The maintained, reproducible path is the project's Colab notebook. You need a
GPU runtime; Colab's free T4 tier works but is slow.

### 0. Prerequisites

- A Google account (for Colab + saving outputs to Google Drive).
- A target phrase chosen per the rules above. The examples below use
  **`okay sentinel`** — replace it everywhere with yours.
- ~30–60 min of attention spread over a couple of hours of wall-clock time.

### 1. Open the notebook and select a GPU

1. Open the official basic training notebook:
   https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb
   (open it in Colab via *File → Open notebook → GitHub*, or click the
   "Open in Colab" badge in that repo).
2. **Runtime → Change runtime type → Hardware accelerator: GPU → Save.**
3. Confirm the GPU is live:
   ```python
   !nvidia-smi
   ```
   You should see a T4 (or better). If it errors, the runtime isn't on GPU.

### 2. Install the framework and dependencies

The first cells clone and install everything. Run them in order:

```bash
!git clone https://github.com/kahrendt/microWakeWord
%cd microWakeWord
!pip install -e .
# Piper sample generator (the TTS that makes your positive examples)
!git clone https://github.com/rhasspy/piper-sample-generator
!pip install -r piper-sample-generator/requirements.txt
```

Restart the runtime if Colab prompts you after the installs, then re-run the
`%cd microWakeWord` cell so you're back in the working directory.

### 3. Generate positive samples (your phrase, many voices)

Piper synthesizes thousands of spoken variants of the phrase across many voices,
speeds, and intonations. This is your positive class.

```bash
!python piper-sample-generator/generate_samples.py \
    --text "okay sentinel" \
    --max-samples 2000 \
    --batch-size 100 \
    --output-dir generated_samples/okay_sentinel
```

- `--max-samples 2000` is a reasonable start. More samples (5000+) generally
  improves robustness at the cost of generation + training time.
- Listen to a few of the generated `.wav` files. If Piper mispronounces your
  phrase, fix the spelling phonetically (e.g. "sentinel" → "sen tin ell") until
  it sounds right — the model can only learn what the TTS actually says.

### 4. Get the negative / background data

The model needs to learn what *isn't* the wake word: ambient noise, music,
speech, and room acoustics. The notebook downloads these standard datasets —
run the cells as written. They include:

- **Negative speech / ambient** — large corpora of general audio so the model
  learns to stay quiet during normal life (e.g. AudioSet subsets, the
  "dinner party" / DiPCo conversational set).
- **Background noise & music** — for mixing under the positive samples.
- **Room impulse responses (RIRs)** — convolved in to simulate reverberation,
  so the model generalizes beyond a clean recording booth.

These downloads are several GB and are the slowest non-training step. Let them
finish before moving on.

### 5. Augment + compute spectrogram features

The notebook then mixes the positive samples with noise/music/reverb at varying
SNRs and converts everything to the spectrogram features the model trains on.
Run the augmentation and feature-generation cells. This produces the ragged
feature files (positive, negative, and background) that feed training.

You don't normally need to edit this stage; just verify each cell completes
without throwing, and sanity-check the reported sample counts (you should see
your ~2000 positives plus the much larger negative/background pools).

### 6. Configure training parameters

There's a `training_parameters.yaml` (the notebook writes/edits it). The knobs
worth knowing:

- `window_step_ms` — sliding-window stride; leave at the notebook default
  (commonly 20 ms) unless you know why you're changing it.
- `train_dir` / feature paths — point at the features generated in step 5
  (the notebook sets these for you).
- training steps / `maximum_files` / batch sizes — controls how long it trains.
  Defaults are tuned to be sane; raise step counts only if the model is
  underfitting (see step 8).

Most first runs should use the defaults as-is. Change one thing at a time when
iterating.

### 7. Train

Run the training cell. It trains a small streaming MixedNet-style model, logging
loss and accuracy as it goes:

```python
# (this is the notebook's training invocation — run its cell, don't retype)
!python -m microwakeword.model_train_eval \
    --training_config training_parameters.yaml \
    --train 1 --restore_checkpoint 1 \
    --train_dir trained_models/okay_sentinel
```

- On a free T4 this is the long pole — tens of minutes to a couple of hours.
- Watch the validation metrics. The notebook reports **accuracy, recall, and
  false-accepts-per-hour** on a held-out set. Recall = how often it catches the
  word; false-accepts/hour = how often it fires when it shouldn't.

### 8. Evaluate and read the metrics

After training the notebook quantizes the model to int8 and evaluates the
`.tflite`. Aim for:

- **High recall** (catches you most of the time) **with low false-accepts/hour**
  (ideally well under ~1/hour for a relay you don't want flipping at night).
- If recall is low → the model is too deaf: add more positive samples (step 3),
  train for more steps, or pick a more phonetically distinct phrase.
- If false-accepts/hour is high → too trigger-happy: more/varied negative data,
  more training, or raise the probability cutoff at deploy time (cheaper, see
  the tuning section).

### 9. Export the two files

The final cells produce the deployment artifacts:

- `okay_sentinel.tflite` — the quantized model.
- `okay_sentinel.json` — the manifest. It references the `.tflite` by name and
  carries the metadata ESPHome needs, including the **`probability_cutoff`** and
  sliding-window settings.

Download both (or save them to Google Drive). **Keep them side-by-side** — the
manifest points at the `.tflite` by filename, so they must live in the same
folder.

---

## Local training (CUDA / NVIDIA GPU, e.g. RTX 3060 12GB)

Training locally is a better fit than Colab once you start iterating: datasets
download **once** instead of every session, there are no timeouts, and a 12GB
Ampere card is massively overprovisioned for this (the model is a few hundred
KB and uses ~1–2GB of VRAM). The real bottleneck is **not** the GPU — it's the
several-GB dataset downloads and the CPU/RAM-bound feature generation. GPU
training itself takes minutes.

The RTX 3060 is compute capability 8.6 (Ampere), fully supported by current
CUDA 11.8/12.x and the TensorFlow/PyTorch wheels.

### OS matters

- **Linux** → straightforward; follow the steps below.
- **Windows** → native TensorFlow GPU support ended at TF 2.10 (too old). Install
  the NVIDIA driver on the Windows host, then do everything inside **WSL2
  (Ubuntu)** using the Linux steps. Don't fight native-Windows TF.

### Setup with a plain venv

```bash
# 0. Confirm the driver sees the card (need driver >= 525 for CUDA 12)
nvidia-smi

# 1. System deps (Ubuntu/Debian)
sudo apt install python3.11 python3.11-venv git ffmpeg

# 2. microWakeWord training env
python3.11 -m venv ~/mww && source ~/mww/bin/activate
pip install --upgrade pip
git clone https://github.com/kahrendt/microWakeWord && cd microWakeWord
pip install -e .

# 3. GPU TensorFlow — bundles matching CUDA + cuDNN as pip wheels (Linux only).
#    This is the trick that sidesteps the classic CUDA-version mismatch hell.
pip install 'tensorflow[and-cuda]'

# 4. Verify TF actually sees the GPU BEFORE wasting time training:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#    -> must print a list with one GPU. Empty list = CUDA not wired up; stop and fix.
```

Then Piper for positive-sample generation:

```bash
git clone https://github.com/rhasspy/piper-sample-generator
pip install -r piper-sample-generator/requirements.txt
# If torch installs CPU-only, force the CUDA build:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

From here, **steps 3–9 above are identical** — just run the same
`generate_samples.py` / training commands locally instead of in notebook cells.
Datasets cache on disk, so iterate freely.

### Does conda help with the version issues?

Yes, but use it for the *right* job. Conda's real value here is **isolating the
Python version** (you get a clean 3.11 without touching system Python) and giving
you a reproducible `environment.yml`.

The important caveat: with TensorFlow ≥ 2.15, **do not** also `conda install
cudatoolkit cudnn`. The modern, supported way to get CUDA for TF is the pip
wheels (`tensorflow[and-cuda]`), even inside a conda env. Installing CUDA *both*
via conda *and* via the pip wheels is the #1 way to create the exact version
conflict you're trying to avoid — two copies of the CUDA libs fighting over which
one gets loaded.

So the clean recipe is "conda for Python, pip for CUDA-bearing packages":

```bash
conda create -n mww python=3.11
conda activate mww
git clone https://github.com/kahrendt/microWakeWord && cd microWakeWord
pip install -e .
pip install 'tensorflow[and-cuda]'        # CUDA comes from pip, NOT conda
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

(The old `conda install -c conda-forge cudatoolkit=11.8 cudnn` approach still
works for *older* TF versions, but microWakeWord tracks recent TF, so the pip-CUDA
path is the one to use.)

### Notes specific to a 12GB card

- **VRAM is a non-issue.** Defaults fit with room to spare; you *could* raise the
  training batch size for marginally faster training, but it won't change much —
  the model is tiny.
- **Budget ~20–30GB of disk** for the downloaded negative/background datasets
  plus the generated spectrogram features.
- **TF + PyTorch in one env is heavy but coexists fine.** If you hit a CUDA-lib
  conflict between them, the clean fix is two separate envs — one for
  Piper/torch generation, one for TF training — since those steps run
  sequentially anyway.
- If `pip install -e .` and `tensorflow[and-cuda]` disagree on a TF version,
  trust the repo's pin and let `[and-cuda]` pull the matching CUDA wheels.

---

## Local training (CPU-only, no GPU)

Training entirely on CPU is **viable**, not a hack. The GPU only accelerates one
step (training, step 7), and that step is the *small* part — the model is a few
hundred KB. The expensive work (dataset downloads, Piper generation, and the
augmentation + feature stage) is CPU/RAM-bound **regardless** of whether you
have a GPU, so on a CPU-only box you're only paying extra on the training loop
itself.

pip install torchcodec torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### What changes vs. the CUDA setup

Just one thing: install the CPU-only `tensorflow-cpu` package instead of the
CUDA wheels.

```bash
# 1. System deps (Ubuntu/Debian)
sudo apt install python3.11 python3.11-venv git ffmpeg

# 2. microWakeWord training env
python3.11 -m venv ~/mww && source ~/mww/bin/activate
pip install --upgrade pip
git clone https://github.com/kahrendt/microWakeWord && cd microWakeWord
pip install -e .

# 3. CPU-only TensorFlow — the `-cpu` package, NOT plain `tensorflow`.
#    Plain `tensorflow` pulls in the GPU/CUDA machinery; `tensorflow-cpu` is the
#    slim CPU build with none of it.
pip install tensorflow-cpu

# 4. Piper for positive-sample generation (CPU torch is fine here)
git clone https://github.com/rhasspy/piper-sample-generator
pip install -r piper-sample-generator/requirements.txt
```

There's **no GPU-verify step** to run and nothing to configure — TensorFlow
falls back to CPU automatically. From here, **steps 3–9 above are identical**;
run the same `generate_samples.py` / training commands, just on CPU.

### Realistic expectations on CPU

- **Training (step 7)** is the only meaningfully slower step. Budget a few hours
  to overnight on a typical multi-core CPU, vs. tens of minutes on a free T4.
  Slow, not blocked.
- **RAM matters more than cores.** Feature generation and the data pipeline are
  memory-hungry — 16GB is tight, 32GB is comfortable. This is the constraint
  most likely to bite you, not CPU speed.
- **Disk** is the same ~20–30GB as the GPU path.
- **Iteration is the real cost.** Wake-word work is tweak-and-retrain; each
  retrain is the multi-hour CPU loop, so the loop gets long.

### Hybrid option

Since generation + features are CPU-bound either way, a good middle path is:
generate samples and compute features locally on CPU (they cache to disk once),
then upload just the cached feature files to Colab and run **only** the training
cell on the free T4. You get the slow-but-free local prep without paying the
training penalty on every iteration.

---

## Deploying into this project

1. Create a `my_models/` folder next to `wakeword-relay.yaml` and drop both
   files in:
   ```
   wakeword_relay/
     wakeword-relay.yaml
     my_models/
       okay_sentinel.tflite
       okay_sentinel.json
   ```
2. In `wakeword-relay.yaml`, swap the stock model for your manifest (the file
   already has this stubbed in a comment):
   ```yaml
   micro_wake_word:
     microphone: mic
     models:
       - model: ./my_models/okay_sentinel.json   # local manifest -> baked in at compile, offline at runtime
     on_wake_word_detected:
       then:
         - switch.toggle: relay
   ```
   `model:` also accepts a URL or a shorthand name (`okay_nabu`, etc.), but a
   local path keeps everything offline and self-contained.
3. Recompile and flash:
   ```bash
   esphome run wakeword-relay.yaml
   esphome logs wakeword-relay.yaml   # watch for "Detected 'okay sentinel'"
   ```

## Tuning after deployment

This tweak-and-reflash loop is where most of the real work happens:

- Open `my_models/okay_sentinel.json` and adjust **`probability_cutoff`**:
  - **Raise it** (e.g. 0.85 → 0.95) if it false-triggers.
  - **Lower it** (e.g. 0.85 → 0.7) if it misses you.
- Re-run `esphome run` to bake the change in, then test.
- If cutoff tuning alone can't get both good recall *and* few false accepts,
  go back and retrain with more/better data (steps 3–4) — the cutoff only
  trades one error type for the other; it can't fix a fundamentally weak model.

## Honest expectations

- The **training is the heavy part, not the deployment** — deployment is just
  swapping one line. Once you have a `.tflite` + `.json`, flashing is identical
  to the stock-model flow.
- A self-trained model is usually decent but not as polished as the official
  `okay_nabu` / `hey_jarvis` ones, which were trained with much larger
  pipelines. Budget time for cutoff tuning and probably one or two retrains.
- **Everything stays offline** — the model is compiled into the firmware; Colab
  is only used once, at training time.

**Practical suggestion:** flash with the stock `okay_nabu` model first to
confirm your mic, wiring, and relay all work end-to-end. Then swap in your
custom word — that way, if the custom model misbehaves, you know it's the model,
not the hardware.