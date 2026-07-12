# Licenses

The original code in this repository is MIT-licensed (text below). Vendored
third-party components keep their own licenses — see the inventory that follows.

## MIT (original code)

MIT License

Copyright © 2026 adji (aadjiman@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Third-party licenses

The repository also **redistributes** the vendored third-party components below,
each under its own license.

## Redistributed components

| Component                                             | Location in repo                                          | Upstream                                                                                                                   | License                                                                                                                  |
|-------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| espeak-ng (library sources)                           | `web_trainer/vendor/espeak-ng/`                           | [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)                                                              | **GPL-3.0-or-later** ([text](licenses/GPL-3.0.txt))                                                                      |
| espeak-ng voice/dictionary data                       | `docs/trainer/espeak-ng-data/`                            | [espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng)                                                              | **GPL-3.0-or-later**                                                                                                     |
| piper-phonemize (compiled WASM links espeak-ng)       | `docs/trainer/piper_phonemize.{js,wasm,data}`             | [rhasspy/piper-phonemize](https://github.com/rhasspy/piper-phonemize)                                                      | MIT sources; the distributed binary links espeak-ng, so the combined work is governed by **GPL-3.0**                     |
| ESPHome micro_wake_word component (patched)           | `firmware/aivox3/components/micro_wake_word/`             | [esphome/esphome](https://github.com/esphome/esphome)                                                                      | **GPL-3.0** (ESPHome C++), MIT (ESPHome Python) — [ESPHome license](https://github.com/esphome/esphome/blob/dev/LICENSE) |
| microWakeWord (training framework, patched)           | `train/microwakeword/`                                    | [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord)                                                        | Apache-2.0 ([text](licenses/Apache-2.0.txt))                                                                             |
| TensorFlow Lite Micro + kernels                       | `web_trainer/vendor/tensorflow/`                          | [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro)                                                      | Apache-2.0                                                                                                               |
| TFLite Micro audio frontend                           | `web_trainer/vendor/micro_frontend/`                      | [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro)                                                      | Apache-2.0                                                                                                               |
| FlatBuffers headers                                   | `web_trainer/vendor/flatbuffers/`                         | [google/flatbuffers](https://github.com/google/flatbuffers)                                                                | Apache-2.0                                                                                                               |
| TFLite schema (generated header)                      | `web_trainer/vendor/tflite_schema/`                       | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)                                                          | Apache-2.0                                                                                                               |
| KissFFT                                               | `web_trainer/vendor/kissfft/`                             | [mborgerding/kissfft](https://github.com/mborgerding/kissfft)                                                              | BSD-3-Clause (see below)                                                                                                 |
| piper-sample-generator (vendored, incl. impulse WAVs) | `train/piper_sample_generator/`, `docs/trainer/impulses/` | [rhasspy/piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)                                        | MIT                                                                                                                      |
| Piper / VITS utilities                                | `train/piper_train/vits/`                                 | [rhasspy/piper](https://github.com/rhasspy/piper) (derived from [jaywalnut310/vits](https://github.com/jaywalnut310/vits)) | MIT                                                                                                                      |

## Fetched at build/run time (not redistributed here)

| Component                                              | How it is used                                                    | License                                                                                                                                                                   |
|--------------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TensorFlow.js + tfjs-tflite                            | loaded from CDN by `docs/emulator/index.html`                     | Apache-2.0                                                                                                                                                                |
| onnxruntime-web                                        | loaded from CDN by the trainer's Piper worker                     | MIT                                                                                                                                                                       |
| Piper TTS voice models                                 | downloaded on demand (browser or `--piper_model`)                 | **per-voice** — check each voice's MODEL_CARD on [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) before redistributing audio or models |
| Negative spectrogram datasets (`negatives.mwwn` input) | downloaded by `train/train.py`; bundle is git-ignored             | see the [dataset card](https://huggingface.co/datasets/kahrendt/microwakeword)                                                                                            |
| MIT RIRs, AudioSet, FMA (training augmentation)        | downloaded by `train/train.py`, never redistributed               | per-dataset (CC-BY and similar) — relevant only if you redistribute them                                                                                                  |
| Roboto font                                            | fetched by ESPHome at compile time, embedded in firmware binaries | Apache-2.0                                                                                                                                                                |

## What this means in practice (obligations)

- **Source availability (GPL-3.0):** distributing this repo, the GitHub Pages site
  (which serves `espeak-ng-data` and `piper_phonemize.wasm`), or compiled ESP32
  firmware (which contains ESPHome C++ code) requires the corresponding source to
  be available. This public repository satisfies that as long as it stays public
  and contains the sources of any modifications (the patches to `micro_wake_word`
  are committed here in full).
- **Notice retention (all licenses):** the license texts in `licenses/` and this
  inventory must ship with redistributions. Do not strip copyright headers from
  vendored files.
- **Stating changes (Apache-2.0 / GPL):** modified vendored files are documented in
  `CLAUDE.md` ("Patches in train/microwakeword") and in the component READMEs;
  keep those notes up to date when patching further.
- **Piper voices:** the trained wake-word models in `firmware/models/` were
  generated using TTS output from Piper voices; model weights trained on TTS audio
  are generally fine to distribute, but if you redistribute the raw generated
  audio, check the individual voice licenses.

## KissFFT license (BSD-3-Clause)

Copyright © 2003–2010 Mark Borgerding. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions, and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions, and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
- Neither the author nor the names of any contributors may be used to endorse or
  promote products derived from this software without specific prior written
  permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER, CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## MIT license (raspy piper / piper-phonemize / piper-sample-generator sources)

Copyright © Michael Hansen and contributors.

Licensed under the MIT License; the full MIT text is in the MIT text above
(same terms, different copyright holder).
