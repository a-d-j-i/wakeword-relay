# Nulllab AI-VOX3 — offline wake-word → relay (ESP32-S3 + ES8311)

ESPHome firmware for the **Nulllab AI-VOX3** board running the Spanish wake words
**"chispa magica"** (turn on) and **"buenas noches"** (turn off) fully on-device.

- Main config: [`aivox3.yaml`](aivox3.yaml)
- Models live in `../models/` (`turn_on.tflite`/`.json` = chispa magica, `turn_off.*` = buenas noches),
  shared with the DevKitC-1 build. Trained with `train/train.py` (98.7% / 99% test recall).

## Hardware

Nulllab **AI-VOX3** (silk: "AI-VOX 3.0", board rev V0.5; schematic rev V1.1, maker EmakeFun).
ESP32-S3-R8 (16 MB flash / 8 MB octal PSRAM), **ES8311** mono audio codec with built-in analog
mic + a **NS4150B** Class-D speaker amp, 1.54" **ST7789** 240×240 LCD (FPC connector), WS2812 RGB
LED, BOOT + A + B buttons + power/reset switch, SD-card slot, and a Li-ion battery input with
on-board charger. PCB ≈ **56 × 40 mm** (white component outline 34 × 32 mm).

### Board features (numbered as on the vendor photo)

| # | Feature | Notes |
|---|---|---|
| 1 | Component outline | 34 × 32 mm keep-out box |
| 2 | RGB indicator LED | WS2812 on GPIO41 |
| 3 | Microphone | built-in analog mic → ES8311 |
| 4 | MD40 motor-driver pads (back) | `G / 5V / 5 / 6` — same IO5/IO6 nets as the PH2.0 port |
| 5 | OLED solder pads / expansion (back) | `13 / 12 / 5V / G` = I²C (SDA13/SCL12) — shared with codec, don't reuse |
| 6 | BOOT button | GPIO0 (strapping) |
| 7 | A button | GPIO46 — user/custom |
| 8 | B button | GPIO45 — user/custom |
| 9 | USB-C | flashing + 5 V / 1 A power (USB-Serial-JTAG on GPIO19/20) |
| 10 | External-mic connector | MX1.25 — analog mic in |
| 11 | **PH2.0 4-pin expansion port** | `G / 5V / 5 / 6` = GND / +5V / GPIO5 / GPIO6 — **relay lives here** |
| 12 | Speaker connector | MX1.25 — to NS4150B amp output |
| 13 | LCD FPC connector | ST7789 240×240 color panel |
| 14 | **2×7 2.54 mm header** | expansion / 4G module; pins below |
| 15 | Power switch + reset | on/off + reset |
| 16 | SD-card slot | mass storage (GPIO38/39/40) |
| 17 | Battery connector | ZH1.5 — Li-ion in |
| 18 | ES8311 | audio codec @ I²C `0x18` |
| 19 | NS4150B | Class-D speaker amplifier |
| 20 | ESP32-S3-R8 | MCU (8 MB PSRAM) |
| 21 | 16 MB flash | external SPI flash |

### Expansion connectors

**PH2.0 4-pin (item 11)** — keyed plug-in, the easy way to add external I/O:

| Pin | Net |
|---|---|
| 1 | `G` (GND) |
| 2 | `5V` (= USB VBUS) |
| 3 | `5` (GPIO5) |
| 4 | `6` (GPIO6) |

**2×7 2.54 mm header (item 14)** — needs soldering/dupont; carries (silk): `5V`, `3V3`, `VBAT`,
`G`, and GPIOs **2, 3, 4, 42, 43, 44, 48**. GPIO48 is the cleanest free pin here; avoid GPIO3
(strapping). `5V`/`VBAT` are USB-VBUS / battery rails respectively.

### Full GPIO map (from the schematic "IO 占用一览表" + `xiaozhi` `config.h`)

| Function | GPIO |
|---|---|
| I²S MCLK / BCLK / WS (LRCLK) | 11 / 10 / 8 |
| I²S DOUT (to codec DAC) / DIN (mic) | 7 / 9 |
| I²C SDA / SCL  (ES8311 @ `0x18`) | 13 / 12 |
| LCD SPI: CLK / MOSI / CS / DC / backlight | 17 / 21 / 15 / 14 / 16  (RST = NC) |
| WS2812 RGB LED | 41 |
| Buttons: BOOT / A / B | 0 / 46 / 45  (+ power/reset switch) |
| Speaker amp (NS4150B) CTRL | 1 (analog-amp enable; pulled high = on) |
| Battery level (ADC) | 18 |
| SD card (DAT/CMD/CLK) | 38 / 39 / 40 |
| USB D− / D+ | 19 / 20 |
| SPI flash | 30–35 |
| PH2.0 connector | 5 / 6 |
| 2×7 header (free GPIOs) | 2 / 3 / 4 / 42 / 43 / 44 / 48 |

## Quick start

```bash
esphome run firmware/aivox3/aivox3.yaml      # compile + flash over USB
esphome logs firmware/aivox3/aivox3.yaml     # watch detections
```
Boots **red 🔴 / 🌙** (off). Say **"chispa magica"** → **green 🟢 / ☀️** (relay on);
say **"buenas noches"** → red/moon (relay off).

> **Logs not showing?** The native-USB port (`/dev/ttyACM*`) is USB-Serial-JTAG; it drops on
> reset. `logger: hardware_uart: USB_SERIAL_JTAG` routes app logs there. To catch boot logs,
> start `esphome logs` **then press RESET**. To survive drops:
> `while true; do esphome logs firmware/aivox3/aivox3.yaml; sleep 1; done`

## How it works — one model at a time

The ESP32-S3 **cannot run both streaming wake-word models at once** in ESPHome: the inference
task runs every model each frame, falls behind, the audio ring buffer drops samples, and the
2nd model collapses to ~0 probability (the 1st still fires). Proven: each model is flawless
**alone**.

**Fix (the key design):** only one model is enabled at a time, switched by light state —
you only need "turn on" while off and "turn off" while on:

- boot: `micro_wake_word.start` + `disable_model: turn_off_model`  → listen for **turn_on** only
- on **turn_on**: light on, `disable turn_on_model` + `enable turn_off_model`
- on **turn_off**: light off, `disable turn_off_model` + `enable turn_on_model`

A disabled model is **unloaded** (zero inference). This also prevents redundant re-fires.

## Hard-won gotchas (don't relearn these)

1. **`micro_wake_word` does NOT auto-start.** Without `micro_wake_word.start` (on_boot or
   voice_assistant) it sits STOPPED and never detects. This was the cause of "nothing works."
2. **Two models don't fit** → use the one-at-a-time switch above (not two models, not a
   2-output model: ESPHome requires a 1×1 output and microWakeWord trains binary).
3. **ES8311 mic** = analog → `audio_dac: es8311` with `use_microphone: false`; mic via
   `i2s_audio` microphone on DIN=9. `mic_gain` is the analog PGA (0–42 dB; 42 dB clips).
4. **ES8311 speaker needs 3 things** (see `aivox3-speaker.yaml`): `timeout: never` on the
   i2s speaker (default 500 ms truncates playback), explicit `set_mute_off()`+`set_volume()`
   at boot (ESPHome never unmutes a bare speaker), and a silent "prime" play at boot (the DAC
   only wakes once the I²S clock runs).
5. **Mic + speaker can't run simultaneously** (one I²S port) — record↔play loopback is
   unreliable; the wake-word app is mic-only so it's fine.
6. **LCD**: RST is NC → use the `mipi_spi` driver (`model: ST7789V`), not the legacy
   `st7789v` (which requires a reset pin). `invert_colors: true`. Full-screen redraw
   on detection costs ~51 ms.
7. **LED**: WS2812 on GPIO41 — has a 1 s default light transition, so quick flashes look
   "dead"; use `transition_length: 0s`. Also, RGB values are a *color ratio* normalized
   independently of brightness — `15%/8%` looks identical to `30%/15%`; **dim via the
   separate `brightness:` parameter**, not by scaling the RGB channels down.
8. **`micro_wake_word` persists model enable/disable state to NVS** (survives reboot *and*
   reflash). If `on_boot` only *disables* the off-model, a "turn_on disabled" left over from
   a previous ON session keeps the on-model dead after a reflash — the symptom is "I must
   erase flash every run or okay_nabu won't fire." **Fix (two parts, both applied):**
   - **Deterministic boot** — `on_boot` must explicitly `enable_model: turn_on_model` **and**
     `disable_model: turn_off_model`, not just one.
   - **Go RAM-only** — set `internal: true` on each model. In `streaming_model.cpp` the
     `pref_.save()` calls are gated by `if (!internal_only_)`, so the enabled flag is never
     *written* to NVS. (`setup()` still *reads* it via `pref_.load()`, so do **one** flash-erase
     to clear pre-existing values; after that nothing is written and each model uses its
     compiled `default_enabled` — first model on, rest off.) Likewise keep other state in RAM:
     relay switch `restore_mode: ALWAYS_OFF`, `onboard_led` light `restore_mode: ALWAYS_OFF`,
     `scene` global `restore_value: no`. Result: no NVS writes, deterministic boot, no
     flash-erase between runs.

## Relay

Wired to the **PH2.0 4-pin expansion connector** (board item #11, next to the USB-C port):
`G`/`5V`/`5`/`6` = GND / +5V / GPIO5 / GPIO6. Drive an **opto-isolated relay *module***
(it has the transistor + flyback diode + opto built in — never wire a bare relay coil to a GPIO):

| Relay module | AI-VOX3 PH2.0 pin |
|---|---|
| GND | `G` |
| VCC | `5V` |
| IN  | `5` (GPIO5) |

`chispa magica` → `switch.turn_on: relay`; `buenas noches` → `switch.turn_off: relay`. Boots OFF
(matches the moon state). GPIO6 on the same connector is a free spare for a 2nd relay.

> ⚠️ The connector's `5V` rail is **USB VBUS** — present only on USB power, not battery-only.
> For battery operation use a 3.3 V-coil relay off `+3V3` (on the 2×7 header, item #14).

## Tuning

- `mic_gain` (in `aivox3.yaml`): currently **42 dB**. The frontend (PCAN) loudness-normalizes,
  so *raising* analog gain mostly adds clipping + noise and **hurt** detection in testing. For
  real "talk near or far" the **ES8311 hardware ALC is now enabled** (see below), not a bigger
  fixed gain.
- **ES8311 ALC (auto level control)** is enabled from a late (`priority: -200`) `on_boot` lambda
  that writes the codec's ADC registers over the `bus_a` I²C bus *after* the `es8311` component
  has configured itself: `0x17` ADC_VOLUME = ALC max gain ceiling (`0xBF` ≈ 0 dB, raise toward
  `0xFF` = +32 dB for more far-speech boost), `0x19` ALC target window (max/min level, `0x77`),
  `0x18` ALC_EN + ADC_AUTOMUTE_EN (`0xC0`; automute keeps ALC from boosting silence into noise).
  Tune `0x17`/`0x19` against `aivox3-micmon.yaml` and re-verify wake-word detection — ALC's
  silence-boost can raise false fires.
- `probability_cutoff` per model: currently **0.5** (lowered for easier triggering). Raise
  (0.6–0.8) if you get false fires.

## Diagnostic configs (kept for future bring-up / debugging)

| File | Purpose |
|---|---|
| `aivox3.yaml` | **The app** — wake words → LED + LCD + relay, one-model-at-a-time |
| `aivox3-okaynabu.yaml` | Known-good English model + heartbeat — isolate pipeline vs our models |
| `aivox3-buenas.yaml` | buenas-noches model only — test one model in isolation |
| `aivox3-nolcd.yaml` | both models, no LCD — test whether the display starves inference |
| `aivox3-micmon.yaml` | live mic level meter (peak/rms) — confirm continuous capture |
| `aivox3-loopback.yaml` | hold-to-record, release-to-play — mic + speaker test |
| `aivox3-speaker.yaml` | speaker-only tone/sweep — DAC/amp test |
| `aivox3-buttons.yaml` | map the 3 buttons (GPIO0/45/46) to LED colors |
| `test_clips/` | `chispa_magica.wav` / `buenas_noches.wav` — play from a phone for repeatable tests |

## TODO

- ~~**Dynamic mic gain** via the ES8311 hardware **ALC** (registers REG18–1B).~~ **Done** —
  enabled from a `priority: -200` `on_boot` lambda over `bus_a` (see *Tuning* above). Still needs
  field tuning of the `0x17`/`0x19` levels against real near/far speech.
- Make the LCD redraw non-blocking (the ~51 ms stall on detection).
