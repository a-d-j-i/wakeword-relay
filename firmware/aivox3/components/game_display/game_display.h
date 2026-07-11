#pragma once
#include "esphome/components/ili9xxx/ili9xxx_display.h"
#include "esphome/core/log.h"
#include <cmath>
#include <vector>

static const char *GAME_TAG = "game_disp";

// Thermometer strip position and size in display coordinates.
// Layout (240px wide): left strip at edge | 148px scene gap | right strip at edge
static constexpr int TX = 4, TY = 20, TW = 44, TH = 200;

// Thermometer geometry in strip-local coordinates (origin = display TX,TY).
static constexpr int T_TX  = 12, T_TW = 20;
static constexpr int T_TOP = 14, T_BOT = 164, T_TH = T_BOT - T_TOP;
static constexpr int T_BX  = T_TX + T_TW / 2;  // bulb centre x = 22
static constexpr int T_BY  = T_BOT + 13;        // bulb centre y = 177
static constexpr int T_BR  = 13;                 // bulb radius

// Right thermometer strip (mic amplitude) — same geometry as left, mirrored.
static constexpr int TX2 = 192;  // 240 - TW - 4

// WIN probability threshold. Tier bands below are frío/tibio/caliente/WIN.
// Retuned 2026-07-11 after the fda slot_mask fix (audio was 2x slow before;
// the old 0.35 threshold + 0.12-0.35 bands were calibrated against that broken
// audio and made the game trivial once the mic came clean). Clean-mic reference:
// casual utterances log avg 0.30-0.53, good ones higher. Raise/lower if WIN
// is still too easy/hard — bands in the loop() tier ladder scale with this.
static constexpr float WIN_THRESH = 0.45f;

// GameDisplay — runs entirely on Core 0.
//
// Polls micro_wake_word probability every 100 ms via id(mww).get_max_probability().
// Owns: thermometer strip (SPI partial update), LED color, relay toggle, audio tones.
// Core 1 (ESPHome main loop + mww inference) is never stalled by any of this.
//
// Thread-safety notes:
//  - get_max_probability(): read-only ring buffer scan, safe from Core 0.
//  - id(relay): GPIO write, atomic on ESP32.
//  - id(onboard_led).make_call().perform(): writes LightState from Core 0 while Core 1
//    reads it. Accepted minor race — worst case one frame wrong color (~16 ms).
//  - id(fda).play(): xQueueSend, fully thread-safe.
//  - WakeWordModel::enable/disable: sets a bool + NVS write (NVS is thread-safe).
//  - id(scene): 32-bit int, atomic on ESP32.

class GameDisplay : public esphome::Component {
 public:
  explicit GameDisplay(esphome::ili9xxx::ILI9XXXDisplay *disp) : disp_(disp) {}

  float get_setup_priority() const override { return esphome::setup_priority::LATE; }

  void setup() override {
    xTaskCreatePinnedToCore(
        [](void *arg) { static_cast<GameDisplay *>(arg)->loop_(); },
        "game_disp", 8192, this, 2, &task_, 0 /* Core 0 */);
  }

 private:
  esphome::ili9xxx::ILI9XXXDisplay *disp_;
  TaskHandle_t task_{nullptr};
  uint16_t buf_[TW * TH]{};
  uint16_t buf2_[TW * TH]{};
  uint16_t *cur_buf_{buf_};  // switched by push_mic_thermo_ so drawing primitives target buf2_
  int last_scene_{-1};
  int last_tier_{0};    // for tier-rise tone gating
  bool was_win_{false};
  int cooldown_{0};     // ticks to ignore prob after WIN (prevents fanfare from triggering opposite word)
  float peak_prob_{0.0f};  // peak-hold: decays slowly so thermometer stays up after speaking
  int peak_hold_{0};       // ticks remaining before peak_prob_ starts decaying
  float bonus_{0.0f};      // "getting warmer" streak: each caliente attempt banks +0.04 (cap 0.12)
                           // onto the score, so insisting on the word closes the gap to WIN.
  int idle_ticks_{0};      // consecutive idle ticks; long silence forfeits the bonus

  // ---- drawing primitives ------------------------------------------------

  static uint16_t c565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t v = (static_cast<uint16_t>(r >> 3) << 11) |
                 (static_cast<uint16_t>(g >> 2) << 5) | (b >> 3);
    return static_cast<uint16_t>(v >> 8) | static_cast<uint16_t>(v << 8);
  }

  void px(int x, int y, uint16_t c) {
    if (static_cast<unsigned>(x) < TW && static_cast<unsigned>(y) < TH)
      cur_buf_[y * TW + x] = c;
  }

  void fill_rect(int x, int y, int w, int h, uint16_t c) {
    for (int dy = 0; dy < h; dy++)
      for (int dx = 0; dx < w; dx++)
        px(x + dx, y + dy, c);
  }

  void draw_rect(int x, int y, int w, int h, uint16_t c) {
    for (int i = 0; i < w; i++) { px(x + i, y, c); px(x + i, y + h - 1, c); }
    for (int i = 1; i < h - 1; i++) { px(x, y + i, c); px(x + w - 1, y + i, c); }
  }

  void fill_circle(int cx, int cy, int r, uint16_t c) {
    for (int dy = -r; dy <= r; dy++)
      for (int dx = -r; dx <= r; dx++)
        if (dx * dx + dy * dy <= r * r) px(cx + dx, cy + dy, c);
  }

  void draw_circle(int cx, int cy, int r, uint16_t c) {
    int x = 0, y = r, d = 3 - 2 * r;
    while (x <= y) {
      px(cx+x, cy+y, c); px(cx-x, cy+y, c);
      px(cx+x, cy-y, c); px(cx-x, cy-y, c);
      px(cx+y, cy+x, c); px(cx-y, cy+x, c);
      px(cx+y, cy-x, c); px(cx-y, cy-x, c);
      if (d < 0) d += 4 * x + 6; else { d += 4 * (x - y) + 10; y--; }
      x++;
    }
  }

  void hline(int x, int y, int len, uint16_t c) {
    for (int i = 0; i < len; i++) px(x + i, y, c);
  }

  // ---- thermometer -------------------------------------------------------

  void render_(float prob, int tier) {
    uint16_t fill_c;
    switch (tier) {
      case 1:  fill_c = c565(0x20, 0x60, 0xFF); break;  // frío     — blue
      case 2:  fill_c = c565(0xFF, 0xC0, 0x00); break;  // tibio    — amber
      case 3:  fill_c = c565(0xFF, 0x40, 0x00); break;  // caliente — red-orange
      case 4:  fill_c = c565(0x00, 0xFF, 0x00); break;  // win      — green
      default: fill_c = c565(0x60, 0x60, 0x60); break;  // idle     — grey
    }
    uint16_t W = c565(0xFF, 0xFF, 0xFF);
    uint16_t K = c565(0x00, 0x00, 0x00);
    uint16_t R = c565(0xFF, 0x20, 0x20);

    fill_rect(0, 0, TW, TH, K);

    // Square-root scale: sqrt(prob/WIN) compresses the top and expands the bottom,
    // so weak detections (0.10–0.20) produce visible bar movement instead of near-nothing.
    float t = prob / WIN_THRESH;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float frac = sqrtf(t) * 0.85f;
    int level = T_BOT - static_cast<int>(T_TH * frac);
    fill_rect(T_TX + 2, level, T_TW - 4, T_BOT - level, fill_c);

    draw_rect(T_TX, T_TOP, T_TW, T_TH, W);

    int win_y = T_BOT - static_cast<int>(T_TH * 0.85f);
    hline(T_TX - 4, win_y, T_TW + 8, R);

    fill_circle(T_BX, T_BY, T_BR, fill_c);
    draw_circle(T_BX, T_BY, T_BR, W);
  }

  void push_thermo_(float prob, int tier) {
    render_(prob, tier);
    disp_->draw_pixels_at(TX, TY, TW, TH,
                          reinterpret_cast<const uint8_t *>(buf_),
                          esphome::display::COLOR_ORDER_RGB,
                          esphome::display::COLOR_BITNESS_565,
                          true, 0, 0, 0);
  }

  // Right thermometer: raw mic amplitude (0..1 smoothed peak).
  // cyan = normal, amber = loud, red = clipping territory.
  void push_mic_thermo_(float level) {
    cur_buf_ = buf2_;

    float frac = sqrtf(level) * 0.85f;
    if (frac > 1.0f) frac = 1.0f;
    uint16_t fill_c = (level > 0.85f) ? c565(0xFF, 0x00, 0x00) :
                      (level > 0.50f) ? c565(0xFF, 0xA0, 0x00) :
                                        c565(0x00, 0xB0, 0xFF);
    uint16_t W = c565(0xFF, 0xFF, 0xFF);
    uint16_t K = c565(0x00, 0x00, 0x00);

    int lv = T_BOT - static_cast<int>(T_TH * frac);
    fill_rect(0, 0, TW, TH, K);
    fill_rect(T_TX + 2, lv, T_TW - 4, T_BOT - lv, fill_c);
    draw_rect(T_TX, T_TOP, T_TW, T_TH, W);
    fill_circle(T_BX, T_BY, T_BR, fill_c);
    draw_circle(T_BX, T_BY, T_BR, W);

    cur_buf_ = buf_;
    disp_->draw_pixels_at(TX2, TY, TW, TH,
                          reinterpret_cast<const uint8_t *>(buf2_),
                          esphome::display::COLOR_ORDER_RGB,
                          esphome::display::COLOR_BITNESS_565,
                          true, 0, 0, 0);
  }

  // ---- LED ---------------------------------------------------------------

  void update_led_(float prob, int tier) {
    auto call = id(onboard_led).make_call();
    call.set_transition_length(0);
    if (tier == 4) {
      call.set_rgb(0.0f, 1.0f, 0.0f);   // green = WIN
    } else {
      float t = prob / WIN_THRESH;
      if (t < 0.0f) t = 0.0f;
      if (t > 1.0f) t = 1.0f;
      call.set_rgb(t, 0.0f, 1.0f - t);  // blue (cold) → red (hot)
    }
    call.perform();
  }

  // ---- audio -------------------------------------------------------------

  static void add_note_(std::vector<uint8_t> &buf, float freq, int samples) {
    for (int i = 0; i < samples; i++) {
      float env = (i < (int)(samples * 0.88f)) ? 1.0f : (float)(samples - i) / (samples * 0.12f);
      int16_t s = (int16_t)(4000.0f * env * sinf(2.0f * 3.14159265f * freq * i / 16000.0f));
      buf.push_back(s & 0xFF);
      buf.push_back((s >> 8) & 0xFF);
    }
  }

  void play_tone_(int tier) {
    std::vector<uint8_t> tone;
    if (tier == 4) {
      tone.reserve((2400 + 2400 + 2400 + 6400) * 2);
      add_note_(tone, 523.0f,  2400);  // C5  0.15 s
      add_note_(tone, 659.0f,  2400);  // E5  0.15 s
      add_note_(tone, 784.0f,  2400);  // G5  0.15 s
      add_note_(tone, 1047.0f, 6400);  // C6  0.40 s — win fanfare
    } else {
      float freq = (tier == 1) ? 330.0f : (tier == 2) ? 523.0f : 880.0f;
      tone.reserve(3200 * 2);
      add_note_(tone, freq, 3200);     // 0.2 s tier tone
    }
    id(fda).play(tone);
  }

  // ---- WIN ---------------------------------------------------------------

  void on_win_() {
    int sc = id(scene);
    const std::string word = (sc == 2) ? "turn_on" : "turn_off";
    id(fda).save_clip(word, id(mww).get_max_probability(), 4);

    if (sc == 2) {  // OFF → ON
      id(scene) = 1;
      id(relay).turn_on();
      id(turn_off_model).enable();
      id(turn_on_model).disable();
    } else {        // ON → OFF
      id(scene) = 2;
      id(relay).turn_off();
      id(turn_on_model).enable();
      id(turn_off_model).disable();
    }
  }

  // ---- main loop (Core 0) ------------------------------------------------

  void loop_() {
    vTaskDelay(pdMS_TO_TICKS(200));   // let boot actions finish
    disp_->update();                  // initial full scene draw
    last_scene_ = id(scene);
    push_thermo_(0.0f, 0);

    int diag_ticks = 0;
    while (true) {
      vTaskDelay(pdMS_TO_TICKS(100));

      float prob = id(mww).get_max_probability();
      // Peak-hold: rise instantly, hold ~1.2s, then decay 10% per 100ms (half-life ~0.7s).
      // Without this, the thermometer flickers — the mww sliding window (2 frames, ~100ms)
      // goes back to zero as soon as speaking stops, so a single 100ms poll often misses the peak.
      // The hold gives the player time to actually read their score off the bar.
      if (prob > peak_prob_) {
        peak_prob_ = prob;
        peak_hold_ = 12;  // 12 × 100ms
      } else if (peak_hold_ > 0) {
        peak_hold_--;
      } else {
        peak_prob_ *= 0.90f;
      }
      // Score = current peak + streak bonus. The bar/LED show the score, so the
      // player watches their banked progress build across attempts.
      float score = peak_prob_ + bonus_;
      if (score > 1.0f) score = 1.0f;
      id(game_prob) = score;

      int tier;
      if      (score >= WIN_THRESH) tier = 4;   // WIN    ≥ 0.45
      else if (score >= 0.35f)      tier = 3;   // hot    0.35–0.45
      else if (score >= 0.22f)      tier = 2;   // warm   0.22–0.35
      else if (score >= 0.12f)      tier = 1;   // cold   0.12–0.22
      else                          tier = 0;   // idle   < 0.12

      // Long silence forfeits the streak bonus (15 s).
      if (tier == 0) {
        if (++idle_ticks_ >= 150 && bonus_ != 0.0f) bonus_ = 0.0f;
      } else {
        idle_ticks_ = 0;
      }

      update_led_(score, tier);

      // Post-win cooldown FIRST: ignore detections (fanfare echo can re-trigger),
      // and gate the tone player too — it used to run before this check, so a
      // residual tier-4 during cooldown played the WIN fanfare without on_win_()
      // ever running ("win sound but no state switch").
      if (cooldown_ > 0) {
        cooldown_--;
        last_tier_ = 4;   // no tones until the bar returns to idle after cooldown
        was_win_ = true;  // no re-win until prob drops below WIN and comes back
        peak_prob_ = 0.0f;
        peak_hold_ = 0;
        bonus_ = 0.0f;   // win consumes the streak — next word starts from scratch
        id(game_prob) = 0.0f;
        push_thermo_(0.0f, 0);
        push_mic_thermo_(id(fda).get_mic_level());
        continue;
      }

      if (tier > last_tier_) {
        play_tone_(tier);
        last_tier_ = tier;
        // Reaching caliente banks a bonus for the NEXT attempts (once per attempt:
        // last_tier_ only re-arms after the bar returns to idle).
        if (tier == 3) {
          bonus_ += 0.04f;
          if (bonus_ > 0.12f) bonus_ = 0.12f;
        }
      }
      if (tier == 0) last_tier_ = 0;

      bool is_win = (tier == 4);
      if (is_win && !was_win_) { on_win_(); cooldown_ = 20; }  // 20 × 100ms = 2s cooldown
      was_win_ = is_win;

      // Every 5s diagnostic (always, so we know the loop is alive even when silent).
      if (++diag_ticks >= 50) {
        ESP_LOGD(GAME_TAG, "prob=%.3f peak=%.3f bonus=%.2f tier=%d mww=%d",
                 prob, peak_prob_, bonus_, tier, id(mww).is_running() ? 1 : 0);
        diag_ticks = 0;
      }
      // Immediate log for any non-zero probability (catches brief 100ms spikes).
      if (prob > 0.01f) {
        ESP_LOGI(GAME_TAG, "ACTIVE prob=%.3f peak=%.3f bonus=%.2f tier=%d", prob, peak_prob_, bonus_, tier);
      }

      // Full background only on scene change — avoids overwriting the thermometer strip,
      // which is what causes the flicker (23ms full-screen SPI + 2.6ms strip SPI).
      // push_thermo_() renders to a buffer and does one draw_pixels_at — already flicker-free.
      int cur = id(scene);
      if (cur != last_scene_) {
        ESP_LOGD(GAME_TAG, "scene %d→%d prob=%.3f peak=%.3f mww=%d",
                 last_scene_, cur, prob, peak_prob_, id(mww).is_running() ? 1 : 0);
        disp_->update();
        last_scene_ = cur;
      }

      push_thermo_(score, tier);
      push_mic_thermo_(id(fda).get_mic_level());
    }
  }
};
