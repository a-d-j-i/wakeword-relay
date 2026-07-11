#pragma once

#include "esphome/components/audio/audio.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/core/component.h"
#include "esphome/core/log.h"

#include <driver/i2s_std.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <string>
#include <vector>

namespace esphome::full_duplex_audio {

// Full-duplex I2S audio component for the ES8311 codec on AI-VOX3.
//
// Uses a single i2s_new_channel() call to create both TX and RX handles on
// I2S_NUM_0, enabling simultaneous mic capture and speaker playback on separate
// GPIO data lines (DIN / DOUT). All I/O tasks run on Core 0 so Core 1
// (micro_wake_word TFLite inference + ESPHome main loop) is never stalled.
//
// Sample collection: a 3-second PSRAM ring buffer captures every RX byte.
// Call save_clip(wake_word, probability, tier) from on_wake_word_detected to
// snapshot the buffer and queue it for HTTP POST to upload_url (as a WAV file).

struct Clip {
  std::vector<uint8_t> wav;  // 44-byte header + raw PCM
  std::string wake_word;
  float probability;
  int tier;
};

class FullDuplexAudio : public Component, public microphone::Microphone {
 public:
  void setup() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::BUS; }

  // microphone::Microphone
  void start() override;
  void stop() override;

  // Enqueue raw 16-bit PCM for playback; returns immediately.
  void play(const std::vector<uint8_t> &data);

  // Snapshot the last 3 s from the PSRAM ring buffer and queue it for upload.
  void save_clip(const std::string &wake_word, float probability, int tier);

  // DIAGNOSTIC: dump the last `ms` of the PSRAM ring buffer to the log as base64
  // lines ("PCMDUMP <id> <seq> <b64>"). Decode with tools/decode_pcm_log.py.
  // Call from the main loop (e.g. an interval: lambda) so log lines aren't dropped
  // by the task log buffer.
  void dump_pcm_b64(uint32_t ms);

  // Config setters (called by generated code)
  void set_mclk_pin(int pin) { mclk_pin_ = static_cast<gpio_num_t>(pin); }
  void set_bclk_pin(int pin) { bclk_pin_ = static_cast<gpio_num_t>(pin); }
  void set_ws_pin(int pin) { ws_pin_ = static_cast<gpio_num_t>(pin); }
  void set_dout_pin(int pin) { dout_pin_ = static_cast<gpio_num_t>(pin); }
  void set_din_pin(int pin) { din_pin_ = static_cast<gpio_num_t>(pin); }
  void set_sample_rate(uint32_t rate) { sample_rate_ = rate; }
  void set_upload_url(const std::string &url) { upload_url_ = url; }
  void set_mic_gain(float gain) { mic_gain_ = gain; }
  // Smoothed peak amplitude 0..1 (fast rise, ~270ms half-life decay). Thread-safe read.
  float get_mic_level() const { return mic_level_; }

 protected:
  // I2S handles — both created in one i2s_new_channel() call for full-duplex
  i2s_chan_handle_t tx_handle_{nullptr};
  i2s_chan_handle_t rx_handle_{nullptr};

  // TX: queue of heap-allocated audio chunks; upload: queue of Clip*
  QueueHandle_t tx_queue_{nullptr};
  QueueHandle_t upload_queue_{nullptr};

  // PSRAM ring buffer — filled continuously by rx_task
  static constexpr size_t RING_SECS = 3;
  uint8_t *ring_buf_{nullptr};
  volatile uint32_t ring_head_{0};   // next write index (mod RING_BUF_BYTES)

  std::string upload_url_;
  float mic_gain_{1.0f};          // linear multiplier applied to RX samples before mww; 2.0 ≈ +6dB
  volatile float mic_level_{0.0f}; // smoothed peak 0..1, updated per RX frame by rx_task

  gpio_num_t mclk_pin_{GPIO_NUM_11};
  gpio_num_t bclk_pin_{GPIO_NUM_10};
  gpio_num_t ws_pin_{GPIO_NUM_8};
  gpio_num_t dout_pin_{GPIO_NUM_7};
  gpio_num_t din_pin_{GPIO_NUM_9};
  uint32_t sample_rate_{16000};

  size_t ring_buf_bytes_() const { return RING_SECS * sample_rate_ * 2; }

  static std::vector<uint8_t> make_wav_(const uint8_t *pcm, size_t len, uint32_t rate);
  void do_upload_(Clip *clip);

  static void rx_task_fn(void *arg);
  static void tx_task_fn(void *arg);
  static void upload_task_fn(void *arg);
};

}  // namespace esphome::full_duplex_audio
