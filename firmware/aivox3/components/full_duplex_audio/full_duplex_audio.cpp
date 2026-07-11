#include "full_duplex_audio.h"

#include "esphome/core/hal.h"

#include <esp_http_client.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace esphome::full_duplex_audio {

static const char *const TAG = "full_duplex_audio";

static constexpr size_t DMA_DESC_NUM = 4;
static constexpr size_t DMA_FRAME_NUM = 256;
static constexpr size_t RX_BUF_BYTES = DMA_FRAME_NUM * 2;  // 16-bit mono, one DMA frame = 16 ms

void FullDuplexAudio::setup() {
  // One i2s_new_channel() call with both handles = full-duplex on a single I2S controller.
  // TX (DOUT GPIO7) and RX (DIN GPIO9) share MCLK/BCLK/WS but use independent DMA rings.
  i2s_chan_config_t chan_cfg = {
      .id = I2S_NUM_0,
      .role = I2S_ROLE_MASTER,
      .dma_desc_num = DMA_DESC_NUM,
      .dma_frame_num = DMA_FRAME_NUM,
      .auto_clear = true,  // TX: output zeros when idle (no noise)
  };
  esp_err_t err = i2s_new_channel(&chan_cfg, &tx_handle_, &rx_handle_);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_new_channel: %s", esp_err_to_name(err));
    this->mark_failed();
    return;
  }

  i2s_std_config_t std_cfg = {
      .clk_cfg = {.sample_rate_hz = this->sample_rate_,
                  .clk_src = I2S_CLK_SRC_DEFAULT,
                  .mclk_multiple = I2S_MCLK_MULTIPLE_256},
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg = {.mclk = mclk_pin_,
                   .bclk = bclk_pin_,
                   .ws = ws_pin_,
                   .dout = dout_pin_,
                   .din = din_pin_,
                   .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false}},
  };

  // ESP32-S3 quirk: I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG leaves slot_mask at
  // I2S_STD_SLOT_BOTH even in MONO mode (the classic-ESP32 variant of the macro
  // selects LEFT). With BOTH, RX captures both stereo slots and the ES8311
  // mirrors its mono ADC onto both -> every sample arrives twice -> audio is
  // stretched 2x and wake word detection sees half-speed speech. Force LEFT,
  // exactly like the stock i2s_audio microphone does for channel: left.
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

  if ((err = i2s_channel_init_std_mode(tx_handle_, &std_cfg)) != ESP_OK) {
    ESP_LOGE(TAG, "TX init: %s", esp_err_to_name(err));
    this->mark_failed();
    return;
  }
  if ((err = i2s_channel_init_std_mode(rx_handle_, &std_cfg)) != ESP_OK) {
    ESP_LOGE(TAG, "RX init: %s", esp_err_to_name(err));
    this->mark_failed();
    return;
  }
  if ((err = i2s_channel_enable(tx_handle_)) != ESP_OK) {
    ESP_LOGE(TAG, "TX enable: %s", esp_err_to_name(err));
    this->mark_failed();
    return;
  }
  if ((err = i2s_channel_enable(rx_handle_)) != ESP_OK) {
    ESP_LOGE(TAG, "RX enable: %s", esp_err_to_name(err));
    this->mark_failed();
    return;
  }

  this->audio_stream_info_ = audio::AudioStreamInfo(16, 1, this->sample_rate_);

  // 3-second PSRAM ring buffer for sample collection
  ring_buf_ = static_cast<uint8_t *>(heap_caps_malloc(ring_buf_bytes_(), MALLOC_CAP_SPIRAM));
  if (!ring_buf_) {
    ESP_LOGW(TAG, "PSRAM ring buffer alloc failed — clip upload disabled");
  } else {
    memset(ring_buf_, 0, ring_buf_bytes_());
  }

  tx_queue_ = xQueueCreate(4, sizeof(std::vector<uint8_t> *));
  upload_queue_ = xQueueCreate(4, sizeof(Clip *));

  // All tasks on Core 0: keeps Core 1 free for TFLite inference
  xTaskCreatePinnedToCore(rx_task_fn, "fda_rx", 4096, this, 5, nullptr, 0);
  xTaskCreatePinnedToCore(tx_task_fn, "fda_tx", 4096, this, 3, nullptr, 0);
  xTaskCreatePinnedToCore(upload_task_fn, "fda_up", 8192, this, 1, nullptr, 0);
}

void FullDuplexAudio::start() {
  this->state_ = microphone::STATE_RUNNING;
}

void FullDuplexAudio::stop() {
  this->state_ = microphone::STATE_STOPPED;
}

void FullDuplexAudio::play(const std::vector<uint8_t> &data) {
  auto *chunk = new std::vector<uint8_t>(data);
  if (xQueueSend(tx_queue_, &chunk, pdMS_TO_TICKS(50)) != pdTRUE) {
    delete chunk;
    ESP_LOGW(TAG, "TX queue full, dropping audio");
  }
}

void FullDuplexAudio::save_clip(const std::string &wake_word, float probability, int tier) {
  if (!ring_buf_ || upload_url_.empty()) return;

  size_t buf_bytes = ring_buf_bytes_();
  uint32_t head = ring_head_;  // snapshot (atomic 32-bit read on ESP32)

  // Reconstruct ring buffer in chronological order (head = oldest byte)
  auto *clip = new Clip;
  clip->wake_word = wake_word;
  clip->probability = probability;
  clip->tier = tier;

  std::vector<uint8_t> pcm(buf_bytes);
  size_t tail = buf_bytes - head;
  memcpy(pcm.data(), ring_buf_ + head, tail);
  memcpy(pcm.data() + tail, ring_buf_, head);

  clip->wav = make_wav_(pcm.data(), buf_bytes, sample_rate_);

  if (xQueueSend(upload_queue_, &clip, 0) != pdTRUE) {
    delete clip;
    ESP_LOGW(TAG, "Upload queue full, dropping clip");
  } else {
    ESP_LOGD(TAG, "Clip queued: %s p=%.2f tier=%d", wake_word.c_str(), probability, tier);
  }
}

void FullDuplexAudio::dump_pcm_b64(uint32_t ms) {
  if (!ring_buf_) {
    ESP_LOGW(TAG, "PCMDUMP: no ring buffer");
    return;
  }
  static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  static uint32_t dump_id = 0;

  size_t rb = ring_buf_bytes_();
  size_t want = (size_t) sample_rate_ * 2 * ms / 1000;
  if (want > rb) want = rb;
  want -= want % 6;  // multiple of 3 keeps each b64 line unpadded; of 2 keeps int16 samples whole

  // Copy the window out first: encoding + logging takes ~0.5 s and rx_task keeps
  // writing into the ring, so encoding in place would race with the writer.
  auto *snap = static_cast<uint8_t *>(heap_caps_malloc(want, MALLOC_CAP_SPIRAM));
  if (!snap) {
    ESP_LOGW(TAG, "PCMDUMP: snapshot alloc failed");
    return;
  }
  uint32_t head = ring_head_;  // oldest byte of our window:
  size_t start = (head + rb - want) % rb;
  size_t first = rb - start < want ? rb - start : want;
  memcpy(snap, ring_buf_ + start, first);
  memcpy(snap + first, ring_buf_, want - first);

  dump_id++;
  constexpr size_t BYTES_PER_LINE = 72;  // -> 96 b64 chars
  size_t total_lines = (want + BYTES_PER_LINE - 1) / BYTES_PER_LINE;
  ESP_LOGI(TAG, "PCMDUMP %u BEGIN rate=%u bytes=%u lines=%u", (unsigned) dump_id, (unsigned) sample_rate_,
           (unsigned) want, (unsigned) total_lines);

  char line[BYTES_PER_LINE / 3 * 4 + 1];
  for (size_t ln = 0; ln < total_lines; ln++) {
    size_t off = ln * BYTES_PER_LINE;
    size_t n = want - off < BYTES_PER_LINE ? want - off : BYTES_PER_LINE;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
      uint8_t b0 = snap[off + i];
      uint8_t b1 = i + 1 < n ? snap[off + i + 1] : 0;
      uint8_t b2 = i + 2 < n ? snap[off + i + 2] : 0;
      uint32_t v = ((uint32_t) b0 << 16) | ((uint32_t) b1 << 8) | b2;
      line[o++] = B64[(v >> 18) & 63];
      line[o++] = B64[(v >> 12) & 63];
      line[o++] = i + 1 < n ? B64[(v >> 6) & 63] : '=';
      line[o++] = i + 2 < n ? B64[v & 63] : '=';
    }
    line[o] = '\0';
    ESP_LOGI(TAG, "PCMDUMP %u %u %s", (unsigned) dump_id, (unsigned) ln, line);
    if ((ln & 31) == 31)
      delay(2);  // let the USB-JTAG buffer drain so lines aren't dropped
  }
  ESP_LOGI(TAG, "PCMDUMP %u END", (unsigned) dump_id);
  heap_caps_free(snap);
}

// --- static helpers ---

std::vector<uint8_t> FullDuplexAudio::make_wav_(const uint8_t *pcm, size_t len, uint32_t rate) {
  std::vector<uint8_t> wav;
  wav.reserve(44 + len);

  auto u32 = [&](uint32_t v) {
    wav.push_back(v & 0xFF); wav.push_back((v >> 8) & 0xFF);
    wav.push_back((v >> 16) & 0xFF); wav.push_back((v >> 24) & 0xFF);
  };
  auto u16 = [&](uint16_t v) {
    wav.push_back(v & 0xFF); wav.push_back((v >> 8) & 0xFF);
  };
  auto str4 = [&](const char *s) {
    wav.push_back(s[0]); wav.push_back(s[1]); wav.push_back(s[2]); wav.push_back(s[3]);
  };

  str4("RIFF"); u32(36 + len);  str4("WAVE");
  str4("fmt "); u32(16);        // PCM chunk
  u16(1);                        // format: PCM
  u16(1);                        // channels: mono
  u32(rate);                     // sample rate
  u32(rate * 2);                 // byte rate
  u16(2);                        // block align
  u16(16);                       // bits per sample
  str4("data"); u32(len);

  wav.insert(wav.end(), pcm, pcm + len);
  return wav;
}

void FullDuplexAudio::do_upload_(Clip *clip) {
  char prob_buf[12], tier_buf[4];
  snprintf(prob_buf, sizeof(prob_buf), "%.3f", clip->probability);
  snprintf(tier_buf, sizeof(tier_buf), "%d", clip->tier);

  esp_http_client_config_t cfg{};
  cfg.url = upload_url_.c_str();
  cfg.timeout_ms = 15000;

  auto *client = esp_http_client_init(&cfg);
  esp_http_client_set_method(client, HTTP_METHOD_POST);
  esp_http_client_set_header(client, "Content-Type", "audio/wav");
  esp_http_client_set_header(client, "X-Wake-Word", clip->wake_word.c_str());
  esp_http_client_set_header(client, "X-Probability", prob_buf);
  esp_http_client_set_header(client, "X-Tier", tier_buf);
  esp_http_client_set_post_field(client,
      reinterpret_cast<const char *>(clip->wav.data()),
      clip->wav.size());

  esp_err_t err = esp_http_client_perform(client);
  int status = esp_http_client_get_status_code(client);
  esp_http_client_cleanup(client);

  if (err == ESP_OK && status == 200) {
    ESP_LOGI(TAG, "Uploaded: %s p=%.2f tier=%d (%zu B)",
             clip->wake_word.c_str(), clip->probability, clip->tier, clip->wav.size());
  } else {
    ESP_LOGW(TAG, "Upload failed: %s HTTP %d", esp_err_to_name(err), status);
  }
}

// --- FreeRTOS task functions ---

void FullDuplexAudio::rx_task_fn(void *arg) {
  auto *self = static_cast<FullDuplexAudio *>(arg);
  static uint8_t buf[RX_BUF_BYTES];

  while (true) {
    if (!self->is_running()) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    size_t bytes_read = 0;
    esp_err_t err = i2s_channel_read(self->rx_handle_, buf, sizeof(buf), &bytes_read, pdMS_TO_TICKS(100));
    if (err != ESP_OK && err != ESP_ERR_TIMEOUT)
      continue;
    if (bytes_read == 0)
      continue;

    // Apply software mic gain (clamp to int16 range to avoid wrap-around distortion)
    if (self->mic_gain_ != 1.0f) {
      float g = self->mic_gain_;
      int16_t *s = reinterpret_cast<int16_t *>(buf);
      for (size_t i = 0; i < bytes_read / 2; i++) {
        float v = s[i] * g;
        s[i] = static_cast<int16_t>(v > 32767.0f ? 32767.0f : v < -32768.0f ? -32768.0f : v);
      }
    }

    // Single-pass: compute frame peak + RMS accumulator for 2s log + update smoothed display level.
    {
      static uint32_t mon_frames = 0;
      static int16_t mon_peak = 0;
      static int64_t mon_rms_sq = 0;
      static size_t mon_samples = 0;

      int16_t frame_peak = 0;
      const int16_t *smp = reinterpret_cast<const int16_t *>(buf);
      size_t n = bytes_read / 2;
      for (size_t i = 0; i < n; i++) {
        int16_t v = smp[i];
        int16_t av = v < 0 ? -v : v;
        if (av > frame_peak) frame_peak = av;
        if (av > mon_peak) mon_peak = av;
        mon_rms_sq += (int64_t)v * v;
      }
      mon_samples += n;

      // Smoothed peak for display thermometer: fast rise, ~270ms half-life decay at 16ms/frame.
      float fp = frame_peak / 32767.0f;
      float cur = self->mic_level_;
      self->mic_level_ = (fp > cur) ? fp : (cur * 0.96f);

      if (++mon_frames >= 128) {  // 128 × 16ms ≈ 2 s
        float peak_db = mon_peak > 0 ? 20.0f * log10f((float)mon_peak / 32767.0f) : -96.0f;
        float rms = mon_samples > 0 ? sqrtf((float)mon_rms_sq / mon_samples) : 0.0f;
        float rms_db = rms > 0.5f ? 20.0f * log10f(rms / 32767.0f) : -96.0f;
        ESP_LOGD(TAG, "mic: peak=%.1fdBFS rms=%.1fdBFS raw_peak=%d",
                 peak_db, rms_db, (int)mon_peak);
        mon_frames = 0; mon_peak = 0; mon_rms_sq = 0; mon_samples = 0;
      }
    }

    // Feed micro_wake_word (callback runs on Core 0; mww loop() consumes on Core 1)
    self->data_callbacks_.call(std::vector<uint8_t>(buf, buf + bytes_read));

    // Update PSRAM ring buffer for clip collection
    if (self->ring_buf_) {
      size_t rb = self->ring_buf_bytes_();
      uint32_t head = self->ring_head_;
      size_t space = rb - head;
      if (bytes_read <= space) {
        memcpy(self->ring_buf_ + head, buf, bytes_read);
        self->ring_head_ = (head + bytes_read) % rb;
      } else {
        memcpy(self->ring_buf_ + head, buf, space);
        memcpy(self->ring_buf_, buf + space, bytes_read - space);
        self->ring_head_ = bytes_read - space;
      }
    }
  }
}

void FullDuplexAudio::tx_task_fn(void *arg) {
  auto *self = static_cast<FullDuplexAudio *>(arg);
  std::vector<uint8_t> *chunk = nullptr;

  while (true) {
    if (xQueueReceive(self->tx_queue_, &chunk, portMAX_DELAY) != pdTRUE)
      continue;
    size_t offset = 0;
    while (offset < chunk->size()) {
      size_t written = 0;
      esp_err_t err = i2s_channel_write(self->tx_handle_, chunk->data() + offset,
                                         chunk->size() - offset, &written, pdMS_TO_TICKS(200));
      if (err != ESP_OK || written == 0) break;
      offset += written;
    }
    delete chunk;
    chunk = nullptr;
  }
}

void FullDuplexAudio::upload_task_fn(void *arg) {
  auto *self = static_cast<FullDuplexAudio *>(arg);
  Clip *clip = nullptr;

  while (true) {
    if (xQueueReceive(self->upload_queue_, &clip, portMAX_DELAY) != pdTRUE)
      continue;
    self->do_upload_(clip);
    delete clip;
    clip = nullptr;
  }
}

void FullDuplexAudio::dump_config() {
  ESP_LOGCONFIG(TAG, "Full-Duplex Audio (I2S_NUM_0):");
  ESP_LOGCONFIG(TAG, "  MCLK GPIO%d  BCLK GPIO%d  WS GPIO%d", mclk_pin_, bclk_pin_, ws_pin_);
  ESP_LOGCONFIG(TAG, "  DOUT GPIO%d (DAC)  DIN GPIO%d (ADC)  %u Hz 16-bit mono",
                dout_pin_, din_pin_, sample_rate_);
  if (!upload_url_.empty())
    ESP_LOGCONFIG(TAG, "  Upload: %s (ring %zu s / %zu KB PSRAM)", upload_url_.c_str(),
                  RING_SECS, ring_buf_bytes_() / 1024);
}

}  // namespace esphome::full_duplex_audio
