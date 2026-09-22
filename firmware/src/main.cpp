// Anuvani S3 listen node.
//
// Core 0 (this loop): I2S capture, energy gate, ring buffer, KWS1 stream,
//   TEL1 telemetry, serial commands. One 20 ms hop must never block.
// Core 1 (kws_task): MFCC + INT8 DS-CNN-S through TFLM/ESP-NN. Takes tens of
//   milliseconds, so it cannot run inside the audio loop.
//
// Idle (no speech): capture + RMS + cheap gate + ring push. Target < 10% CPU.
// The wake clip and its score trail go out on USB serial (W / TRAIL lines).
// Only post-wake audio opens the KWS1 stream to the Pi.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiUdp.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <driver/i2s.h>
#include <math.h>

#include "energy_gate.h"
#include "kws.h"
#include "kws_verify.h"
#include "oled.h"
#include "pins.h"
#include "wifi_secrets.h"

#define PIN_I2S_DUMMY_DOUT 11
#define TEL_PORT 8766

#ifndef ASR_PORT
#define ASR_PORT 8765
#endif

static bool g_i2s = false;
static bool g_kws_ready = false;
static WiFiUDP g_udp;
static WiFiClient g_asr_tcp;
static IPAddress g_asr;

static int16_t g_ring[CLIP_SAMPLES];
static int g_ring_fill = 0;
static int g_ring_write = 0;  // next write index; wrap, not a sliding memmove
static EnergyGate g_gate;
static int g_speech_hops = 0;
static uint32_t g_last_wake_ms = 0;
static bool g_streaming = false;
static uint32_t g_stream_start_ms = 0;
static uint32_t g_last_voice_ms = 0;

// Core 0 fills g_infer_clip and signals; core 1 fills g_infer_out and raises
// g_infer_fresh. One writer per field, published behind a barrier.
static int16_t g_infer_clip[CLIP_SAMPLES];
static KwsResult g_infer_out = {};
static volatile bool g_infer_busy = false;
static volatile bool g_infer_fresh = false;
static SemaphoreHandle_t g_infer_go = nullptr;

static float g_kw = 0.0f;
static uint32_t g_oled_wake_until = 0;
static uint32_t g_last_mfcc_us = 0;
static uint32_t g_last_invoke_us = 0;
static uint32_t g_wake_count = 0;
static float g_idle_cpu = 0.0f;

// Once-a-second listen diagnostic. The hop loop only stores integers.
// The print runs on the inference core so it is outside idle-CPU accounting.
static std::atomic<uint32_t> g_diag_invokes{0};
static std::atomic<uint32_t> g_diag_max_milli{0};
static std::atomic<uint32_t> g_diag_rms_milli{0};
static std::atomic<uint32_t> g_diag_open{0};

static void diag_note_score(float score) {
  uint32_t milli = score > 0.0f ? (uint32_t)(score * 1000.0f + 0.5f) : 0;
  uint32_t cur = g_diag_max_milli.load();
  while (milli > cur && !g_diag_max_milli.compare_exchange_weak(cur, milli)) {
  }
}
static uint32_t g_last_lat_ms = 0;
static uint32_t g_wake_detect_ms = 0;

// Serial-driven modes, used by tools/mic_check.py and tools/record_s3.py.
static bool g_mic_monitor = false;
static bool g_capture = false;
static float g_mic_peak = 0.0f;
static int32_t g_mic_dc = 0;
static int g_i2s_slot = 0;
static float g_slot_rms_a = 0.0f;
static float g_slot_rms_b = 0.0f;

static void log_line(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  Serial.println(buf);
  Serial0.println(buf);
}

static bool asr_probe(IPAddress ip) {
  if (!ip) {
    return false;
  }
  WiFiClient c;
  c.setTimeout(2);
  const bool ok = c.connect(ip, ASR_PORT);
  if (ok) {
    c.stop();
  }
  return ok;
}

static void pick_asr() {
  IPAddress primary, fallback, mdns;
  primary.fromString(ASR_HOST);
#ifdef ASR_HOST_FALLBACK
  fallback.fromString(ASR_HOST_FALLBACK);
#endif
  if (asr_probe(primary)) {
    g_asr = primary;
    return;
  }
  if (WiFi.hostByName("sahayak.local", mdns) && asr_probe(mdns)) {
    g_asr = mdns;
    return;
  }
#ifdef ASR_HOST_FALLBACK
  if (asr_probe(fallback)) {
    g_asr = fallback;
    return;
  }
#endif
  g_asr = primary ? primary : fallback;
}

static bool wifi_up() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }
  // Non-blocking: a 20 s WiFi.begin wait froze MIC/STATUS and the audio loop.
  static uint32_t attempt_ms = 0;
  static bool joining = false;
  const uint32_t now = millis();
  if (!joining || now - attempt_ms >= 25000) {
    log_line("wifi joining %s", WIFI_SSID);
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(true);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    attempt_ms = now;
    joining = true;
  }
  if (now - attempt_ms < 20000) {
    return false;
  }
  joining = false;
  if (WiFi.status() != WL_CONNECTED) {
    log_line("wifi FAIL status=%d", (int)WiFi.status());
    return false;
  }
  pick_asr();
  g_udp.begin(TEL_PORT);
  log_line("wifi ok ip=%s asr=%s", WiFi.localIP().toString().c_str(), g_asr.toString().c_str());
  return true;
}

static bool init_i2s() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 4;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;
  cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
  cfg.bits_per_chan = I2S_BITS_PER_CHAN_32BIT;
  cfg.chan_mask = (i2s_channel_t)(I2S_TDM_ACTIVE_CH0 | I2S_TDM_ACTIVE_CH1);
  cfg.total_chan = 2;
  cfg.left_align = true;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_I2S_SCK;
  pins.ws_io_num = PIN_I2S_WS;
  pins.data_out_num = PIN_I2S_DUMMY_DOUT;
  pins.data_in_num = PIN_I2S_SD;

  if (i2s_driver_install(I2S_PORT, &cfg, 0, nullptr) != ESP_OK) {
    return false;
  }
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) {
    return false;
  }
  return i2s_start(I2S_PORT) == ESP_OK;
}

static void ring_clear() {
  memset(g_ring, 0, sizeof(g_ring));
  g_ring_fill = 0;
  g_ring_write = 0;
}

static void ring_push(const int16_t *hop) {
  // Circular: one 640-byte memcpy per hop instead of sliding 31 KB.
  const int space = CLIP_SAMPLES - g_ring_write;
  if (space >= HOP_SAMPLES) {
    memcpy(g_ring + g_ring_write, hop, HOP_SAMPLES * sizeof(int16_t));
    g_ring_write += HOP_SAMPLES;
    if (g_ring_write >= CLIP_SAMPLES) {
      g_ring_write = 0;
    }
  } else {
    memcpy(g_ring + g_ring_write, hop, space * sizeof(int16_t));
    memcpy(g_ring, hop + space, (HOP_SAMPLES - space) * sizeof(int16_t));
    g_ring_write = HOP_SAMPLES - space;
  }
  if (g_ring_fill < CLIP_SAMPLES) {
    g_ring_fill = min(CLIP_SAMPLES, g_ring_fill + HOP_SAMPLES);
  }
}

// Oldest-to-newest linear copy. CLIP_SAMPLES is not a multiple of HOP_SAMPLES,
// so a wrap can fall mid-hop; handle that here, not on the 20 ms listen path.
static void ring_copy_linear(int16_t *dst, int n) {
  int start = g_ring_write - n;
  if (start < 0) {
    start += CLIP_SAMPLES;
  }
  if (start + n <= CLIP_SAMPLES) {
    memcpy(dst, g_ring + start, n * sizeof(int16_t));
    return;
  }
  const int first = CLIP_SAMPLES - start;
  memcpy(dst, g_ring + start, first * sizeof(int16_t));
  memcpy(dst + first, g_ring, (n - first) * sizeof(int16_t));
}

// 2.5 KB each. On the stack they overflowed loopTask's 8 KB; as statics they also
// stop us memsetting the silence buffer on every 20 ms hop.
static int32_t g_i2s_zeros[HOP_SAMPLES * 2];
static int32_t g_i2s_rx[HOP_SAMPLES * 2];
static int32_t g_hop_raw[HOP_SAMPLES];
// Wall time spent blocked in the I2S driver. The DMA wait is not CPU: the
// listen-loop duty we report subtracts this so idle CPU is real work, not the
// 20 ms hop period.
static uint32_t g_i2s_wait_us = 0;

static float hop_capture(int16_t *hop_out) {
  if (!g_i2s) {
    memset(hop_out, 0, HOP_SAMPLES * sizeof(int16_t));
    g_i2s_wait_us = 0;
    return 0.0f;
  }
  size_t nw = 0, nr = 0;
  const uint32_t tw = micros();
  i2s_write(I2S_PORT, g_i2s_zeros, sizeof(g_i2s_zeros), &nw, pdMS_TO_TICKS(5));
  int32_t *rx = g_i2s_rx;
  i2s_read(I2S_PORT, rx, sizeof(g_i2s_rx), &nr, pdMS_TO_TICKS(25));
  g_i2s_wait_us = micros() - tw;
  const int n = (int)(nr / sizeof(int32_t));
  int got = 0;
  int64_t energy_a = 0;
  int64_t energy_b = 0;
  for (int i = 0; i + 1 < n && got < HOP_SAMPLES; i += 2) {
    const int32_t a24 = rx[i] >> 8;
    const int32_t b24 = rx[i + 1] >> 8;
    energy_a += (int64_t)a24 * (int64_t)a24;
    energy_b += (int64_t)b24 * (int64_t)b24;
    got++;
  }
  // One slot for the whole hop. Picking the louder sample each step splices
  // the empty channel into silence and turns any word into a keyword-like hash.
  const bool use_b = energy_b > energy_a;
  g_i2s_slot = use_b ? 1 : 0;
  g_slot_rms_a = sqrtf((float)energy_a / (float)(got > 0 ? got : 1)) / 8388608.0f;
  g_slot_rms_b = sqrtf((float)energy_b / (float)(got > 0 ? got : 1)) / 8388608.0f;
  int64_t sum = 0;
  got = 0;
  for (int i = 0; i + 1 < n && got < HOP_SAMPLES; i += 2) {
    const int32_t s24 = use_b ? (rx[i + 1] >> 8) : (rx[i] >> 8);
    g_hop_raw[got++] = s24;
    sum += s24;
  }
  if (got <= 0) {
    memset(hop_out, 0, HOP_SAMPLES * sizeof(int16_t));
    return 0.0f;
  }
  if (got < HOP_SAMPLES) {
    memset(hop_out + got, 0, (HOP_SAMPLES - got) * sizeof(int16_t));
  }
  const int32_t mean = (int32_t)(sum / got);
  int64_t acc = 0;
  float peak = 0.0f;
  for (int i = 0; i < got; ++i) {
    int32_t s = (g_hop_raw[i] - mean) * (int32_t)MIC_GAIN;
    if (s > 8388607) {
      s = 8388607;
    } else if (s < -8388608) {
      s = -8388608;
    }
    hop_out[i] = (int16_t)(s >> 8);
    const int32_t s16 = hop_out[i];
    acc += (int64_t)s16 * (int64_t)s16;
    if (g_mic_monitor) {
      const float a = fabsf((float)s * (1.0f / 8388608.0f));
      if (a > peak) {
        peak = a;
      }
    }
  }
  if (g_mic_monitor) {
    g_mic_peak = peak;
    g_mic_dc = mean;
  }
  // hop_out is s>>8, so /32768 matches the old 24-bit / 8388608 energy scale.
  return sqrtf((float)acc / (float)got) * (1.0f / 32768.0f);
}

// --- inference on core 0 (Arduino loopTask is on core 1) ---------------------

static void diag_task(void *) {
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    const uint32_t invokes = g_diag_invokes.exchange(0);
    const uint32_t max_m = g_diag_max_milli.exchange(0);
    const uint32_t rms_m = g_diag_rms_milli.load();
    const bool open = g_diag_open.load() != 0;
    char line[128];
    snprintf(line, sizeof(line),
             "DIAG1 t=%u gate=%s rms=%.3f thr=%.3f invokes=%u max_pkw=%.3f",
             (unsigned)millis(), open ? "open" : "closed", rms_m / 1000.0f,
             (double)ENERGY_RMS_THRESHOLD, (unsigned)invokes, max_m / 1000.0f);
    Serial.println(line);
    Serial0.println(line);
  }
}

static void kws_task(void *) {
  for (;;) {
    xSemaphoreTake(g_infer_go, portMAX_DELAY);
    g_infer_out = kws_infer_clip(g_infer_clip);
    g_diag_invokes.fetch_add(1);
    diag_note_score(g_infer_out.score);
    __sync_synchronize();
    g_infer_fresh = true;
    g_infer_busy = false;
  }
}

// Hands the current ring to core 1. The memcpy is ~30 KB, well inside one hop,
// and means core 1 never reads a buffer core 0 is still writing.
// Speech Commands and our S3 clips are close-mic. Far speech is the same
// word at a lower RMS; boosting the 1 s clip to ~0.10 FS before MFCC makes
// 1 m look like the training set. Silence is left alone so hiss is not a keyword.
static void clip_match_train_level(int16_t *clip, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const float f = (float)clip[i] / 32768.0f;
    acc += (double)f * f;
  }
  const float rms = (float)sqrt(acc / (double)n);
  // Room tone after gain 6 sits near 0.04. Boosting that back to 0.10
  // made hiss look like the close-mic keyword. Only lift real speech.
  if (rms < 0.070f) {
    return;
  }
  const float target = 0.10f;
  float g = target / rms;
  if (g <= 1.05f) {
    return;
  }
  if (g > 2.0f) {
    g = 2.0f;
  }
  for (int i = 0; i < n; ++i) {
    int y = (int)lroundf((float)clip[i] * g);
    if (y > 32767) {
      y = 32767;
    } else if (y < -32767) {
      y = -32767;
    }
    clip[i] = (int16_t)y;
  }
}

static bool infer_request() {
  if (g_infer_busy) {
    return false;
  }
  ring_copy_linear(g_infer_clip, CLIP_SAMPLES);
  clip_match_train_level(g_infer_clip, CLIP_SAMPLES);
  g_infer_busy = true;
  xSemaphoreGive(g_infer_go);
  return true;
}

// --- KWS1 stream to the Pi ---------------------------------------------------

static void send_tel(const char *state, float cpu_pct, float rms) {
  char line[256];
  snprintf(
      line,
      sizeof(line),
      "TEL1 state=%s cpu=%.1f idle=%.1f heap=%u rssi=%d rms=%.5f kw=%.3f mfcc_us=%u inv_us=%u arena=%u wakes=%u lat_ms=%u",
      state,
      cpu_pct,
      g_idle_cpu,
      (unsigned)(ESP.getFreeHeap() / 1024),
      WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0,
      rms,
      g_kw,
      (unsigned)g_last_mfcc_us,
      (unsigned)g_last_invoke_us,
      (unsigned)kws_arena_used(),
      (unsigned)g_wake_count,
      (unsigned)g_last_lat_ms);
  // USB tools (bench, far_test) parse this; the Pi STATS page listens on UDP.
  Serial.println(line);
  Serial0.println(line);
  if (WiFi.status() == WL_CONNECTED) {
    g_udp.beginPacket(g_asr, TEL_PORT);
    g_udp.write((const uint8_t *)line, strlen(line));
    g_udp.endPacket();
  }
}

static void send_u16(uint16_t v) {
  uint8_t b[2] = {(uint8_t)(v & 0xff), (uint8_t)(v >> 8)};
  g_asr_tcp.write(b, 2);
}

static bool stream_begin() {
  if (!g_asr_tcp.connect(g_asr, ASR_PORT)) {
    log_line("kws1 connect fail");
    return false;
  }
  g_asr_tcp.setNoDelay(true);
  uint8_t hdr[11];
  hdr[0] = 'K';
  hdr[1] = 'W';
  hdr[2] = 'S';
  hdr[3] = '1';
  hdr[4] = 1;
  const uint16_t preroll_ms = 500;
  hdr[5] = (uint8_t)(preroll_ms & 0xff);
  hdr[6] = (uint8_t)(preroll_ms >> 8);
  const uint32_t sr = SAMPLE_RATE;
  hdr[7] = (uint8_t)(sr & 0xff);
  hdr[8] = (uint8_t)((sr >> 8) & 0xff);
  hdr[9] = (uint8_t)((sr >> 16) & 0xff);
  hdr[10] = (uint8_t)((sr >> 24) & 0xff);
  g_asr_tcp.write(hdr, sizeof(hdr));

  // g_infer_clip was linearized at the wake infer; last 500 ms is the preroll.
  const int16_t *pre = g_infer_clip + (CLIP_SAMPLES - PREROLL_SAMPLES);
  for (int off = 0; off < PREROLL_SAMPLES; off += HOP_SAMPLES) {
    send_u16((uint16_t)(HOP_SAMPLES * 2));
    g_asr_tcp.write((const uint8_t *)(pre + off), HOP_SAMPLES * 2);
  }
  g_streaming = true;
  g_stream_start_ms = millis();
  g_last_voice_ms = millis();
  if (g_wake_detect_ms) {
    g_last_lat_ms = g_stream_start_ms - g_wake_detect_ms;
  }
  log_line("kws1 stream open lat_ms=%u", (unsigned)g_last_lat_ms);
  return true;
}

static void stream_hop(const int16_t *hop) {
  if (!g_asr_tcp.connected()) {
    g_streaming = false;
    return;
  }
  send_u16((uint16_t)(HOP_SAMPLES * 2));
  g_asr_tcp.write((const uint8_t *)hop, HOP_SAMPLES * 2);
}

static void stream_end() {
  if (g_asr_tcp.connected()) {
    send_u16(0);
    g_asr_tcp.stop();
  }
  g_streaming = false;
  g_last_wake_ms = millis();
  kws_reset();
  ring_clear();
  g_speech_hops = 0;
  energy_gate_reset_hits(&g_gate);
  log_line("kws1 stream end");
}

// --- serial console ----------------------------------------------------------

static const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// One hop as `A <base64>`. tools/record_s3.py reassembles these into WAVs, so
// dataset clips come through the real mic without needing WiFi or the Pi.
static void emit_pcm_b64(char tag, const int16_t *pcm, int samples) {
  const uint8_t *src = (const uint8_t *)pcm;
  const int n = samples * 2;
  char out[((HOP_SAMPLES * 2 + 2) / 3) * 4 + 4];
  int o = 0;
  for (int i = 0; i < n; i += 3) {
    const uint32_t a = src[i];
    const uint32_t b = (i + 1 < n) ? src[i + 1] : 0;
    const uint32_t c = (i + 2 < n) ? src[i + 2] : 0;
    const uint32_t v = (a << 16) | (b << 8) | c;
    out[o++] = kB64[(v >> 18) & 0x3f];
    out[o++] = kB64[(v >> 12) & 0x3f];
    out[o++] = (i + 1 < n) ? kB64[(v >> 6) & 0x3f] : '=';
    out[o++] = (i + 2 < n) ? kB64[v & 0x3f] : '=';
  }
  out[o] = 0;
  Serial.print(tag);
  Serial.print(' ');
  Serial.println(out);
  Serial0.print(tag);
  Serial0.print(' ');
  Serial0.println(out);
}

static void emit_hop_b64(const int16_t *hop) {
  emit_pcm_b64('A', hop, HOP_SAMPLES);
}

// Wake audio is copied out of g_infer_clip because the next inference overwrites it.
static int16_t g_wake_pcm[CLIP_SAMPLES];
static int g_wake_off = -1;
static uint32_t g_wake_pcm_t = 0;

static void wake_clip_arm(uint32_t t_ms) {
  memcpy(g_wake_pcm, g_infer_clip, sizeof(g_wake_pcm));
  g_wake_off = 0;
  g_wake_pcm_t = t_ms;
  log_line("WAKECLIP t=%u samples=%d", (unsigned)t_ms, CLIP_SAMPLES);
}

// Extra inferences after the wake decision, so the saved trail can show the
// score falling once the word leaves the window. Does not change the decision.
static int g_post_left = 0;

static void log_trail(uint32_t t_ms, bool post) {
  float pkw[16], pun[16], psil[16];
  const int n = kws_copy_trail(pkw, pun, psil, 16);
  char kw[8 * 16], un[8 * 16], sil[8 * 16];
  kw[0] = un[0] = sil[0] = 0;
  for (int i = 0; i < n; ++i) {
    char one[16];
    snprintf(one, sizeof(one), "%s%.3f", i ? "," : "", (double)pkw[i]);
    strncat(kw, one, sizeof(kw) - strlen(kw) - 1);
    snprintf(one, sizeof(one), "%s%.3f", i ? "," : "", (double)pun[i]);
    strncat(un, one, sizeof(un) - strlen(un) - 1);
    snprintf(one, sizeof(one), "%s%.3f", i ? "," : "", (double)psil[i]);
    strncat(sil, one, sizeof(sil) - strlen(sil) - 1);
  }
  log_line("TRAIL t=%u n=%d pkw=%s pun=%s psil=%s%s", (unsigned)t_ms, n, kw, un, sil,
           post ? " post=1" : "");
}

static void wake_clip_pump() {
  if (g_wake_off < 0 || g_capture) {
    return;
  }
  const int left = CLIP_SAMPLES - g_wake_off;
  // Hold the last hop until the post-wake scores are logged, so the collector
  // stores the longer trail. Give up after 2.5 s so a stuck infer still finishes.
  if (left <= HOP_SAMPLES && g_post_left > 0 && millis() - g_wake_pcm_t < 2500) {
    return;
  }
  const int n = left > HOP_SAMPLES ? HOP_SAMPLES : left;
  emit_pcm_b64('W', g_wake_pcm + g_wake_off, n);
  g_wake_off += n;
  if (g_wake_off >= CLIP_SAMPLES) {
    g_wake_off = -1;
    log_line("WAKECLIP end t=%u", (unsigned)g_wake_pcm_t);
  }
}

static void print_status() {
  log_line(
      "STATUS keyword=%s model=%u bytes arena=%u/%u i2s=%d kws=%d wakes=%u "
      "thr=%.2f mfcc_us=%u inv_us=%u heap=%u",
      KEYWORD,
      (unsigned)kws_model_bytes(),
      (unsigned)kws_arena_used(),
      (unsigned)KWS_ARENA_BYTES,
      (int)g_i2s,
      (int)g_kws_ready,
      (unsigned)g_wake_count,
      (double)WAKE_SCORE_THRESHOLD,
      (unsigned)g_last_mfcc_us,
      (unsigned)g_last_invoke_us,
      (unsigned)ESP.getFreeHeap());
}

// Steady-state MFCC and invoke cost, measured on the node itself.
//
// The warm-up number at boot is dominated by cold flash cache and is roughly 3x
// pessimistic, and the energy gate means a node with no speech in front of it
// never reports a real one. BENCH forces the work so the efficiency figures in
// the deck come from the device rather than from a desktop estimate.
static void run_bench(int reps) {
  if (!g_kws_ready) {
    log_line("BENCH unavailable: tflm not ready");
    return;
  }
  uint32_t mfcc_min = UINT32_MAX, mfcc_max = 0, inv_min = UINT32_MAX, inv_max = 0;
  uint64_t mfcc_sum = 0, inv_sum = 0;
  for (int i = 0; i < reps; ++i) {
    const KwsResult r = kws_infer_clip(g_ring);
    mfcc_sum += r.mfcc_us;
    inv_sum += r.invoke_us;
    if (r.mfcc_us < mfcc_min) mfcc_min = r.mfcc_us;
    if (r.mfcc_us > mfcc_max) mfcc_max = r.mfcc_us;
    if (r.invoke_us < inv_min) inv_min = r.invoke_us;
    if (r.invoke_us > inv_max) inv_max = r.invoke_us;
  }
  kws_reset();

  const uint32_t mfcc_avg = (uint32_t)(mfcc_sum / reps);
  const uint32_t inv_avg = (uint32_t)(inv_sum / reps);
  // One clip is evaluated every INFER_EVERY_HOPS * HOP_MS of audio, so this is
  // the share of one core the detector costs while speech is present.
  const float period_us = (float)INFER_EVERY_HOPS * (float)HOP_SAMPLES * 1e6f / (float)SAMPLE_RATE;
  const float duty = 100.0f * (float)(mfcc_avg + inv_avg) / period_us;
  log_line(
      "BENCH reps=%d mfcc_us=%u/%u/%u inv_us=%u/%u/%u total_us=%u "
      "period_us=%u speech_cpu=%.1f%% arena=%u/%u",
      reps,
      (unsigned)mfcc_min, (unsigned)mfcc_avg, (unsigned)mfcc_max,
      (unsigned)inv_min, (unsigned)inv_avg, (unsigned)inv_max,
      (unsigned)(mfcc_avg + inv_avg),
      (unsigned)period_us,
      (double)duty,
      (unsigned)kws_arena_used(),
      (unsigned)KWS_ARENA_BYTES);
}

static void handle_cmd(const char *buf) {
  if (!strcasecmp(buf, "MIC")) {
    g_mic_monitor = !g_mic_monitor;
    log_line("mic monitor %s", g_mic_monitor ? "on" : "off");
  } else if (!strcasecmp(buf, "REC")) {
    g_capture = true;
    log_line("capture on");
  } else if (!strcasecmp(buf, "STOP")) {
    g_capture = false;
    log_line("capture off");
  } else if (!strcasecmp(buf, "?") || !strcasecmp(buf, "STATUS")) {
    print_status();
  } else if (!strcasecmp(buf, "BENCH")) {
    run_bench(20);
  } else {
    log_line("commands: MIC | REC | STOP | STATUS | BENCH");
  }
}

static void poll_port(Stream &s, char *buf, int &len) {
  while (s.available()) {
    const char c = (char)s.read();
    if (c == '\n' || c == '\r') {
      buf[len] = 0;
      if (len) {
        handle_cmd(buf);
      }
      len = 0;
    } else if (len < 15) {
      buf[len++] = c;
    }
  }
}

static void poll_serial() {
  static char usb_buf[16];
  static int usb_len = 0;
  static char uart_buf[16];
  static int uart_len = 0;
  poll_port(Serial, usb_buf, usb_len);
  poll_port(Serial0, uart_buf, uart_len);
}

// Arduino's loopTask is pinned to ARDUINO_RUNNING_CORE (core 1 by default), so
// the inference task has to go on the *other* core or it would preempt the audio
// loop instead of running beside it.
#ifndef ARDUINO_RUNNING_CORE
#define ARDUINO_RUNNING_CORE 1
#endif
#define KWS_TASK_CORE (1 - ARDUINO_RUNNING_CORE)

// Each stage logs before it runs. If the node ever goes silent at boot, the last
// line printed says which stage hung, which beats guessing over a dead USB port.
static void boot_step(const char *what) {
  log_line("boot: %s", what);
  Serial.flush();
  delay(30);
}

void setup() {
  Serial.begin(115200);
  // UART-USB (cu.usbserial) must carry 16 kHz hops during REC. 115200 is too
  // slow (~11 kB/s); base64 audio needs ~43 kB/s. USB-CDC ignores this rate.
  Serial0.begin(921600);
  // The host CDC needs time to attach after a reset or the banner is lost.
  delay(1500);
  log_line("=== ANUVANI S3 KWS ===");
  log_line("keyword=%s  INT8 DS-CNN-S on TFLM+ESP-NN  idle target <10%% CPU", KEYWORD);
  log_line("loop core=%d  inference core=%d  cpu=%u MHz",
           (int)xPortGetCoreID(), KWS_TASK_CORE, (unsigned)getCpuFrequencyMhz());

  boot_step("clearing ring");
  ring_clear();
  kws_reset();
  energy_gate_init(&g_gate);

  boot_step("i2s");
  log_line("i2s pins SCK=%d WS=%d SD=%d (header numbers, not silkscreen)", PIN_I2S_SCK, PIN_I2S_WS, PIN_I2S_SD);
  g_i2s = init_i2s();
  log_line("i2s %s", g_i2s ? "ok" : "off");

  boot_step("oled");
  log_line("oled %s", oled_begin() ? "ok" : "off");

  boot_step("tflm allocate");
  g_kws_ready = kws_begin();
  if (g_kws_ready) {
    log_line("tflm ok model=%u bytes arena=%u/%u bytes",
             (unsigned)kws_model_bytes(),
             (unsigned)kws_arena_used(),
             (unsigned)KWS_ARENA_BYTES);
    boot_step("verify stage");
    kws_verify_begin();

    boot_step("first inference (warm-up)");
    // Run once here, on a known-quiet buffer, so a fault in the kernels shows up
    // at boot with a log line next to it instead of mid-demo.
    const KwsResult warm = kws_infer_clip(g_ring);
    log_line("warm-up ok mfcc_us=%u inv_us=%u pkw=%.3f",
             (unsigned)warm.mfcc_us, (unsigned)warm.invoke_us, warm.p_keyword);
    kws_reset();

    boot_step("inference task");
    g_infer_go = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(kws_task, "kws", 8192, nullptr, 2, nullptr, KWS_TASK_CORE);
  } else {
    log_line("tflm FAILED — node will capture but never wake");
  }
  xTaskCreatePinnedToCore(diag_task, "diag", 3072, nullptr, 1, nullptr, KWS_TASK_CORE);
  log_line("serial: MIC (mic monitor) | REC/STOP (clip capture) | STATUS | BENCH");
  log_line("boot: done");
  Serial.flush();
}

void loop() {
  poll_serial();
  wake_clip_pump();

  static int16_t hop[HOP_SAMPLES];
  const uint32_t t0 = micros();
  const float rms = hop_capture(hop);
  ring_push(hop);
  const uint32_t listen_us = micros() - t0;
  const uint32_t work_us = listen_us > g_i2s_wait_us ? listen_us - g_i2s_wait_us : 0;
  const float cpu = 100.0f * (float)work_us / 20000.0f;

  if (g_capture) {
    // Recording a dataset: stream audio out, do not run the detector.
    emit_hop_b64(hop);
    return;
  }

  if (g_mic_monitor) {
    static uint32_t last_mic = 0;
    if (millis() - last_mic >= 250) {
      last_mic = millis();
      log_line("MIC rms=%.6f peak=%.4f dc=%.1f floor=%.5f speech=%.2f slot=%d a=%.4f b=%.4f",
               rms, g_mic_peak, (double)g_mic_dc, g_gate.noise_floor, g_gate.last_speech_ratio,
               g_i2s_slot, g_slot_rms_a, g_slot_rms_b);
    }
  }

  // Join Wi-Fi in the background. Keyword spotting must not wait on it:
  // a reconnect after flash used to skip infer entirely, so marvin never fired.
  wifi_up();

  const bool maybe_word = energy_gate_update(&g_gate, hop, HOP_SAMPLES, rms);
  g_diag_open.store(maybe_word ? 1 : 0);
  g_diag_rms_milli.store((uint32_t)(rms * 1000.0f + 0.5f));

  const char *state = "listen";

  if (g_post_left > 0 && g_kws_ready && g_ring_fill >= CLIP_SAMPLES) {
    infer_request();
  }

  if (g_streaming) {
    state = "wake";
    stream_hop(hop);
    if (rms > STREAM_VOICE_RMS) {
      g_last_voice_ms = millis();
    }
    if (millis() - g_last_voice_ms >= SILENCE_END_MS || millis() - g_stream_start_ms >= STREAM_TIMEOUT_MS) {
      stream_end();
      state = "listen";
    }
  } else if (g_kws_ready && maybe_word && g_ring_fill >= CLIP_SAMPLES &&
             millis() - g_last_wake_ms >= REFRACTORY_MS) {
    // Still LISTEN: looking for marvin. No bytes leave the chip.
    if (g_speech_hops == 0 && g_post_left == 0) {
      kws_reset();
    }
    g_speech_hops++;
    infer_request();
  } else {
    g_speech_hops = 0;
    if (!maybe_word) {
      g_kw *= 0.85f;
    }
  }

  if (g_infer_fresh) {
    __sync_synchronize();
    const KwsResult r = g_infer_out;
    g_infer_fresh = false;
    g_kw = r.score;
    g_last_mfcc_us = r.mfcc_us;
    g_last_invoke_us = r.invoke_us;
    const bool candidate = (r.awake || r.rejected) && !g_streaming &&
                           millis() - g_last_wake_ms >= REFRACTORY_MS;
    if (candidate && r.rejected) {
      log_line("REJECT t=%u score=%.3f pkw=%.3f pun=%.3f psil=%.3f",
               (unsigned)millis(), r.score, r.p_keyword, r.p_unknown, r.p_silence);
      log_trail((unsigned)millis(), false);
      g_post_left = 8;
      wake_clip_arm((unsigned)millis());
      g_last_wake_ms = millis();
    } else if (candidate) {
      g_wake_count++;
      g_wake_detect_ms = millis();
      // Timestamped on the node so tools/far_test.py can align serial with audio.
      log_line("WAKE t=%u score=%.3f pkw=%.3f pun=%.3f psil=%.3f mfcc_us=%u inv_us=%u",
               (unsigned)millis(), r.score, r.p_keyword, r.p_unknown, r.p_silence,
               (unsigned)r.mfcc_us, (unsigned)r.invoke_us);
      log_trail((unsigned)millis(), false);
      g_post_left = 8;
      wake_clip_arm((unsigned)millis());
      state = "wake";
      g_oled_wake_until = millis() + 2500;
      if (!stream_begin()) {
        g_last_wake_ms = millis();
      }
    } else if (g_post_left > 0) {
      g_post_left--;
      if (g_post_left == 0) {
        log_trail((unsigned)millis(), true);
      }
    }
  }

  static uint32_t last_tel = 0;
  static uint32_t last_oled = 0;
  if (!g_streaming) {
    g_idle_cpu = cpu;
  }

  const bool show_wake = g_streaming || strcmp(state, "wake") == 0 ||
                         millis() < g_oled_wake_until;
  if (show_wake) {
    state = "wake";
  }

  int mood = show_wake ? 2 : 0;
  if (millis() - last_oled >= 1000) {
    last_oled = millis();
    oled_show(mood, g_kw, cpu, false);
  }

  if (millis() - last_tel >= 1000) {
    last_tel = millis();
    send_tel(state, cpu, rms);
  }

  if (listen_us < 18000) {
    delayMicroseconds(18000 - listen_us);
  }
}
