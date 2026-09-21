#pragma once

/*
 * ESP32-S3-DevKitC (not Nano). Use the GPIO numbers printed on the header.
 *
 * INMP441 (3.3 V only). Moved off 4/5/6 after those pins stayed silent:
 *   VDD -> 3V3
 *   GND -> GND
 *   L/R -> GND
 *   SCK / BCLK -> GPIO 15
 *   WS  / LRCK -> GPIO 16
 *   SD  / DOUT -> GPIO 17
 *
 * SSD1306 128×64 OLED: SDA GPIO 8, SCL GPIO 9 (optional; firmware degrades)
 * HW-104 stays on the Pi jack.
 */

#define PIN_I2S_SCK 15
#define PIN_I2S_WS 16
#define PIN_I2S_SD 17

#define PIN_OLED_SDA 8
#define PIN_OLED_SCL 9

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define HOP_SAMPLES 320
#define CLIP_SAMPLES 15840
#define PREROLL_SAMPLES 8000
// Skip the CNN only on true hiss so idle hop-CPU stays under 10%.
// Do not raise this to keep a loud room at 0.4% — that misses marvin.
#define ENERGY_RMS_THRESHOLD 0.015f
#define ENERGY_CONSECUTIVE 2
// Adaptive floor / onset / speech-band. Quiet rooms (floor below
// ENERGY_NOISY_FLOOR) keep the two-hop RMS gate above, so TPR/FAR at
// 0.65/3/3 do not move. Noisy rooms only *skip* inference on steady
// non-speech; they never open the gate when the old rule would not.
#define ENERGY_NOISY_FLOOR 0.030f
#define ENERGY_ONSET_ABS 0.006f
#define ENERGY_ONSET_REL 0.25f
#define ENERGY_FLOOR_DOWN 0.90f
#define ENERGY_FLOOR_UP 0.995f
#define ENERGY_SPEECH_HP 0.8889f  // exp(-2*pi*300/16000)
#define ENERGY_SPEECH_RATIO 0.55f
#define INFER_EVERY_HOPS 5
#define REFRACTORY_MS 1200
#define SILENCE_END_MS 3500
#define STREAM_VOICE_RMS 0.028f
#define STREAM_TIMEOUT_MS 10000

// S3 firmware-faithful sweep (results/tuning.json, 20 Sep 23:42 IST):
// 0.35/1/1 → TPR 0.94 but 200 FA/h. 0.65/3/3 → TPR 0.80 at 15 FA/h.
#define WAKE_SCORE_THRESHOLD 0.65f
#define SMOOTH_WINDOW 3
#define DEBOUNCE_HITS 3
#define KWS_MARGIN 0.02f

// After DC-block. 12x: 1 m speech lands near the close-mic MFCCs the net saw.
#define MIC_GAIN 12

// TFLM scratch. The model's tensors need ~50 KB and the interpreter's own
// bookkeeping sits in here too. Boot prints the high-water mark; trim to it.
#define KWS_ARENA_BYTES (64 * 1024)
