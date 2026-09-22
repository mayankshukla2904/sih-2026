#include "kws.h"

#include <Arduino.h>

#include "kws_model.h"
#include "kws_verify.h"
#include "mfcc.h"
#include "pins.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef KWS_PROFILE
#define KWS_PROFILE 0
#endif
#if KWS_PROFILE
#include "tensorflow/lite/micro/micro_profiler.h"
static tflite::MicroProfiler g_profiler;
#endif

// Internal SRAM, not PSRAM: TFLM touches the arena on every invoke and PSRAM
// would cost more than the kernels save.
// 16-byte aligned: TFLM requires it, and ESP-NN's S3 assembly kernels only take
// their aligned load/store path when the tensors they are handed are aligned.
alignas(16) static uint8_t g_arena[KWS_ARENA_BYTES];

static tflite::MicroInterpreter *g_interp = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;
static float g_in_scale = 1.0f;
static int g_in_zp = 0;
static float g_out_scale = 1.0f;
static int g_out_zp = 0;

static float g_feats[MFCC_FEAT_LEN];
static float g_scores[SMOOTH_WINDOW];
static int g_n = 0;
static int g_pos = 0;
static int g_hits = 0;

// One entry per inference while the gate is open. Printed on wake so a false
// accept keeps its score trail, not only the final number.
static constexpr int kTrail = 16;
static float g_trail_kw[kTrail];
static float g_trail_un[kTrail];
static float g_trail_sil[kTrail];
static int g_trail_n = 0;

bool kws_begin() {
  const tflite::Model *model = tflite::GetModel(KWS_MODEL);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("tflm schema %lu != %d\n", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Exactly the operators DS-CNN-S needs. A full resolver would waste flash.
  // PAD is the explicit ZeroPadding2D in front of each VALID depthwise conv.
  static tflite::MicroMutableOpResolver<7> resolver;
  if (resolver.AddConv2D() != kTfLiteOk) return false;
  if (resolver.AddDepthwiseConv2D() != kTfLiteOk) return false;
  if (resolver.AddPad() != kTfLiteOk) return false;
  if (resolver.AddRelu() != kTfLiteOk) return false;
  if (resolver.AddMean() != kTfLiteOk) return false;
  if (resolver.AddFullyConnected() != kTfLiteOk) return false;
  if (resolver.AddSoftmax() != kTfLiteOk) return false;

#if KWS_PROFILE
  // Per-layer timings, to find which kernel misses its ESP-NN fast path. Off in
  // the demo build: the profiler adds a timer read around every operator.
  static tflite::MicroInterpreter interp(model, resolver, g_arena, sizeof(g_arena), nullptr, &g_profiler);
#else
  static tflite::MicroInterpreter interp(model, resolver, g_arena, sizeof(g_arena));
#endif
  if (interp.AllocateTensors() != kTfLiteOk) {
    Serial.println("tflm AllocateTensors failed — raise KWS_ARENA_BYTES");
    return false;
  }
  g_interp = &interp;
  g_input = interp.input(0);
  g_output = interp.output(0);

  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8) {
    Serial.println("tflm model is not fully int8");
    return false;
  }
  if (g_input->bytes != (size_t)MFCC_FEAT_LEN) {
    Serial.printf("tflm input %u bytes, expected %d\n", (unsigned)g_input->bytes, MFCC_FEAT_LEN);
    return false;
  }

  g_in_scale = g_input->params.scale;
  g_in_zp = g_input->params.zero_point;
  g_out_scale = g_output->params.scale;
  g_out_zp = g_output->params.zero_point;
  return true;
}

void kws_reset() {
  g_n = 0;
  g_pos = 0;
  g_hits = 0;
  g_trail_n = 0;
}

int kws_copy_trail(float *pkw, float *pun, float *psil, int maxn) {
  const int stored = g_trail_n < kTrail ? g_trail_n : kTrail;
  const int n = stored < maxn ? stored : maxn;
  const int begin = g_trail_n - n;
  for (int i = 0; i < n; ++i) {
    const int j = (begin + i) % kTrail;
    pkw[i] = g_trail_kw[j];
    pun[i] = g_trail_un[j];
    psil[i] = g_trail_sil[j];
  }
  return n;
}

size_t kws_arena_used() {
  return g_interp ? g_interp->arena_used_bytes() : 0;
}

size_t kws_model_bytes() {
  return KWS_MODEL_LEN;
}

static inline int8_t quantize(float v) {
  const int q = (int)lroundf(v / g_in_scale) + g_in_zp;
  return (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}

KwsResult kws_infer_clip(const int16_t *clip) {
  KwsResult r{};
  if (g_interp == nullptr) {
    return r;
  }

  uint32_t t0 = micros();
  mfcc_from_clip(clip, g_feats);
  r.mfcc_us = micros() - t0;

  int8_t *in = g_input->data.int8;
  for (int i = 0; i < MFCC_FEAT_LEN; ++i) {
    in[i] = quantize(g_feats[i]);
  }

  t0 = micros();
#if KWS_PROFILE
  g_profiler.ClearEvents();
#endif
  if (g_interp->Invoke() != kTfLiteOk) {
    return r;
  }
  r.invoke_us = micros() - t0;
#if KWS_PROFILE
  g_profiler.Log();
#endif

  const int8_t *out = g_output->data.int8;
  r.p_keyword = ((float)out[0] - g_out_zp) * g_out_scale;
  r.p_unknown = ((float)out[1] - g_out_zp) * g_out_scale;
  r.p_silence = ((float)out[2] - g_out_zp) * g_out_scale;
  const int slot = g_trail_n % kTrail;
  g_trail_kw[slot] = r.p_keyword;
  g_trail_un[slot] = r.p_unknown;
  g_trail_sil[slot] = r.p_silence;
  if (g_trail_n < 100000) {
    g_trail_n++;
  }

  g_scores[g_pos] = r.p_keyword;
  g_pos = (g_pos + 1) % SMOOTH_WINDOW;
  if (g_n < SMOOTH_WINDOW) {
    g_n++;
  }
  float sum = 0.0f;
  for (int i = 0; i < g_n; ++i) {
    sum += g_scores[i];
  }
  r.score = sum / (float)g_n;

  if (r.score >= WAKE_SCORE_THRESHOLD && r.p_keyword >= r.p_unknown + KWS_MARGIN &&
      r.p_keyword >= r.p_silence) {
    g_hits++;
  } else {
    g_hits = 0;
  }
  r.awake = (g_hits >= DEBOUNCE_HITS);
  r.rejected = false;
  // The verifier only sees clips the first model already called a keyword.
  // It cannot recover a marvin the first model missed. A rejection still
  // leaves `rejected` set so the clip and score trail are saved.
  if (r.awake && !kws_verify_feats(g_feats, MFCC_FEAT_LEN)) {
    r.awake = false;
    r.rejected = true;
  }
  return r;
}
