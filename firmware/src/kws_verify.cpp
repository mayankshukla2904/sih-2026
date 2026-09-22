#include "kws_verify.h"

#include <Arduino.h>
#include <cstdio>
#include <math.h>

#include "mfcc_shape.h"

#if __has_include("kws_verify_model.h")
#include "kws_verify_model.h"
#define KWS_HAS_VERIFY 1
#else
#define KWS_HAS_VERIFY 0
#endif

#if KWS_HAS_VERIFY
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

alignas(16) static uint8_t g_verify_arena[16 * 1024];
static float g_verify_delta[48 * 10];
static tflite::MicroInterpreter *g_verify = nullptr;
static TfLiteTensor *g_verify_in = nullptr;
static TfLiteTensor *g_verify_out = nullptr;
static float g_verify_in_scale = 1.0f;
static int g_verify_in_zp = 0;
static float g_verify_out_scale = 1.0f;
static int g_verify_out_zp = 0;

static int8_t quantize_verify(float v) {
  const int q = (int)lroundf(v / g_verify_in_scale) + g_verify_in_zp;
  return (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
}
#endif

static void verify_log(const char *line) {
  Serial.println(line);
  Serial0.println(line);
}

bool kws_verify_begin() {
#if !KWS_HAS_VERIFY
  verify_log("verify: off (no second-stage model)");
  return true;
#else
  const tflite::Model *model = tflite::GetModel(KWS_VERIFY_MODEL);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    verify_log("verify: schema mismatch, candidates pass through");
    return false;
  }
  static tflite::MicroMutableOpResolver<6> resolver;
  if (resolver.AddConv2D() != kTfLiteOk) return false;
  if (resolver.AddRelu() != kTfLiteOk) return false;
  if (resolver.AddMean() != kTfLiteOk) return false;
  if (resolver.AddFullyConnected() != kTfLiteOk) return false;
  if (resolver.AddSoftmax() != kTfLiteOk) return false;
  if (resolver.AddReshape() != kTfLiteOk) return false;
  static tflite::MicroInterpreter interp(
      model, resolver, g_verify_arena, sizeof(g_verify_arena));
  if (interp.AllocateTensors() != kTfLiteOk) {
    verify_log("verify: arena too small, candidates pass through");
    return false;
  }
  g_verify = &interp;
  g_verify_in = interp.input(0);
  g_verify_out = interp.output(0);
  g_verify_in_scale = g_verify_in->params.scale;
  g_verify_in_zp = g_verify_in->params.zero_point;
  g_verify_out_scale = g_verify_out->params.scale;
  g_verify_out_zp = g_verify_out->params.zero_point;
  char line[96];
  snprintf(line, sizeof(line), "verify: on model=%u arena=%u",
           (unsigned)KWS_VERIFY_MODEL_LEN, (unsigned)interp.arena_used_bytes());
  verify_log(line);
  return true;
#endif
}

bool kws_verify_feats(const float *feats, int n) {
#if !KWS_HAS_VERIFY
  (void)feats;
  (void)n;
  return true;
#else
  if (g_verify == nullptr || feats == nullptr || n != MFCC_FEAT_LEN) {
    return true;
  }
  // 49 frames x 10 coeffs, frame-major. The verifier sees the change from
  // one frame to the next, not the MFCC the main model just scored.
  const int coeffs = 10;
  const int frames = n / coeffs;
  const int n_delta = (frames - 1) * coeffs;
  if (frames != 49 || g_verify_in->bytes != n_delta) {
    return true;
  }
  for (int t = 0; t < frames - 1; ++t) {
    for (int c = 0; c < coeffs; ++c) {
      g_verify_delta[t * coeffs + c] =
          feats[(t + 1) * coeffs + c] - feats[t * coeffs + c];
    }
  }
  int8_t *in = g_verify_in->data.int8;
  for (int i = 0; i < n_delta; ++i) {
    in[i] = quantize_verify(g_verify_delta[i]);
  }
  if (g_verify->Invoke() != kTfLiteOk) {
    return true;
  }
  const int8_t *out = g_verify_out->data.int8;
  const float p_marvin = ((float)out[0] - g_verify_out_zp) * g_verify_out_scale;
  const float p_reject = ((float)out[1] - g_verify_out_zp) * g_verify_out_scale;
  const bool accept = p_marvin >= p_reject + KWS_VERIFY_MARGIN;
  if (!accept) {
    char line[80];
    snprintf(line, sizeof(line), "REJECT p_marvin=%.3f p_reject=%.3f", p_marvin, p_reject);
    verify_log(line);
  }
  return accept;
#endif
}
