#pragma once

#include <stddef.h>
#include <stdint.h>

struct KwsResult {
  bool awake;
  // First model said keyword, second model said no. The clip is still logged.
  // The stream to the Pi does not open.
  bool rejected;
  float score;  // smoothed p_keyword
  float p_keyword;
  float p_unknown;
  float p_silence;
  uint32_t mfcc_us;
  uint32_t invoke_us;
};

// Bring up TFLM on the embedded INT8 model. False means the arena is too small
// or an operator is missing; main() reports that instead of pretending to listen.
bool kws_begin();

void kws_reset();

// clip is CLIP_SAMPLES int16 samples at 16 kHz.
KwsResult kws_infer_clip(const int16_t *clip);

// Oldest-first scores since the last kws_reset, up to maxn.
int kws_copy_trail(float *pkw, float *pun, float *psil, int maxn);

// For the boot banner and the deck: arena high-water mark and model size.
size_t kws_arena_used();
size_t kws_model_bytes();
