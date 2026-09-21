#include "energy_gate.h"

#include <math.h>

#include "pins.h"

void energy_gate_init(EnergyGate *g) {
  g->noise_floor = 0.0f;
  g->last_speech_ratio = 0.0f;
  g->hp_x1 = 0.0f;
  g->hp_y1 = 0.0f;
  g->hits = 0;
}

void energy_gate_reset_hits(EnergyGate *g) {
  g->hits = 0;
}

static float speech_ratio(EnergyGate *g, const int16_t *hop, int n, float rms) {
  if (n <= 0 || rms < 1e-8f) {
    g->hp_x1 = 0.0f;
    g->hp_y1 = 0.0f;
    return 0.0f;
  }
  double acc = 0.0;
  float x1 = g->hp_x1;
  float y1 = g->hp_y1;
  for (int i = 0; i < n; ++i) {
    const float x = (float)hop[i] * (1.0f / 32768.0f);
    const float y = x - x1 + ENERGY_SPEECH_HP * y1;
    x1 = x;
    y1 = y;
    acc += (double)y * (double)y;
  }
  g->hp_x1 = x1;
  g->hp_y1 = y1;
  const float hp_rms = sqrtf((float)(acc / (double)n));
  return hp_rms / rms;
}

bool energy_gate_update(EnergyGate *g, const int16_t *hop, int n, float rms) {
  g->last_speech_ratio = speech_ratio(g, hop, n, rms);

  const bool loud = rms > ENERGY_RMS_THRESHOLD;
  if (loud) {
    if (g->hits < ENERGY_CONSECUTIVE) {
      g->hits++;
    }
  } else if (g->hits > 0) {
    g->hits--;
  }

  bool open = false;
  if (g->hits >= ENERGY_CONSECUTIVE) {
    // Extra checks only once the slow floor says the room is actually noisy.
    // A 3 m word in a quiet room never sees them.
    if (g->noise_floor <= ENERGY_NOISY_FLOOR) {
      open = true;
    } else {
      const float margin = fmaxf(ENERGY_ONSET_ABS, ENERGY_ONSET_REL * g->noise_floor);
      const bool onset = rms >= g->noise_floor + margin;
      const bool speech_onset =
          g->last_speech_ratio >= ENERGY_SPEECH_RATIO && rms >= g->noise_floor + 0.5f * margin;
      open = onset || speech_onset;
    }
  }

  // Update the floor after the decision so this hop cannot lift the floor
  // into its own onset check.
  if (rms < g->noise_floor) {
    g->noise_floor = ENERGY_FLOOR_DOWN * g->noise_floor + (1.0f - ENERGY_FLOOR_DOWN) * rms;
  } else {
    g->noise_floor = ENERGY_FLOOR_UP * g->noise_floor + (1.0f - ENERGY_FLOOR_UP) * rms;
  }
  return open;
}
