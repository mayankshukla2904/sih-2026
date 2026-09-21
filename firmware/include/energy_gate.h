#pragma once

#include <stdint.h>

// Listen-path gate. Quiet rooms match the old RMS > 0.015 for 2 hops.
// Noisy rooms add an onset / speech-band check so a fan does not keep the
// CNN awake. maybe_word is always a subset of the old rule, so this cannot
// add false wakes; it can only skip obvious steady noise.

struct EnergyGate {
  float noise_floor;
  float last_speech_ratio;
  float hp_x1;
  float hp_y1;
  int hits;
};

void energy_gate_init(EnergyGate *g);

// Clear hit counter after a wake. Keep the floor so a noisy room does not
// look "quiet" for the next few seconds.
void energy_gate_reset_hits(EnergyGate *g);

// hop is DC-blocked int16, n = HOP_SAMPLES, rms is hop_capture's return.
bool energy_gate_update(EnergyGate *g, const int16_t *hop, int n, float rms);
