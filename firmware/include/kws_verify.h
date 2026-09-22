#pragma once

// Second-stage check. Runs only after the main DS-CNN has already called a
// clip a keyword candidate. With no verify model built yet, every candidate
// is accepted and the wake behaviour stays the one already on the chip.
bool kws_verify_begin();
bool kws_verify_feats(const float *feats, int n);
