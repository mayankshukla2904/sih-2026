#pragma once

#include <stdint.h>

// SSD1306 128×64 on PIN_OLED_SDA / PIN_OLED_SCL. Safe if the panel is missing:
// oled_begin() returns false and every later call is a no-op.

bool oled_begin();
bool oled_present();

// mood: 0 listen (waiting for marvin), 2 awake (keyword hit). cpu_pct is
// hop-loop idle duty. Redraws when mood/score/cpu (0.1%) change, or force.
void oled_show(int mood, float kw_score, float cpu_pct, bool force);
