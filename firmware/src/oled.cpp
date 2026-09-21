#include "oled.h"

#include <Arduino.h>
#include <Wire.h>
#include <string.h>

#include "pins.h"

#define OLED_ADDR 0x3C
#define OLED_W 128
#define OLED_H 64

static bool g_ok = false;
static int g_mood = -1;
static int g_kw_i = -1;
static int g_cpu_i = -1;
static uint8_t g_buf[OLED_W * OLED_H / 8];

static bool wr_cmd(uint8_t c) {
  Wire.beginTransmission(OLED_ADDR);
  Wire.write(0x00);
  Wire.write(c);
  return Wire.endTransmission() == 0;
}

static bool wr_data(const uint8_t *p, size_t n) {
  while (n) {
    const size_t chunk = n > 16 ? 16 : n;
    Wire.beginTransmission(OLED_ADDR);
    Wire.write(0x40);
    Wire.write(p, chunk);
    if (Wire.endTransmission() != 0) {
      return false;
    }
    p += chunk;
    n -= chunk;
  }
  return true;
}

static void px(int x, int y, bool on) {
  if (x < 0 || y < 0 || x >= OLED_W || y >= OLED_H) {
    return;
  }
  const int i = x + (y / 8) * OLED_W;
  const uint8_t bit = (uint8_t)(1 << (y & 7));
  if (on) {
    g_buf[i] |= bit;
  } else {
    g_buf[i] &= (uint8_t)~bit;
  }
}

static void hline(int x0, int x1, int y) {
  if (x0 > x1) {
    int t = x0;
    x0 = x1;
    x1 = t;
  }
  for (int x = x0; x <= x1; ++x) {
    px(x, y, true);
  }
}

static void circle(int cx, int cy, int r, bool fill) {
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      const int d = dx * dx + dy * dy;
      if (fill) {
        if (d <= r * r) {
          px(cx + dx, cy + dy, true);
        }
      } else if (d >= (r - 1) * (r - 1) && d <= r * r) {
        px(cx + dx, cy + dy, true);
      }
    }
  }
}

static void face(int mood) {
  const int cx = 32;
  const int cy = 34;
  circle(cx, cy, 22, false);
  if (mood == 0) {
    hline(cx - 12, cx - 4, cy - 4);
    hline(cx + 4, cx + 12, cy - 4);
    hline(cx - 6, cx + 6, cy + 8);
  } else if (mood == 1) {
    circle(cx - 8, cy - 4, 3, false);
    circle(cx + 8, cy - 4, 3, false);
    px(cx - 8, cy - 4, true);
    px(cx + 8, cy - 4, true);
    hline(cx - 5, cx + 5, cy + 8);
  } else {
    circle(cx - 8, cy - 5, 4, true);
    circle(cx + 8, cy - 5, 4, true);
    for (int i = -8; i <= 8; ++i) {
      const int y = cy + 7 + (i * i) / 20;
      px(cx + i, y, true);
    }
  }
}

// 5×7 glyphs we actually print. Enough for the status strip.
static const uint8_t kFont[][5] = {
    {0x3E, 0x51, 0x49, 0x45, 0x3E},  // 0
    {0x00, 0x42, 0x7F, 0x40, 0x00},  // 1
    {0x42, 0x61, 0x51, 0x49, 0x46},  // 2
    {0x21, 0x41, 0x45, 0x4B, 0x31},  // 3
    {0x18, 0x14, 0x12, 0x7F, 0x10},  // 4
    {0x27, 0x45, 0x45, 0x45, 0x39},  // 5
    {0x3C, 0x4A, 0x49, 0x49, 0x30},  // 6
    {0x01, 0x71, 0x09, 0x05, 0x03},  // 7
    {0x36, 0x49, 0x49, 0x49, 0x36},  // 8
    {0x06, 0x49, 0x49, 0x29, 0x1E},  // 9
    {0x00, 0x60, 0x60, 0x00, 0x00},  // .
    {0x00, 0x00, 0x00, 0x00, 0x00},  // space
    {0x7C, 0x12, 0x11, 0x12, 0x7C},  // A
    {0x7F, 0x49, 0x49, 0x49, 0x36},  // B
    {0x3E, 0x41, 0x41, 0x41, 0x22},  // C
    {0x7F, 0x41, 0x41, 0x22, 0x1C},  // D
    {0x7F, 0x49, 0x49, 0x49, 0x41},  // E
    {0x7F, 0x09, 0x09, 0x09, 0x01},  // F
    {0x3E, 0x41, 0x49, 0x49, 0x7A},  // G
    {0x7F, 0x08, 0x08, 0x08, 0x7F},  // H
    {0x00, 0x41, 0x7F, 0x41, 0x00},  // I
    {0x20, 0x40, 0x41, 0x3F, 0x01},  // J
    {0x7F, 0x08, 0x14, 0x22, 0x41},  // K
    {0x7F, 0x40, 0x40, 0x40, 0x40},  // L
    {0x7F, 0x02, 0x0C, 0x02, 0x7F},  // M
    {0x7F, 0x04, 0x08, 0x10, 0x7F},  // N
    {0x3E, 0x41, 0x41, 0x41, 0x3E},  // O
    {0x7F, 0x09, 0x09, 0x09, 0x06},  // P
    {0x3E, 0x41, 0x51, 0x21, 0x5E},  // Q
    {0x7F, 0x09, 0x19, 0x29, 0x46},  // R
    {0x46, 0x49, 0x49, 0x49, 0x31},  // S
    {0x01, 0x01, 0x7F, 0x01, 0x01},  // T
    {0x3F, 0x40, 0x40, 0x40, 0x3F},  // U
    {0x1F, 0x20, 0x40, 0x20, 0x1F},  // V
    {0x3F, 0x40, 0x38, 0x40, 0x3F},  // W
    {0x63, 0x14, 0x08, 0x14, 0x63},  // X
    {0x07, 0x08, 0x70, 0x08, 0x07},  // Y
    {0x61, 0x51, 0x49, 0x45, 0x43},  // Z
    {0x00, 0x36, 0x36, 0x00, 0x00},  // :
    {0x08, 0x08, 0x08, 0x08, 0x08},  // -
    {0x23, 0x13, 0x08, 0x64, 0x62},  // %
    {0x00, 0x07, 0x00, 0x07, 0x00},  // !
};

static int glyph_index(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c == '.') {
    return 10;
  }
  if (c == ' ') {
    return 11;
  }
  if (c >= 'A' && c <= 'Z') {
    return 12 + (c - 'A');
  }
  if (c >= 'a' && c <= 'z') {
    return 12 + (c - 'a');
  }
  if (c == ':') {
    return 38;
  }
  if (c == '-') {
    return 39;
  }
  if (c == '%') {
    return 40;
  }
  if (c == '!') {
    return 41;
  }
  return 11;
}

static void text(int x, int y, const char *s) {
  while (*s) {
    const uint8_t *g = kFont[glyph_index(*s++)];
    for (int col = 0; col < 5; ++col) {
      uint8_t bits = g[col];
      for (int row = 0; row < 7; ++row) {
        if (bits & (1 << row)) {
          px(x + col, y + row, true);
        }
      }
    }
    x += 6;
  }
}

static bool flush() {
  if (!wr_cmd(0x21) || !wr_cmd(0) || !wr_cmd(OLED_W - 1)) {
    return false;
  }
  if (!wr_cmd(0x22) || !wr_cmd(0) || !wr_cmd(7)) {
    return false;
  }
  return wr_data(g_buf, sizeof(g_buf));
}

bool oled_begin() {
  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL, 400000);
  delay(50);
  Wire.beginTransmission(OLED_ADDR);
  if (Wire.endTransmission() != 0) {
    g_ok = false;
    return false;
  }
  const uint8_t init[] = {
      0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00,
      0xA1, 0xC8, 0xDA, 0x12, 0x81, 0x7F, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF,
  };
  for (unsigned i = 0; i < sizeof(init); ++i) {
    if (!wr_cmd(init[i])) {
      g_ok = false;
      return false;
    }
  }
  g_ok = true;
  g_mood = -1;
  memset(g_buf, 0, sizeof(g_buf));
  text(4, 4, "ANUVANI");
  face(0);
  text(64, 28, "LISTEN");
  flush();
  return true;
}

bool oled_present() {
  return g_ok;
}

void oled_show(int mood, float kw_score, float cpu_pct, bool force) {
  if (!g_ok) {
    return;
  }
  const int kw_i = (int)(kw_score * 100.0f + 0.5f);
  // Tenths, so 0.4% idle is not rounded to C 0%.
  const int cpu_i = (int)(cpu_pct * 10.0f + (cpu_pct >= 0 ? 0.5f : -0.5f));
  if (!force && mood == g_mood && kw_i == g_kw_i && cpu_i == g_cpu_i) {
    return;
  }
  g_mood = mood;
  g_kw_i = kw_i;
  g_cpu_i = cpu_i;

  memset(g_buf, 0, sizeof(g_buf));
  text(2, 2, "ANUVANI");
  face(mood);
  if (mood == 2) {
    text(64, 20, "AWAKE!");
  } else {
    text(64, 20, "LISTEN");
  }
  char line[24];
  snprintf(line, sizeof(line), "K %.2f", (double)kw_score);
  text(64, 36, line);
  snprintf(line, sizeof(line), "C %.1f%%", (double)cpu_pct);
  text(64, 50, line);
  flush();
}
