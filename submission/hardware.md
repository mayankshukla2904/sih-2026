# Hardware

Listen node that the problem statement scores: **ESP32-S3**. The Raspberry Pi is the remote ASR host and demo screen, not the idle-listen device.

Evaluated against PS 26172 in [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md). Efficiency limits (< 256 KB RAM, < 10% CPU) apply to the S3 while it is idling in continuous listening mode.

---

## 1. Roles

| Piece | Role | PS metric |
| --- | --- | --- |
| ESP32-S3-DevKitC-1 | Always-on KWS, energy gate, MFCC, INT8 CNN, KWS1 client, OLED | RAM, idle CPU, model size |
| INMP441 | 16 kHz I2S MEMS microphone | Capture path |
| SSD1306 128×64 | Status (LISTEN / AWAKE, score, hop CPU). Optional; firmware runs without it | Not scored |
| Raspberry Pi 4B | Offline Vosk ASR, 3.5″ UI | Remote ASR after wake |
| HW-104 / speaker | Pi-side audio out for TTS replies | Not scored |

Wake-word audio never leaves the S3. The Pi socket opens only after a local detection.

---

## 2. ESP32-S3 listen node

**Board:** ESP32-S3-DevKitC-1, 16 MB flash, PSRAM enabled in the build but **TFLM arena is internal SRAM** (PSRAM would slow every invoke). Clock logged at boot: **240 MHz**. PlatformIO env `s3node`.

### 2.1 INMP441 (3.3 V only)

Moved off GPIO 4/5/6 after those pins stayed silent. Header numbers from `firmware/include/pins.h`:

| INMP441 | ESP32-S3 |
| --- | --- |
| VDD | 3V3 |
| GND | GND |
| L/R | GND (left channel) |
| SCK / BCLK | GPIO **15** |
| WS / LRCK | GPIO **16** |
| SD / DOUT | GPIO **17** |

Dummy I2S DOUT on GPIO **11** (driver is RX+TX). Capture: 32-bit slots, 24-bit MSB-aligned samples, left channel, DC block, `MIC_GAIN 12`.

### 2.2 SSD1306 OLED

| OLED | ESP32-S3 |
| --- | --- |
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO **8** |
| SCL | GPIO **9** |

I2C address `0x3C`. If the panel is missing, `oled_begin()` fails closed and KWS continues.

### 2.3 Measured footprint (20 Sep 2026 23:45 IST flash)

| Item | Value |
| --- | --- |
| INT8 model in flash | 43,152 B (42.1 KB) |
| Firmware image | 894,525 B (873.6 KB) of 3.34 MB app partition |
| Firmware static RAM | **187.6 KB** (192,152 B of 320 KB) |
| TFLM arena | 57,148 / 65,536 B used |
| Free heap at STATUS | 108,996 B |

Static RAM is the PlatformIO `.dram0` total after link. Under the 256 KB official cap.

---

## 3. Raspberry Pi ASR host

Not the idle-listen device. Do not report whole-system Linux RSS against the 256 KB quota.

| Item | Detail |
| --- | --- |
| Board | Raspberry Pi 4B |
| ASR | Vosk, offline, after KWS1 connect |
| Transport | TCP port **8765**, magic `KWS1`, 16 kHz int16, 20 ms frames, 500 ms preroll |
| Display | 3.5″ SPI panel for live transcript (HDMI is a valid fallback) |
| Speaker | HW-104 on the Pi jack |

Pi GPIO notes for the **legacy Pi-as-node** wiring (INMP441 on BCM 18/19/20) live in `docs/pinout.md`. That path is a reference implementation. The scored node is the S3.

---

## 4. Power and always-on behaviour

- S3 stays in continuous listen. Wi-Fi STA with modem sleep (`setSleep(true)`). KWS1 connects only on wake.
- CNN is off while the energy gate is closed. Quiet room (floor ≤ 0.030): two hops at RMS > 0.015 FS, same as the 0.4% bench. Noisy room: those two hops **and** an onset or 300 Hz speech-band hop. The listen core still captures every 20 ms. A tiny always-on keyword net is not on the chip.
- OLED redraw is 1 Hz except on a forced AWAKE frame. I2C is not on the 20 ms hop path.

---

## 5. What is not hardware-claimed

- Battery-life hours: not measured.
- Peak current during Wi-Fi TX: not measured.
- “The Pi uses < 256 KB”: false. Linux cannot honestly claim that.
