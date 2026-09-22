# Anuvaani — SIH 26172

Custom wake word **marvin** on an ESP32-S3. Audio stays on the chip until the word is accepted, then 16 kHz mono PCM streams to a Raspberry Pi running offline Vosk on the LAN. Open-source only (TensorFlow Lite for Microcontrollers, ESP-NN, Vosk). Not a pretrained Hey Google / Alexa model.

## Hardware

| Piece | Role |
| --- | --- |
| ESP32-S3-DevKitC-1, 240 MHz | Always-on listen, INT8 DS-CNN, stream client |
| INMP441 | Mic. 3V3, GND, L/R to GND, SCK 15, WS 16, SD 17 |
| SSD1306 | Optional status. SDA 8, SCL 9 |
| Raspberry Pi 4B | Remote ASR after wake. TCP 8765, 20 ms frames |

Listen core captures a 20 ms hop, checks energy, and keeps a 1 s ring. The CNN runs on the other core, and only when the energy gate opens.

## Numbers

Limits are the official idle-listen caps: under 256 KB RAM and under 10% CPU. They apply to quiet listening, not to speech or streaming.

| | Limit | Measured |
| --- | --- | --- |
| Idle-listen CPU | < 10% | **0.4%** |
| Static RAM | < 256 KB | **218.9 KB** (224,136 B) |
| INT8 model | — | **42.1 KB** |
| Firmware flash | — | 878 KB (898,525 B) |
| Clip TPR at 0.65 / 3 / 3 | high TPR | **0.80** |
| False accepts on those clips | near-zero | **15 / hour** |
| Live, 1 m, 22 Sep | — | **7 / 10** marvin, **6 / 10** other words also woke |
| Keyword-end → ASR | — | not measured |

On-device knobs: score 0.65, smooth 3, debounce 3. The keyword must beat unknown by 0.08. Mic gain is 6×. The 0.80 / 15-per-hour line is the clip sweep on the 20 Sep INT8 model, which is still the one on the chip.

ASR is Vosk on the Pi over LAN, not a public cloud. Battery life and Wi-Fi current were not measured.

## Second stage

A 2.6 KB INT8 net that runs only after the main model has already called a clip marvin. It sees the frame-to-frame change of the MFCC, not the same 49×10 picture as the always-on net. It does not replace `models/marvin.int8.tflite`.

On the holdout (later false-wake clips from the same day), INT8 keyword hit rate stayed at **0.253** against **0.26** for the main model alone, and **90%** of those confuser candidates were rejected. Margin **−0.05**. File: [`second-stage/marvin.verify.tflite`](second-stage/marvin.verify.tflite). Full sweep: [`second-stage/verify_stage.json`](second-stage/verify_stage.json).

That net is **not** in the firmware on the board. Live, it also dropped real “marvin”, so `kws_verify_model.h` was removed. With the header absent, every main-model candidate passes. Boot log: `verify: off (no second-stage model)`.
