# Hardware architecture

How the listen node is split across chips, cores, and buses. Companion to [`hardware.md`](hardware.md). PS evaluation: [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md). Why idle CPU can be 0.4% while still scoring `marvin`: [`cpu-utilisation.md`](cpu-utilisation.md).

---

## 1. System

```
INMP441  --I2S-->  ESP32-S3 core 1 (capture, energy, ring, KWS1, OLED)
                      |
                      | snapshot 0.99 s clip, only if RMS gate open
                      v
                   ESP32-S3 core 0 (MFCC + INT8 DS-CNN-S, TFLM + ESP-NN)
                      |
                      | on wake only
                      v
                   Wi-Fi STA  --KWS1 TCP-->  Raspberry Pi 4B (Vosk + UI)
```

Nothing leaves the S3 until the on-device model accepts `marvin`.

---

## 2. Dual-core split

Arduino `loopTask` is pinned to **ARDUINO_RUNNING_CORE** (core 1 by default). Inference is a FreeRTOS task on the **other** core (`KWS_TASK_CORE = 1 - ARDUINO_RUNNING_CORE`).

| Task | Core | Stack | Work |
| --- | --- | --- | --- |
| `loop()` | 1 | 16,384 B | I2S hop, energy gate, circular ring, stream, serial, OLED @ 1 Hz |
| `kws` | 0 | 8,192 B | `mfcc_from_clip` + TFLM `Invoke` |

Signalling: binary semaphore. Core 1 copies the ring into `g_infer_clip[15840]`, gives the semaphore. Core 0 writes `g_infer_out` and sets `g_infer_fresh`. One in-flight clip; no queue. If core 0 is busy, that hop skips a new request.

This is why idle-listen CPU on the listen core can be **0.4%** while a speech-state invoke on the inference core is **~66%** of a 100 ms slot: they are different cores and different PS clauses. Checking for the wake word on a quiet hop is RMS + a counter, not a neural-net forward pass.

---

## 3. Capture and ring

| Buffer | Size | Notes |
| --- | --- | --- |
| I2S DMA | 4 × 256 samples | Level-1 interrupt; dummy TX auto-clear |
| `g_i2s_rx` | 640 × int32 | One hop of stereo 32-bit |
| Hop PCM | 320 × int16 | 20 ms @ 16 kHz |
| Circular ring | 15,840 × int16 = 31,680 B | 0.99 s; wrap index, no `memmove` |
| `g_infer_clip` | 31,680 B | Linear snapshot for the inference core |

Idle listen used to `memmove` 31 KB every hop (~10.6% CPU). The ring is now circular; copy-to-linear happens only when inference is requested.

Feature window: 30 ms Hamming, 20 ms hop, 49 frames, 10 MFCC, 40 mel bands, 512-point radix-2 FFT, 70 Hz high-pass, pre-emphasis 0.97. Same recipe in Python (`shared/feature_spec.py`) and C (`firmware/src/mfcc.cpp`). Before MFCC, speech clips are gain-matched toward ~0.10 FS so 1 m speech looks like the close-mic training set; true silence is left alone.

---

## 4. Energy gate

Quiet room after `MIC_GAIN 12` sits near 0.01–0.02 FS. Implementation: `firmware/src/energy_gate.cpp`.

Quiet rooms (slow noise floor ≤ **0.030**): same as 20 Sep — RMS > **0.015** for **2** hops opens `maybe_word`. That is the path `results/bench.json` measured at **0.4%** idle.

Noisy rooms (floor above 0.030): still requires those two loud hops, **and** an onset (RMS ≥ floor + max(0.006, 0.25×floor)) or speech-band energy (300 Hz high-pass ratio ≥ 0.55). Steady fan/TV after the floor has adapted does not keep the CNN on. The gate is always a **subset** of the old two-hop rule, so it cannot add false wakes.

CNN stays off while the gate is closed. Do not raise `ENERGY_RMS_THRESHOLD` to force 0.4% in a loud room — that misses `marvin`.

A tiny always-on keyword net was considered and **not** shipped: DS-CNN invoke is ~36 ms; 10% of a 20 ms hop is 2 ms.

---

## 5. Detector

Architecture: **DS-CNN-S** (Hello Edge / ARM TinyML). Three classes: `keyword`, `unknown`, `silence`. ~23.7k Keras parameters; BN folded at export.

On device:

- Full INT8 TFLite, TFLM interpreter, **7** ops: `CONV_2D`, `DEPTHWISE_CONV_2D`, `PAD`, `RELU`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`.
- Arena **64 KB**, 16-byte aligned, internal SRAM. High-water **57,148 B**.
- ESP-NN `CONFIG_NN_OPTIMIZED` + `ARCH_ESP32_S3` for SIMD INT8 conv (~36 ms invoke).
- Smoothing window **3**, debounce **3**, threshold **0.65**, keyword-vs-unknown margin **0.02**.
- Refractory **1,200 ms** after hang-up.

Wake state machine:

```
listen  --energy 2 hops-->  score clips  --3 consecutive KWS hits-->  wake (KWS1 open)
   ^                                                                   |
   +----- 3.5 s quiet or 10 s timeout ---------------------------------+
```

OLED shows only **LISTEN** or **AWAKE**. Scoring still happens in LISTEN; there is no third SPEECH mood.

---

## 6. KWS1 stream (after wake only)

TCP, little-endian, no timestamps, no sequence numbers.

| Field | Layout |
| --- | --- |
| Header | `KWS1` + version 1 + preroll_ms + sample_rate |
| Preroll | 500 ms (8,000 samples) from the ring, including audio *before* the trigger |
| Frames | uint16 length + PCM; 20 ms int16 mono |
| End | length 0 |

TEL1 is a separate 1 Hz telemetry line (state, RMS, idle CPU, scores). It is not the ASR path.

Wi-Fi: STA, credentials in `firmware/include/wifi_secrets.h` (not in this pack). `.local` mDNS fallback for the ASR host.

---

## 7. OLED (status only)

128×64 SSD1306. Face + product name + mood + `K` score + hop CPU:

| Mood | Line | Meaning |
| --- | --- | --- |
| LISTEN | `C 0.4%` | Hop-loop duty, one decimal (the scored idle metric when the gate is closed) |
| AWAKE | `AWAKE!` | Same hop stamp; inference duty is on the other core |

Redraw cached on mood / score / 0.1% CPU; 1 Hz from `loop()`, plus a forced frame on wake.

---

## 8. Why this split satisfies the efficiency clause

The official text scores **idle listening**. Architecture choices that serve that clause:

1. Microcontroller, not a Linux SBC, as the always-on node.
2. CNN off the listen core; listen core waits on I2S for most of the 20 ms.
3. Energy gate so the inference core is asleep on hiss — and, once the room is actually loud, on steady non-speech. The wake word is still *watched for* via RMS every hop.
4. Circular ring instead of a 31 KB shift every hop.
5. Integer RMS (no double `sqrt` on the hop).
6. Model in flash; arena in SRAM under 64 KB; total static RAM 187.6 KB.

Accuracy is a separate clause. Firmware-faithful TPR at 0.65/3/3 is 0.80 at 15 FA/h; live-room TPR at those knobs is unmeasured. Architecture does not hide that.
