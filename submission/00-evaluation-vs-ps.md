# Evaluation against SIH Problem Statement 26172

**Source of truth:** official text of PS **SIH26172** (ISRO) on [sih.gov.in/sih2026PS](https://sih.gov.in/sih2026PS), copied verbatim in `docs/problem-statement.md`. Team-chosen targets that are **not** in that text are labelled as such.

**Snapshot date:** 21 September 2026. Keyword on device: `marvin`. Firmware knobs: `WAKE_SCORE_THRESHOLD 0.65`, `SMOOTH_WINDOW 3`, `DEBOUNCE_HITS 3`, `ENERGY_RMS_THRESHOLD 0.015`, `ENERGY_CONSECUTIVE 2`, `KWS_MARGIN 0.02`. Adaptive gate (noisy rooms only): `ENERGY_NOISY_FLOOR 0.030`, onset 0.006 / 0.25× floor, speech-band 300 Hz. INT8 on the chip is still the **20 Sep 22:46 IST** export. A 21 Sep from-scratch train on new 3 m / lookalike / noise clips has **not** replaced it.

**Overall:** the pipeline matches the binding software and architecture requirements. Official idle-efficiency limits are met on the ESP32-S3. Firmware-faithful clip TPR at the deployed knobs is **0.80** with **15 false accepts / hour** on S3-mic negatives. Live room TPR/FAR at **0.65 / 3 / 3** has **not** been re-measured. Keyword-end → ASR latency has **not** been measured. ASR is LAN-hosted Vosk, not cloud — disclosed below.

---

## 1. Verbatim problem (what we are scored on)

> Build an ultra-lightweight, highly accurate keyword spotting (KWS) model that runs locally on a low-power device. Upon detecting the keyword, the system must instantly and efficiently stream the subsequent audio to a remote Automated Speech Recognition (ASR) server with minimal data overhead and latency.

**Key metrics**

- Efficiency: Model size (RAM/Flash footprint) and CPU usage during idle listening.
- Accuracy: High true-positive rate for the keyword with near-zero false activations.
- Latency: The time delta between the keyword ending and the cloud ASR receiving the audio stream.

**Restrictions**

- Open-source only. No proprietary / commercial voice-activation SDK.
- TinyML frameworks (TFLM, PyTorch Mobile, or similar).
- No pre-trained global keywords (`Hey Google`, `Alexa`). Train a custom keyword.

**Expected Solution (sih.gov.in)**

> The edge software application must run smoothly within an environment restricted to less than 256KB of RAM and consume under 10% CPU utilization while idling in continuous listening mode.

Those two numbers apply **only to idle listen**. After a wake, the node is allowed to use more CPU while streaming.

---

## 2. Binding requirements (R1–R7)

| # | Requirement | Verdict | Evidence |
| --- | --- | --- | --- |
| R1 | KWS runs **locally** on a low-power device | **Pass** | INT8 DS-CNN-S on ESP32-S3. No audio leaves the chip until a detection. |
| R2 | Model is **ultra-lightweight** | **Pass** | INT8 graph **43,152 B** (`models/marvin.int8.tflite`). “Highly accurate” is scored separately. |
| R3 | On detection, stream **subsequent** audio to a **remote** ASR | **Pass, with disclosed deviation** | KWS1 TCP to Raspberry Pi. Pi is remote-over-network, not internet. See §5. |
| R4 | Streaming is instant, minimal data overhead | **Pass (design)** | 16 kHz mono int16, 20 ms frames, no re-encode. Connection opens on wake. **Latency number not measured.** |
| R5 | Open-source only; no commercial wake SDK | **Pass** | TFLM (Apache-2.0), ESP-NN, Vosk (Apache-2.0). No Porcupine / Snowboy / Alexa / Google wake. |
| R6 | Built on open-source TinyML | **Pass** | TensorFlow Lite for Microcontrollers + ESP-NN S3 kernels. |
| R7 | **Custom keyword**, not a pretrained global one | **Pass** | `marvin`, trained by us (Speech Commands v2 + S3-mic recordings). Product name Anuvaani is not the wake word. |

Prohibited anti-pattern (detecting the word by string-matching an ASR transcript) is not used.

---

## 3. Official efficiency limits

| Limit | Official | Measured | Verdict |
| --- | --- | --- | --- |
| RAM while listening | **< 256 KB** | Firmware static RAM **187.6 KB** (192,152 B of 327,680 B SRAM, PlatformIO 20 Sep 23:45 IST). TFLM arena high-water **55.8 / 64 KB**. | **Pass** |
| Idle-listen CPU | **< 10%** | Mean **0.4%**, max **0.4%** (`results/bench.json`, 20 Sep 2026 12:17 UTC). | **Pass** |
| Model flash | scored, no numeric cap | INT8 **42.1 KB**; firmware image **873.6 KB** of the 3.34 MB app partition. | **Pass** |

Idle CPU is hop capture + circular ring push, **minus I2S wait**, divided by 20 ms. That is the listen core. MFCC and TFLM invoke run on the **other** core, and only when the energy gate is open. They are **not** part of the idle number.

How idle stays at 0.4% **while still checking for `marvin`:** the listen core never runs the CNN. Every 20 ms it captures a hop, computes integer RMS, and writes 320 samples into a circular ring. In a quiet room (noise floor ≤ 0.030) the gate is still **RMS > 0.015 for two hops** — same as the 20 Sep bench. If the floor has risen (fan / TV), the CNN is requested only on an onset or speech-band hop, so steady loud noise does not keep invoking. When speech is present, the other core spends ~66 ms on MFCC + INT8 invoke; that duty is **not** held to 10%. Full accounting is in [`cpu-utilisation.md`](cpu-utilisation.md).

Do not cite the older FAR-run idle of 9.5% (`results/far.json`): that log mixed hop-loop samples with a period when the energy gate was open.

---

## 4. Accuracy

PS wording: “High true-positive rate for the keyword with near-zero false activations.” No official numeric TPR/FAR. The team operationalized a working FAR cap as **≤ 15 false accepts per hour** of non-keyword audio for knob selection (not “near-zero,” and not a PS number). A stricter **< 1 / hour** budget is still unmet.

### 4.1 Offline (held-out Speech Commands testing list)

Source: on-device INT8 checkpoint (exported 20 Sep 22:46 IST). Speaker-disjoint val, 2,238 real speakers in the corpus. Deployed firmware: **0.65**. `models/marvin.metrics.json` on disk currently describes a **later fine-tune that was not exported**; do not quote that file as the chip. Sidecar: `models/marvin.metrics.rejected.json`, `results/far_aware_finetune.json`.

| Split | n | Accuracy | TPR @ 0.65 | Clip FAR @ 0.65 |
| --- | --- | --- | --- | --- |
| train | 12,827 | 0.7598 | (see file) | (see file) |
| val | 11,164 | 0.9467 | 0.5077 | 0.0233 |

S3-mic `keyword_real` disk scores on that INT8: n=**290** at export (mean p(keyword)=**0.654**, TPR@0.5=**0.672**). Clip FAR on Speech Commands is **not** false wakes per hour of continuous audio.

S3-mic files on disk (21 Sep), **not** used to pick 0.65 / 3 / 3:

| Folder | n | Notes |
| --- | --- | --- |
| `keyword_real` | **330** | 210 close (`mayank`) + 80 at 1 m + 40 at 3 m |
| `unknown_real` | **189** | non-keyword speech through the same mic |
| `lookalike_real` | **45** | `martin` / `marvel` |
| `silence_real` | **55** | room hush |
| `noise_real` | **4** | includes a ~10 min room take |

### 4.2 Firmware-faithful knob sweep (`results/tuning.json`)

INT8 model, S3-mic clips, firmware smoothing + debounce + sliding windows. **290** `keyword_real` (the set at export), 207 S3 negatives (**20.08 min**), 144 combinations. FAR budget used by the tuner: **15 / hour**. The 21 Sep 3 m / lookalike recordings are not in this sweep.

| Setting | TPR | FA / hour | On device? |
| --- | --- | --- | --- |
| **0.65 / 3 / 3** (recommended, flashed) | **0.7966** | **14.939** | **Yes** |
| 0.70 / 5 / 3 | 0.7448 | 8.963 | No |
| 0.35 / 1 / 1 (max TPR on this sweep) | 0.9379 | 200.179 | No — blows the FAR cap |

This is the current accuracy evidence at the knobs that actually run. It is **not** a live-room measurement.

### 4.3 Live true-positive rate (`results/tpr.json`)

One speaker (mayank). Two sessions. Both used **0.84**, not the current 0.65.

| When | Threshold | 1 m | 3 m | Overall |
| --- | --- | --- | --- | --- |
| 2026-08-31 12:03 UTC | **0.84** (retired) | 2 / 10 = 0.20 | 0 / 10 = 0.00 | **2 / 20 = 0.10** |
| 2026-08-31 12:19 UTC | 0.55 (never deployed) | 4 / 10 = 0.40 | 3 / 10 = 0.30 | 7 / 20 = 0.35 |

Live TPR at **0.65 / 3 / 3** has **not** been re-run. Do not quote 10% or 35% as the current deployed figure.

### 4.4 Live false accepts (`results/far.json`)

89 wakes in **31.62 min** of ambient room audio with no wake word, at **0.84 / 7 / 4** (retired knobs):

**168.864 false accepts per hour.**

That log does **not** apply to 0.65 / 3 / 3. The current FAR evidence is the S3-negative sweep in §4.2 (**15 / hour**). A new live FAR run has not been taken.

### 4.5 Verdict on accuracy

**Partial.** Firmware-faithful TPR at the deployed knobs is 0.80 with FAR held to the 15/h cap. That is not yet “high TPR with near-zero false activations,” and live-room confirmation is missing. Do not say FAR is near zero.

FAR-aware S3 fine-tune (20 Sep 23:11–23:41 IST) did **not** export. Peak S3 TPR@0.55 was 0.43 with FAR 0.003; disk TPR@0.5 fell to 0.46. A 21 Sep from-scratch train on the larger S3 set is **running** and has **not** replaced the chip; export still requires TPR@0.5 ≥ **0.67**. The chip still runs the 20 Sep 22:46 INT8. Log of the refused fine-tune: `results/far_aware_finetune.json`.

Always-on second neural net: **not implemented**. DS-CNN invoke is ~36 ms; a 20 ms hop only allows ~2 ms under a 10% idle cap. First stage on device is the energy gate, not a tiny keyword CNN.

---

## 5. Latency

PS: “time delta between the **keyword ending** and the **cloud ASR receiving** the audio stream.”

| Item | Status |
| --- | --- |
| Harness | `server/latency_harness.py` exists |
| Measured keyword-end → first ASR byte | **Not run** |
| Detection-to-serial (when a hit occurs) | 0.66–0.85 s median at 1 m in the retired 0.84 TPR session — this is **not** the PS latency |

**Unscored.** Do not invent a millisecond figure.

Disclosed deviation: the PS says “cloud ASR.” Our ASR is **Vosk on a Raspberry Pi on the same LAN**. The hop that would be measured is node → LAN server, not internet transit. State this in Q&A; do not imply a public cloud.

---

## 6. INT8 vs float (supporting, not a PS metric)

`models/marvin.int8_parity.json`, 200 clips from `marvin.rep.npy`:

| | |
| --- | --- |
| Argmax agreement | 0.98 |
| Mean \|Δ p_keyword\| | 0.01578 |
| Max \|Δ p_keyword\| | 0.24516 |

---

## 7. What we will and will not say to judges

**Say**

- Local custom KWS on ESP32-S3; nothing is sent until `marvin`.
- Idle listen **0.4% CPU**, static RAM **187.6 KB**, INT8 model **42.1 KB**.
- The listen core only captures + RMS + ring; the CNN runs on the other core and only when the energy gate opens (quiet room: two hops above 0.015; noisy room: plus onset or speech-band).
- Open-source TFLM + ESP-NN; keyword trained by us.
- After wake we stream raw 16 kHz PCM to a remote ASR (Pi / Vosk, LAN).
- Firmware-faithful TPR **0.80** at **15 FA/h** on S3-mic clips at 0.65 / 3 / 3.

**Do not say**

- “Near-zero false wakes” or “FAR 0.”
- Live TPR of 10% or 35% as if it were the current 0.65 setting.
- Idle 10.625% (old PDF) or “we always stay under 10% including speech.”
- A keyword-end → ASR latency number.
- “Cloud ASR” without the LAN caveat.
- “The Raspberry Pi uses 256 KB RAM.” The 256 KB / 10% limits are the **ESP32-S3 listen node**.
- “The CNN runs continuously” or that a tiny always-on keyword net is on the chip. It is not.

---

## 8. Scorecard

| PS item | Result |
| --- | --- |
| R1 local KWS | Pass |
| R2 ultra-lightweight model | Pass (size) |
| R3 remote stream after wake | Pass (LAN ASR disclosed) |
| R4 minimal-overhead stream | Pass (design); latency unmeasured |
| R5–R6 open-source TinyML | Pass |
| R7 custom keyword | Pass |
| Efficiency: RAM < 256 KB idle | **Pass (187.6 KB)** |
| Efficiency: CPU < 10% idle | **Pass (0.4%)** |
| Accuracy: high TPR, near-zero FA | **Not met as written** (clip TPR 0.80 @ 15 FA/h; live at 0.65 unmeasured) |
| Latency: keyword-end → ASR | **Not measured** |
