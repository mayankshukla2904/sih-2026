# Research report — on-device custom keyword spotting for SIH 26172

Anuvaani is a complete listen-then-stream pipeline: a custom-word DS-CNN on an ESP32-S3, then raw audio to a remote ASR. This note records the research choices and the honest gap versus the problem statement. It is not a claim that accuracy is solved.

PS evaluation (read first): [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md).

---

## 1. Problem, as written

ISRO PS **SIH26172** asks for:

1. An ultra-lightweight KWS model that runs **locally**.
2. After the word, **stream subsequent audio** to a remote ASR with little overhead.
3. Open-source TinyML only; **custom keyword** (not Hey Google / Alexa).
4. Score **efficiency** (RAM/flash + idle CPU), **accuracy** (high TPR, near-zero false wakes), **latency** (keyword-end → ASR receipt).

Official idle envelope: **< 256 KB RAM** and **< 10% CPU** in continuous listening.

The research question is therefore not “best ASR on a Pi.” It is: **can a microcontroller keep a custom word detector always on inside that envelope, then hand the following speech to ASR without sending audio before the word.**

---

## 2. Related work we actually used

| Line | What we took | What we did not take |
| --- | --- | --- |
| Zhang / ARM *Hello Edge* DS-CNN-S | Depthwise-separable CNN sized for KWS; 49×10-class of features | Their Google Speech Commands trained **global** words as a shipped wake engine |
| Warden *Speech Commands* v0.02 | Keyword / unknown / silence recipe; official testing list | Using `marvin` as a pretrained commercial wake |
| TensorFlow Lite for Microcontrollers | INT8 interpreter on MCU | Cloud STT for wake |
| ESP-NN | S3 SIMD INT8 conv | Floating-point CNN on device |
| Vosk | Offline ASR **after** wake | Vosk (or any ASR) as the wake detector |

We rejected commercial wake SDKs (Porcupine, Sensory, Alexa/Google engines) because R5 forbids them. We rejected “run ASR always and grep the transcript” because then there is no local model to size and no idle-CPU story.

---

## 3. Design decisions

### 3.1 MCU as the scored node

A Raspberry Pi cannot honestly report a sub-256 KB process. The always-on detector moved to **ESP32-S3**. The Pi is the remote ASR + screen. That is the architecture the efficiency clause is written for.

### 3.2 DS-CNN-S, not a transformer

On a 240 MHz MCU with a 64 KB arena, attention-based KWS is the wrong first model. DS-CNN-S is ~24k parameters, folds to **43 KB INT8**, and maps onto seven TFLM ops that ESP-NN accelerates. Depthwise layers use explicit `PAD` + `VALID` so the S3 SIMD path is taken (SAME-padded depthwise was a 36 ms fallback).

### 3.3 Features: 10 MFCC, not raw waveform

Same 30/20 ms, 40-mel, 10-coeff, 70 Hz high-pass recipe in Python and C. A parallel log-Mel / residual experiment (Sanjeet branch) scored at chance on the regenerated val set (`results/branch_compare.json`) and was **not** flashed. Device dtype is INT8; a teammate FP16 graph is host-only.

### 3.4 Full INT8, arena in SRAM

TFLM touches the arena every invoke; PSRAM would erase the kernel win. Arena high-water **57,148 B**. INT8 vs float argmax agreement **0.98** on 200 clips; remaining drift is accepted in exchange for size and speed.

### 3.5 Why idle CPU is 0.4% while still checking for the word

Idle CPU is a **listen-core hop** metric, not “CNN duty all day.” Three facts:

1. **Split cores.** Capture / RMS / ring stay on core 1. MFCC + TFLM stay on core 0. The scored 0.4% cannot include an invoke that never runs on that core.
2. **Energy gate.** RMS is computed every hop. In a quiet room the CNN is requested only after two hops above 0.015 FS. Once the slow floor is above 0.030, an onset or speech-band hop is also required. A quiet room is still *listening*; it is not *classifying*. A tiny always-on keyword net is **not** on the chip (invoke ~36 ms vs a 2 ms 10% budget).
3. **Cheap hop path.** Circular ring (no 31 KB `memmove`) and int64 RMS replaced the 10.6% hop of 19 Sep.

When the gate is open, invoke + MFCC occupy **~66 ms**, about **66%** of a 100 ms slot on the inference core. That is allowed: the PS caps idle listen, not keyword classification. Full write-up: [`cpu-utilisation.md`](cpu-utilisation.md).

The energy gate is **not** a substitute for a calibrated FAR. In a quiet room it only skips hiss. Talk radio still reaches the CNN.

### 3.6 Streaming protocol

KWS1 is length-prefixed PCM. No codec. 500 ms preroll so the command that follows the word is not clipped. Latency the PS wants is still **unmeasured**.

### 3.7 LAN ASR, not cloud

The PS says “cloud ASR.” Demo ASR is **Vosk on LAN** so the booth works without internet. This is a disclosed deviation, not a claim of internet transit.

### 3.8 Thresholds from a firmware-faithful Pareto, not a guess

`tools/tune_threshold.py` slides S3-mic clips the way the firmware does (smooth + debounce + hop windows). On 20 Sep:

- 0.35 / 1 / 1 → TPR 0.94 but **200 FA/h**.
- **0.65 / 3 / 3 → TPR 0.80 at 15 FA/h** (deployed; tuner budget 15/h).
- 0.70 / 5 / 3 → TPR 0.74 at 9 FA/h.

Training now checkpoints the epoch that maximises S3 TPR@0.55 **only while S3-negative FAR ≤ 0.03**, and keeps Google Speech Commands negatives (`drop_gsc=False`) so TPR is not bought by ignoring unknowns. A 10-epoch run on 20 Sep hit that FAR cap (FAR 0.003) but S3 TPR@0.5 only 0.46, so INT8 was **not** replaced.

---

## 4. Results that survive the PS test

**Efficiency — pass**

- Static RAM 187.6 KB < 256 KB.
- Idle CPU 0.4% < 10%.
- INT8 model 42.1 KB.

**Accuracy — not met as written**

- Firmware-faithful S3 sweep at deployed 0.65/3/3: TPR **0.80**, **15 FA/h** (`results/tuning.json`). That is the current on-device operating point.
- Speech Commands val TPR 0.51 @ 0.65 looks weaker because that split is a different microphone.
- Live TPR at 0.84 (retired): **2 / 20 = 0.10**. Live TPR at 0.65: **not re-run**.
- Live FAR at 0.84/7/4 (retired): **168.9 / hour**. Do not cite as current.

The mismatch that remains: **clip-level Speech Commands accuracy did not fully transfer to the S3 microphone in a live room.** Causes we can name without guessing numbers:

- Domain shift (close-mic v0.02 vs INMP441 + gain 12).
- Interferers (phone / laptop playback) still thin in training.
- Raising threshold/debounce to hold FAR also suppresses TPR — hence the 0.80 / 15/h Pareto point rather than 0.94 / 200/h.

**Latency — unscored.**

---

## 5. Open questions (not claimed as done)

1. The 20 Sep FAR-aware fine-tune was refused (TPR@0.5 0.46 < 0.55). Export now requires TPR@0.5 ≥ **0.67** (the on-device INT8). A 21 Sep from-scratch run on 3 m + lookalike + 10 min noise was started; it does not replace the chip unless it beats that gate. `models/marvin.keras` on disk may be a **refused** checkpoint — the flashed graph is `models/marvin.int8.tflite` (20 Sep 22:46 IST).
2. Measure live TPR and live FAR at the **deployed** 0.65 / 3 / 3 knobs, 1 m and 3 m, more than one speaker. Disk now has 40 `mayank3m` and 80 `mayank1m` clips; `tuning.json` is still the older 290-clip sweep.
3. Measure keyword-end → first KWS1 byte with `server/latency_harness.py`.
4. Always-on tiny keyword net: not shipped (idle CPU). Data (3 m, lookalikes, room noise) is the current accuracy lever.

---

## 6. Conclusion

The stack is a legitimate TinyML answer to the **architecture and efficiency** half of PS 26172: local custom KWS, open-source TFLM, idle envelope met *because the listen core never runs the CNN*, audio held on-device until the word. The **accuracy** half is improved on S3 clips (TPR 0.80 at 15 FA/h) but is not “near-zero false activations,” and **latency** is not yet a number. Submission materials that follow this report use those sentences as the ceiling of what may be claimed.
