# CPU utilisation report

Idle-listen CPU is one of the two official efficiency numbers on PS 26172 (the other is RAM). This report defines the metric, shows the measured 0.4%, and answers the question: **how can the node sit at a fraction of a percent while still checking for the wake word?**

PS evaluation: [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md). Raw JSON: `results/bench.json`. Architecture: [`hardware-architecture.md`](hardware-architecture.md).

---

## 1. What the problem statement scores

From sih.gov.in Expected Solution:

> consume under 10% CPU utilization **while idling in continuous listening mode**.

| State | Under the 10% cap? | What is running |
| --- | --- | --- |
| LISTEN, energy gate closed | **Yes** | Listen core: capture hop, integer RMS, circular ring push |
| LISTEN, energy gate open (`marvin` being scored) | No (other core) | Inference core: MFCC + TFLM INT8 invoke |
| AWAKE, KWS1 streaming | No | Listen core: capture + TCP send |

Speech-state **66.2%** on the inference core is expected and allowed. The 10% clause is idle listen, not “always, including while the word is being classified.”

---

## 2. Why idle CPU is 0.4% even though we listen for `marvin`

The detector is always **armed**. It is not always **running the CNN**. Four mechanisms stack:

### 2.1 Two cores, two jobs

Arduino `loopTask` is on **core 1**. `kws_task` is pinned to the other core (`KWS_TASK_CORE = 1 - ARDUINO_RUNNING_CORE`, typically **core 0**).

Every 20 ms the listen core:

1. Reads 320 samples from I2S (most of the 20 ms is **blocked waiting** on the codec).
2. DC-blocks, converts, integer RMS.
3. Pushes the hop into a circular ring (wrap index, no 31 KB `memmove`).
4. Looks at RMS. That is the whole wake-word “check” on this core.

The 0.4% figure is **that** work, minus I2S wait, over 20 ms. The CNN is physically unable to inflate this number because it is not scheduled here.

### 2.2 Energy gate: CNN off in a quiet room

In a quiet room (noise floor ≤ 0.030) the decision is still:

```
if RMS > 0.015: hits++
else:           hits--
maybe_word = (hits >= 2)
```

If the floor has risen, the same two-hop loud rule must pass **and** the hop must be an onset or speech-like. Steady loud noise does not keep requesting the CNN.

If the room is hiss, `maybe_word` stays false. `infer_request()` is never called. Core 0 sleeps on a semaphore. The node is still listening: the next loud hop can open the gate in 40 ms. It is **not** invoking DS-CNN-S fifty times a second on silence.

The scored 0.4% in `results/bench.json` is capture + RMS + ring (the stamp is *before* `energy_gate_update`). The extra 300 Hz high-pass is a few hundred MACs on the hop and was not re-BENCH’d; do not invent a new idle number.

This is why a quiet bench log sits at 0.4% and a noisy FAR log (mean RMS 0.162, gate open 1,658 times) reported ~9.5% hop CPU plus a busy inference core. Different rooms, different cores.

### 2.3 Inference is gated a second time by “busy”

When the gate is open, `infer_request()` copies the 0.99 s ring and gives the semaphore **only if** the other core is free. MFCC is ~30 ms and INT8 invoke is ~36 ms (ESP-NN SIMD), so a new clip is accepted at most about every 66 ms. The listen core’s copy is ~31 KB once per invoke, still inside one hop. Skipped hops do not stall audio.

### 2.4 Cheap listen-path math

| Change vs the 19 Sep PDF | Effect on the 20 ms hop |
| --- | --- |
| Circular ring instead of `memmove` of 31,680 B | Dominant. The old shift was most of the 10.6%. |
| int64 sum-of-squares + `sqrtf` instead of `double` RMS | Removes soft-float from the hop. |
| OLED at 1 Hz, after the CPU stamp | I2C is not on the 20 ms path. |
| Wi-Fi modem sleep; KWS1 closed until wake | No TCP while LISTEN. |

The 10% quota is for **idle**. 0.4% vs 10% is about **25×** headroom. That budget belongs to accuracy work, not to further idle micro-optimisations.

---

## 3. Definition used in firmware

In `firmware/src/main.cpp` `loop()`:

```
t0 = micros()
rms = hop_capture(hop)     // includes blocking i2s_read / dummy i2s_write
ring_push(hop)
listen_us = micros() - t0
work_us   = listen_us - g_i2s_wait_us     // waiting on the codec is not CPU work
cpu       = 100 * work_us / 20_000        // 20 ms hop
```

If the node is not streaming, that `cpu` is copied to `g_idle_cpu`.

**In the metric:** I2S sample conversion, DC block, int64 RMS + `sqrtf`, circular ring write.

**Out of the metric:** time blocked in I2S, MFCC, TFLM, Wi-Fi, OLED I2C (OLED runs after the stamp, at 1 Hz).

This matches “CPU usage during idle listening,” not “fraction of wall time the hop occupies.”

---

## 4. Measured values (20 Sep 2026)

Quiet listen, `results/bench.json`:

| | |
| --- | --- |
| Mean idle CPU | **0.4%** |
| Max idle CPU | **0.4%** |
| n | 1 STATUS sample after BENCH |
| Clock | 240 MHz |

Speech-state (same file, 20 BENCH reps). Period used for duty = `INFER_EVERY_HOPS × 20 ms` = 100 ms:

| | mean | max |
| --- | --- | --- |
| MFCC | 30.1 ms | 32.2 ms |
| Invoke | 36.1 ms | 36.8 ms |
| Duty (sum / 100 ms) | **66.2%** | — |

Live FAR log (`results/far.json`, retired 0.84 knobs) reported idle **9.5%** because the room was not idle: mean RMS 0.162 opened the energy gate 1,658 times. Do not mix that with the 10% quota.

---

## 5. OLED display

Two moods only: **LISTEN** and **AWAKE**. The CPU line is hop-loop `cpu` printed to **one decimal** (`C 0.4%`), so 0.4% is not rounded to `C 0%`.

Serial `TEL1` still carries hop-loop `cpu` plus `g_idle_cpu`.

Redraw at 1 Hz when mood, keyword score, or CPU tenths change. Forced redraw on wake.

---

## 6. Claims allowed in Q&A

- “Idle listening on the S3 is **0.4%** of a 20 ms hop, under the official 10%.”
- “We still check for `marvin`: RMS every hop, CNN on the other core only when the room is loud enough (and, if the room is already loud, only on an onset or speech-band hop).”
- “When we actually run the network, that core is about **66%** busy; the PS does not cap that.”
- “The OLED LISTEN line is the same hop CPU, to one decimal.”

Not allowed:

- “We use 10.625% idle” (old PDF).
- “CPU is always under 10%, including speech.”
- Subtracting OLED time to make idle look smaller (OLED is already outside the stamp).
- “The CNN runs continuously at 0.4%.” It does not. 0.4% is the listen hop.
