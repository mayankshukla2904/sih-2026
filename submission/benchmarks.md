# Benchmarks

Replaces the one-page snapshot `DOC-20260919-WA0007.pdf` (captured ~19 Sep 2026). That PDF is a historical artefact. **This file is the current table.** Evaluation against PS 26172 is in [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md); do not read a pass into a fail.

Every current figure has a file in `results/` or `models/` unless marked unmeasured.

---

## 0. Why the PDF is retired

| Cell in the 19 Sep PDF | What it was | Why it is not current |
| --- | --- | --- |
| Idle CPU **10.625%** | Hop loop that `memmove`d 31 KB + `double` RMS | Circular ring + int64 RMS: idle **0.4%** (`results/bench.json`, 20 Sep 12:17 UTC) |
| FAR **0 / hour** | Offline clip sweep on easy negatives | Retired live ambient at 0.84/7/4: **168.864 / hour**. Current knob sweep at 0.65/3/3: **14.939 / hour** (`results/tuning.json`) |
| TPR **10–35%** | Two live sessions mixed | Those sessions were 0.84 and 0.55. Deployed knobs are **0.65 / 3 / 3**. Firmware-faithful TPR **0.7966**. Live TPR at 0.65 **not re-run** |
| No Pi / no keyword-end latency | Incomplete vs PS | Still **unmeasured**; listed as such instead of omitted |
| Speech CPU omitted | PDF showed only idle | Speech-state duty **66.2%** on the inference core (not under the 10% cap) |

---

## 1. Official efficiency (idle listen)

Device: ESP32-S3-DevKitC-1 @ 240 MHz. Quota from sih.gov.in Expected Solution. Firmware flashed 20 Sep 2026 23:45 IST.

| Metric | Quota | 19 Sep PDF | 20 Sep measured | Source |
| --- | --- | --- | --- | --- |
| Firmware static RAM | < 256 KB | — | **187.6 KB** (192,152 B) | PlatformIO size after `pio run -t upload` |
| Idle-listen CPU mean | < 10% | 10.625% | **0.4%** | `results/bench.json` `idle_cpu_pct` |
| Idle-listen CPU max | < 10% | — | **0.4%** | same (`n=1` quiet STATUS sample) |
| INT8 model | scored | — | **42.1 KB** (43,152 B) | `models/marvin.int8.tflite` |
| Firmware flash | app partition 3.34 MB | — | 873.6 KB (894,525 B) | PlatformIO `Flash` line |
| TFLM arena used | inside RAM quota | — | 55.8 / 64 KB | `bench.json` `tflm.arena_used` |

Idle formula: `(hop_capture + ring_push − I2S wait) / 20 ms`. OLED is not in the stamp. CNN is on the other core and only when the energy gate is open. See [`cpu-utilisation.md`](cpu-utilisation.md).

---

## 2. Speech-state compute (not the 10% quota)

`BENCH` on device, 20 reps after warm-up. Period = `INFER_EVERY_HOPS × 20 ms` = 100 ms.

| | min | mean | max |
| --- | --- | --- | --- |
| MFCC | 29.993 ms | **30.137 ms** | 32.231 ms |
| TFLM invoke | 35.995 ms | **36.108 ms** | 36.796 ms |
| MFCC + invoke | — | 66.245 ms | — |
| Duty vs 100 ms slot | — | **66.2%** | — |

Warm-up (first invoke after boot): MFCC 30.172 ms, invoke 36.755 ms.

---

## 3. Accuracy — offline clips

Held-out `speech_commands_testing_list`, speaker-disjoint. Figures below are from the **on-device INT8** checkpoint (20 Sep 22:46 IST). `models/marvin.metrics.json` was later overwritten by a fine-tune that **did not export**; see `results/far_aware_finetune.json`.

| Split | n | Accuracy | TPR @ 0.65 | Clip FAR @ 0.65 |
| --- | --- | --- | --- | --- |
| train | 12,827 | 0.7598 | — | — |
| val | 11,164 | 0.9467 | 0.5077 | 0.0233 |

S3-mic `keyword_real` **at that export**: n=290, mean p(keyword)=0.6542, TPR@0.5=0.6724, argmax recall=0.6828.

On disk 21 Sep, not yet in this table or on the chip: **330** keyword clips (`mayank` 210, `mayank1m` 80, `mayank3m` 40), **189** unknown, **45** lookalike (`martin` / `marvel`), **55** silence, **4** noise wavs (longest ~10 min). A 25-epoch from-scratch train on that set **refused** INT8 export (TPR@0.5=0.455 < 0.67).

INT8 vs float, 200 representative clips (`marvin.int8_parity.json`): argmax agreement **0.98**; mean \|Δ p_keyword\| 0.01578; max 0.24516.

---

## 4. Accuracy — firmware-faithful knob sweep

`results/tuning.json`, 20 Sep 2026 17:42 UTC. 290 keyword clips, 20.08 min S3 negatives, 144 combinations. Replays firmware smoothing + debounce. Tuner FAR budget: **15 / hour**.

| Knobs (thr / window / hits) | TPR | FA / hour | Used on device? |
| --- | --- | --- | --- |
| **0.65 / 3 / 3** | **0.7966** | **14.939** | **Yes** (flashed 20 Sep 23:45 IST) |
| 0.70 / 5 / 3 | 0.7448 | 8.963 | No |
| 0.35 / 1 / 1 (max TPR) | 0.9379 | 200.179 | No — fails the 15/h cap |
| 0.84 / 7 / 4 (retired) | — | — | No — previous live FAR was 168.9/h |

---

## 5. Accuracy — live (not yet at current knobs)

### True positives (`results/tpr.json`)

Keyword spoken by one person. Hit = serial `WAKE` during the trial window. **Both sessions used retired thresholds.**

| Session | Threshold | Distance | Hits | TPR | Median detect latency* |
| --- | --- | --- | --- | --- | --- |
| 2026-08-31 12:03 UTC | **0.84** | 1 m | 2 / 10 | 0.20 | 0.78 s |
| 2026-08-31 12:03 UTC | **0.84** | 3 m | 0 / 10 | 0.00 | — |
| 2026-08-31 12:03 UTC | **0.84** | overall | **2 / 20** | **0.10** | — |
| 2026-08-31 12:19 UTC | 0.55 | 1 m | 4 / 10 | 0.40 | 0.845 s |
| 2026-08-31 12:19 UTC | 0.55 | 3 m | 3 / 10 | 0.30 | 1.404 s |
| 2026-08-31 12:19 UTC | 0.55 | overall | 7 / 20 | 0.35 | — |

\*Time from trial start to `WAKE` line. **Not** keyword-end → ASR.

Live TPR at deployed **0.65 / 3 / 3**: **not measured**.

### False accepts (`results/far.json`)

| | |
| --- | --- |
| Material | Ambient room, no wake word; note “balanced S3 FT; thr=0.84/7/4” |
| Duration | 31.62 min |
| Wakes | 89 |
| **FA / hour** | **168.864** (retired knobs) |

Current FAR evidence is the S3-negative sweep in §4 (**14.939 / hour** at 0.65/3/3). During the old FAR run the energy gate was often open (mean RMS 0.162, 1,658 invokes). Idle-listen CPU from that file (mean 9.507%) is **not** the quiet-listen figure; use `bench.json` 0.4% for idle.

---

## 6. Latency (PS definition)

Keyword **ending** → first audio byte at the ASR server.

| | |
| --- | --- |
| Tool | `python -m server.latency_harness --wav <clip> --trials 12` |
| Result | **Not measured** |

ASR path: LAN Vosk on Raspberry Pi, not a public cloud. Disclose that if a number is later reported.

---

## 7. OLED vs serial CPU (20 Sep firmware)

| State | OLED | Matches |
| --- | --- | --- |
| LISTEN | `LISTEN` + `C 0.4%` | hop-loop CPU, one decimal |
| AWAKE | `AWAKE!` + hop CPU | same hop stamp (inference duty is on the other core) |

No third SPEECH mood.

---

## 8. Score vs PS metrics

| PS metric | Current evidence | Call |
| --- | --- | --- |
| Efficiency (size + idle CPU) | 42.1 KB model, 187.6 KB RAM, 0.4% idle | **Meets official numeric limits** |
| Accuracy | Clip TPR 0.80 @ 15 FA/h (0.65/3/3); live at those knobs unmeasured | **Does not meet** “high TPR / near-zero FA” as written |
| Latency | Unmeasured | **Unscored** |
