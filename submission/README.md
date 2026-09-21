# Anuvaani — SIH 26172 submission pack

**Read first:** [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md).

That evaluation treats [sih.gov.in/sih2026PS](https://sih.gov.in/sih2026PS) (PS **SIH26172**, ISRO) and the verbatim copy in `docs/problem-statement.md` as the source of truth. Every other file in this folder is written **after** that verdict. Nothing here upgrades a fail into a pass.

Product name: **Anuvaani**. Spoken keyword: **marvin**. Listen node: **ESP32-S3**. Remote ASR: Raspberry Pi 4B running offline Vosk over LAN.

| File | What it is |
| --- | --- |
| [`00-evaluation-vs-ps.md`](00-evaluation-vs-ps.md) | Requirement-by-requirement score against the official PS |
| [`hardware.md`](hardware.md) | Bill of materials, pinout, what runs where |
| [`hardware-architecture.md`](hardware-architecture.md) | Dual-core firmware, energy gate, MFCC, TFLM, KWS1 stream |
| [`benchmarks.md`](benchmarks.md) | Replaces the 19 Sep 2026 snapshot PDF with current measured numbers |
| [`cpu-utilisation.md`](cpu-utilisation.md) | How idle CPU is 0.4% while the node still checks for `marvin` |
| [`research.md`](research.md) | Why DS-CNN-S / INT8 / energy gating, and what is still open |

Figures come from `results/bench.json`, `results/far.json`, `results/tpr.json`, `results/tuning.json`, `models/marvin.int8_parity.json`, and the 20 Sep 23:45 IST PlatformIO size dump. Do **not** quote `models/marvin.metrics.json` as the chip — that file is a refused 20 Sep 23:41 fine-tune. Estimates are labelled as estimates.

**Headline numbers (chip still 20 Sep 2026 22:46 IST INT8; pack dated 21 Sep):** idle listen **0.4% CPU**, RAM **187.6 KB**, INT8 **42.1 KB**. Deployed knobs **0.65 / 3 / 3**. Firmware-faithful TPR **0.80** at **15 FA/h** on **290** S3-mic keyword clips. Disk now has **330** keyword clips (210 close + 80 at 1 m + 40 at 3 m), **189** unknown, **45** lookalike, **55** silence, **4** noise (including a 10 min room take). Those extra clips are **not** in `tuning.json`. A 21 Sep from-scratch train on them **refused** export (TPR@0.5=0.455 < 0.67). Live TPR/FAR at 0.65 not re-measured. Keyword-end → ASR latency unmeasured. Tiny always-on keyword net **not** shipped.

**Do not quote the 19 Sep PDF as current.** It recorded idle listen at 10.625% CPU and a clip-sweep FAR of 0. Those numbers are superseded.
