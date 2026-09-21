# SIH Idea PPT Brief — PS 26172

Fill this into the **official SIH idea template**. Do not invent extra slides.

**Deck type:** official idea submission only (6 slides, PDF).  
**Not this file:** finale 10-slide pitch, custom templates, walls of text.

---

## Hard rules (from SIH 2025/2026 idea format)

- Maximum **6 slides including the title slide**.
- Use the **ministry/SIH template**. Do not rename the official headings.
- Export **PDF** for the portal. PPT/Word is typically rejected.
- No paragraphs. Points, diagrams, infographics, photos only.
- Max ~6 bullets per slide. Body text ≥ 14 pt.
- Evaluators spend ~2–3 minutes. Put PS 26172 numbers on judged slides.
- Idea must be novel. Do not copy a previous SIH voice-assistant deck.

**Source of truth:** the verbatim PS in [GUIDE.md](../GUIDE.md) and [problem-statement.md](problem-statement.md). Official Expected Solution (SIH 2026 portal): edge app **&lt; 256 KB RAM** and **&lt; 10% CPU while idling in continuous listening mode**. After the keyword, stream CPU is not scored.

- Efficiency: model RAM/Flash footprint + idle-listening CPU (we target &lt; 256 KB subsystem, &lt; 10% idle)
- High true-positive rate, **near-zero false activations**
- Latency judged as: keyword ends → remote ASR receives the stream
- Open-source TinyML only (TFLite Micro / similar)
- **No proprietary voice SDKs**
- **No models pretrained on Alexa / Hey Google / similar global keywords**
- Custom keyword must be trained

Never claim a 25.6 KB RAM cap. That was a misreading.

---

## Where the numbers come from

Do not type a metric into a slide by hand. [results.md](results.md) is generated
from the measurement files in `results/` by `python -m tools.results_report`, and
it says explicitly which metrics have not been measured yet. Copy from there.

| Judged metric | Produced by |
|---|---|
| Efficiency — INT8 model size, static RAM, TFLM arena | `python -m tools.results_report` (reads the built ELF) |
| Efficiency — idle listening CPU | `python -m tools.bench` (TEL1 samples; `far_test` also records this) |
| Accuracy — true positive rate | `python -m tools.tpr_test --speaker <name> --per-distance 10` |
| Accuracy — false activations per hour | `python -m tools.far_test --minutes 60 --note "<material>"` |
| Latency — keyword end → ASR first byte | `python -m server.latency_harness --wav <clip> --trials 12` |
| Threshold / debounce choice | `python -m tools.tune_threshold` |

Two supporting checks worth a sentence on the technical slide, because they are
what makes the numbers trustworthy: `python -m tools.mfcc_parity` proves the C
feature frontend matches the Python one the model was trained on, and the INT8
export prints its own agreement with the float model on real MFCCs.

---

## Placeholders to fill before export

| Placeholder | Fill with |
|---|---|
| `[TEAM]` | Registered team name |
| `[TEAM ID]` | Portal team ID |
| `[INSTITUTE]` | College / university |
| `[THEME]` | Official theme as listed on the portal |
| `[WAKE WORD]` | Custom keyword / short phrase (not Alexa, not Hey Google) |
| `[MENTOR]` | Faculty mentor if required on the title slide |

Suggested idea title (edit if needed):

**Edge Voice Node — Ultra-Low-Resource On-Device Wake Word**

---

## Slide 1 — Title

**Official heading:** title page (template fields)

| Field | Value |
|---|---|
| Problem Statement ID | **26172** |
| Problem Statement Title | Ultra-lightweight on-device keyword spotting; stream audio to remote ASR only after wake |
| Theme | `[THEME]` |
| PS Category | **Hardware** |
| Team ID | `[TEAM ID]` |
| Team Name | `[TEAM]` |
| Institute | `[INSTITUTE]` |
| Idea title | Edge Voice Node — Ultra-Low-Resource On-Device Wake Word |

Keep this slide sparse. No architecture dump here.

**Speaker notes (~20 s):** We are a hardware team on PS 26172. Always-on custom wake word on a microcontroller. Cloud ASR starts only after the keyword, so idle RAM and CPU stay inside the problem limits.

---

## Slide 2 — Proposed Solution

**Official heading:** Proposed Solution (Describe your Idea / Solution / Prototype)

**One-line idea:** Always-on custom wake-word detection runs locally on an ESP32-S3; only after wake does the node stream later audio to a remote ASR server.

**Bullets (max 6):**

- Local TinyML keyword spotting — **no continuous cloud audio**
- Custom keyword `[WAKE WORD]` — **not** Alexa / Hey Google pretrained models
- Two stages: cheap energy gate → INT8 DS-CNN only on speech candidates
- After wake: OLED `AWAKE` + start post-wake audio stream (ASR later)
- Open-source stack only: Arduino / TFLM + ESP-NN
- Measured idle: **187.6 KB** RAM, **0.4%** CPU; clip TPR **0.80** at **15 FA/h** (not near-zero)

**Diagram to draw (left-to-right, two boxes):**

```text
ALWAYS-ON (local, cheap)
  INMP441 → I2S DMA → energy gate → MFCC → INT8 DS-CNN → debounce
                                              │
                                         wake? ── no → keep listening
                                              │
                                             yes
                                              ▼
POST-WAKE (only then)
  OLED AWAKE → stream subsequent audio → remote ASR
```

**Uniqueness line:** Not a generic voice assistant. It is an **embedded TinyML optimization** problem with **measured** RAM, CPU, FAR/FRR, and wake-to-stream latency.

**Speaker notes (~30 s):** Assistants that stream all day fail privacy, power, and this problem statement. We detect `[WAKE WORD]` on the chip. Silence skips the CNN, which is how idle CPU stays under 10%. The radio and ASR path stay off until wake.

---

## Slide 3 — Technical Approach

**Official heading:** Technical Approach

**Hardware (why each part exists):**

| Part | Why |
|---|---|
| ESP32-S3-DevKitC (scored device) | Dual core at 240 MHz with SIMD that ESP-NN uses for INT8 conv; 512 KB SRAM; Wi-Fi idles until wake |
| INMP441 I2S MEMS mic | Digital 16 kHz audio, no analog front-end, India-available |
| Raspberry Pi + 480×320 display | Runs Vosk offline ASR after a wake, and shows the node's live stats to judges |

**Software:** Arduino/ESP-IDF (C/C++), I2S DMA capture, float32 MFCC with a
512-point radix-2 FFT from generated tables, INT8 DS-CNN-S (Hello Edge / MLPerf
Tiny family) on TensorFlow Lite Micro with Espressif's ESP-NN kernels, static
tensor arena, no heap in the hot path, no proprietary SDK. Vosk on the Pi. The
model is trained by us on Speech Commands v2 *marvin* (INT8 DS-CNN-S) — nothing
pretrained on a global wake word is used anywhere. Product name is Anuvaani.

**Methodology flowchart:**

```text
INMP441
   → I2S DMA, 20 ms hops                          core 1 (Arduino loop)
   → energy gate (no neural net during silence)   core 1
   → 70 Hz highpass + MFCC, 512-pt radix-2 FFT    core 0 (kws task)
   → INT8 DS-CNN-S on TFLM + ESP-NN               core 0
   → score smoothing + debounce                   core 0
   → KWS1 stream opens only now  → Pi → Vosk      core 1
```

Splitting the work across cores is what keeps the numbers honest: inference takes
tens of milliseconds, so pinning it to the other core means the 20 ms audio hop
is never delayed and no audio is dropped while thinking.

**Implementation phases (tiny timeline, not a 36-hour dump):** mic bring-up →
features with a C-vs-Python parity test → custom dataset recorded through the
same mic → train with SpecAugment and realistic-SNR noise → INT8 quantize
calibrated on real MFCCs → on-device KWS → measure RAM/CPU/FAR/TPR/latency.

**Speaker notes (~30 s):** TFLM with ESP-NN because it is open and gives us
Espressif's SIMD INT8 kernels for free. The energy gate means the network never
runs in a quiet room, which is where the idle CPU budget is won. The model lives
in flash and activations sit in a small static arena. Wi-Fi carries nothing until
the word is detected, so wake-word audio physically cannot leave the device.

---

## Slide 4 — Feasibility and Viability

**Official heading:** Feasibility and Viability

**Feasibility**

- BOM: ESP32-S3-DevKitC-1 + INMP441 + SSD1306 + Raspberry Pi 4B (ASR host)
- Dual-core S3: listen hop on one core, INT8 CNN on the other
- Dataset: `marvin` clips on **this** INMP441, Speech Commands as unknown/silence

**Risks**

- Idle CPU if the CNN ran every 20 ms hop (~36 ms invoke)
- False accepts from TV / similar words (`martin`, `marvel`)
- Far-field TPR at 3 m (live at 0.65 not re-measured)

**Mitigations**

- Energy gate (quiet: two hops RMS > 0.015; noisy: plus onset / speech-band)
- Knobs 0.65 / 3 / 3 from firmware-faithful sweep (TPR 0.80 @ 15 FA/h)
- Extra 1 m / 3 m / lookalike / 10 min noise clips on disk; 21 Sep train refused export (TPR@0.5 0.455 < 0.67)

**Speaker notes (~30 s):** This is buildable with parts already in Indian maker shops. The real risk is idle CPU, not RAM — a full KWS stack is tens of KB against a 256 KB cap. We do not run the CNN on silence, and we pick the M33 board so the 10% idle number is honest.

---

## Slide 5 — Impact and Benefits

**Official heading:** Impact and Benefits

**Who benefits**

- Privacy-sensitive kiosks, field devices, and civic hardware that cannot stream room audio 24/7
- Indic / institutional custom keywords that commercial assistant models do not cover
- Students and labs that need a measurable open TinyML reference, not a closed SDK

**Benefits**

- **Privacy:** wake-word stays on-device; cloud sees audio only after an explicit keyword
- **Energy / cost:** MCU-class node vs always-on phone/cloud pipeline
- **Accuracy under the rules:** custom word + same-mic data → lower false accepts than a mismatched pretrained model
- **Scale:** same pipeline retargets to a new keyword without a vendor license

**Optional SDG tags (only if the template wants them):** SDG 9 (industry, innovation, infrastructure). Do not overclaim social impact.

**Speaker notes (~30 s):** The impact is not “another chatbot.” It is a node that can sit in a room, stay under 256 KB and 10% idle CPU, and only then open a network path. That is the difference between a demo assistant and a deployable edge device.

---

## Slide 6 — Research and References

**Official heading:** Research and References

Use links / citations, not a bibliography paragraph.

- Zhang, Y. et al. **Hello Edge: Keyword Spotting on Microcontrollers** (2017) — DS-CNN architecture  
- Banbury, C. et al. **MLPerf Tiny** — DS-CNN KWS reference (~38.6K params, INT8)  
- **TensorFlow Lite for Microcontrollers** + **ESP-NN** (S3 SIMD INT8 kernels)
- Warden, P. **Speech Commands** dataset — unknown / silence / negatives only
- Espressif ESP32-S3 + I2S INMP441 (no closed voice SDK)

**Do not list:** Picovoice Porcupine, Alexa Voice Service, Google Assistant SDK, Hey Google / Alexa checkpoints.

**Speaker notes (~20 s):** We reuse an open architecture and open negative speech data. The keyword class is trained by us. That satisfies the “no pretrained global keywords” rule without inventing a network from scratch.

---

## Visual / layout checklist

- One accent color + dark/light from the official template. Do not restyle the chrome.
- Architecture on slides 2 and 3 as **boxes and arrows**, not sentences.
- Slide 4: three columns — Feasible / Risk / Mitigation.
- Slide 5: measured RAM **187.6 KB**, idle CPU **0.4%**, clip TPR **0.80** @ **15 FA/h**. Label live TPR/FAR and latency as unmeasured.
- OLED mock: `LISTEN` + hop CPU, `AWAKE!` on a hit.
- File size &lt; 10 MB.

---

## After the draft

Copy bullets from this file into the **official SIH template**. Numbers must match [`submission/00-evaluation-vs-ps.md`](../submission/00-evaluation-vs-ps.md). Do not invent TPR, FAR, idle CPU, or latency.

1. Fill `[TEAM]`, `[TEAM ID]`, `[INSTITUTE]`, `[THEME]`. Spoken keyword is `marvin`.
2. Confirm the wake word is not a global assistant phrase.
3. Export PDF and check it is exactly 6 slides, file < 10 MB.
