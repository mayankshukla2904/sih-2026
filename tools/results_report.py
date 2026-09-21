"""Render docs/results.md from whatever has actually been measured.

Every number in the deck should be traceable to a file on disk. Each measurement
tool writes results/<name>.json; this collects them, adds model and firmware
sizes, and writes the report. Measurements that have not been run are listed as
missing with the command that produces them, so nothing gets quietly invented.

  python -m tools.results_report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import (  # noqa: E402
    DEBOUNCE_HITS,
    KEYWORD,
    MODELS_DIR,
    SMOOTH_WINDOW,
    WAKE_SCORE_THRESHOLD,
)
from shared.feature_spec import CLIP_SAMPLES, FEATURE_SHAPE, SPEC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
ELF = ROOT / "firmware" / ".pio" / "build" / "s3node" / "firmware.elf"

# quota / target, from the problem statement
QUOTA_RAM_KB = 256
QUOTA_IDLE_CPU = 10.0

MISSING = {
    "far": ("false accepts per hour", "python -m tools.far_test --minutes 60"),
    "tpr": ("true positive rate", "python -m tools.tpr_test --speaker <name> --per-distance 10"),
    "latency": (
        "keyword-end to ASR first byte",
        f"python -m server.latency_harness --wav data/{KEYWORD}/keyword/*.wav --trials 12",
    ),
    "tuning": ("threshold/debounce sweep", "python -m tools.tune_threshold"),
    "bench": ("on-device MFCC/invoke + idle CPU", "python -m tools.bench"),
}


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def elf_sizes() -> dict | None:
    """RAM and flash from the built firmware, via the Xtensa size tool."""
    if not ELF.exists():
        return None
    candidates = list((Path.home() / ".platformio" / "packages").glob("toolchain-*/bin/*-size"))
    size_bin = str(candidates[0]) if candidates else "size"
    try:
        proc = subprocess.run([size_bin, "-A", str(ELF)], capture_output=True, text=True, check=True)
    except Exception:
        return None
    sections: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            try:
                sections[parts[0]] = int(parts[1])
            except ValueError:
                continue
    ram = sum(v for k, v in sections.items() if k in (".dram0.data", ".dram0.bss", ".noinit"))
    flash = sum(v for k, v in sections.items() if k in (".flash.text", ".flash.rodata", ".iram0.text"))
    return {"static_ram_bytes": ram, "flash_bytes": flash, "sections": sections}


def model_sizes() -> dict:
    out = {}
    for suffix in ("int8.tflite", "float.tflite", "keras"):
        path = MODELS_DIR / f"{KEYWORD}.{suffix}"
        if path.exists():
            out[suffix] = path.stat().st_size
    return out


def fmt_kb(n: int | None) -> str:
    return f"{n / 1024:.1f} KB" if n else "—"


def section_efficiency(fw: dict | None, models: dict, far: dict | None, bench: dict | None) -> list[str]:
    lines = ["## Efficiency", ""]
    rows = [
        ("INT8 model (flash)", fmt_kb(models.get("int8.tflite")), "—"),
        ("Float32 model, for comparison", fmt_kb(models.get("float.tflite")), "—"),
    ]
    if fw:
        rows.append(("Firmware static RAM", fmt_kb(fw["static_ram_bytes"]), f"< {QUOTA_RAM_KB} KB"))
        rows.append(("Firmware flash", fmt_kb(fw["flash_bytes"]), "16 MB available"))
    idle = None
    if bench and bench.get("idle_cpu_pct"):
        idle = bench["idle_cpu_pct"]
    elif far and far.get("idle_cpu_pct"):
        idle = far["idle_cpu_pct"]
    if idle:
        rows.append(("Idle listening CPU (mean)", f"{idle['mean']} %", f"< {QUOTA_IDLE_CPU} %"))
        rows.append(("Idle listening CPU (max)", f"{idle['max']} %", f"< {QUOTA_IDLE_CPU} %"))
    if bench and bench.get("bench"):
        b = bench["bench"]
        rows.append(("MFCC (steady-state mean)", f"{b['mfcc_avg'] / 1000:.1f} ms", "—"))
        rows.append(("TFLM invoke (steady-state mean)", f"{b['inv_avg'] / 1000:.1f} ms", "—"))
        rows.append(("Speech-state CPU (one core)", f"{b['speech_cpu']} %", "—"))
        rows.append(
            ("TFLM arena", f"{b['arena_used'] / 1024:.1f} / {b['arena_bytes'] / 1024:.0f} KB", f"< {QUOTA_RAM_KB} KB"),
        )
    elif far and far.get("invoke_us") and far["invoke_us"]["n"]:
        i = far["invoke_us"]
        rows.append(("TFLM invoke (median)", f"{i['p50'] / 1000:.1f} ms", "—"))

    lines.append("| Metric | Measured | Quota |")
    lines.append("| --- | --- | --- |")
    for name, value, quota in rows:
        lines.append(f"| {name} | {value} | {quota} |")
    lines.append("")
    if not fw:
        lines.append("_Firmware sizes unavailable: build with `pio run` in `firmware/` first._")
        lines.append("")
    return lines


def section_accuracy(metrics: dict | None, tpr: dict | None, far: dict | None) -> list[str]:
    lines = ["## Accuracy", ""]

    if metrics:
        n_spk = metrics.get("n_real_speakers")
        speakers = metrics.get("real_speakers") or []
        if n_spk is None:
            n_spk = len(speakers)
        hold = metrics.get("holdout_speaker") or "none — synthetic only"
        lines.append(f"Held-out split: `{hold}`  ")
        if n_spk and n_spk > 12:
            lines.append(f"Real speakers in dataset: {n_spk}")
        else:
            lines.append(f"Real speakers in dataset: {', '.join(speakers) or 'none'}")
        lines.append("")
        lines.append("| Split | n | Accuracy | TPR @ thr | FAR @ thr |")
        lines.append("| --- | --- | --- | --- | --- |")
        for split in metrics.get("splits", []):
            d = split.get("at_default_threshold", {})
            lines.append(
                f"| {split['split']} | {split['n']} | {split['accuracy']} | "
                f"{d.get('tpr')} | {d.get('far')} |"
            )
        lines.append("")
        src = metrics.get("source")
        note = f"Threshold used above: {WAKE_SCORE_THRESHOLD}."
        if src:
            note += f" Source `{src}`."
        if metrics.get("val_is_speaker_disjoint"):
            note += " Validation is speaker-disjoint."
        lines.append(note)
        lines.append("")
    else:
        lines.append("_No offline metrics: run `python -m training.train`._")
        lines.append("")

    if tpr:
        lines.append("### Live trial (true positive rate)")
        lines.append("")
        lines.append("| Speaker | Distance | Hits / n | TPR | Median wake latency |")
        lines.append("| --- | --- | --- | --- | --- |")
        for run in tpr.get("runs", []):
            for dist, d in run.get("by_distance", {}).items():
                lat = f"{d['median_wake_latency_s']} s" if d.get("median_wake_latency_s") else "—"
                lines.append(
                    f"| {run['speaker']} | {dist} | {d['hits']} / {d['n']} | "
                    f"{d['tpr']} | {lat} |"
                )
        lines.append("")

    if far:
        rate = far.get("false_accepts_per_hour")
        lines.append("### False accepts")
        lines.append("")
        lines.append(
            f"{far['false_accepts']} wake(s) in {far['duration_min']} min of material the node must "
            f"ignore = **{rate} per hour**."
        )
        if far.get("note"):
            lines.append(f"  Material: {far['note']}")
        lines.append("")

    return lines


def section_latency(lat: dict | None) -> list[str]:
    lines = ["## Latency", ""]
    if not lat or not lat.get("median_ms"):
        lines.append("_Not measured: run `python -m server.latency_harness --wav <clip> --trials 12`._")
        lines.append("")
        return lines
    lines.append("Keyword end to the first audio byte at the ASR, timed on one clock.")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("| --- | --- |")
    for key, label in (
        ("median_ms", "Median"),
        ("p95_ms", "p95"),
        ("min_ms", "Min"),
        ("max_ms", "Max"),
    ):
        lines.append(f"| {label} | {lat[key]} ms |")
    lines.append(f"| Trials woken | {lat['woke']} / {lat['trials']} |")
    lines.append("")
    return lines


def section_tuning(tuning: dict | None) -> list[str]:
    lines = ["## Threshold and debounce", ""]
    if not tuning:
        lines.append(
            f"Currently deployed: threshold {WAKE_SCORE_THRESHOLD}, smoothing window "
            f"{SMOOTH_WINDOW}, debounce {DEBOUNCE_HITS} — defaults, not yet tuned. "
            "Run `python -m tools.tune_threshold`."
        )
        lines.append("")
        return lines
    best = tuning.get("recommended")
    lines.append(
        f"Swept {len(tuning['sweep'])} combinations over {tuning['n_positive_clips']} keyword clips "
        f"and {tuning['negative_minutes']} min of negatives, replaying the firmware's own smoothing "
        "and debounce logic."
    )
    if not best:
        lowest = tuning.get("lowest_far") or {}
        budget = tuning.get("far_budget_per_hour", 1.0)
        lines.append("")
        lines.append(
            f"No setting met FA/hr ≤ {budget}. Lowest FAR in the sweep: "
            f"{lowest.get('far_per_hour')} /hr at threshold {lowest.get('threshold')} "
            f"(TPR {lowest.get('tpr')}). Firmware keeps the deployed defaults until more "
            "`noise_real` / `lookalike_real` is recorded through the S3 mic."
        )
        lines.append("")
        return lines
    lines.append("")
    lines.append("| Knob | Deployed | Recommended |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| WAKE_SCORE_THRESHOLD | {WAKE_SCORE_THRESHOLD} | {best['threshold']} |")
    lines.append(f"| SMOOTH_WINDOW | {SMOOTH_WINDOW} | {best['smooth_window']} |")
    lines.append(f"| DEBOUNCE_HITS | {DEBOUNCE_HITS} | {best['debounce_hits']} |")
    lines.append("")
    lines.append(
        f"At the recommended setting: TPR {best['tpr']}, {best['far_per_hour']} false accepts/hour."
    )
    tied = tuning.get("n_tied_with_recommended")
    swept = tuning.get("n_settings_swept")
    if tied and swept and tied > swept // 4:
        lines.append("")
        lines.append(
            f"_{tied} of {swept} settings tie on TPR and false accepts, so the sweep cannot "
            "discriminate. Firmware keeps the deployed defaults until real recorded negatives "
            "exist (`python -m tools.record_s3 --label noise --continuous 600`)._"
        )
    lines.append("")
    return lines


def section_pipeline() -> list[str]:
    return [
        "## What runs where",
        "",
        "| Stage | Device | Detail |",
        "| --- | --- | --- |",
        "| Capture | ESP32-S3 | INMP441 I2S, 16 kHz mono, 20 ms hops |",
        "| Energy gate | ESP32-S3 core 1 | RMS threshold; no neural net during silence |",
        f"| Features | ESP32-S3 core 0 | {SPEC.n_mels}-band log-mel to {SPEC.n_mfcc} MFCC, "
        f"{SPEC.n_frames} frames, {SPEC.n_fft}-point radix-2 FFT, sparse mel filterbank |",
        "| Keyword spotting | ESP32-S3 core 0 | DS-CNN-S, INT8, TFLM + ESP-NN |",
        "| Speech recognition | Raspberry Pi | Vosk, offline, only after a wake |",
        "",
        f"Clip length {CLIP_SAMPLES} samples ({CLIP_SAMPLES / SPEC.sample_rate:.2f} s), "
        f"model input {FEATURE_SHAPE}.",
        "",
        "Wake-word audio never leaves the ESP32. The KWS1 stream to the Pi opens only "
        "after a detection, and the Pi runs Vosk locally, so no audio reaches a network "
        "service at any point.",
        "",
    ]


def section_missing(present: dict) -> list[str]:
    absent = [(k, v) for k, v in MISSING.items() if not present.get(k)]
    if not absent:
        return []
    lines = ["## Not yet measured", ""]
    for key, (what, cmd) in absent:
        lines.append(f"- **{what}** — `{cmd}`")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DOCS / "results.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    far = load("far")
    tpr = load("tpr")
    lat = load("latency")
    tuning = load("tuning")
    bench = load("bench")
    metrics_path = MODELS_DIR / f"{KEYWORD}.metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    parity_path = MODELS_DIR / f"{KEYWORD}.int8_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else None

    fw = elf_sizes()
    models = model_sizes()

    extra: list[str] = []
    if parity:
        extra = [
            "## INT8 vs float",
            "",
            f"Calibrated on `{Path(parity['calibrated_on']).name}`. "
            f"Argmax agreement {parity['argmax_agreement'] * 100:.1f}% over {parity['n']} clips; "
            f"p_keyword drift mean {parity['p_keyword_mean_abs_drift']}, "
            f"max {parity['p_keyword_max_abs_drift']}.",
            "",
        ]

    lines = [
        f"# Measured results — keyword `{KEYWORD}`",
        "",
        f"Generated by `python -m tools.results_report` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "Every figure below comes from a file in `results/` or `models/`; nothing is estimated.",
        "",
        *section_pipeline(),
        *section_efficiency(fw, models, far, bench),
        *extra,
        *section_accuracy(metrics, tpr, far),
        *section_latency(lat),
        *section_tuning(tuning),
        *section_missing({"far": far, "tpr": tpr, "latency": lat, "tuning": tuning, "bench": bench}),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    for name, data in (("far", far), ("tpr", tpr), ("latency", lat), ("tuning", tuning), ("bench", bench)):
        print(f"  {name}: {'present' if data else 'MISSING'}")
    print(f"  offline metrics: {'present' if metrics else 'MISSING'}")
    print(f"  int8 parity: {'present' if parity else 'MISSING'}")
    print(f"  firmware sizes: {'present' if fw else 'MISSING (build the firmware)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
