"""Pick WAKE_SCORE_THRESHOLD, SMOOTH_WINDOW and DEBOUNCE_HITS from data.

The default 0.72 / 5 / 3 was a guess. This runs the deployed INT8 model over
clips the way the firmware actually does it — sliding the clip window forward one
hop at a time, averaging the last SMOOTH_WINDOW scores, and requiring
DEBOUNCE_HITS consecutive crossings — then sweeps the three knobs together.

Two kinds of material:
  * short keyword clips  -> a hit means the word was detected (TPR)
  * long negative audio  -> each detection is a false accept, reported per hour

  python -m tools.tune_threshold
  python -m tools.tune_threshold --negative-dir data/sahayak/noise_real

It prints the pins.h block to paste, so the firmware defaults come from a
measurement rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import load_wav_mono_16k  # noqa: E402
from node.features import features_from_clip  # noqa: E402
from shared.config import DATA_DIR, KEYWORD, MODELS_DIR  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, HOP_SAMPLES, SPEC  # noqa: E402
from training.augment import match_train_level  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# The firmware only infers every INFER_EVERY_HOPS hops; mirror that or the sweep
# would report a detector that runs 5x more often than the real one.
INFER_EVERY_HOPS = 5


class Int8Model:
    """The exact .tflite that runs on the S3, in the desktop interpreter."""

    def __init__(self, path: Path) -> None:
        import tensorflow as tf

        self.interp = tf.lite.Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def p_keyword(self, feat: np.ndarray) -> float:
        q = np.round(feat[None, ...] / self.in_scale + self.in_zp).clip(-128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
        return float((raw[0] - self.out_zp) * self.out_scale)


    def p_all(self, feat: np.ndarray) -> tuple[float, float, float]:
        q = np.round(feat[None, ...] / self.in_scale + self.in_zp).clip(-128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
        p = (raw - self.out_zp) * self.out_scale
        return float(p[0]), float(p[1]), float(p[2])


def noise_floor(audio: np.ndarray) -> float:
    """RMS of the quietest 100 ms, i.e. the room the clip was recorded in."""
    win = SPEC.sample_rate // 10
    if len(audio) < win * 2:
        return 1e-4
    frames = audio[: len(audio) // win * win].reshape(-1, win)
    return float(np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).min() + 1e-9)


def pad_with_room(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surround a one-shot clip with its own noise floor.

    The firmware never sees an isolated clip: it holds a rolling CLIP_SAMPLES
    window and infers every INFER_EVERY_HOPS hops, so a real utterance passes
    through many window offsets and produces a run of consecutive scores. Scoring
    a bare clip yields exactly one position, which makes any DEBOUNCE_HITS above 1
    look like it never fires. Padding restores the run.
    """
    pad_n = CLIP_SAMPLES
    floor = noise_floor(audio)
    lead = rng.normal(0.0, floor, pad_n).astype(np.float32)
    tail = rng.normal(0.0, floor, pad_n).astype(np.float32)
    return np.concatenate([lead, audio.astype(np.float32), tail])


def score_track(model: Int8Model, audio: np.ndarray) -> np.ndarray:
    """p_keyword for every position the firmware would evaluate."""
    if len(audio) < CLIP_SAMPLES:
        pad = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        pad[: len(audio)] = audio
        audio = pad
    step = HOP_SAMPLES * INFER_EVERY_HOPS
    starts = range(0, len(audio) - CLIP_SAMPLES + 1, step)
    out = []
    for s in starts:
        clip = match_train_level(audio[s : s + CLIP_SAMPLES].astype(np.float32))
        out.append(model.p_keyword(features_from_clip(clip)))
    return np.array(out)


def count_fires(track: np.ndarray, threshold: float, window: int, hits_needed: int) -> int:
    """Replays firmware/src/kws.cpp smoothing + debounce over a score track.

    Counts distinct wake events: after firing, the run must drop below threshold
    before it can fire again, which is what the refractory period enforces.
    """
    if len(track) == 0:
        return 0
    fires = 0
    hits = 0
    armed = True
    buf: list[float] = []
    for s in track:
        buf.append(float(s))
        if len(buf) > window:
            buf.pop(0)
        avg = sum(buf) / len(buf)
        if avg >= threshold:
            hits += 1
            if hits >= hits_needed and armed:
                fires += 1
                armed = False
        else:
            hits = 0
            armed = True
    return fires


def load_positive_tracks(model: Int8Model, folders: list[Path], cap: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    tracks = []
    for folder in folders:
        if not folder.exists():
            continue
        for wav in sorted(folder.glob("*.wav"))[:cap]:
            tracks.append(score_track(model, pad_with_room(load_wav_mono_16k(wav), rng)))
    return tracks


def load_negative_tracks(model: Int8Model, folders: list[Path], cap: int) -> tuple[list[np.ndarray], float]:
    rng = np.random.default_rng(1)
    tracks = []
    total_s = 0.0
    for folder in folders:
        if not folder.exists():
            continue
        for wav in sorted(folder.glob("*.wav"))[:cap]:
            audio = load_wav_mono_16k(wav)
            # Short negatives get the same padding as positives so both sides are
            # judged over comparable numbers of window positions; long recordings
            # already slide on their own.
            if len(audio) < 3 * CLIP_SAMPLES:
                audio = pad_with_room(audio, rng)
            total_s += len(audio) / SPEC.sample_rate
            tracks.append(score_track(model, audio))
    return tracks, total_s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--positive-cap", type=int, default=120, help="keyword clips to score")
    p.add_argument("--negative-cap", type=int, default=200, help="negative clips per folder")
    p.add_argument("--negative-dir", type=Path, nargs="*", default=None)
    p.add_argument(
        "--max-far-per-hour",
        type=float,
        default=1.0,
        help="never recommend a setting above this false-accept rate (problem statement: near-zero)",
    )
    p.add_argument(
        "--s3-only",
        action="store_true",
        help="score only INMP441 keyword_real vs unknown_real/silence_real/noise_real",
    )
    p.add_argument("--out", type=Path, default=RESULTS / "tuning.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model or (MODELS_DIR / f"{args.keyword}.int8.tflite")
    if not model_path.exists():
        raise SystemExit(f"no {model_path}; run python -m training.train first")

    root = DATA_DIR / args.keyword
    model = Int8Model(model_path)
    print(f"model {model_path.name}  input scale={model.in_scale:.5f} zp={model.in_zp}")

    pos_dirs = [root / "keyword_real", root / "keyword"]
    neg_dirs = args.negative_dir or [
        root / "noise_real",
        root / "unknown_real",
        root / "lookalike_real",
        root / "unknown",
        root / "lookalike",
        root / "silence",
    ]
    if args.s3_only:
        pos_dirs = [root / "keyword_real"]
        neg_dirs = [
            root / "noise_real",
            root / "unknown_real",
            root / "silence_real",
        ]

    print("scoring positives...")
    pos = load_positive_tracks(model, pos_dirs, args.positive_cap)
    print("scoring negatives...")
    neg, neg_seconds = load_negative_tracks(model, neg_dirs, args.negative_cap)
    if not pos or not neg:
        raise SystemExit("need both keyword clips and negative clips to tune")
    neg_hours = neg_seconds / 3600.0
    print(f"{len(pos)} positive clips, {len(neg)} negative clips ({neg_seconds / 60:.1f} min)\n")

    rows = []
    for window in (1, 2, 3, 5):
        for hits_needed in (1, 2, 3):
            for threshold in np.arange(0.30, 0.86, 0.05):
                thr = float(round(threshold, 2))
                tp = sum(1 for t in pos if count_fires(t, thr, window, hits_needed) > 0)
                fa = sum(count_fires(t, thr, window, hits_needed) for t in neg)
                rows.append(
                    {
                        "threshold": thr,
                        "smooth_window": window,
                        "debounce_hits": hits_needed,
                        "tpr": round(tp / len(pos), 4),
                        "false_accepts": fa,
                        "far_per_hour": round(fa / neg_hours, 3) if neg_hours > 0 else None,
                    }
                )

    # Highest TPR among settings that stay under the FAR budget. If nothing
    # meets the budget, do not silently recommend a 100+/hr operating point —
    # report the lowest-FAR setting and refuse to treat it as deployable.
    def rank_tpr(r: dict) -> tuple:
        return (
            r["tpr"],
            -(r["far_per_hour"] or 0.0),
            r["threshold"],
            r["debounce_hits"],
            r["smooth_window"],
        )

    def rank_far(r: dict) -> tuple:
        return (
            -(r["far_per_hour"] or 0.0),
            r["tpr"],
            r["threshold"],
            r["debounce_hits"],
            r["smooth_window"],
        )

    budget = args.max_far_per_hour
    clean = [r for r in rows if (r["far_per_hour"] or 0) <= budget]
    budget_met = bool(clean)
    pool = clean if budget_met else rows
    best = max(pool, key=rank_tpr if budget_met else rank_far)
    n_tied = sum(
        1 for r in pool if r["tpr"] == best["tpr"] and (r["far_per_hour"] or 0) == (best["far_per_hour"] or 0)
    )

    print(f"{'thr':>5} {'win':>4} {'hits':>5} {'TPR':>7} {'FA':>5} {'FA/hr':>8}")
    shown = sorted(pool, key=rank_tpr if budget_met else rank_far, reverse=True)[:12]
    for r in shown:
        print(
            f"{r['threshold']:>5.2f} {r['smooth_window']:>4} {r['debounce_hits']:>5} "
            f"{r['tpr']:>7.3f} {r['false_accepts']:>5} {r['far_per_hour']:>8.2f}"
        )

    if budget_met:
        print("\nrecommended — paste into firmware/include/pins.h:")
        print(f"  #define WAKE_SCORE_THRESHOLD {best['threshold']:.2f}f")
        print(f"  #define SMOOTH_WINDOW {best['smooth_window']}")
        print(f"  #define DEBOUNCE_HITS {best['debounce_hits']}")
        print("and mirror them in shared/config.py.")
        print(f"  at that setting: TPR={best['tpr']:.3f}  false accepts={best['far_per_hour']}/hr")
    else:
        print(
            f"\nNO setting meets FA/hr <= {budget:g}. Lowest FAR in the sweep is "
            f"{best['far_per_hour']}/hr at thr={best['threshold']} win={best['smooth_window']} "
            f"hits={best['debounce_hits']} (TPR={best['tpr']:.3f})."
        )
        print(
            "Do not paste that into pins.h. Record more noise_real / lookalike_real "
            "through the S3 mic (tools.record_s3 --label noise --continuous 600) and retrain."
        )
    if n_tied > len(pool) // 4:
        print(
            f"\n  WARNING: {n_tied}/{len(pool)} settings tie on TPR and false accepts, so this\n"
            "  sweep cannot actually discriminate. That happens when the negatives are too\n"
            "  easy — re-run against real recorded negatives (tools.record_s3 --label noise)\n"
            "  before trusting the recommendation."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model_path.name,
                "n_positive_clips": len(pos),
                "n_negative_clips": len(neg),
                "negative_minutes": round(neg_seconds / 60, 2),
                "infer_every_hops": INFER_EVERY_HOPS,
                "n_tied_with_recommended": n_tied,
                "n_settings_swept": len(rows),
                "far_budget_per_hour": budget,
                "budget_met": budget_met,
                "recommended": best if budget_met else None,
                "lowest_far": best if not budget_met else None,
                "sweep": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
