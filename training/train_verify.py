"""Stage 4.2. A tiny second classifier that runs only on first-stage candidates.

The always-on model stays the chip DS-CNN and still sees the 49x10 MFCC.
This net sees the frame-to-frame change of that MFCC (48x10), so a completed
word that moves and then leaves is not the same picture as a partial "mar-"
that sits still. It is exported to its own tflite and header. It never goes
through export_all, so a failed run cannot replace the INT8 on the ESP.

A miss by the first model is still a miss. This stage cannot recover it.

  python -m training.train_verify
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import load_wav_mono_16k  # noqa: E402
from node.features import features_from_clip  # noqa: E402
from shared.config import DATA_DIR, MODELS_DIR  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES  # noqa: E402
from training.augment import match_train_level  # noqa: E402
from training.confuser_set import (  # noqa: E402
    false_wake_paths,
    inventory,
    lookalike_paths,
    lookalike_word,
    session_holdout,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "verify_stage.json"
HEADER = ROOT / "firmware" / "include" / "kws_verify_model.h"
TFLITE = MODELS_DIR / "marvin.verify.tflite"
CANDIDATE_THR = 0.65
MARGIN = 0.08
# How much p(marvin) must beat p(reject). 0.0 is the old bare 0.5.
MARGIN_GRID = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
# A one-clip wobble is not a reason to flash a second model.
MATERIAL_REJECT = 0.10
VERIFY_SHAPE = (48, 10, 1)
ALLOWED_OPS = {"CONV_2D", "RELU", "MEAN", "FULLY_CONNECTED", "SOFTMAX", "RESHAPE"}


def _clip(path: Path) -> np.ndarray:
    audio = load_wav_mono_16k(path)
    if len(audio) < CLIP_SAMPLES:
        pad = np.zeros(CLIP_SAMPLES, np.float32)
        pad[: len(audio)] = audio
        return pad
    return audio[:CLIP_SAMPLES].astype(np.float32)


def _mfcc(path: Path) -> np.ndarray:
    return features_from_clip(match_train_level(_clip(path))).astype(np.float32)


def _delta(mfcc: np.ndarray) -> np.ndarray:
    """First difference along time. Shape (48, 10, 1), not the stage-1 MFCC."""
    spec = mfcc[:, :, 0]
    change = (spec[1:] - spec[:-1]).astype(np.float32)
    return change[:, :, None]


def _kw_distance(path: Path) -> str:
    name = path.name
    if "3m" in name:
        return "3m"
    if "1m" in name:
        return "1m"
    return "close"


def _holdout(paths: list[Path], key):
    return session_holdout(paths, key)


def _false_order(path: Path) -> int:
    tail = path.stem.split("_")[-1]
    if tail.isdigit():
        return int(tail)
    return int(path.stat().st_mtime)


def _false_split(paths: list[Path]):
    """Keep the latest 20 percent of a one-day false-wake set out of training.

    A session holdout would hide every clip from the net, which is what made
    the previous run reject every keyword.
    """
    if not paths:
        return [], [], []
    from datetime import datetime

    by_day: dict[str, list[Path]] = {}
    for path in paths:
        day = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(path)
    if len(by_day) > 1:
        return session_holdout(paths, lambda _p: "false")
    ordered = sorted(paths, key=_false_order)
    n_te = max(10, int(round(len(ordered) * 0.2)))
    n_te = min(n_te, max(1, len(ordered) // 2))
    held = ordered[-n_te:]
    kept = ordered[:-n_te]
    bucket = {
        "group": "false",
        "holdout_session": "latest 20% of the only day, by wake time",
        "n": len(held),
        "train_n": len(kept),
        "sessions": sorted(by_day),
        "split": "within_day_latest",
    }
    return kept, held, [bucket]


def _is_candidate(prob: np.ndarray) -> bool:
    pkw, pun, psil = (float(prob[0]), float(prob[1]), float(prob[2]))
    return pkw >= CANDIDATE_THR and pkw >= pun + MARGIN and pkw >= psil


def _rows() -> list[tuple]:
    root = DATA_DIR / "marvin"
    kw = sorted((root / "keyword_real").glob("*.wav"))
    look = [p for p, _w in lookalike_paths()]
    false_w = false_wake_paths()
    unk = sorted((root / "unknown_real").glob("*.wav"))
    groups = (
        (_holdout(kw, _kw_distance), "keyword", 0),
        (_holdout(look, lambda p: lookalike_word(p) or "lookalike"), "lookalike", 1),
        (_false_split(false_w), "false_wake", 1),
        (_holdout(unk, lambda p: "unknown"), "unknown", 1),
    )
    rows = []
    buckets = []
    for (train, test, bucket), role, label in groups:
        buckets.extend(bucket)
        for split, paths in (("train", train), ("test", test)):
            for p in paths:
                if role == "keyword":
                    kind = "keyword"
                elif role == "lookalike":
                    kind = lookalike_word(p) or "lookalike"
                elif role == "false_wake":
                    kind = "false_wake"
                else:
                    kind = "unknown"
                rows.append((split, p, label, kind, _kw_distance(p) if label == 0 else ""))
    return rows, buckets


def _build():
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=VERIFY_SHAPE, name="delta_mfcc")
    x = keras.layers.Conv2D(8, (8, 4), strides=2, padding="same", use_bias=True, name="vconv")(inp)
    x = keras.layers.ReLU(name="vrelu")(x)
    x = keras.layers.GlobalAveragePooling2D(name="vgap")(x)
    x = keras.layers.Dense(2, name="vlogits")(x)
    out = keras.layers.Softmax(name="vprobs")(x)
    return keras.Model(inp, out), tf, keras


def _convert(model, rep: np.ndarray, tf) -> bytes:
    conv_w, conv_b = model.get_layer("vconv").get_weights()
    dense_w, dense_b = model.get_layer("vlogits").get_weights()

    class Frozen(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 48, 10, 1], tf.float32)])
        def __call__(self, x):
            y = tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(
                        x, tf.constant(conv_w), strides=[1, 2, 2, 1], padding="SAME"
                    ),
                    tf.constant(conv_b),
                )
            )
            y = tf.reduce_mean(y, axis=[1, 2])
            logits = tf.matmul(y, tf.constant(dense_w)) + tf.constant(dense_b)
            return tf.nn.softmax(logits)

    frozen = Frozen()
    sample = rep[:1].astype(np.float32)
    keras_p = model.predict(sample, verbose=0)[0]
    frozen_p = frozen(tf.constant(sample)).numpy()[0]
    err = float(np.max(np.abs(keras_p - frozen_p)))
    print(f"verify fold check max|keras-frozen|={err:.6f}", flush=True)
    if err > 1e-4:
        raise SystemExit("frozen verifier does not match the trained keras net")
    concrete = frozen.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], frozen)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [row[None].astype(np.float32)] for row in rep
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _ops(blob: bytes, tf) -> list[str]:
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    return sorted({d["op_name"] for d in interp._get_ops_details()} - {"DELEGATE"})


def _write_header(blob: bytes, margin: float) -> None:
    rows = ",\n  ".join(
        ", ".join(f"0x{b:02x}" for b in blob[i : i + 12]) for i in range(0, len(blob), 12)
    )
    HEADER.write_text(
        "\n".join(
            [
                "#pragma once",
                "// Generated by training.train_verify — do not edit.",
                f"#define KWS_VERIFY_MODEL_LEN {len(blob)}",
                f"#define KWS_VERIFY_MARGIN {margin:.2f}f",
                "",
                f"alignas(16) static const unsigned char KWS_VERIFY_MODEL[{len(blob)}] = {{",
                f"  {rows}",
                "};",
                "",
            ]
        )
    )


def _int8_probs(blob: bytes, feats: np.ndarray, tf) -> np.ndarray:
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    oscale, ozp = out["quantization"]
    got = []
    for row in feats:
        q = np.clip(np.round(row.reshape(inp["shape"]) / scale) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out["index"]).astype(np.float32).reshape(-1)
        got.append((raw - ozp) * oscale)
    return np.stack(got)


def _save(report: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))


def main() -> None:
    status = inventory()
    print("confuser inventory", json.dumps(status), flush=True)
    rows, buckets = _rows()
    small = [b for b in buckets if b["n"] < 10]
    # A bucket with n>=10 can still be a single recording day. That day is the
    # holdout, so the group contributes nothing to training. Report it. The
    # export block for size is n<10, as specified for this run.
    single_day = [b for b in buckets if b["train_n"] == 0]
    advisory = bool(small)
    print("holdout buckets", json.dumps(buckets), flush=True)
    if advisory:
        print(
            "WARNING: a holdout bucket has fewer than 10 clips. Result is advisory. "
            "No verifier header will be written.",
            flush=True,
        )
    if single_day:
        print(
            f"WARNING: {len(single_day)} groups have only one recording day, "
            "so that whole group is held out.",
            flush=True,
        )
    if not status["ready_for_stage4"]:
        missing = ", ".join(status["missing_words"]) or "none"
        report = {
            "ready_for_stage4": False,
            "missing_words": status["missing_words"],
            "false_wake_clips": status["false_wake_clips"],
            "accept": False,
            "exported": False,
            "reason": f"missing words: {missing}. false-wake clips: {status['false_wake_clips']}.",
        }
        _save(report)
        raise SystemExit(report["reason"])

    from training.chip_weights import load_chip_fused
    print(f"featurizing {len(rows)} clips", flush=True)
    mfcc = np.stack([_mfcc(r[1]) for r in rows]).astype(np.float32)
    x = np.stack([_delta(row) for row in mfcc]).astype(np.float32)
    chip = load_chip_fused("marvin")
    chip_p = chip.predict(mfcc, verbose=0)
    cand = np.array([_is_candidate(p) for p in chip_p])
    # Copies in false_wake_real are clips the device already woke on.
    for i, row in enumerate(rows):
        if row[3] == "false_wake":
            cand[i] = True
    print(f"first-stage candidates {int(cand.sum())}/{len(rows)}", flush=True)

    # Only clips stage 1 already accepted. Easy unknowns are not the reject
    # class: training on them made the net reject every keyword.
    train_idx = [i for i, r in enumerate(rows) if r[0] == "train" and cand[i]]
    n_pos = sum(rows[i][2] == 0 for i in train_idx)
    n_neg = sum(rows[i][2] == 1 for i in train_idx)
    train_scope = "first-stage candidates only, false wakes split within their recording day"
    test_idx = [i for i, r in enumerate(rows) if r[0] == "test"]
    base = {
        "view": "frame-delta of the 49x10 MFCC, shape 48x10. Stage 1 still sees the MFCC.",
        "candidate_rule": "pkw>=0.65 and pkw>=pun+0.08 and pkw>=psil; false_wake_real counted as candidates because the device woke",
        "train_scope": train_scope,
        "train_candidates": {"keyword": n_pos, "reject": n_neg},
        "holdout_buckets": buckets,
        "accept_advisory": advisory,
        "single_day_groups": len(single_day),
        "exported": False,
    }
    if n_pos < 8 or n_neg < 4:
        base["accept"] = False
        base["reason"] = (
            f"not enough first-stage candidates in the train split "
            f"(keyword {n_pos}, reject {n_neg}). Single-day confuser groups "
            "are entirely in the holdout."
        )
        _save(base)
        print(json.dumps(base, indent=2), flush=True)
        raise SystemExit(base["reason"])

    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    model, tf, keras = _build()
    order = np.random.default_rng(0).permutation(len(train_idx))
    train_idx = [train_idx[i] for i in order]
    x_tr = x[train_idx]
    y_tr = np.array([rows[i][2] for i in train_idx], np.int32)
    # Match the two class totals, with a small lean toward keeping keywords.
    # The holdout may drop at most 0.02 of keyword TPR, so the net has to
    # stay willing to accept a real marvin.
    n_pos_f = max(1, int((y_tr == 0).sum()))
    n_neg_f = max(1, int((y_tr == 1).sum()))
    sw = np.array([1.25 * n_neg_f if y == 0 else n_pos_f for y in y_tr], np.float32)
    sw *= len(y_tr) / sw.sum()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    model.fit(
        x_tr, y_tr, sample_weight=sw, validation_split=0.15,
        epochs=30, batch_size=16, callbacks=[stop], verbose=2,
    )

    # Holdout keyword TPR is over every held-out marvin, not only candidates.
    # The verifier never runs when the first stage says no.
    kw_te = [i for i in test_idx if rows[i][2] == 0]
    neg_te = [i for i in test_idx if rows[i][2] == 1 and rows[i][3] != "unknown" and cand[i]]
    unk_te = [i for i in test_idx if rows[i][3] == "unknown" and cand[i]]
    if not kw_te or not neg_te:
        base["accept"] = False
        base["holdout_keyword_n"] = len(kw_te)
        base["holdout_false_candidates"] = len(neg_te)
        base["reason"] = "holdout has no keyword clips or no first-stage false candidates."
        _save(base)
        print(json.dumps(base, indent=2), flush=True)
        raise SystemExit(base["reason"])

    kw_probs = model.predict(x[kw_te], verbose=0)
    neg_probs = model.predict(x[neg_te], verbose=0)
    unk_probs = model.predict(x[unk_te], verbose=0) if unk_te else None

    def accept_rate(idxs: list[int], probs: np.ndarray, margin: float) -> float:
        both = []
        for j, i in enumerate(idxs):
            stage2 = float(probs[j, 0]) >= float(probs[j, 1]) + margin
            both.append(bool(cand[i]) and stage2)
        return float(np.mean(both))

    tpr_s1 = float(np.mean([cand[i] for i in kw_te]))
    sweep = []
    chosen = None
    for margin in MARGIN_GRID:
        tpr_both = accept_rate(kw_te, kw_probs, margin)
        far_s2 = accept_rate(neg_te, neg_probs, margin)
        unk_s2 = accept_rate(unk_te, unk_probs, margin) if unk_te else 0.0
        row = {
            "margin": margin,
            "stage1_and_stage2_keyword_tpr": round(tpr_both, 4),
            "confuser_candidate_accept": round(far_s2, 4),
            "confuser_candidate_reject": round(1.0 - far_s2, 4),
            "unknown_candidate_accept": round(unk_s2, 4),
            "tpr_ok": tpr_both >= tpr_s1 - 0.02,
        }
        sweep.append(row)
        if row["tpr_ok"] and (
            chosen is None
            or row["confuser_candidate_reject"] > chosen["confuser_candidate_reject"]
            or (
                row["confuser_candidate_reject"] == chosen["confuser_candidate_reject"]
                and row["margin"] > chosen["margin"]
            )
        ):
            chosen = row
    # When no margin keeps keyword TPR, report the bare 0.5 point so the
    # headline matches the sweep instead of a blank fallback.
    reported = chosen if chosen is not None else sweep[0]
    tpr_both = reported["stage1_and_stage2_keyword_tpr"]
    far_s2 = reported["confuser_candidate_accept"]
    reject = reported["confuser_candidate_reject"]
    margin = reported["margin"] if chosen is not None else None
    metric_pass = chosen is not None and reject >= MATERIAL_REJECT
    accept = metric_pass and not advisory
    report = {
        **base,
        "holdout_keyword_n": len(kw_te),
        "holdout_false_candidates": len(neg_te),
        "stage1_keyword_tpr": round(tpr_s1, 4),
        "stage1_and_stage2_keyword_tpr": tpr_both,
        "stage2_far_on_false_candidates": far_s2,
        "confuser_candidate_reject": reject,
        "chosen_margin": margin,
        "margin_sweep": sweep,
        "material_reject": MATERIAL_REJECT,
        "metric_pass": metric_pass,
        "accept": accept,
        "rule": "keyword TPR drops by at most 0.02, confuser-candidate rejection >= 0.10, every holdout bucket n>=10",
    }
    _save(report)
    print(json.dumps(report, indent=2), flush=True)
    if not accept:
        raise SystemExit("verifier did not clear the holdout bar; no header written")

    rep = x_tr[: min(200, len(x_tr))]
    blob = _convert(model, rep, tf)
    ops = _ops(blob, tf)
    report["ops"] = ops
    report["bytes"] = len(blob)
    if not set(ops) <= ALLOWED_OPS:
        _save(report)
        raise SystemExit(f"verifier graph uses {ops}, which the firmware resolver does not have")
    kw_i = _int8_probs(blob, x[kw_te], tf)
    neg_i = _int8_probs(blob, x[neg_te], tf)
    tpr_i = accept_rate(kw_te, kw_i, float(margin))
    far_i = accept_rate(neg_te, neg_i, float(margin))
    report["int8_keyword_tpr"] = round(tpr_i, 4)
    report["int8_confuser_reject"] = round(1.0 - far_i, 4)
    if tpr_i < tpr_s1 - 0.02 or (1.0 - far_i) < MATERIAL_REJECT:
        report["accept"] = False
        report["reason"] = "float holdout passed but the INT8 graph did not"
        _save(report)
        raise SystemExit(report["reason"])
    TFLITE.write_bytes(blob)
    model.save(MODELS_DIR / "marvin.verify.keras")
    _write_header(blob, float(margin))
    report["exported"] = str(HEADER)
    _save(report)
    print(f"wrote {TFLITE} ({len(blob)} bytes) and {HEADER}", flush=True)
    print("flash the node to turn the second stage on", flush=True)


if __name__ == "__main__":
    main()
