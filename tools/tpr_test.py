"""Gate F: true positive rate from a scripted live trial.

Accuracy on held-out WAVs is not the number judges care about; the number is
"how often does it wake when I say the word, standing where I actually stand".
So this prompts a human, waits for the node to wake, and records hit or miss per
distance.

  python -m tools.tpr_test --speaker mayank --per-distance 10
  python -m tools.tpr_test --distances 1 3 5 --per-distance 10

Results land in results/tpr.json, keyed by speaker and distance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import KEYWORD, WAKE_SCORE_THRESHOLD  # noqa: E402
from tools.s3_link import DEFAULT_BAUD, S3Link  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--speaker", default="anon")
    p.add_argument("--distances", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--per-distance", type=int, default=10)
    p.add_argument("--window", type=float, default=3.0, help="seconds allowed for a wake")
    p.add_argument(
        "--auto",
        action="store_true",
        help="countdown prompts instead of waiting for Enter (for unattended/agent runs)",
    )
    p.add_argument("--idle-seconds", type=float, default=20.0, help="quiet listen before trials (auto mode)")
    p.add_argument("--countdown", type=float, default=4.0, help="seconds to get ready before each say")
    p.add_argument("--out", type=Path, default=RESULTS / "tpr.json")
    return p.parse_args()


def drain(link: S3Link) -> None:
    for _ in link.read(0.4):
        pass


def _tel_snapshot(evs: list) -> dict:
    cpus, idles, rms, states = [], [], [], []
    for ev in evs:
        if ev.kind != "tel":
            continue
        f = ev.fields
        if "cpu" in f:
            cpus.append(float(f["cpu"]))
        if "idle" in f:
            idles.append(float(f["idle"]))
        if "rms" in f:
            rms.append(float(f["rms"]))
        if "state" in f:
            states.append(str(f["state"]))
    def stat(xs: list[float]) -> dict | None:
        if not xs:
            return None
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 3),
            "max": round(max(xs), 3),
        }
    return {
        "cpu": stat(cpus),
        "idle_cpu": stat(idles),
        "rms": stat(rms),
        "states": sorted(set(states)),
    }


def listen_idle(link: S3Link, seconds: float) -> dict:
    """Room with fan, do not say the keyword. Records idle CPU / RMS / false wakes."""
    print(f"\n=== IDLE LISTEN {seconds:.0f}s — do NOT say '{KEYWORD}' ===", flush=True)
    print("Leave the room as it is (fan/AC is the point).\n", flush=True)
    t0 = time.monotonic()
    evs = []
    wakes = []
    for ev in link.read(seconds):
        evs.append(ev)
        if ev.kind == "wake":
            wakes.append(
                {
                    "at_s": round(ev.wake.host_time - t0, 2),
                    "score": ev.wake.score,
                    "p_keyword": ev.wake.p_keyword,
                }
            )
            print(f"    FALSE WAKE at {wakes[-1]['at_s']}s score={ev.wake.score:.3f}", flush=True)
    snap = _tel_snapshot(evs)
    snap["false_wakes"] = wakes
    snap["duration_s"] = seconds
    print(
        f"    idle cpu={snap['idle_cpu']}  hop cpu={snap['cpu']}  "
        f"rms={snap['rms']}  FA={len(wakes)}",
        flush=True,
    )
    return snap


def one_trial(link: S3Link, window: float) -> dict:
    """Wait for a wake after the prompt. The refractory period is 1.5 s, so one
    utterance can only ever produce one WAKE line."""
    t0 = time.monotonic()
    evs = []
    hit = None
    for ev in link.read(window):
        evs.append(ev)
        if ev.kind == "wake" and hit is None:
            w = ev.wake
            hit = {
                "hit": True,
                "latency_s": round(w.host_time - t0, 3),
                "score": w.score,
                "p_keyword": w.p_keyword,
                "invoke_us": w.invoke_us,
                "mfcc_us": w.mfcc_us,
            }
    if hit is None:
        hit = {"hit": False}
    hit["tel"] = _tel_snapshot(evs)
    return hit


def main() -> int:
    args = parse_args()

    with S3Link(args.port, args.baud) as link:
        status = link.status()
        print(f"node status: {status or 'no reply'}", flush=True)
        print(
            f"\nkeyword={args.keyword!r} speaker={args.speaker} threshold={WAKE_SCORE_THRESHOLD}",
            flush=True,
        )

        idle = None
        if args.auto and args.idle_seconds > 0:
            idle = listen_idle(link, args.idle_seconds)

        if args.auto:
            print(
                f"\nAUTO: {args.per_distance} trials per distance. "
                f"Get ready for {args.countdown:.0f}s, then say '{args.keyword}' ONCE.\n",
                flush=True,
            )
        else:
            print("Enter = I am about to say it, s = skip, q = quit this distance\n", flush=True)

        by_distance = {}
        for dist in args.distances:
            print(f"--- stand {dist:g} m from the mic ---", flush=True)
            trials = []
            while len(trials) < args.per_distance:
                n = len(trials) + 1
                if args.auto:
                    print(
                        f"[{n}/{args.per_distance}] {dist:g} m  GET READY ({args.countdown:.0f}s)",
                        flush=True,
                    )
                    time.sleep(args.countdown)
                    drain(link)
                    print(f"    SAY '{args.keyword}' NOW", flush=True)
                else:
                    cmd = input(f"[{n}/{args.per_distance}] {dist:g} m > ").strip().lower()
                    if cmd == "q":
                        break
                    if cmd == "s":
                        continue
                    drain(link)
                    print(f"    say '{args.keyword}' now...", flush=True)
                r = one_trial(link, args.window)
                trials.append(r)
                if r["hit"]:
                    print(
                        f"    HIT  score={r['score']:.3f} after {r['latency_s']:.2f}s",
                        flush=True,
                    )
                else:
                    print("    MISS", flush=True)
                # Clear the refractory window before the next prompt.
                time.sleep(1.7)

            hits = sum(1 for t in trials if t["hit"])
            n = len(trials)
            tpr = hits / n if n else None
            lat = sorted(t["latency_s"] for t in trials if t["hit"])
            by_distance[f"{dist:g}m"] = {
                "n": n,
                "hits": hits,
                "tpr": round(tpr, 4) if tpr is not None else None,
                "median_wake_latency_s": lat[len(lat) // 2] if lat else None,
                "trials": trials,
            }
            if n:
                print(f"    {dist:g} m: TPR {hits}/{n} = {tpr * 100:.0f}%\n")

    total_n = sum(v["n"] for v in by_distance.values())
    total_hits = sum(v["hits"] for v in by_distance.values())
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keyword": args.keyword,
        "speaker": args.speaker,
        "threshold": WAKE_SCORE_THRESHOLD,
        "idle_listen": idle,
        "overall": {
            "n": total_n,
            "hits": total_hits,
            "tpr": round(total_hits / total_n, 4) if total_n else None,
        },
        "by_distance": by_distance,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep earlier speakers so several team members accumulate into one file.
    existing = json.loads(args.out.read_text()) if args.out.exists() else {"runs": []}
    if "runs" not in existing:
        existing = {"runs": [existing]}
    existing["runs"].append(result)
    args.out.write_text(json.dumps(existing, indent=2))

    if total_n:
        print(f"overall TPR {total_hits}/{total_n} = {total_hits / total_n * 100:.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
