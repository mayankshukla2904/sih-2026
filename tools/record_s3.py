"""Record dataset clips through the ESP32-S3's own INMP441, over USB.

Why through the S3 and not the laptop mic: the model has to survive this mic,
this preamp, this room. Clips recorded on a MacBook teach the network the wrong
noise floor and the wrong frequency response.

The firmware streams base64 hops on the USB serial after a `REC` command
(see firmware/src/main.cpp). Nothing goes over WiFi, so recording works even
with the Pi switched off.

Guided clips (what you want for keyword/lookalike/unknown):
  python -m tools.record_s3 --label keyword --speaker mayank --count 30
  python -m tools.record_s3 --label lookalike --speaker mayank --count 15 --prompt "sahayata"

Continuous background (what you want for noise_real / the FAR test material):
  python -m tools.record_s3 --label noise --continuous 600
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import DATA_DIR, KEYWORD  # noqa: E402
from shared.feature_spec import HOP_SAMPLES, SPEC  # noqa: E402
from tools.s3_link import DEFAULT_BAUD, default_port  # noqa: E402

# keyword/unknown/lookalike/silence become <name>_real/; noise is already "real".
LABEL_DIRS = {
    "keyword": "keyword_real",
    "unknown": "unknown_real",
    "lookalike": "lookalike_real",
    "silence": "silence_real",
    "noise": "noise_real",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--label", choices=tuple(LABEL_DIRS), default="keyword")
    p.add_argument("--speaker", default="anon", help="one token, no underscores")
    p.add_argument("--count", type=int, default=30, help="clips in guided mode")
    p.add_argument("--clip-ms", type=int, default=1400)
    p.add_argument("--lead-ms", type=int, default=250, help="silence kept before you speak")
    p.add_argument("--prompt", default=None, help="what to say; defaults to the spoken keyword")
    p.add_argument("--continuous", type=float, default=0.0, help="seconds of one long recording")
    p.add_argument("--out", type=Path, default=None, help="override output directory")
    return p.parse_args()


class S3Audio:
    """Talks the tiny serial protocol in firmware/src/main.cpp."""

    def __init__(self, port: str, baud: int) -> None:
        import serial

        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 2.0
        # Toggling these resets the S3 over USB-JTAG.
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def start(self) -> None:
        self.ser.write(b"REC\n")
        self.ser.flush()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            line = self.ser.readline()
            if b"capture on" in line:
                return
        raise SystemExit("firmware never acknowledged REC; is the capture build flashed?")

    def stop(self) -> None:
        self.ser.write(b"STOP\n")
        self.ser.flush()
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def read_samples(self, n: int) -> np.ndarray:
        """Block until n samples of int16 arrive. Non-audio serial lines are ignored."""
        out = np.zeros(n, dtype=np.int16)
        got = 0
        deadline = time.monotonic() + (n / SPEC.sample_rate) * 4 + 5.0
        while got < n and time.monotonic() < deadline:
            line = self.ser.readline()
            if not line.startswith(b"A "):
                continue
            try:
                raw = base64.b64decode(line[2:].strip())
            except Exception:
                continue
            if len(raw) != HOP_SAMPLES * 2:
                continue
            hop = np.frombuffer(raw, dtype="<i2")
            take = min(n - got, len(hop))
            out[got : got + take] = hop[:take]
            got += take
        if got < n:
            raise SystemExit(
                f"serial underrun: {got}/{n} samples. "
                "Hops must arrive as 'A …' lines. Unplug other serial monitors, "
                "reflash if this firmware is older than dual-UART capture, then retry."
            )
        return out

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.ser.close()


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SPEC.sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())


def level(pcm: np.ndarray) -> tuple[float, float]:
    f = pcm.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))), float(np.max(np.abs(f)))


def bar(rms: float, width: int = 28) -> str:
    filled = int(min(1.0, rms / 0.15) * width)
    return "#" * filled + "." * (width - filled)


def guided(dev: S3Audio, args: argparse.Namespace, out_dir: Path) -> None:
    prompt = args.prompt or (KEYWORD if args.label == "keyword" else args.label)
    lead = int(SPEC.sample_rate * args.lead_ms / 1000)
    clip_n = int(SPEC.sample_rate * args.clip_ms / 1000)
    existing = len(list(out_dir.glob(f"{args.label}_{args.speaker}_*.wav")))
    print(f"\n{args.count} clips of {prompt!r}, speaker={args.speaker}, {args.clip_ms} ms each")
    print(f"writing to {out_dir} (starting at index {existing})")
    print("Vary distance (1 m and 3 m), speed, and loudness between takes.")
    print("Enter = record, s = skip, q = quit\n")

    saved = 0
    i = existing
    while saved < args.count:
        cmd = input(f"[{saved + 1}/{args.count}] say '{prompt}' > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        dev.read_samples(lead)  # drop the keypress noise
        pcm = dev.read_samples(clip_n)
        rms, peak = level(pcm)
        # Firmware applies MIC_GAIN 12 before USB. A plosive or key-click that
        # is only ~0.08 FS at the mic becomes peak 1.0 here, while 3 m speech
        # RMS can still sit at 0.01. Redo only when the *whole clip* is loud.
        if peak >= 0.99 and rms >= 0.05:
            print(f"    actually loud (peak {peak:.2f} rms {rms:.4f}) — move back and redo")
            continue
        if peak >= 0.99:
            print(
                f"    peak {peak:.2f} with rms {rms:.4f} — click/plosive after 12x mic gain, keeping"
            )
        if rms < 0.004:
            print(f"    too quiet (rms {rms:.4f}) — probably missed it, redo")
            continue
        path = out_dir / f"{args.label}_{args.speaker}_{i:03d}.wav"
        write_wav(path, pcm)
        print(f"    saved {path.name}  rms={rms:.4f} peak={peak:.2f}  {bar(rms)}")
        saved += 1
        i += 1
    print(f"\n{saved} clips written to {out_dir}")


def continuous(dev: S3Audio, args: argparse.Namespace, out_dir: Path) -> None:
    total = int(SPEC.sample_rate * args.continuous)
    print(f"\nrecording {args.continuous:.0f}s continuously into {out_dir}")
    print("Play whatever the node must ignore: TV, podcast, Hindi/English chatter.\n")
    chunks: list[np.ndarray] = []
    got = 0
    t0 = time.monotonic()
    while got < total:
        want = min(SPEC.sample_rate, total - got)
        pcm = dev.read_samples(want)
        chunks.append(pcm)
        got += want
        rms, _ = level(pcm)
        elapsed = time.monotonic() - t0
        print(f"\r  {elapsed:6.1f}s / {args.continuous:.0f}s  rms={rms:.4f} {bar(rms)}", end="", flush=True)
    print()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{args.label}_{args.speaker}_{stamp}.wav"
    write_wav(path, np.concatenate(chunks))
    mins = got / SPEC.sample_rate / 60.0
    print(f"saved {path} ({mins:.1f} min)")


def main() -> None:
    args = parse_args()
    out_dir = args.out or (DATA_DIR / args.keyword / LABEL_DIRS[args.label])
    if "_" in args.speaker:
        raise SystemExit("--speaker must not contain underscores (it is parsed out of the filename)")

    dev = S3Audio(args.port or default_port(), args.baud)
    try:
        dev.start()
        print("capture on")
        if args.continuous > 0:
            continuous(dev, args, out_dir)
        else:
            guided(dev, args, out_dir)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
