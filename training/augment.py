"""Augmentation shared by dataset generation and training.

Two levels:
  * waveform — gain, shift, stretch, and noise mixed at INMP441-realistic SNRs.
    Used when clips are written to disk (training.generate_dataset).
  * feature  — SpecAugment-style time/frequency masking on the 49x10 MFCC.
    Used per batch during training (training.train), so every epoch sees a
    different mask. Costs nothing at inference time.

Noise comes from data/<keyword>/noise_real/*.wav when it exists (recorded
through the S3 mic, so it carries the real mic/room colour) and falls back to
shaped synthetic noise otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import load_wav_mono_16k  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, SPEC  # noqa: E402

# INMP441 in a room, at 1-3 m, with a laptop fan / TV / chatter around.
SNR_DB_RANGE = (0.0, 18.0)

# Must match firmware clip_match_train_level (target 0.10, floor 0.008, max 10x).
TRAIN_RMS_TARGET = 0.10
TRAIN_RMS_FLOOR = 0.008
TRAIN_RMS_MAX_GAIN = 10.0


def match_train_level(
    x: np.ndarray,
    target: float = TRAIN_RMS_TARGET,
    floor: float = TRAIN_RMS_FLOOR,
    max_g: float = TRAIN_RMS_MAX_GAIN,
) -> np.ndarray:
    """Boost quiet speech to the RMS the S3 uses before MFCC. Never attenuate."""
    y = x.astype(np.float32)
    r = rms(y)
    if r < floor:
        return y
    g = target / r
    if g <= 1.05:
        return y
    return np.clip(y * min(g, max_g), -1.0, 1.0).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


class NoiseBank:
    """Random CLIP_SAMPLES windows of background noise."""

    def __init__(self, dirs: list[Path] | None = None, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self._clips: list[np.ndarray] = []
        for folder in dirs or []:
            if not folder.exists():
                continue
            for wav in sorted(folder.glob("*.wav")):
                try:
                    audio = load_wav_mono_16k(wav)
                except Exception as exc:  # a truncated recording should not kill a run
                    print(f"noise skip {wav.name}: {exc}")
                    continue
                if len(audio) >= CLIP_SAMPLES // 2:
                    self._clips.append(audio.astype(np.float32))
        self.n_real = len(self._clips)

    def __len__(self) -> int:
        return self.n_real

    def _synthetic(self) -> np.ndarray:
        """Pink-ish noise: white noise with a 1/f tilt, plus optional mains hum."""
        white = self.rng.normal(0, 1.0, CLIP_SAMPLES).astype(np.float32)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(CLIP_SAMPLES, 1.0 / SPEC.sample_rate)
        tilt = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
        shaped = np.fft.irfft(spec * tilt, n=CLIP_SAMPLES).astype(np.float32)
        shaped /= rms(shaped)
        if self.rng.random() < 0.3:
            t = np.arange(CLIP_SAMPLES, dtype=np.float32) / SPEC.sample_rate
            shaped += 0.2 * np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        return shaped

    def sample(self) -> np.ndarray:
        if not self._clips:
            return self._synthetic()
        clip = self._clips[int(self.rng.integers(len(self._clips)))]
        if len(clip) <= CLIP_SAMPLES:
            out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            out[: len(clip)] = clip
            return out
        start = int(self.rng.integers(0, len(clip) - CLIP_SAMPLES))
        return clip[start : start + CLIP_SAMPLES].copy()


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so speech sits snr_db above it, then add."""
    s_rms = rms(speech)
    n_rms = rms(noise)
    if s_rms < 1e-6 or n_rms < 1e-6:
        return speech.astype(np.float32)
    target = s_rms / (10.0 ** (snr_db / 20.0))
    return (speech + noise * (target / n_rms)).astype(np.float32)


def augment_waveform(
    clip: np.ndarray,
    noise_bank: NoiseBank | None = None,
    rng: np.random.Generator | None = None,
    p_noise: float = 0.8,
    gain_lo: float = 0.08,
    gain_hi: float = 1.15,
    snr_db_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Gain / shift / stretch / noise. Keeps length at CLIP_SAMPLES."""
    rng = rng or np.random.default_rng()
    x = clip.astype(np.float32).copy()

    x *= float(rng.uniform(gain_lo, gain_hi))

    if rng.random() < 0.7:
        x = np.roll(x, int(rng.integers(-1600, 1600)))

    if rng.random() < 0.35:
        from scipy import signal as sps

        factor = float(rng.uniform(0.88, 1.12))
        y = sps.resample(x, max(1, int(len(x) * factor))).astype(np.float32)
        x = _fit(y, rng)

    if noise_bank is not None and rng.random() < p_noise:
        lo, hi = snr_db_range if snr_db_range is not None else SNR_DB_RANGE
        snr = float(rng.uniform(lo, hi))
        x = mix_at_snr(x, noise_bank.sample(), snr)

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _fit(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(x) >= CLIP_SAMPLES:
        start = int(rng.integers(0, len(x) - CLIP_SAMPLES + 1))
        return x[start : start + CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    pad = (CLIP_SAMPLES - len(x)) // 2
    out[pad : pad + len(x)] = x
    return out


def spec_augment(
    feat: np.ndarray,
    rng: np.random.Generator,
    n_time_masks: int = 2,
    max_time_mask: int = 8,
    n_freq_masks: int = 1,
    max_freq_mask: int = 2,
    p: float = 0.8,
) -> np.ndarray:
    """Mask random time frames and MFCC coefficients in place-safe fashion.

    feat is (n_frames, n_mfcc, 1). Masked regions take the per-clip mean so the
    model cannot use "exactly zero" as a shortcut.
    """
    if rng.random() > p:
        return feat
    out = feat.copy()
    fill = float(out.mean())
    n_frames, n_mfcc = out.shape[0], out.shape[1]

    for _ in range(n_time_masks):
        width = int(rng.integers(0, max_time_mask + 1))
        if width == 0:
            continue
        start = int(rng.integers(0, max(1, n_frames - width)))
        out[start : start + width, :, 0] = fill

    for _ in range(n_freq_masks):
        width = int(rng.integers(0, max_freq_mask + 1))
        if width == 0:
            continue
        # Coefficient 0 is overall energy; masking it too often just adds noise.
        start = int(rng.integers(1, max(2, n_mfcc - width)))
        out[:, start : start + width, 0] = fill

    return out


def spec_augment_batch(batch: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
    return np.stack([spec_augment(b, rng, **kwargs) for b in batch])
