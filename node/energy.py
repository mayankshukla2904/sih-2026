"""Listen-path gate. Quiet rooms: RMS above threshold for N hops.

Noisy rooms add an onset / speech-band check so a fan does not keep the
CNN awake. maybe_word is always a subset of the old two-hop rule, so this
cannot add false wakes. Must stay in lockstep with firmware/src/energy_gate.cpp.
"""

from __future__ import annotations

import math

import numpy as np

from shared.config import (
    ENERGY_CONSECUTIVE_FRAMES,
    ENERGY_FLOOR_DOWN,
    ENERGY_FLOOR_UP,
    ENERGY_NOISY_FLOOR,
    ENERGY_ONSET_ABS,
    ENERGY_ONSET_REL,
    ENERGY_RMS_THRESHOLD,
    ENERGY_SPEECH_HP,
    ENERGY_SPEECH_RATIO,
)


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))


class EnergyGate:
    def __init__(
        self,
        threshold: float = ENERGY_RMS_THRESHOLD,
        consecutive: int = ENERGY_CONSECUTIVE_FRAMES,
        noisy_floor: float = ENERGY_NOISY_FLOOR,
        onset_abs: float = ENERGY_ONSET_ABS,
        onset_rel: float = ENERGY_ONSET_REL,
        speech_ratio: float = ENERGY_SPEECH_RATIO,
    ) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self.noisy_floor = noisy_floor
        self.onset_abs = onset_abs
        self.onset_rel = onset_rel
        self.speech_ratio_thr = speech_ratio
        self._hits = 0
        self.noise_floor = 0.0
        self.last_rms = 0.0
        self.last_speech_ratio = 0.0
        self._hp_x1 = 0.0
        self._hp_y1 = 0.0

    def _speech_ratio(self, frame: np.ndarray, energy: float) -> float:
        if energy < 1e-8 or frame.size == 0:
            self._hp_x1 = 0.0
            self._hp_y1 = 0.0
            return 0.0
        x1 = self._hp_x1
        y1 = self._hp_y1
        acc = 0.0
        for x in frame.astype(np.float64, copy=False).ravel():
            y = x - x1 + ENERGY_SPEECH_HP * y1
            x1 = float(x)
            y1 = y
            acc += y * y
        self._hp_x1 = x1
        self._hp_y1 = y1
        hp_rms = math.sqrt(acc / frame.size)
        return hp_rms / energy

    def speechy(self, frame: np.ndarray) -> bool:
        self.last_rms = rms(frame)
        self.last_speech_ratio = self._speech_ratio(frame, self.last_rms)

        if self.last_rms > self.threshold:
            self._hits = min(self._hits + 1, self.consecutive)
        else:
            self._hits = max(self._hits - 1, 0)

        open_ = False
        if self._hits >= self.consecutive:
            if self.noise_floor <= self.noisy_floor:
                open_ = True
            else:
                margin = max(self.onset_abs, self.onset_rel * self.noise_floor)
                onset = self.last_rms >= self.noise_floor + margin
                speech_onset = (
                    self.last_speech_ratio >= self.speech_ratio_thr
                    and self.last_rms >= self.noise_floor + 0.5 * margin
                )
                open_ = onset or speech_onset

        if self.last_rms < self.noise_floor:
            self.noise_floor = ENERGY_FLOOR_DOWN * self.noise_floor + (
                1.0 - ENERGY_FLOOR_DOWN
            ) * self.last_rms
        else:
            self.noise_floor = ENERGY_FLOOR_UP * self.noise_floor + (
                1.0 - ENERGY_FLOOR_UP
            ) * self.last_rms
        return open_

    def reset(self) -> None:
        self._hits = 0
        # Keep the floor; a wake should not make a noisy room look quiet.
