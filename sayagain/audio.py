"""Small audio helpers shared by the scripts: load/save mono float32, resample, RMS."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000


def load(path: str | Path, sr: int = SR) -> np.ndarray:
    """Mono float32 at `sr`."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample(x, file_sr, sr)


def save(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def active_rms(x: np.ndarray, sr: int = SR, frame_s: float = 0.02, rel_db: float = -35.0) -> float:
    """RMS over frames louder than `rel_db` below the loudest frame — speech level without the silences."""
    n = max(1, int(frame_s * sr))
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    e = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    if not e.size or e.max() == 0:
        return 0.0
    keep = e >= e.max() * 10 ** (rel_db / 20)
    return float(np.sqrt(np.mean(e[keep] ** 2)))
