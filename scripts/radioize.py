"""Simulated radio path: 300–3400 Hz band-pass, μ-law 8 kHz round-trip, squelch tail. Badge: radio·simulated.

uv run python scripts/radioize.py IN.wav OUT.wav [--seed 1018]
uv run python scripts/radioize.py --dir data/human/a2 --out data/mix/radio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sayagain.audio import SR, load, resample, save  # noqa: E402

MU = 255.0


def bandpass(x: np.ndarray, sr: int = SR, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """8-bit μ-law encode/decode (G.711-style companding, quantised to 256 levels)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return (np.sign(q) * (np.power(1 + MU, np.abs(q)) - 1) / MU).astype(np.float32)


def squelch_tail(
    x: np.ndarray, rng: np.random.Generator, sr: int = SR, ms: float = 150.0, level: float = 0.05
) -> np.ndarray:
    n = int(sr * ms / 1000)
    tail = rng.standard_normal(n).astype(np.float32) * level * np.linspace(1, 0, n, dtype=np.float32)
    return np.concatenate([x, bandpass(tail, sr)])


def radioize(x: np.ndarray, seed: int = 1018, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peak = float(np.max(np.abs(x))) or 1.0
    y = bandpass(x / peak * 0.8, sr)
    y = resample(mulaw_roundtrip(resample(y, sr, 8000)), 8000, sr)
    return squelch_tail(y, rng, sr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", nargs="?", type=Path)
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--dir", type=Path, help="radioize every .wav in this folder")
    ap.add_argument("--seed", type=int, default=1018)
    args = ap.parse_args(argv)
    if args.dir:
        if args.out is None and args.inp is not None:
            args.out = args.inp
        outdir = args.out or (args.dir.parent / (args.dir.name + "-radio"))
        files = sorted(args.dir.glob("*.wav"))
        for f in files:
            save(outdir / f.name, radioize(load(f), args.seed))
        print(f"radioized {len(files)} files -> {outdir}")
        return 0
    if args.inp is None or args.out is None:
        ap.error("give IN.wav OUT.wav, or --dir")
    save(args.out, radioize(load(args.inp), args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
