"""Add industrial noise at a fixed SNR (clean / +6 / 0 / −6 dB), deterministic per seed.

    uv run python scripts/mix.py --speech data/human/a2 --noise data/noise --out data/mix --snr clean 6 0 -6
    uv run python scripts/mix.py --speech data/human/hero --noise synthetic --out data/demo --snr clean -6

SNR is measured against the speech's active level (silences excluded). Eval mixes use MIMII (CC BY-SA 4.0, fetched by
scripts/fetch_mimii.sh, never committed). `--noise synthetic` generates machine-like noise in code (pump hum harmonics
+ broadband flow noise) — ours, MIT — for the few demo clips that are committed.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sayagain.audio import SR, active_rms, load, rms, save  # noqa: E402


def synthetic_noise(n: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Pump-like hum (50 Hz mains-motor harmonics with slow wobble) over pink-ish broadband flow noise."""
    t = np.arange(n) / sr
    f0 = 50.0 * (1 + 0.002 * np.sin(2 * np.pi * 0.3 * t))
    hum = sum((0.6 / k) * np.sin(2 * np.pi * k * np.cumsum(f0) / sr + rng.uniform(0, 2 * np.pi)) for k in range(1, 9))
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[1:] /= np.sqrt(freqs[1:] / 100.0)
    spec[0] = 0
    pink = np.fft.irfft(spec, n)
    x = hum / (rms(hum) or 1) + 0.8 * pink / (rms(pink) or 1)
    return (x / (np.max(np.abs(x)) or 1)).astype(np.float32)


def noise_segment(n: int, pool: list[Path] | None, rng: np.random.Generator) -> np.ndarray:
    if pool is None:
        return synthetic_noise(n, rng)
    src = load(pool[int(rng.integers(len(pool)))])
    if len(src) < n:
        src = np.tile(src, n // len(src) + 1)
    start = int(rng.integers(0, len(src) - n + 1))
    return src[start : start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, normalize: bool = True) -> np.ndarray:
    s = active_rms(speech)
    nr = rms(noise)
    if s == 0 or nr == 0:
        raise ValueError("silent speech or noise: an SNR mix would be mislabelled")
    gain = s / (nr * 10 ** (snr_db / 20))
    y = speech + gain * noise
    peak = float(np.max(np.abs(y)))
    if normalize and peak > 0.95:
        y = y / peak * 0.95  # one gain for speech and noise together: the SNR is unchanged
    return y.astype(np.float32)


def snr_label(snr: str) -> str:
    """'clean', '+6dB', '0dB', '-6dB'."""
    if snr == "clean":
        return "clean"
    v = int(snr)
    return "0dB" if v == 0 else f"{v:+d}dB"


def noisy_copy(speech: np.ndarray, stem: str, snr: str, pool: list[Path] | None, seed: int = 1018) -> np.ndarray:
    """The one way every tool builds an SNR copy: same exchange + same SNR -> same noise slice."""
    if snr == "clean":
        return speech
    rng = np.random.default_rng([seed, zlib.crc32(stem.encode()), int(snr) + 100])
    return mix_at_snr(speech, noise_segment(len(speech), pool, rng), float(snr))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech", type=Path, required=True, help="folder of speech WAVs (or one file)")
    ap.add_argument("--noise", default="synthetic", help="folder of noise WAVs (searched recursively), or 'synthetic'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snr", nargs="+", default=["clean", "6", "0", "-6"])
    ap.add_argument("--seed", type=int, default=1018)
    args = ap.parse_args(argv)
    files = [args.speech] if args.speech.is_file() else sorted(args.speech.glob("*.wav"))
    pool = None if args.noise == "synthetic" else sorted(Path(args.noise).rglob("*.wav"))
    if pool is not None and not pool:
        ap.error(f"no noise WAVs under {args.noise} (run scripts/fetch_mimii.sh, or use --noise synthetic)")
    for f in files:
        speech = load(f)
        for snr in args.snr:
            save(args.out / snr_label(snr) / f.name, noisy_copy(speech, f.stem, snr, pool, args.seed))
    print(f"mixed {len(files)} files × {len(args.snr)} SNR -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
