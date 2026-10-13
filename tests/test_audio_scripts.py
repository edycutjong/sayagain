"""Radio-path simulation and the SNR mixer (synthetic test tones, no speech)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mix  # noqa: E402
import radioize  # noqa: E402

from sayagain.audio import SR, active_rms, load, rms, save  # noqa: E402


def tone_bursts(seconds: float = 2.0) -> np.ndarray:
    """Two 'utterances' of mixed tones separated by silence, like a turn pair."""
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t)
    gate = ((t > 0.2) & (t < 0.8)) | ((t > 1.2) & (t < 1.8))
    return (x * gate).astype(np.float32)


def band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[(f >= lo) & (f < hi)].sum())


def test_radioize_removes_content_outside_the_radio_band() -> None:
    x = tone_bursts()
    y = radioize.radioize(x)
    before = band_energy(x, 5000, 8000) / band_energy(x, 300, 3400)
    after = band_energy(y, 5000, 8000) / band_energy(y, 300, 3400)
    assert after < before * 1e-3
    assert len(y) > len(x)  # squelch tail appended


def test_mulaw_roundtrip_is_close_but_quantised() -> None:
    x = np.linspace(-0.9, 0.9, 1001, dtype=np.float32)
    y = radioize.mulaw_roundtrip(x)
    assert np.max(np.abs(y - x)) < 0.02
    assert len(np.unique(y)) <= 256


def test_radioize_is_deterministic() -> None:
    x = tone_bursts()
    assert np.array_equal(radioize.radioize(x, seed=1018), radioize.radioize(x, seed=1018))


@pytest.mark.parametrize("snr", [6.0, 0.0, -6.0])
def test_mix_hits_the_requested_snr(snr: float) -> None:
    speech = tone_bursts()
    noise = mix.synthetic_noise(len(speech), np.random.default_rng(0))
    y = mix.mix_at_snr(speech, noise, snr, normalize=False)
    added = y - speech
    measured = 20 * np.log10(active_rms(speech) / rms(added))
    assert abs(measured - snr) < 0.1


def test_mix_cli_is_deterministic(tmp_path: Path) -> None:
    save(tmp_path / "in" / "X-01.wav", tone_bursts())
    for run in ("a", "b"):
        mix.main(["--speech", str(tmp_path / "in"), "--out", str(tmp_path / run), "--snr", "clean", "-6"])
    a, b = load(tmp_path / "a" / "-6dB" / "X-01.wav"), load(tmp_path / "b" / "-6dB" / "X-01.wav")
    assert np.array_equal(a, b)
    assert (tmp_path / "a" / "clean" / "X-01.wav").exists()
