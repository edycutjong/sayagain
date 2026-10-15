"""Gap-closing tests for scripts/check_submission_readiness.py, scripts/mix.py and scripts/radioize.py."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_submission_readiness as cr  # noqa: E402
import mix  # noqa: E402
import radioize  # noqa: E402

from sayagain.audio import SR, load, save  # noqa: E402


def tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------- readiness


def test_finding_str_format() -> None:
    f = cr.Finding("number", "README.md", 7, "90 % not found")
    assert str(f) == "FAIL [number] README.md:7: 90 % not found"


def test_load_receipts_without_results_dir_is_empty(tmp_path: Path) -> None:
    assert cr.load_receipts(tmp_path) == []


def test_load_receipts_skips_invalid_json_and_reads_valid(tmp_path: Path) -> None:
    res = tmp_path / "bench" / "results"
    res.mkdir(parents=True)
    (res / "bad.json").write_text("{not json", encoding="utf-8")
    (res / "good.json").write_text(json.dumps({"stage": "asr"}), encoding="utf-8")
    (res / "rows.jsonl").write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    got = cr.load_receipts(tmp_path)
    assert {"stage": "asr"} in got
    assert [r for r in got if "a" in r] == [{"a": 1}, {"a": 2}]
    assert len(got) == 3


def test_extraction_on_amd_without_receipt_is_flagged(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Extraction runs on AMD.\n", encoding="utf-8")
    findings = cr.check_files(tmp_path, "build")
    assert [(f.check, f.text) for f in findings] == [("amd-stage", "extraction on AMD with no AMD extraction receipt")]


def test_collect_tests_parses_pytest_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(stdout="x\n42 tests collected in 0.1s\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cr.collect_tests(tmp_path) == 42
    assert calls[0][1]["cwd"] == tmp_path
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="no summary"))
    assert cr.collect_tests(tmp_path) is None


def test_main_pass_and_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "README.md").write_text("Plain page.\n", encoding="utf-8")
    monkeypatch.setattr(cr, "collect_tests", lambda root: 5)
    assert cr.main(["--root", str(good)]) == 0
    out = capsys.readouterr().out
    assert "readiness (build): PASS · 0 finding(s) · 5 tests collected" in out

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "README.md").write_text("This is TODO.\nWe have 9 tests.\n", encoding="utf-8")
    assert cr.main(["--root", str(bad), "--no-collect"]) == 1
    out = capsys.readouterr().out
    assert "FAIL [placeholder] README.md:1: TODO" in out
    assert "readiness (build): FAIL · 1 finding(s)" in out
    assert "tests collected" not in out


# ---------------------------------------------------------------- mix


def test_noise_segment_from_pool_short_and_long(tmp_path: Path) -> None:
    short = tmp_path / "short.wav"
    long = tmp_path / "long.wav"
    save(short, tone(0.25))
    save(long, tone(2.0))
    n = SR  # longer than the short file: tiled
    seg = mix.noise_segment(n, [short], np.random.default_rng(0))
    assert seg.shape == (n,)
    assert float(np.abs(seg).max()) > 0.1
    seg2 = mix.noise_segment(n, [long], np.random.default_rng(0))
    assert seg2.shape == (n,)


def test_noisy_copy_clean_is_identity_and_noisy_is_deterministic() -> None:
    sp = tone(1.0)
    assert mix.noisy_copy(sp, "stem", "clean", None) is sp
    a = mix.noisy_copy(sp, "stem", "0", None)
    b = mix.noisy_copy(sp, "stem", "0", None)
    c = mix.noisy_copy(sp, "other", "0", None)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, sp)


def test_mix_cli_with_noise_folder_and_single_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    sp = tmp_path / "hero.wav"
    save(sp, tone(1.0))
    noise_dir = tmp_path / "noise" / "sub"
    save(noise_dir / "n.wav", tone(2.0) * 0.5)
    out = tmp_path / "out"
    rc = mix.main(["--speech", str(sp), "--noise", str(tmp_path / "noise"), "--out", str(out), "--snr", "clean", "6"])
    assert rc == 0
    assert "mixed 1 files × 2 SNR" in capsys.readouterr().out
    assert (out / "clean" / "hero.wav").exists()
    assert (out / "+6dB" / "hero.wav").exists()
    assert len(load(out / "+6dB" / "hero.wav")) == len(load(sp))


def test_mix_cli_empty_noise_folder_errors(tmp_path: Path) -> None:
    sp = tmp_path / "hero.wav"
    save(sp, tone(0.5))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        mix.main(["--speech", str(sp), "--noise", str(tmp_path / "empty"), "--out", str(tmp_path / "o")])
    assert e.value.code == 2


def test_mix_normalizes_clipping_peak() -> None:
    sp = tone(1.0) * 2.0
    y = mix.mix_at_snr(sp, mix.synthetic_noise(len(sp), np.random.default_rng(1)), 0.0)
    assert float(np.max(np.abs(y))) == pytest.approx(0.95, abs=1e-4)


# ---------------------------------------------------------------- radioize


def test_squelch_tail_extends_signal() -> None:
    x = tone(0.5)
    y = radioize.squelch_tail(x, np.random.default_rng(0), ms=100.0)
    assert len(y) == len(x) + int(SR * 0.1)
    assert np.array_equal(y[: len(x)], x)


def test_radioize_cli_single_file(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    save(src, tone(0.5))
    dst = tmp_path / "sub" / "out.wav"
    assert radioize.main([str(src), str(dst)]) == 0
    y = load(dst)
    assert len(y) > int(0.5 * SR)


def test_radioize_cli_dir_default_and_explicit_out(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    d = tmp_path / "speech"
    save(d / "a.wav", tone(0.4))
    save(d / "b.wav", tone(0.4))
    assert radioize.main(["--dir", str(d)]) == 0
    assert "radioized 2 files" in capsys.readouterr().out
    default_out = tmp_path / "speech-radio"
    assert sorted(p.name for p in default_out.glob("*.wav")) == ["a.wav", "b.wav"]

    explicit = tmp_path / "explicit"
    assert radioize.main([str(d), str(explicit), "--dir", str(d)]) == 0
    assert (explicit / "a.wav").exists()

    positional = tmp_path / "positional"  # the lone positional doubles as the output folder
    assert radioize.main([str(positional), "--dir", str(d)]) == 0
    assert (positional / "b.wav").exists()


def test_radioize_cli_requires_inputs() -> None:
    with pytest.raises(SystemExit) as e:
        radioize.main([])
    assert e.value.code == 2
