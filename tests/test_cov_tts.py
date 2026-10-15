"""Coverage for the remaining synthetic-speech branches: ElevenLabs 5xx retry and fatal HTTP errors, the periodic
progress lines of the synthesiser and the ASR pass, and the re-audit path of `eval_tts --score`.
No network, no sleeping: urlopen, time.sleep and the extraction runner are faked; every file lands under tmp_path."""

import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from sayagain.audio import SR, save

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval as ev  # noqa: E402
import eval_tts  # noqa: E402
import tts_cards  # noqa: E402
from test_tts_exploratory import CARDS, FakeSynth, FakeWhisper, tts_env, write_transcripts  # noqa: E402,F401


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _synth(tmp_path, monkeypatch) -> "tts_cards.Synth":
    creds = tmp_path / "c.json"
    creds.write_text(json.dumps({"keys": [{"name": "a", "key": "sk_a"}]}))
    monkeypatch.setattr(tts_cards, "CREDS", creds)
    return tts_cards.Synth(tmp_path / "cache")


def test_synth_sleeps_and_retries_on_a_server_error(tmp_path, monkeypatch) -> None:
    s = _synth(tmp_path, monkeypatch)
    sleeps, calls = [], []

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "down", {}, io.BytesIO(b"upstream down"))
        return Resp(b"ID3ok")

    monkeypatch.setattr(tts_cards.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tts_cards.time, "sleep", sleeps.append)
    p = s.mp3("tech", "Copy.")
    assert p.read_bytes() == b"ID3ok" and len(calls) == 2 and sleeps == [10]
    assert s.k == 0 and s.chars_billed == 5  # same key retried, not abandoned


def test_synth_exits_with_the_message_on_another_http_error(tmp_path, monkeypatch) -> None:
    s = _synth(tmp_path, monkeypatch)

    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 422, "bad", {}, io.BytesIO(b"voice not found"))

    monkeypatch.setattr(tts_cards.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tts_cards.time, "sleep", lambda s: pytest.fail("must not retry a 4xx"))
    with pytest.raises(SystemExit) as exc:
        s.mp3("tech", "Copy.")
    assert str(exc.value) == "ElevenLabs HTTP 422: voice not found"


def test_synth_main_prints_progress_every_20_cards(tmp_path, monkeypatch, capsys) -> None:
    cards = [{"exchange_id": f"P-{i:02d}", "turns": [{"role": "operator", "text": "ab"}]} for i in range(20)]
    cp = tmp_path / "cards.jsonl"
    cp.write_text("\n".join(json.dumps(c) for c in cards), encoding="utf-8")
    monkeypatch.setattr(tts_cards, "Synth", lambda cache: FakeSynth(tmp_path))
    out, man = tmp_path / "clean", tmp_path / "manifest.json"
    assert tts_cards.main(["--cards", str(cp), "--out", str(out), "--manifest", str(man)]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "cards ·" in ln]
    assert lines == ["20/20 cards · 0 chars billed this run"]


def test_transcribe_prints_progress_every_10_cards(tts_env, monkeypatch, capsys) -> None:  # noqa: F811
    root, cards = tts_env
    import sayagain.asr as asr_mod

    monkeypatch.setattr(asr_mod, "LocalWhisper", FakeWhisper)
    many = [{**CARDS["H-01"], "exchange_id": f"Q-{i:02d}"} for i in range(10)]
    monkeypatch.setattr(ev, "load_cards", lambda path=None: many)
    t = np.arange(int(1.2 * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    gap = np.zeros(SR, dtype=np.float32)
    wav = np.concatenate([gap, tone, gap, tone, gap])
    for c in many:
        save(root / "data" / "tts" / "clean" / f"{c['exchange_id']}.wav", wav)
    assert eval_tts.main(["--transcribe", "--snr", "clean"]) == 0
    out = capsys.readouterr().out
    assert any(ln.startswith("radio-clean: 10 cards · ") for ln in out.splitlines())
    assert "radio-clean: done" in out


def test_reaudit_records_its_source_and_hands_prior_extractions_to_model_rungs(tts_env, monkeypatch) -> None:  # noqa: F811
    root, cards = tts_env
    write_transcripts(root, cards)
    prior_rows = [{"exchange_id": "H-01", "marker": "kept"}]
    ev.write_jsonl(root / "bench" / "results" / "tts" / "first-extractions.jsonl", prior_rows)
    first = root / "bench" / "results" / "tts" / "tts-first.json"
    ev.write_json(
        first,
        {"rungs": {"Gemma-4-31B-IT": {"extractions": "bench/results/tts/first-extractions.jsonl"}}},
    )
    monkeypatch.setattr(ev, "check_daily", lambda b: (1.0, {"daily_cost_used_usd": 0.0}))
    seen = {}
    real = ev.run_extraction_rung

    def fake_rung(run, rung, cards, registry, pacer, budget, mode, prior, **kw):
        seen[rung.name] = prior
        return real(run, ev.CANDIDATES["regex"], cards, registry, pacer, budget, mode, None, **kw)

    monkeypatch.setattr(ev, "run_extraction_rung", fake_rung)
    argv = ["--score", "--snr", "clean", "--rungs", "regex,Gemma-4-31B-IT", "--run", "tts-second"]
    assert eval_tts.main([*argv, "--reaudit", str(first)]) == 0
    doc = json.loads((root / "bench" / "results" / "tts" / "tts-second.json").read_text(encoding="utf-8"))
    assert doc["reaudit_of"] == "bench/results/tts/tts-first.json"
    assert seen["regex"] is None  # a regex rung never re-audits
    assert seen["Gemma-4-31B-IT"] == prior_rows
