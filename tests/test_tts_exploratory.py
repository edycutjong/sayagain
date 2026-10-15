"""The exploratory synthetic-speech rows (badge `tts·synthetic`, bench/PROTOCOL.md Deviations 2026-10-15):
speech turns into the referee, WER normalisation, the score / report / transcribe commands and the card synthesiser.
No network and no Whisper: the ASR and the TTS API are replaced by fakes; every file lands under tmp_path."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from referee.registry import Registry
from referee.verdict import ABSTAIN, MISMATCH, NOT_AUDITED, VERIFIED
from sayagain.asr import ASRResult
from sayagain.audio import SR, save
from sayagain.extract import RUNGS, Pacer, regex_extraction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval as ev  # noqa: E402
import eval_tts  # noqa: E402
import tts_cards  # noqa: E402

REG = Registry.load()
CARDS = {c["exchange_id"]: c for c in ev.load_cards()}


def asr_turns(*texts, alts=None, failed=None):
    alts = alts or list(texts)
    return [
        {
            "id": f"X-{k}",
            "t_start": 3.5 * k,
            "t_end": 3.5 * k + 2.5,
            "text": t,
            "alt_text": a,
            "not_audited": (failed or {}).get(k),
        }
        for k, (t, a) in enumerate(zip(texts, alts, strict=True))
    ]


# ------------------------------------------------------------------------------------------- referee on ASR turns


def test_judge_card_on_asr_turns_reads_the_transcripts_not_the_script() -> None:
    card = CARDS["H-01"]  # scripted as a MISMATCH (two-oh-eight)
    turns = asr_turns("Close valve 203, line Bravo.", "Copy, closing 203, line Bravo.")
    rec = ev.judge_card(card, regex_extraction, REG, asr_turns=turns, audio_gate="disagreement")
    assert rec["verdict"] == VERIFIED  # what was heard decides; the card's label stays the expectation
    assert rec["expected_verdict"] == MISMATCH


def test_asr_failure_is_not_audited_and_never_extracted() -> None:
    seen = []

    def extract(text):
        seen.append(text)
        return regex_extraction(text)

    turns = asr_turns("Close valve 203, line Bravo.", "", failed={1: "asr error"})
    rec = ev.judge_card(CARDS["H-01"], extract, REG, asr_turns=turns, audio_gate="disagreement")
    assert seen == ["Close valve 203, line Bravo."]
    assert rec["verdict"] == NOT_AUDITED and rec["not_audited_turns"] == 1


def test_disagreement_gate_abstains_when_the_unprompted_pass_hears_another_tag() -> None:
    turns = asr_turns(
        "Close valve 203, line Bravo.",
        "Copy, closing 203, line Bravo.",
        alts=["Close valve 203, line Bravo.", "Copy, closing 208, line Bravo."],
    )
    rec = ev.judge_card(CARDS["H-01"], regex_extraction, REG, asr_turns=turns, audio_gate="disagreement")
    assert rec["verdict"] == ABSTAIN and "audio" in rec["label"]
    none = ev.judge_card(CARDS["H-01"], regex_extraction, REG, asr_turns=turns, audio_gate="none")
    assert none["verdict"] == VERIFIED  # without the gate the misheard pass would stand


# ---------------------------------------------------------------------------------------------------------- WER


def test_wer_words_collapses_number_formatting() -> None:
    assert eval_tts.wer_words("Close valve two-oh-three, line Bravo.") == ["close", "valve", "203", "line", "bravo"]
    assert eval_tts.wer_words("close valve 203 line bravo") == ["close", "valve", "203", "line", "bravo"]
    assert eval_tts.wer_words("katup empat satu Delta") == ["katup", "41", "delta"]
    assert eval_tts.wer_words("one") == ["1"]


def test_edit_distance_and_card_level_wer() -> None:
    assert eval_tts.edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert eval_tts.edit_distance([], ["a"]) == 1
    card = CARDS["H-01"]
    w = eval_tts.wer_rows([card], {"H-01": asr_turns("Close valve 203, line Bravo.", "Copy, closing 208, line Bravo.")})
    assert w["all"] == {"errors": 0, "ref_words": 10, "cards": 1, "wer": 0.0}
    w = eval_tts.wer_rows([card], {"H-01": asr_turns("Close valve 203, line Bravo.", "Copy, closing 203, lion Bravo.")})
    assert w["held_out"]["errors"] == 2 and w["set:H"]["cards"] == 1 and "lang:en" in w
    assert eval_tts.wer_rows([card], {}) == {}


# ----------------------------------------------------------------------------------------- score / rescore / report


@pytest.fixture
def tts_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "RECEIPTS", tmp_path / "bench" / "results" / "receipts")
    monkeypatch.setattr(ev, "OUT", tmp_path / "bench" / "results" / "text")
    monkeypatch.setattr(eval_tts, "ROOT", tmp_path)
    monkeypatch.setattr(eval_tts, "OUT", tmp_path / "bench" / "results" / "tts")
    monkeypatch.setattr(eval_tts, "TRANSCRIPTS", tmp_path / "bench" / "results" / "tts" / "transcripts")
    monkeypatch.setattr(eval_tts, "TTS", tmp_path / "data" / "tts" / "clean")
    cards = [CARDS[k] for k in ("H-01", "R-01", "C-01", "A2-021")]
    monkeypatch.setattr(ev, "load_cards", lambda path=None: cards)
    return tmp_path, cards


def write_transcripts(root: Path, cards: list[dict], cond: str = "radio-clean") -> Path:
    p = root / "bench" / "results" / "tts" / "transcripts" / f"{cond}.jsonl"
    rows = []
    for c in cards:
        texts = [t["text"] for t in c["turns"] if t["role"] != "pause"]
        rows.append({"exchange_id": c["exchange_id"], "condition": cond, "turns": asr_turns(*texts)})
    ev.write_jsonl(p, rows)
    return p


def test_score_regex_writes_a_rescorable_result_and_a_report(tts_env, capsys) -> None:
    root, cards = tts_env
    write_transcripts(root, cards)
    assert eval_tts.main(["--score", "--snr", "clean", "--rungs", "regex", "--run", "tts-t1"]) == 0
    doc = json.loads((root / "bench" / "results" / "tts" / "tts-t1.json").read_text(encoding="utf-8"))
    assert doc["provenance"] == "tts·synthetic" and doc["condition"] == "radio-clean"
    assert doc["badges"] == ["tts·synthetic", "radio·simulated", "asr·off-AMD"]
    rung = doc["rungs"]["regex"]
    assert rung["transcripts"] == "bench/results/tts/transcripts/radio-clean.jsonl"
    assert rung["audio_gate"] == "disagreement"
    assert rung["cards"].startswith("bench/results/tts/tts-t1/")
    assert doc["wer"]["all"]["wer"] == 0.0
    h01 = [r for r in ev.read_jsonl(root / rung["cards"]) if r["exchange_id"] == "H-01"][0]
    assert h01["label"] == "MISMATCH · tag: XV-203 ≠ XV-208"
    # the default --rescore picks up the tts rows and re-derives them unchanged
    assert ev.main(["--rescore"]) == 0
    assert "OK   tts-t1.json :: regex" in capsys.readouterr().out
    assert eval_tts.main(["--report", str(root / "bench" / "results" / "tts" / "tts-t1.json")]) == 0
    report = (root / "bench" / "results" / "tts" / "RESULTS.md").read_text(encoding="utf-8")
    spend = json.loads((root / "bench" / "results" / "tts" / "spend.json").read_text(encoding="utf-8"))
    assert spend["calls_cost_usd_total"] == 0.0 and spend["files"] == ["bench/results/tts/tts-t1.json"]
    assert "bench/results/tts/spend.json" in report
    assert "radio-clean · tts-t1" in report and "Whisper WER" in report and "no human recording" in report


def test_score_takes_one_condition(tts_env) -> None:
    assert eval_tts.main(["--score", "--snr", "clean", "-6", "--rungs", "regex"]) == 2


def test_score_model_rung_stops_on_the_daily_guard(tts_env, monkeypatch) -> None:
    root, cards = tts_env
    write_transcripts(root, cards)
    monkeypatch.setattr(ev, "check_daily", lambda b: (0.0, {"daily_cost_used_usd": 0.9}))
    assert eval_tts.main(["--score", "--snr", "clean", "--rungs", "regex,Gemma-4-31B-IT"]) == 2


def test_run_extraction_rung_sends_only_heard_turns_to_a_model(tts_env, monkeypatch) -> None:
    root, cards = tts_env
    p = root / "t.jsonl"
    rows = [{"exchange_id": c["exchange_id"], "turns": asr_turns("Close valve 203, line Bravo.", "")} for c in cards]
    for r in rows:
        r["turns"][1]["not_audited"] = "asr error"
    ev.write_jsonl(p, rows)
    sent = []

    class FakeExtractor:
        def __init__(self, *a, **k):
            pass

        def __call__(self, text):
            sent.append(text)
            return regex_extraction(text)

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(ev, "TokenFactoryExtractor", FakeExtractor)
    monkeypatch.setattr(ev, "open_client", lambda *a, **k: FakeClient())
    s = ev.run_extraction_rung(
        "m1", RUNGS["Gemma-4-31B-IT"], cards, REG, Pacer(0.0), ev.Budget(1.0),
        transcripts=p, audio_gate="disagreement", out=root / "o", prefix="tts",
    )  # fmt: skip
    assert sent == ["Close valve 203, line Bravo."]
    assert s["rows"]["held_out"]["not_audited"]["k"] == 2  # H-01 and A2-021: the read-back was not heard


# ------------------------------------------------------------------------------------------------- transcribe


class FakeWhisper:
    def transcribe(self, samples, *, language=None, prompt=None):
        text = "Close valve 203, line Bravo." if prompt else "Close valve 208, line Bravo."
        return ASRResult(text, None, "en", {"stage": "asr", "ok": True, "prompted": bool(prompt)})


def test_transcribe_runs_the_radio_path_and_resumes(tts_env, monkeypatch) -> None:
    root, cards = tts_env
    import sayagain.asr as asr_mod

    monkeypatch.setattr(asr_mod, "LocalWhisper", FakeWhisper)
    t = np.arange(int(1.2 * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    gap = np.zeros(SR, dtype=np.float32)
    save(root / "data" / "tts" / "clean" / "H-01.wav", np.concatenate([gap, tone, gap, tone, gap]))
    assert eval_tts.main(["--transcribe", "--snr", "clean", "--only", "H-01"]) == 0
    rows = ev.read_jsonl(root / "bench" / "results" / "tts" / "transcripts" / "radio-clean.jsonl")
    assert len(rows) == 1 and len(rows[0]["turns"]) == 2
    turn = rows[0]["turns"][0]
    assert turn["text"] == "Close valve 203, line Bravo." and turn["alt_text"] == "Close valve 208, line Bravo."
    rec = ev.read_jsonl(root / "bench" / "results" / "receipts" / "tts-asr-radio-clean.jsonl")
    assert len(rec) == 4 and rec[0]["condition"] == "radio-clean"
    assert eval_tts.main(["--transcribe", "--snr", "clean", "--only", "H-01"]) == 0  # resumable: nothing added
    assert len(ev.read_jsonl(root / "bench" / "results" / "tts" / "transcripts" / "radio-clean.jsonl")) == 1


def test_transcribe_refuses_missing_takes_and_missing_noise(tts_env, monkeypatch) -> None:
    root, _ = tts_env
    assert eval_tts.main(["--transcribe", "--snr", "clean", "--only", "H-01"]) == 2  # no TTS take
    save(root / "data" / "tts" / "clean" / "H-01.wav", np.zeros(SR, dtype=np.float32))
    import sayagain.asr as asr_mod

    monkeypatch.setattr(asr_mod, "LocalWhisper", FakeWhisper)
    assert eval_tts.main(["--transcribe", "--snr", "-6", "--only", "H-01"]) == 2  # no noise under tmp_path


def test_condition_names_follow_the_snr_labels() -> None:
    assert [eval_tts.condition_name(s) for s in ("clean", "6", "0", "-6")] == [
        "radio-clean",
        "radio-+6dB",
        "radio-0dB",
        "radio--6dB",
    ]


# ------------------------------------------------------------------------------------------------- synthesiser


class FakeSynth:
    def __init__(self, tmp: Path):
        self.tmp, self.chars_billed, self.calls = tmp, 0, []

    def mp3(self, role, text):
        self.calls.append(role)
        p = self.tmp / f"{len(self.calls)}.wav"
        save(p, (0.3 * np.ones(int(0.5 * SR))).astype(np.float32))
        return p


def test_build_card_lays_turns_out_with_silences(tmp_path) -> None:
    card = {
        "turns": [{"role": "operator", "text": "a"}, {"role": "pause", "seconds": 2.0}, {"role": "tech", "text": "bc"}]
    }
    synth = FakeSynth(tmp_path)
    x, turns = tts_cards.build_card(card, synth)
    assert synth.calls == ["operator", "tech"]
    assert [t["chars"] for t in turns] == [1, 2]
    expect = tts_cards.LEAD_S + 0.5 + 2.0 + tts_cards.GAP_S + 0.5 + tts_cards.TAIL_S
    assert abs(len(x) / SR - expect) < 0.01
    assert abs(float(np.max(np.abs(x))) - 0.7) < 1e-3


def test_synth_main_writes_wavs_and_a_manifest_without_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tts_cards, "Synth", lambda cache: FakeSynth(tmp_path))
    out, man = tmp_path / "clean", tmp_path / "manifest.json"
    assert tts_cards.main(["--only", "H-01,R-01", "--out", str(out), "--manifest", str(man)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["H-01.wav", "R-01.wav"]
    m = json.loads(man.read_text(encoding="utf-8"))
    assert m["badge"] == "tts·synthetic" and m["audio_committed"] is False and m["n_cards"] == 2
    assert m["voices"]["operator"]["voice_id"] != m["voices"]["tech"]["voice_id"]
    assert "sk_" not in man.read_text(encoding="utf-8")


def test_synth_falls_through_keys_and_caches(tmp_path, monkeypatch) -> None:
    import io
    import urllib.error

    creds = tmp_path / "c.json"
    creds.write_text(json.dumps({"keys": [{"name": "a", "key": "sk_a"}, {"name": "b", "key": "sk_b"}]}))
    monkeypatch.setattr(tts_cards, "CREDS", creds)
    used = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        key = req.headers["Xi-api-key"]
        used.append(key)
        if key == "sk_a":
            raise urllib.error.HTTPError(req.full_url, 401, "quota", {}, io.BytesIO(b'{"detail":"quota_exceeded"}'))
        return Resp(b"ID3fake")

    monkeypatch.setattr(tts_cards.urllib.request, "urlopen", urlopen)
    s = tts_cards.Synth(tmp_path / "cache")
    p = s.mp3("tech", "Copy.")
    assert p.read_bytes() == b"ID3fake" and used == ["sk_a", "sk_b"] and s.chars_billed == 5
    assert s.mp3("tech", "Copy.") == p and len(used) == 2  # cached: no second request

    def all_fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "busy", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(tts_cards.urllib.request, "urlopen", all_fail)
    with pytest.raises(SystemExit):
        tts_cards.Synth(tmp_path / "cache2").mp3("operator", "Open.")
