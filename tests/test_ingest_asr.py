"""Audio path on fixtures: ingest (VAD inside one WAV per card), ASR backends, the disagreement gate end to end.

No human audio and no Whisper weights in CI: tones stand in for speech in the VAD tests, fake backends for the ASR.
The one real-Whisper test is opt-in (SAYAGAIN_RUN_WHISPER=1) and uses a `tts·fixture` clip — never an eval row.
"""

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from referee.audit import audit
from referee.regex_extractor import extract
from referee.verdict import ABSTAIN, MISMATCH, R_AUDIO
from sayagain import asr as asr_mod
from sayagain.asr import ASRFailed, ASRResult, ASRUnavailable, NotebookASR, align_tokens, transcribe_turn
from sayagain.audio import SR, save
from sayagain.ingest import TurnAudio, cut, ingest_card

FIXTURES = Path(__file__).parent / "fixtures"


def two_bursts(seconds: float = 6.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    on = ((t > 0.4) & (t < 2.0)) | ((t > 3.2) & (t < 5.1))
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * on + 0.001 * np.random.default_rng(1).standard_normal(len(t))
    return x.astype(np.float32)


def test_ingest_one_wav_per_card_gives_turns_in_order(tmp_path: Path) -> None:
    p = tmp_path / "A2-031.wav"
    save(p, two_bursts())
    turns = ingest_card(p)
    assert [t.id for t in turns] == ["A2-031-0", "A2-031-1"]
    assert all(t.channel == "A2-031" for t in turns)
    assert abs(turns[0].t_start - 0.4) < 0.2 and abs(turns[1].t_end - 5.1) < 0.2
    assert turns[0].t_end < turns[1].t_start


def test_noisy_variant_is_cut_at_the_clean_boundaries() -> None:
    clean = two_bursts()
    noisy = clean + 0.2 * np.random.default_rng(2).standard_normal(len(clean)).astype(np.float32)
    a, b = cut(clean, "X"), cut(clean, "X", variant=noisy)
    assert [(t.t_start, t.t_end) for t in a] == [(t.t_start, t.t_end) for t in b]
    assert not np.allclose(a[0].samples, b[0].samples)
    with pytest.raises(ValueError):
        cut(clean, "X", variant=noisy[:-10])


def test_align_tokens_offsets_point_into_the_text() -> None:
    vocab = {1: " Copy", 2: ",", 3: " closing", 4: " two", 5: "-oh", 6: "-eight", 99: "<|en|>"}

    def decode(ids):
        return "".join(vocab[i] for i in ids)

    text, toks = align_tokens([99, 1, 2, 3, 4, 5, 6], [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -2.0], decode, {99})
    assert text == "Copy, closing two-oh-eight"
    assert [text[s:e] for s, e, _ in toks] == ["Copy", ",", " closing", " two", "-oh", "-eight"]
    assert toks[-1][2] == -2.0


def test_align_tokens_survives_a_merge_that_rewrites_earlier_text() -> None:
    # a byte-level piece decodes to a replacement char until its continuation arrives
    seq = {(1,): " t\ufffd", (1, 2): " to"}

    def decode(ids):
        return seq[tuple(ids)]

    text, toks = align_tokens([1, 2], [-0.1, -0.7], decode, set())
    assert text == "to" and toks == [(0, 1, -0.1), (1, 2, -0.7)]


def test_notebook_backend_is_off_until_the_human_reports_a_gpu(monkeypatch) -> None:
    monkeypatch.delenv(asr_mod.NOTEBOOK_ENV, raising=False)
    with pytest.raises(ASRUnavailable, match="HUMAN-TODO 2"):
        NotebookASR()


@pytest.mark.parametrize("gpu, on_amd", [("AMD Instinct MI300X", 1), ("NVIDIA A100", 0), ("", 0)])
def test_notebook_on_amd_comes_from_the_service_report(gpu: str, on_amd: int) -> None:
    def h(req):
        return httpx.Response(
            200, json={"text": " Copy, closing two-oh-eight.", "tokens": None, "language": "en", "gpu": gpu}
        )

    nb = NotebookASR("https://notebook.example/asr", transport=httpx.MockTransport(h))
    res = nb.transcribe(np.zeros(SR, dtype=np.float32), language="en")
    assert res.text == "Copy, closing two-oh-eight." and res.receipt["asr_on_amd"] == on_amd


def test_notebook_failure_fails_closed() -> None:
    nb = NotebookASR("https://notebook.example/asr", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(ASRFailed):
        nb.transcribe(np.zeros(SR, dtype=np.float32))
    turn = TurnAudio("X-0", "X", 0.0, 1.0, np.zeros(SR, dtype=np.float32))
    t, recs = transcribe_turn(nb, turn)
    assert t.not_audited == "asr error" and recs and recs[0]["ok"] is False


class FakeASR:
    """Returns scripted text per (turn id, prompted?) — stands in for Whisper in unit tests."""

    def __init__(self, texts: dict):
        self.texts = texts

    def transcribe(self, samples, *, language=None, prompt=None):
        key = (len(samples), prompt is not None)
        return ASRResult(self.texts[key], None, language or "en", {"stage": "asr", "asr_on_amd": 0, "ok": True})


def _turns(texts_prompted: list[str], texts_plain: list[str]):
    out = []
    for k, (p, u) in enumerate(zip(texts_prompted, texts_plain, strict=True)):
        n = SR + k  # distinct lengths key the fake
        fake = FakeASR({(n, True): p, (n, False): u})
        ta = TurnAudio(f"H-01-{k}", "H-01", 3.5 * k, 3.5 * k + 2.5, np.zeros(n, dtype=np.float32))
        t, recs = transcribe_turn(fake, ta, language="en")
        assert len(recs) == 2 and t.alt_transcript == u
        t.extraction = extract(t.transcript)
        out.append(t)
    return out


def test_disagreement_gate_end_to_end(registry) -> None:
    ins, rb = "Close valve two-oh-three, line Bravo.", "Copy, closing two-oh-eight, line Bravo."
    agree = _turns([ins, rb], [ins, rb])
    rep = audit(agree, registry, audio_gate="disagreement")
    assert rep.judged[0].verdict.verdict == MISMATCH
    # the vocabulary prompt "fixed" the read-back toward the instruction; the unprompted pass heard 208
    pulled = _turns([ins, "Copy, closing two-oh-three, line Bravo."], [ins, rb])
    rep = audit(pulled, registry, audio_gate="disagreement")
    v = rep.judged[0].verdict
    assert v.verdict == ABSTAIN and v.reason == R_AUDIO


def test_vocab_prompt_holds_no_tag_number() -> None:
    from referee import aliases as al

    toks = al.tokenize(asr_mod.VOCAB_PROMPT)
    assert al.tag_mentions(toks) == [] and not any(c.isdigit() for c in asr_mod.VOCAB_PROMPT)


@pytest.mark.skipif(os.environ.get("SAYAGAIN_RUN_WHISPER") != "1", reason="opt-in: needs Whisper weights (local)")
def test_local_whisper_on_a_tts_fixture(registry) -> None:
    """tts·fixture: macOS `say` voice reading the hero read-back. Proves the backend runs; never an eval row."""
    clip = FIXTURES / "tts-fixture-hero-readback.wav"
    meta = json.loads((FIXTURES / "tts-fixture-hero-readback.json").read_text())
    assert meta["badge"] == "tts·fixture"
    turns = ingest_card(clip)
    w = asr_mod.LocalWhisper()
    res = w.transcribe(np.concatenate([t.samples for t in turns]), language="en", prompt=asr_mod.VOCAB_PROMPT)
    assert res.receipt["asr_on_amd"] == 0 and res.receipt["badge"] == "asr·off-AMD"
    assert res.tokens and all(0 <= s < e <= len(res.text) and lp <= 0 for s, e, lp in res.tokens)
    assert "eight" in res.text.lower() or "208" in res.text
    # the whole path on the fixture: scripted instruction + Whisper read-back -> regex rung -> referee, floor gate
    from referee.turns import Turn

    ins_text = "Close valve two-oh-three, line Bravo."
    ins = Turn("H-0", "H", 0.0, 2.5, ins_text, extract(ins_text), tokens=[(0, len(ins_text), -0.01)])
    rb = Turn("H-1", "H", 3.5, 6.0, res.text, extract(res.text), tokens=res.tokens)
    v = audit([ins, rb], registry, floor=0.01, audio_gate="floor").judged[0].verdict
    assert v.label == "MISMATCH · tag: XV-203 ≠ XV-208"
