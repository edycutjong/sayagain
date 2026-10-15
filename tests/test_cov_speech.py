"""Coverage for the speech group: LocalWhisper (fake torch/transformers, no weights), the backend factory, the
notebook non-200 path, audio/VAD/ingest edge cases and the kill-test CLI end to end on fakes (no model, no network)."""

import json
import math
import sys
import types
from pathlib import Path

import httpx
import numpy as np
import pytest

from sayagain import asr as asr_mod
from sayagain.asr import ASRFailed, ASRUnavailable, LocalWhisper, NotebookASR, backend
from sayagain.audio import SR, active_rms, save
from sayagain.ingest import TurnAudio
from sayagain.vad import segments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import kill_test_local as kt  # noqa: E402

VOCAB = 200
TOK = {
    "<|startoftranscript|>": 1,
    "<|en|>": 2,
    "<|id|>": 3,
    "<|transcribe|>": 4,
    "<|notimestamps|>": 5,
    "<|0.00|>": 100,
}
PIECES = {10: " Close", 11: " valve", 12: " two"}


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, k):
        r = self.a[k]
        return FakeTensor(r) if isinstance(r, np.ndarray) and r.ndim else r

    def __int__(self):
        return int(self.a)

    def tolist(self):
        return self.a.tolist()

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, *a, **k):
        return self

    @property
    def shape(self):
        return self.a.shape

    def gather(self, dim, idx):
        return FakeTensor(np.take_along_axis(self.a, idx.a, dim))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_torch(mps: bool) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.float16, t.float32 = "float16", "float32"
    t.tensor = lambda data, device=None: FakeTensor(data)
    t.no_grad = _NoGrad

    def log_softmax(x, dim):
        z = x.a - x.a.max(axis=dim, keepdims=True)
        return FakeTensor(z - np.log(np.exp(z).sum(axis=dim, keepdims=True)))

    t.log_softmax = log_softmax
    return t


class FakeTokenizer:
    all_special_ids = [1, 2, 3, 4, 5]

    def convert_tokens_to_ids(self, tok):
        return TOK[tok]

    def convert_ids_to_tokens(self, i):
        return {2: "<|en|>", 3: "<|id|>"}[int(i)]

    def decode(self, ids):
        return "".join(PIECES.get(i, "") for i in ids)


class FakeProc:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def __call__(self, samples, sampling_rate, return_tensors):
        self.calls.append((len(samples), sampling_rate, return_tensors))
        return types.SimpleNamespace(input_features=FakeTensor(np.zeros((1, 3))))

    def get_prompt_ids(self, prompt):
        return [7, 8]


class FakeModel:
    dtype = "fake-dtype"

    def __init__(self, fail=False):
        self.fail, self.generate_kw, self.forward_ids, self.moved, self.evaled = fail, None, None, None, False

    def to(self, device):
        self.moved = device
        return self

    def eval(self):
        self.evaled = True

    def generate(self, feats, **kw):
        if self.fail:
            raise RuntimeError("decoder blew up")
        self.generate_kw = kw
        return FakeTensor([[10, 11, 5, 12, 100]])  # text, special, text, timestamp: the last two kinds are skipped

    def detect_language(self, feats):
        return FakeTensor([3])

    def __call__(self, input_features, decoder_input_ids):
        self.forward_ids = decoder_input_ids.a.tolist()
        return types.SimpleNamespace(logits=FakeTensor(np.zeros((1, decoder_input_ids.shape[1], VOCAB))))


@pytest.fixture
def fakes(monkeypatch):
    state = types.SimpleNamespace(proc=FakeProc(), model=FakeModel(), loaded=[])
    tf = types.ModuleType("transformers")

    class WP:
        @staticmethod
        def from_pretrained(mid):
            state.loaded.append(("proc", mid))
            return state.proc

    class WM:
        @staticmethod
        def from_pretrained(mid, dtype):
            state.loaded.append(("model", mid, dtype))
            return state.model

    tf.WhisperProcessor, tf.WhisperForConditionalGeneration = WP, WM
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False))
    state.tf = tf
    return state


def test_local_whisper_prompted_english_cpu(fakes) -> None:
    w = LocalWhisper()
    res = w.transcribe(np.zeros(SR, dtype=np.float32), language="en", prompt="radio words")
    assert res.text == "Close valve two"
    assert res.language == "en"
    assert fakes.loaded == [("proc", "openai/whisper-large-v3"), ("model", "openai/whisper-large-v3", "float32")]
    assert w.device == "cpu" and fakes.model.moved == "cpu" and fakes.model.evaled
    kw = fakes.model.generate_kw
    assert {k: v for k, v in kw.items() if k != "prompt_ids"} == {"task": "transcribe", "language": "en"}
    assert kw["prompt_ids"].tolist() == [7, 8]
    # decoder context: prompt ids + SOT, language, task, notimestamps, then the generated tokens, minus the last one
    assert fakes.model.forward_ids == [[7, 8, 1, 2, 4, 5, 10, 11, 5, 12]]
    # timestamp / special generated tokens are dropped; the three text tokens carry teacher-forced log-probs
    assert res.tokens == [
        (0, 5, pytest.approx(-math.log(VOCAB))),
        (5, 11, pytest.approx(-math.log(VOCAB))),
        (11, 15, pytest.approx(-math.log(VOCAB))),
    ]
    r = res.receipt
    assert r["ok"] is True and r["asr_on_amd"] == 0 and r["badge"] == "asr·off-AMD"
    assert r["device"] == "cpu" and r["detected_language"] == "en" and r["n_tokens"] == 3
    assert r["asr_language"] == "en" and r["asr_prompt"] == "radio words"
    assert fakes.proc.calls == [(SR, SR, "pt")]


def test_local_whisper_autodetect_no_prompt_reuses_loaded_model(fakes, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    w = LocalWhisper(model="m/x")
    res = w.transcribe(np.zeros(SR, dtype=np.float32))
    assert w.device == "mps" and fakes.loaded[1] == ("model", "m/x", "float16")
    assert res.language == "id" and res.receipt["detected_language"] == "id"
    assert res.receipt["asr_language"] == "auto" and res.receipt["asr_prompt"] is None
    assert fakes.model.generate_kw == {"task": "transcribe"}
    assert fakes.model.forward_ids[0][:4] == [1, 3, 4, 5]  # detected language id (id) in the forced prefix
    w.transcribe(np.zeros(SR, dtype=np.float32))
    assert len(fakes.loaded) == 2  # the second call did not reload


def test_local_whisper_explicit_device_is_kept(fakes) -> None:
    w = LocalWhisper(device="cpu")
    w.transcribe(np.zeros(SR, dtype=np.float32), language="en")
    assert w.device == "cpu"


def test_local_whisper_missing_extra_is_unavailable_not_failed(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` -> ImportError
    with pytest.raises(ASRUnavailable, match="killtest"):
        LocalWhisper().transcribe(np.zeros(SR, dtype=np.float32))


def test_local_whisper_crash_fails_closed_with_receipt(fakes) -> None:
    fakes.model.fail = True
    with pytest.raises(ASRFailed) as ei:
        LocalWhisper().transcribe(np.zeros(SR, dtype=np.float32), language="en")
    assert ei.value.reason == "asr error"
    rec = ei.value.receipt
    assert rec["ok"] is False and rec["error"] == "RuntimeError: decoder blew up" and rec["asr_on_amd"] == 0
    assert "latency_s" in rec


def test_backend_factory(fakes) -> None:
    lw = backend("local-whisper", model="m/y", device="cpu")
    assert isinstance(lw, LocalWhisper) and lw.model_id == "m/y" and lw.device == "cpu"
    nb = backend("notebook", url="http://nb.invalid/asr", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert isinstance(nb, NotebookASR) and nb.url == "http://nb.invalid/asr"
    with pytest.raises(ValueError, match="unknown ASR backend 'nope'"):
        backend("nope")


def test_notebook_non_200_is_asr_failed_with_status() -> None:
    nb = NotebookASR("http://nb.invalid/asr", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(ASRFailed) as ei:
        nb.transcribe(np.zeros(SR // 10, dtype=np.float32))
    rec = ei.value.receipt
    assert rec["http_status"] == 503 and rec["ok"] is False and rec["error"] == "ValueError: status 503"


def test_notebook_non_string_text_is_asr_failed() -> None:
    nb = NotebookASR(
        "http://nb.invalid/asr", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": 42}))
    )
    with pytest.raises(ASRFailed) as ei:
        nb.transcribe(np.zeros(SR // 10, dtype=np.float32))
    assert ei.value.receipt["error"] == "TypeError: text is not a string"
    assert ei.value.receipt["http_status"] == 200


def test_active_rms_of_silence_is_zero() -> None:
    assert active_rms(np.zeros(SR, dtype=np.float32)) == 0.0


def test_vad_shorter_than_one_frame_has_no_segments() -> None:
    assert segments(np.ones(10, dtype=np.float32)) == []


def test_turn_audio_duration() -> None:
    t = TurnAudio("X-0", "X", 1.25, 3.0, np.zeros(4, dtype=np.float32))
    assert t.duration_s == pytest.approx(1.75)


def test_read_turn_without_a_command_is_none(registry) -> None:
    assert kt.read_turn("Roger, copy that.", registry) is None


def _bursts() -> np.ndarray:
    t = np.arange(int(SR * 6.0)) / SR
    on = ((t > 0.4) & (t < 2.0)) | ((t > 3.2) & (t < 5.1))
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * on + 0.001 * np.random.default_rng(1).standard_normal(len(t))
    return x.astype(np.float32)


def _kt_setup(tmp_path: Path, monkeypatch, texts: list[str]):
    clips = tmp_path / "clips"
    save(clips / "KT-01.wav", _bursts())
    cards = tmp_path / "cards.jsonl"
    line = next(ln for ln in (ROOT / "data" / "cards.jsonl").read_text().splitlines() if '"KT-01"' in ln)
    cards.write_text(line + "\n", encoding="utf-8")
    noise = tmp_path / "noise"
    save(noise / "n.wav", np.random.default_rng(3).standard_normal(SR * 2).astype(np.float32) * 0.1)
    calls = []
    tf = types.ModuleType("transformers")

    def pipeline(task, model, device, torch_dtype):
        calls.append(("pipeline", task, model, device, torch_dtype))

        def run(inp, generate_kwargs):
            calls.append(("run", len(inp["raw"]), inp["sampling_rate"], generate_kwargs))
            return {"text": f"  {texts[sum(c[0] == 'run' for c in calls) % 2 - 1]}  "}

        return run

    tf.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True))
    return clips, cards, noise, calls


def test_kill_test_main_runs_modes_and_writes_verdict(tmp_path: Path, monkeypatch, capsys) -> None:
    card = json.loads(next(ln for ln in (ROOT / "data" / "cards.jsonl").read_text().splitlines() if '"KT-01"' in ln))
    i, r = (t["text"] for t in card["turns"])
    clips, cards, noise, calls = _kt_setup(tmp_path, monkeypatch, [i, r])
    out = tmp_path / "out" / "killtest.json"
    argv = ["--clips", str(clips), "--cards", str(cards), "--noise", str(noise), "--out", str(out)]
    rc = kt.main([*argv, "--modes", "en", "auto"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["device"] == "mps" and res["on_amd"] is False and res["model"] == "openai/whisper-large-v3"
    assert calls[0] == ("pipeline", "automatic-speech-recognition", "openai/whisper-large-v3", "mps", "float16")
    kws = [c[3] for c in calls if c[0] == "run"]
    assert {"task": "transcribe", "language": "en"} in kws and {"task": "transcribe"} in kws
    assert set(res["modes"]) == {"en", "auto"}
    for mode in res["modes"].values():
        assert set(mode["rows"]) == {"clean", kt.mix.snr_label("0")}
        clean = mode["rows"]["clean"]
        assert clean["n_exchanges"] == 1 and clean["n_turns"] == 2 and clean["p50_s_per_turn"] is not None
        assert clean["tag_exact"] == 1.0 and clean["paired"] == 1.0
        assert mode["verdict"]["decision"] == "go"
    printed = capsys.readouterr().out
    assert "en go" in printed and f"-> {out}" in printed


def test_kill_test_main_needs_noise_for_non_clean_snr(tmp_path: Path, monkeypatch, capsys) -> None:
    clips, cards, noise, _ = _kt_setup(tmp_path, monkeypatch, ["a", "b"])
    empty = tmp_path / "no-noise"
    empty.mkdir()
    out = tmp_path / "killtest.json"
    rc = kt.main(["--clips", str(clips), "--cards", str(cards), "--noise", str(empty), "--out", str(out)])
    assert rc == 2 and not out.exists()
    assert "fetch_mimii" in capsys.readouterr().err


def test_kill_test_main_without_clips_exits_3(tmp_path: Path, capsys) -> None:
    out = tmp_path / "killtest.json"
    assert kt.main(["--clips", str(tmp_path / "none"), "--out", str(out)]) == 3
    assert "SKIPPED" in capsys.readouterr().err and not out.exists()


def test_kill_test_main_clean_only_skips_noise_pool(tmp_path: Path, monkeypatch) -> None:
    clips, cards, noise, calls = _kt_setup(tmp_path, monkeypatch, ["", ""])
    out = tmp_path / "k.json"
    rc = kt.main(
        ["--clips", str(clips), "--cards", str(cards), "--noise", str(tmp_path / "x"), "--out", str(out)]
        + ["--snr", "clean", "--modes", "id"]
    )
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    row = res["modes"]["id"]["rows"]["clean"]
    assert row["tag_exact"] == 0.0 and row["paired"] == 0.0  # empty transcripts never count as a read tag
    assert res["modes"]["id"]["verdict"]["decision"].startswith("stop")
    assert asr_mod.NOTEBOOK_ENV == "SAYAGAIN_NOTEBOOK_ASR_URL"
