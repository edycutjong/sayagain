"""Coverage-closing tests for sayagain.amd_client, sayagain.extract and scripts/access_gate.py (all offline, mocked)."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from sayagain import amd_client, extract
from sayagain.amd_client import Config, NotAuditing, Receipt, TokenFactory, extract_or_not_audited, load_config
from sayagain.extract import Pacer, Rung, TokenFactoryExtractor, _retry_after

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import access_gate  # noqa: E402

FIX = json.loads((Path(__file__).parent / "fixtures" / "tokenfactory_synthetic.json").read_text())
CFG = Config(api_key="sk-test-not-a-real-key", base_url="https://tokenfactory.example.amd.com/v1")
HERO = "Copy, closing two-oh-eight, line Bravo."


def tf_for(handler, **kw) -> TokenFactory:
    return TokenFactory(CFG, transport=httpx.MockTransport(handler), **kw)


def chat_resp(content: str, **kw) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]}, **kw)


# ------------------------------------------------------------------------------------------ amd_client


def test_env_file_skips_blank_comment_and_keyless_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AMD_TF_API_KEY", raising=False)
    monkeypatch.delenv("AMD_TF_BASE_URL", raising=False)
    env = tmp_path / "env"
    env.write_text(
        "\n# comment\njunk line without equals\nAMD_TF_API_KEY='abc'\nAMD_TF_BASE_URL=https://h.amd.com/v1/\n"
    )
    cfg = load_config(env)
    assert (cfg.api_key, cfg.base_url) == ("abc", "https://h.amd.com/v1")


def test_close_closes_http_client() -> None:
    tf = tf_for(lambda r: httpx.Response(200, json={}))
    tf.close()
    assert tf._http.is_closed


def test_pacing_sleeps_between_calls(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(amd_client.time, "sleep", slept.append)
    tf = tf_for(lambda r: httpx.Response(200, json={"data": []}), min_interval_s=1000.0)
    tf.models()
    tf.models()
    assert len(slept) == 1 and 0 < slept[0] <= 1000.0


def test_network_error_is_not_auditing_and_recorded(tmp_path: Path) -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    tf = tf_for(boom, receipts_path=tmp_path / "r.jsonl")
    with pytest.raises(NotAuditing) as ei:
        tf.models()
    assert ei.value.reason == "network error" and ei.value.body == "ConnectError"
    line = json.loads((tmp_path / "r.jsonl").read_text().splitlines()[0])
    assert line["ok"] is False and line["error"] == "network error" and line["error_body"] == "ConnectError"


def test_usage_returns_dict_or_empty(tmp_path: Path) -> None:
    tf = tf_for(lambda r: httpx.Response(200, json={"spent": 1.5}), receipts_path=tmp_path / "r.jsonl")
    data, rec = tf.usage()
    assert data == {"spent": 1.5} and rec.stage == "usage" and rec.http_status == 200
    assert len(tf.receipts) == 1
    data2, _ = tf_for(lambda r: httpx.Response(200, json=[1, 2])).usage()
    assert data2 == {}


def test_extract_or_not_audited_success_and_failure() -> None:
    tf = tf_for(lambda r: httpx.Response(200, json=FIX["extract"]))
    obj, reason, rec = extract_or_not_audited(tf, HERO, "m")
    assert reason is None and obj["kind"] == "readback" and isinstance(rec, Receipt)
    bad = tf_for(lambda r: httpx.Response(429, text="slow"))
    obj, reason, rec = extract_or_not_audited(bad, HERO, "m")
    assert obj is None and reason == "rate limited" and rec.http_status == 429


# ------------------------------------------------------------------------------------------ extract


def test_retry_after_parsing() -> None:
    assert _retry_after(None) is None

    def rec(h: dict) -> Receipt:
        return Receipt("chat", "/x", "h", False, "m", None, 429, "t", 0.0, False, rate_limit=h)

    assert _retry_after(rec({})) is None
    assert _retry_after(rec({"retry-after": "soon"})) is None
    assert _retry_after(rec({"retry-after": "12"})) == 12.0
    assert _retry_after(rec({"retry-after": "9999"})) == extract.MAX_RETRY_AFTER_S
    assert _retry_after(rec({"retry-after": "-3"})) == 0.0


def test_extractor_constructor_validation() -> None:
    tf = tf_for(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="rung 0"):
        TokenFactoryExtractor(tf, extract.REGEX)
    with pytest.raises(ValueError, match="mode 'bogus'"):
        TokenFactoryExtractor(tf, Rung("r", "m"), mode="bogus")


def test_extractor_non_json_output_is_kept_invalid_and_pacer_is_used(tmp_path: Path) -> None:
    class CountingPacer:
        n = 0

        def wait(self) -> None:
            self.n += 1

    tf = tf_for(lambda r: chat_resp("sorry, no JSON here"), receipts_path=tmp_path / "r.jsonl")
    pacer = CountingPacer()
    ex = TokenFactoryExtractor(tf, Rung("r", "m"), pacer=pacer)  # type: ignore[arg-type]
    out = ex(HERO)
    assert pacer.n == 1 and out.attempts == 1 and out.not_audited is None
    assert out.obj == {"_invalid": "not JSON", "_content": "sorry, no JSON here"}
    assert out.receipt["params"]["schema_valid"] is False
    assert out.receipt["error"].startswith("schema:")


def test_extractor_non_object_json_is_invalid() -> None:
    out = TokenFactoryExtractor(tf_for(lambda r: chat_resp("[1, 2]")), Rung("r", "m"))(HERO)
    assert out.obj["_invalid"] == "not an object"


def test_extractor_retries_429_with_retry_after_then_gives_up() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="limit", headers={"retry-after": "7"})

    ex = TokenFactoryExtractor(tf_for(h), Rung("r", "m"), max_retries=2, sleep=sleeps.append)
    out = ex(HERO)
    assert out.not_audited == "rate limited" and out.obj is None and out.attempts == 3
    assert len(calls) == 3 and sleeps == [7.0, 7.0]


def test_pacer_sleeps_until_next_slot(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(extract.time, "sleep", slept.append)
    p = Pacer(500.0)
    p.wait()
    p.wait()
    assert len(slept) == 1 and 0 < slept[0] <= 500.0


def test_cache_returns_same_extraction_and_counts_hits() -> None:
    n: list[str] = []

    def fn(t: str) -> extract.Extraction:
        n.append(t)
        return extract.Extraction(t, {}, None)

    c = extract.Cache(fn)
    a, b = c("x"), c("x")
    assert a is b and n == ["x"] and c.hits == 1
    assert extract.regex_extraction(HERO).receipt is None


# ------------------------------------------------------------------------------------------ access_gate


def test_reasoning_tokens_shapes() -> None:
    assert access_gate._reasoning_tokens(None) is None
    assert access_gate._reasoning_tokens({"completion_tokens_details": {"reasoning_tokens": 5}}) == 5
    assert access_gate._reasoning_tokens({"reasoning_tokens": 3}) == 3


def _err_body(msg: str, status: int = 400) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": msg}})


def test_extraction_retries_soft_switch_when_thinking_kwarg_rejected() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if "chat_template_kwargs" in body:
            return _err_body("extra inputs are not permitted: chat_template_kwargs")
        return httpx.Response(200, json=FIX["extract"])

    ext = access_gate.probe_extraction(tf_for(h), "m")
    assert ext["ok"] and ext["thinking_off"] == "no_think" and ext["response_mode"] == "json_schema"
    assert ext["rejected"][0]["thinking_off"] == "chat_template_kwargs"


def test_extraction_stops_on_non_400_failure() -> None:
    ext = access_gate.probe_extraction(tf_for(lambda r: httpx.Response(429, text="limit")), "m")
    assert ext["ok"] is False
    assert [r["reason"] for r in ext["rejected"]] == ["rate limited"]


def test_extraction_all_modes_refused() -> None:
    ext = access_gate.probe_extraction(tf_for(lambda r: _err_body("nope")), "m")
    assert ext["ok"] is False
    assert [r["mode"] for r in ext["rejected"]] == list(amd_client.RESPONSE_MODES)


def test_extraction_reports_grounding_error() -> None:
    obj = {
        "kind": "readback",
        "commands": [{"action_span": "closing", "tag_span": "zzz-not-in-text", "state_span": None, "line_span": None}],
        "retracted_spans": [],
    }
    ext = access_gate.probe_extraction(tf_for(lambda r: chat_resp(json.dumps(obj))), "m")
    assert ext["ok"] and ext["schema_valid"] and ext["grounded"] is False and ext["grounding_error"]
    assert "canonical_ok" in ext and ext["canonical_ok"] is False


def test_logprobs_failure_is_reported() -> None:
    out = access_gate.probe_logprobs(tf_for(lambda r: httpx.Response(500, text="boom")), "m")
    assert out == {"ok": False, "reason": "server error", "status": 500, "body": "boom"}


def test_asr_probe_failure_and_rejected_prompt(tmp_path: Path) -> None:
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFF")
    fail = access_gate.probe_asr(tf_for(lambda r: httpx.Response(404, text="nf")), "w", clip)
    assert fail["ok"] is False and fail["status"] == 404 and fail["clip"] == "c.wav"

    def h(req: httpx.Request) -> httpx.Response:
        if b"valve" in req.content:
            return httpx.Response(400, text="no prompt")
        return httpx.Response(200, json={"text": "hi", "segments": [{"tokens_logprobs": [0.1]}]})

    out = access_gate.probe_asr(tf_for(h), "w", clip)
    assert out["ok"] and out["prompt_accepted"] is False and out["prompt_error"]["status"] == 400
    assert out["token_logprobs"] is True and out["segment_avg_logprob"] is False


def test_audio_route_answered() -> None:
    out = access_gate.probe_audio_route(tf_for(lambda r: httpx.Response(200, json={"text": ""})))
    assert out == {"status": 200, "answered": True, "text": ""}


def test_recommend_branch_without_audio() -> None:
    assert access_gate.recommend_branch({}).startswith("B or C")


def test_gate_models_failure_returns_early() -> None:
    gate = access_gate.run_gate(tf_for(lambda r: httpx.Response(500, text="x")), model=None, clip=None)
    assert gate["models"]["ok"] is False and gate["models"]["status"] == 500
    assert "extraction" not in gate


def test_gate_retries_transient_models_and_notes_rate_limit(monkeypatch, tmp_path: Path) -> None:
    slept: list[float] = []
    monkeypatch.setattr(access_gate.time, "sleep", slept.append)
    ext_calls: list[str] = []
    lp_calls: list[str] = []
    limited = {"ok": False, "rejected": [{"reason": "rate limited", "status": 429, "body": "20 req/min"}]}

    def fake_ext(tf, m):
        ext_calls.append(m)
        if len(ext_calls) == 1:
            return limited
        return {"ok": True, "canonical_ok": True, "thinking_off": "no_think", "rejected": []}

    def fake_lp(tf, m, thinking_off="chat_template_kwargs"):
        lp_calls.append(thinking_off)
        if len(lp_calls) == 1:
            return {"ok": False, "reason": "timeout"}
        return {"ok": True}

    monkeypatch.setattr(access_gate, "probe_extraction", fake_ext)
    monkeypatch.setattr(access_gate, "probe_logprobs", fake_lp)

    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "Qwen-X"}, {"id": "whisper-x"}]})

    gate = access_gate.run_gate(tf_for(h), model=None, clip=None, retry_after_s=0.5)
    assert slept == [0.5]
    assert gate["extraction"]["Qwen-X"]["retried"] is True and gate["logprobs"]["Qwen-X"]["retried"] is True
    assert lp_calls[-1] == "no_think"
    assert gate["ladder_candidates"] == ["Qwen-X"]
    assert gate["asr"] == {"ok": None, "reason": "audio model listed, no clip given to probe it"}


def test_gate_rate_limit_note_without_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        access_gate,
        "probe_extraction",
        lambda tf, m: {"ok": False, "rejected": [{"reason": "rate limited", "status": 429, "body": "b" * 300}]},
    )
    monkeypatch.setattr(access_gate, "probe_logprobs", lambda tf, m, t="x": {"ok": True})
    h = lambda r: httpx.Response(200, json={"data": [{"id": "Qwen-X"}]})  # noqa: E731
    gate = access_gate.run_gate(tf_for(h), model="Qwen-X", clip=None, retry_after_s=None)
    assert gate["rate_limit_note"] == "b" * 200 and gate["ladder_candidates"] == []


def test_main_writes_gate_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(access_gate, "load_config", lambda: CFG)
    monkeypatch.setattr(access_gate, "RECEIPTS", tmp_path / "r.jsonl")
    seen: dict = {}

    def fake_run(tf, *, model, clip, retry_after_s):
        seen.update(model=model, clip=clip, retry=retry_after_s, interval=tf.min_interval_s)
        return {"branch": "B", "x": 1}

    monkeypatch.setattr(access_gate, "run_gate", fake_run)
    out = tmp_path / "sub" / "gate.json"
    assert access_gate.main(["--out", str(out), "--model", "M", "--interval", "0", "--retry-after", "1"]) == 0
    assert json.loads(out.read_text()) == {"branch": "B", "x": 1}
    assert seen["model"] == "M" and seen["retry"] == 1.0 and seen["interval"] == 0.0
    assert "branch: B" in capsys.readouterr().out


def test_chat_sends_reasoning_effort_and_rung_describes_itself() -> None:
    bodies: list[dict] = []

    def h(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return chat_resp("ok")

    text, _, rec = tf_for(h, thinking_off=None).chat("m", [{"role": "user", "content": "hi"}], reasoning_effort="low")
    assert text == "ok" and bodies[0]["reasoning_effort"] == "low" and rec.params["reasoning_effort"] == "low"
    assert Rung("n", "m", "low", "on", 2.0, 99).describe() == {
        "name": "n",
        "model": "m",
        "reasoning_effort": "low",
        "thinking": "on",
        "params_b": 2.0,
        "max_tokens": 99,
    }
