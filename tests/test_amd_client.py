"""Token Factory client against a SYNTHETIC fixture (OpenAI response shapes; not a recorded AMD response)."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from sayagain.amd_client import (
    Config,
    MissingCredentials,
    NotAuditing,
    TokenFactory,
    extract_or_not_audited,
    load_config,
)

FIX = json.loads((Path(__file__).parent / "fixtures" / "tokenfactory_synthetic.json").read_text())
CFG = Config(api_key="sk-test-not-a-real-key", base_url="https://tokenfactory.example.amd.com/v1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def client(handler, tmp_path: Path | None = None, **kw) -> TokenFactory:
    rp = tmp_path / "receipts.jsonl" if tmp_path else None
    return TokenFactory(CFG, transport=httpx.MockTransport(handler), receipts_path=rp, **kw)


def ok_handler(seen: list):
    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=FIX["models"])
        if req.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json=FIX["asr"], headers={"x-request-id": "req-asr"})
        body = json.loads(req.content)
        payload = FIX["logprobs"] if body.get("logprobs") else FIX["extract"]
        return httpx.Response(
            200, json=payload, headers={"x-request-id": "req-1", "x-ratelimit-remaining-requests": "59"}
        )

    return h


def test_extraction_request_is_structured_with_thinking_off(tmp_path: Path) -> None:
    seen: list = []
    tf = client(ok_handler(seen), tmp_path)
    obj, rec = tf.extract_spans("Copy, closing two-oh-eight, line Bravo.", "Qwen/Qwen3-32B")
    body = json.loads(seen[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0
    assert "registry" not in json.dumps(body["messages"]).lower()  # the registry lives in code, never in the prompt
    assert obj["commands"][0]["tag_span"] == "two-oh-eight"
    assert rec.params["schema_valid"] and rec.params["think_block_stripped"]


def test_receipt_fields_and_no_key_leak(tmp_path: Path) -> None:
    tf = client(ok_handler([]), tmp_path)
    tf.extract_spans("Copy, closing two-oh-eight, line Bravo.", "Qwen/Qwen3-32B")
    line = (tmp_path / "receipts.jsonl").read_text()
    rec = json.loads(line)
    for k in ("endpoint", "model", "request_id", "latency_s", "usage", "host", "on_amd", "rate_limit", "started_at"):
        assert k in rec
    assert rec["on_amd"] is True and rec["request_id"] == "req-1" and rec["usage"]["total_tokens"] == 222
    assert rec["rate_limit"] == {"x-ratelimit-remaining-requests": "59"}
    assert "sk-test" not in line


def test_no_think_soft_switch() -> None:
    seen: list = []
    tf = client(ok_handler(seen), thinking_off="no_think")
    tf.extract_spans("Copy.", "Qwen/Qwen3-32B")
    body = json.loads(seen[0].content)
    assert "chat_template_kwargs" not in body and body["messages"][-1]["content"].endswith("/no_think")


@pytest.mark.parametrize(
    "response, reason",
    [
        (httpx.Response(429, text="slow down"), "rate limited"),
        (httpx.Response(503), "server error"),
        (httpx.Response(400, text="bad schema"), "rejected (400)"),
    ],
)
def test_failures_are_not_auditing(response: httpx.Response, reason: str, tmp_path: Path) -> None:
    tf = client(lambda req: response, tmp_path)
    ext, why, rec = extract_or_not_audited(tf, "Copy.", "m")
    assert ext is None and why == reason and rec.ok is False
    assert (tmp_path / "receipts.jsonl").exists()


def test_timeout_is_not_auditing() -> None:
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(NotAuditing) as e:
        client(h).extract_spans("Copy.", "m")
    assert e.value.reason == "timeout"


def test_invalid_json_is_returned_for_the_referee_to_abstain() -> None:
    bad = {"choices": [{"message": {"content": "not json"}}]}
    obj, rec = client(lambda r: httpx.Response(200, json=bad)).extract_spans("Copy.", "m")
    assert "_invalid" in obj and rec.params["schema_valid"] is False


def test_transcribe_records_language_and_prompt() -> None:
    seen: list = []
    tf = client(ok_handler(seen))
    wav = Path(__file__).parent / "fixtures" / "tokenfactory_synthetic.json"  # any bytes: the mock ignores them
    out, rec = tf.transcribe(wav, "openai/whisper-large-v3", language="en", prompt="valve, line Bravo")
    assert out["text"].startswith("Copy") and rec.asr_language == "en" and rec.asr_prompt == "valve, line Bravo"


def test_missing_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AMD_TF_API_KEY", raising=False)
    monkeypatch.delenv("AMD_TF_BASE_URL", raising=False)
    with pytest.raises(MissingCredentials):
        load_config(tmp_path / "absent")
    (tmp_path / "env").write_text('export AMD_TF_API_KEY="k"\nAMD_TF_BASE_URL=https://x.amd.com/v1/\n')
    cfg = load_config(tmp_path / "env")
    assert cfg.api_key == "k" and cfg.base_url == "https://x.amd.com/v1" and cfg.on_amd


def test_off_amd_host_is_recorded_as_off_amd() -> None:
    assert not Config("k", "https://api.example.com/v1").on_amd
    assert not Config("k", "https://notamd.com/v1").on_amd


def test_access_gate_against_fixture(tmp_path: Path) -> None:
    import access_gate

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    gate = access_gate.run_gate(client(ok_handler([])), model=None, clip=clip)
    assert gate["models"]["audio"] == ["openai/whisper-large-v3"]
    assert gate["probe_model"] == "Qwen/Qwen3-32B"
    assert gate["extraction"]["ok"] and gate["extraction"]["grounded"]
    assert gate["logprobs"]["returned"] is True
    assert gate["asr"]["ok"] and gate["asr"]["segment_avg_logprob"] and not gate["asr"]["token_logprobs"]
    assert gate["asr"]["prompt_accepted"] is True
    assert gate["branch"].startswith("A")


def test_access_gate_without_key_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    import access_gate

    monkeypatch.setattr(access_gate, "load_config", lambda: (_ for _ in ()).throw(MissingCredentials("no key")))
    out = tmp_path / "gate.json"
    assert access_gate.main(["--out", str(out)]) == 3
    assert not out.exists()
