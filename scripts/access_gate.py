"""Access gate: what does the AMD Token Factory actually serve, and what does each call return?

    uv run python scripts/access_gate.py [--model ID] [--clip data/human/hero/H-01.wav]

Probes (each one receipted): GET /models · one JSON-schema extraction with thinking off (falls back to the /no_think
soft switch if chat_template_kwargs is rejected), per instruct model, falling back from server-enforced JSON schema to
JSON mode to prompt-only · token log-probs on chat · usage token counts · rate-limit headers · an audio route, and
whether it accepts a decoding `prompt`. Writes bench/results/access-gate.json.

Without credentials it writes nothing and exits 3: the key is created by a human, never invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee import coverage, grounding  # noqa: E402
from referee.registry import Registry  # noqa: E402
from referee.turns import Turn, analyze  # noqa: E402
from sayagain.amd_client import RESPONSE_MODES, MissingCredentials, NotAuditing, TokenFactory, load_config  # noqa: E402

OUT = ROOT / "bench" / "results" / "access-gate.json"
RECEIPTS = ROOT / "bench" / "results" / "receipts" / "access-gate.jsonl"
HERO_RB = "Copy, closing two-oh-eight, line Bravo."
HERO_RB_CANON = ("CLOSE", "XV-208", "B")
AUDIO_HINTS = ("whisper", "asr", "audio", "transcri", "speech")
INSTRUCT_HINTS = ("qwen", "llama", "instruct", "deepseek", "glm", "mistral", "gemma", "phi", "minicpm", "mimo")


def classify_models(models: list[dict]) -> dict[str, list[str]]:
    ids = sorted(str(m.get("id", "")) for m in models if m.get("id"))
    audio = [i for i in ids if any(h in i.lower() for h in AUDIO_HINTS)]
    llm = [i for i in ids if i not in audio and any(h in i.lower() for h in INSTRUCT_HINTS)]
    return {"all": ids, "audio": audio, "instruct": llm}


def _err(e: NotAuditing) -> dict:
    return {"ok": False, "reason": e.reason, "status": e.status, "body": e.body}


def _reasoning_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    det = usage.get("completion_tokens_details") or {}
    return det.get("reasoning_tokens", usage.get("reasoning_tokens"))


def _kwarg_rejected(e: NotAuditing) -> bool:
    body = e.body.lower()
    return e.status == 400 and any(
        k in body for k in ("chat_template", "enable_thinking", "extra_forbidden", "extra inputs")
    )


def probe_extraction(tf: TokenFactory, model: str) -> dict:
    """One extraction of the hero read-back per model: the strongest output mode the model accepts
    (server-enforced JSON schema, else JSON mode, else prompt-only), thinking off (chat_template_kwargs, else the
    /no_think soft switch). Records every rejection, schema validity, grounding, latency and reasoning tokens."""
    rejected: list[dict] = []
    for mode in RESPONSE_MODES:
        for think in ("chat_template_kwargs", "no_think"):
            tf.thinking_off = think
            try:
                obj, rec = tf.extract_spans(HERO_RB, model, response_mode=mode)
            except NotAuditing as e:
                rejected.append({"mode": mode, "thinking_off": think, **_err(e)})
                if _kwarg_rejected(e):
                    continue  # only the thinking kwarg was refused: retry this mode with the soft switch
                if e.status == 400:
                    break  # this output mode is refused: next mode
                return {"ok": False, "rejected": rejected}
            out: dict = {
                "ok": True,
                "response_mode": mode,
                "thinking_off": think,
                "schema_valid": rec.params.get("schema_valid"),
                "think_block_stripped": rec.params.get("think_block_stripped"),
                "reasoning_tokens": _reasoning_tokens(rec.usage),
                "latency_s": rec.latency_s,
                "usage": rec.usage,
                "request_id": rec.request_id,
                "output": obj,
                "rejected": rejected,
            }
            if rec.params.get("schema_valid"):
                try:
                    g = grounding.ground(HERO_RB, obj)
                    coverage.check(HERO_RB, g)
                    out["grounded"] = True
                except (grounding.GroundingError, coverage.CoverageError) as e:
                    out["grounded"], out["grounding_error"] = False, str(e)
                # does the referee read the expected tuple from it? (CLOSE, XV-208, line B)
                a = analyze(Turn("gate", "gate", 0, 1, HERO_RB, obj), Registry.load())
                got = [(c.action, c.tag, c.line) for c in a.commands] if a.status == "ok" else [a.reason]
                out["canonical"] = got
                out["canonical_ok"] = got == [HERO_RB_CANON]
            return out
    return {"ok": False, "rejected": rejected}


def probe_logprobs(tf: TokenFactory, model: str, thinking_off: str = "chat_template_kwargs") -> dict:
    tf.thinking_off = thinking_off
    try:
        _, raw, rec = tf.chat(
            model, [{"role": "user", "content": "Reply with the single word: copy"}], logprobs=True, max_tokens=8
        )
    except NotAuditing as e:
        return _err(e)
    lp = (raw.get("choices") or [{}])[0].get("logprobs")
    return {
        "ok": True,
        "returned": bool(lp and lp.get("content")),
        "usage_present": rec.usage is not None,
        "rate_limit_headers": rec.rate_limit,
    }


def probe_asr(tf: TokenFactory, model: str, clip: Path) -> dict:
    out: dict = {"model": model, "clip": clip.name}
    try:
        res, rec = tf.transcribe(clip, model, language="en")
        segs = res.get("segments") or []
        out.update(
            ok=True,
            text=res.get("text"),
            latency_s=rec.latency_s,
            segment_avg_logprob=bool(segs and "avg_logprob" in segs[0]),
            token_logprobs=bool(segs and any("tokens_logprobs" in s or "logprobs" in s for s in segs)),
        )
    except NotAuditing as e:
        return {**out, **_err(e)}
    try:
        res2, _ = tf.transcribe(clip, model, language="en", prompt="valve, pump, line Alpha, Bravo, two-oh-three")
        out["prompt_accepted"] = True
        out["prompted_text"] = res2.get("text")
    except NotAuditing as e:
        out["prompt_accepted"], out["prompt_error"] = False, _err(e)
    return out


def probe_audio_route(tf: TokenFactory) -> dict:
    """Is there an audio route at all? One POST of 0.5 s of silence (a probe, not speech) to /audio/transcriptions."""
    import tempfile
    import wave

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "silence.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 8000)
        try:
            res, rec = tf.transcribe(wav, "whisper-1", language="en")
            return {"status": rec.http_status, "answered": True, "text": res.get("text")}
        except NotAuditing as e:
            return {"status": e.status, "answered": False, "reason": e.reason, "body": e.body[:200]}


def recommend_branch(gate: dict) -> str:
    asr = gate.get("asr") or {}
    if asr.get("ok"):
        return "A (Token Factory serves an audio model)"
    return "B or C — no audio route on the Token Factory; depends on the notebook GPU check (human)"


TRANSIENT = ("rate limited", "timeout", "network error", "server error")


def _transient(ext: dict) -> bool:
    return not ext.get("ok") and any(r.get("reason") in TRANSIENT for r in ext.get("rejected", [])[-1:])


def run_gate(tf: TokenFactory, *, model: str | None, clip: Path | None, retry_after_s: float | None = None) -> dict:
    gate: dict = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": tf.config.host,
        "base_path": urlparse(tf.config.base_url).path,
        "on_amd": tf.config.on_amd,
    }
    try:
        models, rec = tf.models()
        gate["models"] = classify_models(models)
        gate["models_request"] = {"latency_s": rec.latency_s, "rate_limit_headers": rec.rate_limit}
    except NotAuditing as e:
        gate["models"] = _err(e)
        return gate
    probe = [model] if model else gate["models"]["instruct"]
    gate["extraction"] = {}
    gate["logprobs"] = {}
    for m in probe:
        ext = probe_extraction(tf, m)
        gate["extraction"][m] = ext
        gate["logprobs"][m] = probe_logprobs(tf, m, ext.get("thinking_off", "chat_template_kwargs"))
    # one retry pass for models that only failed on the rate limit or a timeout (not on a refusal)
    transient = [
        m for m in probe if _transient(gate["extraction"][m]) or gate["logprobs"][m].get("reason") in TRANSIENT
    ]
    if transient and retry_after_s is not None:
        time.sleep(retry_after_s)
        for m in transient:
            if _transient(gate["extraction"][m]):
                gate["extraction"][m] = {**probe_extraction(tf, m), "retried": True}
            if gate["logprobs"][m].get("reason") in TRANSIENT:
                gate["logprobs"][m] = {
                    **probe_logprobs(tf, m, gate["extraction"][m].get("thinking_off", "chat_template_kwargs")),
                    "retried": True,
                }
    limits = {
        r.get("body", "") for e in gate["extraction"].values() for r in e.get("rejected", []) if r.get("status") == 429
    }
    if limits:
        gate["rate_limit_note"] = sorted(limits)[0][:200]
    gate["ladder_candidates"] = sorted(
        m for m, e in gate["extraction"].items() if e.get("ok") and e.get("canonical_ok")
    )
    audio = gate["models"]["audio"]
    if audio and clip is not None and clip.exists():
        gate["asr"] = probe_asr(tf, audio[0], clip)
    elif audio:
        gate["asr"] = {"ok": None, "reason": "audio model listed, no clip given to probe it"}
    else:
        gate["asr"] = {"ok": False, "reason": "no audio model in /models", "route": probe_audio_route(tf)}
    gate["key_expiry"] = "not exposed by the API — check the Token Factory console"
    gate["branch"] = recommend_branch(gate)
    return gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="instruct model id to probe (default: first instruct model listed)")
    ap.add_argument("--clip", type=Path, default=ROOT / "data" / "human" / "hero" / "H-01.wav")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--interval", type=float, default=3.2, help="seconds between calls (unpaced bursts drew HTTP 429)")
    ap.add_argument("--retry-after", type=float, default=65.0, help="cool-down before re-probing rate-limited models")
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
    except MissingCredentials as e:
        print(f"SKIPPED: {e}. A human creates the Token Factory key; nothing was written.", file=sys.stderr)
        return 3
    tf = TokenFactory(cfg, receipts_path=RECEIPTS, min_interval_s=args.interval)
    try:
        gate = run_gate(tf, model=args.model, clip=args.clip, retry_after_s=args.retry_after)
    finally:
        tf.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"access gate -> {args.out} · branch: {gate.get('branch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
