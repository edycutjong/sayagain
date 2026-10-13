"""Access gate: what does the AMD Token Factory actually serve, and what does each call return?

    uv run python scripts/access_gate.py [--model ID] [--clip data/human/hero/H-01.wav]

Probes (each one receipted): GET /models · one JSON-schema extraction with thinking off (falls back to the /no_think
soft switch if chat_template_kwargs is rejected) · token log-probs on chat · usage token counts · rate-limit headers ·
an audio route, and whether it accepts a decoding `prompt`. Writes bench/results/access-gate.json.

Without credentials it writes nothing and exits 3: the key is created by a human, never invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee import coverage, grounding  # noqa: E402
from sayagain.amd_client import MissingCredentials, NotAuditing, TokenFactory, load_config  # noqa: E402

OUT = ROOT / "bench" / "results" / "access-gate.json"
RECEIPTS = ROOT / "bench" / "results" / "receipts" / "access-gate.jsonl"
HERO_RB = "Copy, closing two-oh-eight, line Bravo."
AUDIO_HINTS = ("whisper", "asr", "audio", "transcri", "speech")
INSTRUCT_HINTS = ("qwen", "llama", "instruct", "deepseek", "glm", "mistral", "gemma", "phi", "minicpm")


def classify_models(models: list[dict]) -> dict[str, list[str]]:
    ids = sorted(str(m.get("id", "")) for m in models if m.get("id"))
    audio = [i for i in ids if any(h in i.lower() for h in AUDIO_HINTS)]
    llm = [i for i in ids if i not in audio and any(h in i.lower() for h in INSTRUCT_HINTS)]
    return {"all": ids, "audio": audio, "instruct": llm}


def _err(e: NotAuditing) -> dict:
    return {"ok": False, "reason": e.reason, "status": e.status, "body": e.body}


def probe_extraction(tf: TokenFactory, model: str) -> dict:
    """One structured extraction on the hero read-back; thinking off via chat_template_kwargs, else /no_think."""
    for mode in ("chat_template_kwargs", "no_think"):
        tf.thinking_off = mode
        try:
            obj, rec = tf.extract_spans(HERO_RB, model)
        except NotAuditing as e:
            if e.status == 400 and mode == "chat_template_kwargs":
                continue  # the server rejected the kwarg: try the soft switch
            return {"thinking_off": mode, **_err(e)}
        out: dict = {
            "ok": True,
            "thinking_off": mode,
            "schema_valid": rec.params.get("schema_valid"),
            "think_block_stripped": rec.params.get("think_block_stripped"),
            "latency_s": rec.latency_s,
            "usage": rec.usage,
            "request_id": rec.request_id,
            "output": obj,
        }
        if rec.params.get("schema_valid"):
            try:
                g = grounding.ground(HERO_RB, obj)
                coverage.check(HERO_RB, g)
                out["grounded"] = True
            except (grounding.GroundingError, coverage.CoverageError) as e:
                out["grounded"], out["grounding_error"] = False, str(e)
        return out
    return {"ok": False, "reason": "thinking could not be turned off"}


def probe_logprobs(tf: TokenFactory, model: str) -> dict:
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


def recommend_branch(gate: dict) -> str:
    asr = gate.get("asr") or {}
    if asr.get("ok"):
        return "A (Token Factory serves an audio model)"
    return "B or C — no audio route on the Token Factory; depends on the notebook GPU check (human)"


def run_gate(tf: TokenFactory, *, model: str | None, clip: Path | None) -> dict:
    gate: dict = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": tf.config.host,
        "on_amd": tf.config.on_amd,
    }
    try:
        models, rec = tf.models()
        gate["models"] = classify_models(models)
        gate["models_request"] = {"latency_s": rec.latency_s, "rate_limit_headers": rec.rate_limit}
    except NotAuditing as e:
        gate["models"] = _err(e)
        return gate
    llm = model or (gate["models"]["instruct"] or gate["models"]["all"] or [None])[0]
    gate["probe_model"] = llm
    if llm:
        gate["extraction"] = probe_extraction(tf, llm)
        gate["logprobs"] = probe_logprobs(tf, llm)
    audio = gate["models"]["audio"]
    if audio and clip is not None and clip.exists():
        gate["asr"] = probe_asr(tf, audio[0], clip)
    elif audio:
        gate["asr"] = {"ok": None, "reason": "audio model listed, no clip given to probe it"}
    else:
        gate["asr"] = {"ok": False, "reason": "no audio model in /models"}
    gate["key_expiry"] = "not exposed by the API — check the Token Factory console"
    gate["branch"] = recommend_branch(gate)
    return gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="instruct model id to probe (default: first instruct model listed)")
    ap.add_argument("--clip", type=Path, default=ROOT / "data" / "human" / "hero" / "H-01.wav")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
    except MissingCredentials as e:
        print(f"SKIPPED: {e}. A human creates the Token Factory key; nothing was written.", file=sys.stderr)
        return 3
    tf = TokenFactory(cfg, receipts_path=RECEIPTS)
    try:
        gate = run_gate(tf, model=args.model, clip=args.clip)
    finally:
        tf.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"access gate -> {args.out} · branch: {gate.get('branch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
