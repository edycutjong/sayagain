"""Extraction on the AMD Token Factory: one turn's transcript -> spans in the span schema, one receipt per call.

    ext = TokenFactoryExtractor(client, rung)
    result = ext(transcript)   # Extraction(obj | None, not_audited | None, receipt)

Rules (bench/PROTOCOL.md §3, build-plan access gate):
  * JSON mode (`json_object`) with the schema in the system prompt — no Token Factory model accepts `json_schema` —
    and the schema is checked in code. Invalid output is kept as-is (never repaired): the referee abstains on it.
  * The prompt carries no registry content: the model only copies words; tags resolve in code.
  * Thinking: off where the model allows it (`reasoning_effort: none`, or a model that never thinks); a model that
    cannot turn it off is run at its lowest effort and labelled a thinking rung.
  * HTTP 429 -> wait (Retry-After, else a fixed backoff) and retry, at most `max_retries` times, then NOT_AUDITED.
    Timeouts, 5xx and malformed responses -> NOT_AUDITED at once. A NOT_AUDITED turn is never a verdict.
  * Every attempt is receipted by the client (`receipts_path`); the returned receipt adds the stage timings,
    the cost in USD from `usage.cost` and `llm_on_amd` (1 when the endpoint is AMD-operated — never a GPU claim).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from referee.regex_extractor import extract as regex_extract
from referee.schema import CANON_PROMPT, SYSTEM_PROMPT, validate_canonical, validate_extraction
from sayagain.amd_client import _FENCE, SCHEMA_IN_PROMPT, NotAuditing, Receipt, TokenFactory

BACKOFF_S = (5.0, 15.0, 30.0)


@dataclass(frozen=True)
class Rung:
    """One rung of the model ladder. `model` None = rung 0, the frozen regex extractor (no model, no call)."""

    name: str
    model: str | None = None
    reasoning_effort: str | None = None  # sent as-is; None sends nothing
    thinking: str = "none"  # "none" (model never thinks) | "off" | "on" (cannot be disabled on this gateway)
    params_b: float | None = None  # parameter count in billions when the model name states it, else None
    max_tokens: int = 512

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "thinking": self.thinking,
            "params_b": self.params_b,
            "max_tokens": self.max_tokens,
        }


REGEX = Rung("regex", None, thinking="none", params_b=0.0)


@dataclass
class Extraction:
    text: str
    obj: dict | None
    not_audited: str | None
    receipt: dict | None = None
    attempts: int = 0


class Pacer:
    """Spaces call starts by `interval_s` across threads (the gateway allows 20 requests per minute per account)."""

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval_s
        if start > now:
            time.sleep(start - now)


def _retry_after(rec: Receipt | None) -> float | None:
    if rec is None:
        return None
    v = (rec.rate_limit or {}).get("retry-after")
    try:
        return max(0.0, float(v)) if v is not None else None
    except ValueError:
        return None


def complete_receipt(rec: Receipt, rung: Rung, attempts: int) -> dict:
    d = rec.to_dict()
    usage = rec.usage or {}
    cost = usage.get("cost")
    details = usage.get("completion_tokens_details") or {}
    d.update(
        rung=rung.name,
        llm_on_amd=1 if rec.on_amd else 0,
        cost_usd=float(cost) if isinstance(cost, int | float) else None,
        reasoning_tokens=usage.get("reasoning_tokens", details.get("reasoning_tokens")),
        stage_timings={"extract_s": rec.latency_s},
        attempts=attempts,
    )
    return d


class TokenFactoryExtractor:
    """Calls one Token Factory model per turn. `mode`: "spans" (the product) or "canonical" (the grounding ablation:
    the model returns normalised values instead of copied words)."""

    def __init__(
        self,
        client: TokenFactory,
        rung: Rung,
        *,
        pacer: Pacer | None = None,
        mode: str = "spans",
        max_retries: int = len(BACKOFF_S),
        backoff_s: tuple[float, ...] = BACKOFF_S,
        sleep=time.sleep,
    ):
        if rung.model is None:
            raise ValueError("rung 0 makes no call")
        if mode not in ("spans", "canonical"):
            raise ValueError(f"mode {mode!r}")
        self.client, self.rung, self.pacer, self.mode = client, rung, pacer, mode
        self.max_retries, self.backoff_s, self.sleep = max_retries, backoff_s, sleep

    def _once(self, text: str) -> tuple[dict, Receipt]:
        system = (SYSTEM_PROMPT + SCHEMA_IN_PROMPT) if self.mode == "spans" else CANON_PROMPT
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"Transcript: {text}"}]
        content, _, rec = self.client.chat(
            self.rung.model,  # type: ignore[arg-type]
            msgs,
            schema={},  # JSON mode; the schema itself travels in the prompt and is checked below
            response_mode="json_object",
            temperature=0.0,
            max_tokens=self.rung.max_tokens,
            stage="extract" if self.mode == "spans" else "extract-canonical",
            record=False,
            reasoning_effort=self.rung.reasoning_effort,
        )
        content = _FENCE.sub("", content).strip()
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            obj = {"_invalid": "not JSON", "_content": content[:200]}
        problem = validate_extraction(obj) if self.mode == "spans" else validate_canonical(obj)
        rec.params["schema_valid"] = problem is None
        rec.params["mode"] = self.mode
        if problem:
            rec.error = f"schema: {problem}"
        self.client._record(rec)
        return obj, rec

    def __call__(self, text: str) -> Extraction:
        attempts = 0
        while True:
            attempts += 1
            if self.pacer is not None:
                self.pacer.wait()
            try:
                obj, rec = self._once(text)
                return Extraction(text, obj, None, complete_receipt(rec, self.rung, attempts), attempts)
            except NotAuditing as e:
                if e.status == 429 and attempts <= self.max_retries:
                    wait = _retry_after(e.receipt)
                    self.sleep(wait if wait is not None else self.backoff_s[min(attempts - 1, len(self.backoff_s) - 1)])
                    continue
                rec = complete_receipt(e.receipt, self.rung, attempts) if e.receipt is not None else None
                return Extraction(text, None, e.reason, rec, attempts)


def regex_extraction(text: str) -> Extraction:
    """Rung 0: the frozen regex + alias extractor. No call, no receipt."""
    return Extraction(text, regex_extract(text), None, None, 0)


@dataclass
class Cache:
    """One extraction per distinct transcript within a run (cards repeat words such as "Ulangi?")."""

    fn: object
    hits: int = 0
    store: dict[str, Extraction] = field(default_factory=dict)

    def __call__(self, text: str) -> Extraction:
        if text in self.store:
            self.hits += 1
            return self.store[text]
        out = self.fn(text)  # type: ignore[operator]
        self.store[text] = out
        return out
