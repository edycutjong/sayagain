"""Client for the AMD Token Factory (OpenAI-compatible API). Every call writes a receipt.

Credentials come only from the environment or `~/.config/amd-tokenfactory/env` (`AMD_TF_API_KEY`, `AMD_TF_BASE_URL`);
nothing is read from or written to the repository, and the key never appears in a receipt or a log line.

Failure policy (fail closed): HTTP 429, timeouts, network errors, 5xx and malformed responses raise `NotAuditing`; the
pipeline records the exchange as NOT_AUDITED — never as a verdict.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from referee.schema import SPAN_SCHEMA, SYSTEM_PROMPT, validate_extraction

ENV_PATH = Path.home() / ".config" / "amd-tokenfactory" / "env"
AMD_DOMAINS = ("amd.com", "amd.com.cn")
RESPONSE_MODES = ("json_schema", "json_object", "prompt")
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
SCHEMA_IN_PROMPT = (
    "\nReturn only one JSON object, no prose, with exactly these keys: "
    '{"kind": "instruction"|"readback"|"say_again"|"other", '
    '"commands": [{"action_span": string|null, "tag_span": string|null, "state_span": string|null, '
    '"line_span": string|null}], "retracted_spans": [string]}'
)
_RATE_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)


class MissingCredentials(Exception):
    """No Token Factory key: the human creates it (HUMAN-TODO) — never fabricated."""


class NotAuditing(Exception):
    def __init__(self, reason: str, receipt: Receipt | None = None, status: int | None = None, body: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.receipt = receipt
        self.status = status
        self.body = body[:500]


@dataclass
class Config:
    api_key: str = field(repr=False)  # never in a repr, a log line or an assertion diff
    base_url: str = ""

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or ""

    @property
    def on_amd(self) -> bool:
        """Served by an AMD-operated endpoint (amd.com / amd.com.cn). It says where the API is hosted — not which GPU
        ran the model, which the API does not expose."""
        return any(self.host == d or self.host.endswith("." + d) for d in AMD_DOMAINS)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


def load_config(path: Path = ENV_PATH) -> Config:
    vals = _parse_env_file(path) if path.exists() else {}
    key = os.environ.get("AMD_TF_API_KEY") or vals.get("AMD_TF_API_KEY")
    base = os.environ.get("AMD_TF_BASE_URL") or vals.get("AMD_TF_BASE_URL")
    if not key or not base:
        raise MissingCredentials(f"AMD_TF_API_KEY / AMD_TF_BASE_URL not set (env or {path})")
    return Config(api_key=key, base_url=base.rstrip("/"))


@dataclass
class Receipt:
    stage: str  # "models" | "extract" | "chat" | "asr"
    endpoint: str  # path only, e.g. "/chat/completions"
    host: str
    on_amd: bool
    model: str | None
    request_id: str | None
    http_status: int | None
    started_at: str
    latency_s: float
    ok: bool
    error: str | None = None
    usage: dict | None = None
    params: dict = field(default_factory=dict)
    rate_limit: dict = field(default_factory=dict)
    asr_language: str | None = None
    asr_prompt: str | None = None
    error_body: str | None = None  # first 300 chars of a failed response's body
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return asdict(self)


class TokenFactory:
    def __init__(
        self,
        config: Config,
        *,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
        receipts_path: Path | None = None,
        thinking_off: str = "chat_template_kwargs",  # or "no_think" (Qwen3 soft switch), chosen by the access gate
        min_interval_s: float = 0.0,  # client-side pacing: unpaced bursts drew HTTP 429 from the Token Factory
    ):
        self.config = config
        self.receipts_path = receipts_path
        self.thinking_off = thinking_off
        self.receipts: list[Receipt] = []
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------------------------------ plumbing

    def _record(self, r: Receipt) -> Receipt:
        self.receipts.append(r)
        if self.receipts_path is not None:
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.receipts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
        return r

    def _fail(self, reason: str, rec: Receipt, status: int | None = None, body: str = "") -> NotAuditing:
        rec.ok, rec.error = False, rec.error or reason
        rec.error_body = body[:300] or None
        self._record(rec)
        return NotAuditing(reason, rec, status, body)

    def _call(
        self,
        stage: str,
        method: str,
        path: str,
        *,
        model: str | None = None,
        params: dict | None = None,
        asr_language: str | None = None,
        asr_prompt: str | None = None,
        **kw,
    ) -> tuple[object, Receipt]:
        """One HTTP call -> (parsed JSON, receipt NOT yet recorded: the caller records it once it is complete).
        Every failure is recorded here and raised as NotAuditing."""
        wait = self._last_call + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        started = datetime.now(UTC).isoformat(timespec="milliseconds")
        t0 = time.perf_counter()

        def receipt(status: int | None, resp: httpx.Response | None = None, data: object = None) -> Receipt:
            body = data if isinstance(data, dict) else {}
            headers = resp.headers if resp is not None else {}
            return Receipt(
                stage=stage,
                endpoint=path,
                host=self.config.host,
                on_amd=self.config.on_amd,
                model=model,
                request_id=(headers.get("x-request-id") or body.get("id")) if resp is not None else None,
                http_status=status,
                started_at=started,
                latency_s=round(time.perf_counter() - t0, 4),
                ok=True,
                usage=body.get("usage"),
                params=params or {},
                rate_limit={h: headers[h] for h in _RATE_HEADERS if h in headers},
                asr_language=asr_language,
                asr_prompt=asr_prompt,
            )

        try:
            resp = self._http.request(method, path, **kw)
        except httpx.TimeoutException as e:
            raise self._fail("timeout", receipt(None)) from e
        except httpx.HTTPError as e:
            raise self._fail("network error", receipt(None), body=type(e).__name__) from e
        code = resp.status_code
        if code == 429:
            raise self._fail("rate limited", receipt(code, resp), code, resp.text)
        if code >= 500:
            raise self._fail("server error", receipt(code, resp), code, resp.text)
        if code >= 300:  # redirects are not followed: a gateway / login page is not an answer
            raise self._fail(f"rejected ({code})", receipt(code, resp), code, resp.text)
        try:
            data = resp.json()
        except ValueError as e:
            raise self._fail("malformed response", receipt(code, resp), code, resp.text) from e
        return data, receipt(code, resp, data)

    # ------------------------------------------------------------------------------------------- calls

    def models(self) -> tuple[list[dict], Receipt]:
        data, rec = self._call("models", "GET", "/models")
        self._record(rec)
        items = data.get("data", []) if isinstance(data, dict) else []
        return [m for m in items if isinstance(m, dict)], rec

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        schema: dict | None = None,
        schema_name: str = "spans",
        response_mode: str = "json_schema",
        temperature: float = 0.0,
        logprobs: bool = False,
        max_tokens: int = 512,
        stage: str = "chat",
        record: bool = True,
    ) -> tuple[str, dict, Receipt]:
        """One chat completion with thinking off. Returns (content without any think block, raw JSON, receipt).
        `record=False` leaves recording to the caller (it adds fields first)."""
        body: dict = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None and response_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        elif schema is not None and response_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        if self.thinking_off == "chat_template_kwargs":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.thinking_off == "no_think":
            body["messages"][-1]["content"] = body["messages"][-1]["content"] + " /no_think"
        if logprobs:
            body["logprobs"], body["top_logprobs"] = True, 1
        params = {
            "temperature": temperature,
            "thinking_off": self.thinking_off,
            "response_mode": response_mode if schema is not None else None,
            "logprobs": logprobs,
            "max_tokens": max_tokens,
        }
        raw, rec = self._call(stage, "POST", "/chat/completions", model=model, params=params, json=body)
        try:
            content = raw["choices"][0]["message"]["content"] or ""  # type: ignore[index]
            if not isinstance(content, str):
                raise TypeError("content is not a string")
        except (KeyError, IndexError, TypeError) as e:
            raise self._fail("malformed response", rec, rec.http_status, json.dumps(raw)[:300]) from e
        stripped = _THINK.sub("", content)
        rec.params["think_block_stripped"] = stripped != content
        if record:
            self._record(rec)
        return stripped, raw, rec  # type: ignore[return-value]

    def extract_spans(
        self, transcript: str, model: str, *, temperature: float = 0.0, response_mode: str = "json_schema"
    ) -> tuple[dict, Receipt]:
        """The extractor rung: spans of the transcript, in the span schema.

        response_mode: "json_schema" (server-enforced), else "json_object" / "prompt" for models that refuse a schema —
        the schema then travels in the system prompt and is checked locally either way. Invalid output is returned
        as-is (never repaired) so the referee abstains on it and the eval counts it.
        """
        system = SYSTEM_PROMPT if response_mode == "json_schema" else SYSTEM_PROMPT + SCHEMA_IN_PROMPT
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"Transcript: {transcript}"}]
        content, _, rec = self.chat(
            model,
            msgs,
            schema=SPAN_SCHEMA,
            temperature=temperature,
            stage="extract",
            response_mode=response_mode,
            record=False,
        )
        content = _FENCE.sub("", content).strip()
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            obj = {"_invalid": "not JSON", "_content": content[:200]}
        problem = validate_extraction(obj)
        rec.params["schema_valid"] = problem is None
        if problem:
            rec.error = f"schema: {problem}"
        self._record(rec)  # complete now: schema validity and think-block stripping are on disk
        return obj, rec

    def transcribe(
        self, wav_path: Path, model: str, *, language: str | None = None, prompt: str | None = None
    ) -> tuple[dict, Receipt]:
        data: dict = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        with open(wav_path, "rb") as f:
            out, rec = self._call(
                "asr",
                "POST",
                "/audio/transcriptions",
                model=model,
                params={"response_format": "verbose_json"},
                asr_language=language or "auto",
                asr_prompt=prompt,
                files={"file": (wav_path.name, f, "audio/wav")},
                data=data,
            )
        self._record(rec)
        return (out if isinstance(out, dict) else {"text": str(out)}), rec


def extract_or_not_audited(
    client: TokenFactory, transcript: str, model: str
) -> tuple[dict | None, str | None, Receipt]:
    """Pipeline helper: (extraction, None, receipt) or (None, not-audited reason, receipt)."""
    try:
        obj, rec = client.extract_spans(transcript, model)
        return obj, None, rec
    except NotAuditing as e:
        assert e.receipt is not None
        return None, e.reason, e.receipt
