"""Speech-to-text backends. The AMD Token Factory serves no transcription model (bench/results/access-gate.json), so:

  local-whisper  Whisper-large-v3 through `transformers` on this machine (branch C). Receipts say `asr_on_amd = 0`
                 and every row it feeds is badged off-AMD. Token log-probabilities are returned (for `span_conf`).
  notebook       branch B stub: a Whisper service on an AMD notebook GPU, enabled only when the human reports one
                 (`SAYAGAIN_NOTEBOOK_ASR_URL`). `asr_on_amd = 1` only if that service names an AMD GPU itself.

A failed transcription is NOT_AUDITED (fail closed), never an empty transcript. The vocabulary prompt holds radio
words only — no tag number, so it cannot pull a misheard read-back toward the instruction.
"""

from __future__ import annotations

import io
import os
import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from referee.turns import Turn
from sayagain.audio import SR
from sayagain.ingest import TurnAudio

VOCAB_PROMPT = (
    "Plant radio. Valve, motor valve, pump, line Alpha, Bravo, Charlie, Delta. Close, open, isolate, start, stop. "
    "Copy, roger, say again, correction. Katup, pompa, jalur, tutup, buka, isolasi, siap, ulangi, koreksi."
)
NOTEBOOK_ENV = "SAYAGAIN_NOTEBOOK_ASR_URL"
_AMD_GPU = re.compile(r"\b(?:AMD|Instinct|Radeon|MI\d{3}[A-Z]?)\b", re.IGNORECASE)
_LANG_TOKEN = re.compile(r"<\|startoftranscript\|><\|(\w{2,3})\|>")


class ASRUnavailable(Exception):
    """The backend cannot run here (no weights, no notebook reported) — a setup fact, not a transcription."""


class ASRFailed(Exception):
    def __init__(self, reason: str, receipt: dict | None = None):
        super().__init__(reason)
        self.reason, self.receipt = reason, receipt


@dataclass
class ASRResult:
    text: str
    tokens: list[tuple[int, int, float]] | None  # (char start, char end, log-prob) aligned to `text`
    language: str | None
    receipt: dict = field(default_factory=dict)


def align_tokens(
    ids: list[int], logprobs: list[float], decode: Callable[[list[int]], str], skip: set[int]
) -> tuple[str, list[tuple[int, int, float]]]:
    """Generated token ids + their log-probs -> (text, [(char start, char end, log-prob)]) with offsets into the
    returned (left-stripped) text. Pieces come from incremental decoding so BPE merges keep exact offsets."""
    kept = [(i, lp) for i, lp in zip(ids, logprobs, strict=True) if i not in skip]
    full, spans, prefix = "", [], []
    for i, lp in kept:
        prefix.append(i)
        now = decode(prefix)
        if not now.startswith(full):  # a merge rewrote earlier characters: attribute the whole tail to this token
            cut = len(os.path.commonprefix([full, now]))
            spans = [(s, min(e, cut), p) for s, e, p in spans if s < cut]
            spans.append((cut, len(now), lp))
        elif len(now) > len(full):
            spans.append((len(full), len(now), lp))
        full = now
    lead = len(full) - len(full.lstrip())
    text = full.strip()
    out = []
    for s, e, lp in spans:
        s2, e2 = max(0, s - lead), min(len(text), e - lead)
        if e2 > s2:
            out.append((s2, e2, float(lp)))
    return text, out


def _receipt(backend: str, model: str, *, on_amd: bool, language: str | None, prompt: str | None, **kw) -> dict:
    return {
        "stage": "asr",
        "backend": backend,
        "model": model,
        "on_amd": on_amd,
        "asr_on_amd": 1 if on_amd else 0,
        "asr_language": language or "auto",
        "asr_prompt": prompt,
        "started_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        **kw,
    }


class LocalWhisper:
    """Branch C: Whisper on this machine (MPS / CPU). Off-AMD by construction."""

    name = "local-whisper"

    def __init__(self, model: str = "openai/whisper-large-v3", device: str | None = None):
        self.model_id, self.device = model, device
        self._model = self._proc = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as e:  # the killtest extra is not installed
            raise ASRUnavailable("local-whisper needs `uv sync --extra killtest`") from e
        self.device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self._proc = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
        self._model.eval()

    def transcribe(self, samples: np.ndarray, *, language: str | None = None, prompt: str | None = None) -> ASRResult:
        rec = _receipt(
            self.name,
            self.model_id,
            on_amd=False,
            language=language,
            prompt=prompt,
            host=platform.machine(),
            badge="asr·off-AMD",
        )
        t0 = time.perf_counter()
        try:
            self._load()
            import torch

            proc, model = self._proc, self._model
            feats = proc(samples, sampling_rate=SR, return_tensors="pt").input_features.to(self.device, model.dtype)
            kw: dict = {"task": "transcribe", "return_dict_in_generate": True, "output_scores": True}
            if language:
                kw["language"] = language
            if prompt:
                kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(prompt), device=self.device)
            with torch.no_grad():
                out = model.generate(feats, **kw)
            scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            gen = out.sequences[0, -scores.shape[0] :].tolist()
            tok = proc.tokenizer
            skip = set(tok.all_special_ids) | {i for i in set(gen) if i >= tok.convert_tokens_to_ids("<|0.00|>")}
            text, tokens = align_tokens(gen, scores.float().cpu().tolist(), lambda ids: tok.decode(ids), skip)
            m = _LANG_TOKEN.search(tok.decode(out.sequences[0].tolist(), skip_special_tokens=False))
            detected = m.group(1) if m else language
        except ASRUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 — fail closed: a crashed ASR is NOT_AUDITED, never an empty transcript
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300], latency_s=round(time.perf_counter() - t0, 4))
            raise ASRFailed("asr error", rec) from e
        rec.update(
            ok=True,
            device=self.device,
            latency_s=round(time.perf_counter() - t0, 4),
            detected_language=detected,
            n_tokens=len(tokens),
        )
        return ASRResult(text, tokens, detected, rec)


class NotebookASR:
    """Branch B stub: POST the turn's WAV to a Whisper service on an AMD notebook GPU.

    Expected reply: {"text": str, "tokens": [[start, end, logprob], ...] | null, "language": str, "gpu": str}.
    Enabled only when `SAYAGAIN_NOTEBOOK_ASR_URL` is set — i.e. after the human's notebook check (HUMAN-TODO 2).
    """

    name = "notebook"

    def __init__(self, url: str | None = None, *, transport=None, timeout: float = 120.0):
        self.url = url or os.environ.get(NOTEBOOK_ENV)
        if not self.url:
            raise ASRUnavailable(f"branch B is off: no notebook GPU reported ({NOTEBOOK_ENV} unset; HUMAN-TODO 2)")
        import httpx

        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

    def transcribe(self, samples: np.ndarray, *, language: str | None = None, prompt: str | None = None) -> ASRResult:
        import httpx
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.clip(samples, -1, 1).astype(np.float32), SR, format="WAV", subtype="PCM_16")
        rec = _receipt(self.name, "openai/whisper-large-v3", on_amd=False, language=language, prompt=prompt)
        data = {k: v for k, v in (("language", language), ("prompt", prompt)) if v}
        t0 = time.perf_counter()
        try:
            r = self._http.post(self.url, files={"file": ("turn.wav", buf.getvalue(), "audio/wav")}, data=data)
            rec.update(http_status=r.status_code, latency_s=round(time.perf_counter() - t0, 4))
            if r.status_code != 200:
                raise ValueError(f"status {r.status_code}")
            body = r.json()
            text = body["text"]
            if not isinstance(text, str):
                raise TypeError("text is not a string")
            toks = body.get("tokens")
            tokens = [(int(s), int(e), float(lp)) for s, e, lp in toks] if toks else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300], latency_s=round(time.perf_counter() - t0, 4))
            raise ASRFailed("asr error", rec) from e
        gpu = str(body.get("gpu") or "")
        on_amd = bool(_AMD_GPU.search(gpu))  # only the service's own report of its device counts
        rec.update(
            ok=True, gpu=gpu or None, on_amd=on_amd, asr_on_amd=int(on_amd), detected_language=body.get("language")
        )
        return ASRResult(text.strip(), tokens, body.get("language"), rec)


def backend(name: str, **kw):
    if name == "local-whisper":
        return LocalWhisper(**kw)
    if name == "notebook":
        return NotebookASR(**kw)
    raise ValueError(f"unknown ASR backend {name!r}")


def transcribe_turn(
    asr, turn: TurnAudio, *, language: str | None = None, prompt: str | None = VOCAB_PROMPT, unprompted: bool = True
) -> tuple[Turn, list[dict]]:
    """One turn -> a referee Turn (extraction still empty) + its ASR receipts.

    With a prompt and `unprompted=True` the turn is transcribed twice; the unprompted text feeds the disagreement
    gate (`Turn.alt_transcript`). Any failure -> the turn is NOT_AUDITED.
    """
    receipts: list[dict] = []
    t = Turn(turn.id, turn.channel, turn.t_start, turn.t_end, "")
    try:
        res = asr.transcribe(turn.samples, language=language, prompt=prompt)
        receipts.append(res.receipt)
        t.transcript, t.tokens = res.text, res.tokens
        t.meta["asr_language"] = res.language
        if prompt and unprompted:
            alt = asr.transcribe(turn.samples, language=language, prompt=None)
            receipts.append(alt.receipt)
            t.alt_transcript = alt.text
    except ASRFailed as e:
        if e.receipt:
            receipts.append(e.receipt)
        t.not_audited = e.reason
    t.meta["asr_receipts"] = [r.get("started_at") for r in receipts]
    return t, receipts
