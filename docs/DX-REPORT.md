# Developer-experience report

One line per friction, written as it happened. Evidence for API behaviour: `bench/results/access-gate.json` and its
receipts in `bench/results/receipts/access-gate.jsonl`.

| Date | Where | Friction | What we did |
|---|---|---|---|
| 2026-10-14 | AMD Token Factory | None of the 11 instruct models we probed accepts `response_format: json_schema`; each answers HTTP 400 "does not support JSON schema output mode" (bench/results/access-gate.json) | The client falls back to `json_object` with the schema in the system prompt; the output is still validated locally and grounded by code |
| 2026-10-14 | AMD Token Factory | HTTP 429 when the first, unpaced probe run called the models back to back; that run's receipts were replaced by later runs, so its error text is not kept here | The client paces calls (one every 3.2 s in scripts/access_gate.py), the gate retries a rate-limited model once after a cool-down, and receipts now keep the first 300 characters of any error body; in the pipeline a 429 is retried with back-off, and an exchange whose retries run out is NOT_AUDITED, never a verdict |
| 2026-10-14 | AMD Token Factory | `logprobs: true` is accepted, but none of the 11 probed models returns token log-probs | The audio gate cannot use model probabilities; the prompted-vs-unprompted ASR disagreement gate is the fallback |
| 2026-10-14 | AMD Token Factory | `/models` lists no speech-to-text model; `/audio/transcriptions` exists but answers "Transcription model not found: whisper-1" (bench/results/access-gate.json) | Speech-to-text is planned elsewhere, decided by the notebook check, and printed per stage |
| 2026-10-14 | AMD Token Factory | The `usage` block carries cost fields incl. `upstream_inference_cost`, and the API does not say which GPU served a request | Receipts record the endpoint host only; nothing here names a GPU |
| 2026-10-14 | AMD Token Factory | Temperature 0, JSON mode: Gemma-4-31B-IT returned a run of repeated characters instead of JSON, and MiniCPM-V-4.7-1B returned keys outside the schema (bench/results/access-gate.json) | Malformed output is never repaired: the referee abstains on it and the eval counts it |
| 2026-10-14 | AMD Token Factory | The key's endpoint host is developer.amd.com.cn; the event FAQ names notebooks.amd.com/tokenfactory | Recorded per receipt (`host`). The Token Factory portal the event FAQ names (notebooks.amd.com/tokenfactory) documents this base URL for its API |
| 2026-10-14 | Local Mac | The global `transformers` install fails to import (tokenizers version pin) | The kill test runs in an isolated `uv` extra (`uv sync --extra killtest`), never on the judged path |
