# Developer-experience report

One line per friction, written as it happened. Evidence for API behaviour: `bench/results/access-gate.json` and its
receipts in `bench/results/receipts/access-gate.jsonl`.

| Date | Where | Friction | What we did |
|---|---|---|---|
| 2026-10-14 | AMD Token Factory | No listed instruct model accepts `response_format: json_schema`; every one returns HTTP 400 "does not support JSON schema output mode" (bench/results/access-gate.json) | Client falls back to `json_object` with the schema in the system prompt; the output is still validated locally and grounded by code |
| 2026-10-14 | AMD Token Factory | Rate limit: HTTP 429 "maximum 20 requests per minute" per model, hit during the first probe run | Client-side pacing (one call every 3.2 s per key) and one retry pass after a cool-down in the gate; in the pipeline a 429 makes the exchange NOT_AUDITED, never a verdict |
| 2026-10-14 | AMD Token Factory | `logprobs: true` is accepted but no token log-probs come back on any listed model | The audio gate cannot use model probabilities; the prompted-vs-unprompted ASR disagreement gate is the fallback |
| 2026-10-14 | AMD Token Factory | No audio / Whisper model in `/models` | Speech-to-text cannot run on the Token Factory; where ASR runs is decided by the notebook check and printed per stage |
| 2026-10-14 | AMD Token Factory | The `usage` block carries cost fields incl. `upstream_inference_cost`, and the API does not say which GPU served a request | Receipts record the host only; nothing claims a GPU model from these receipts |
| 2026-10-14 | AMD Token Factory | Same model, temperature 0, JSON mode: one probe of Gemma-4-31B-IT returned a run of repeated characters instead of JSON (bench/results/access-gate.json) | Malformed output is never repaired: the referee abstains on it and the eval counts it |
| 2026-10-14 | AMD Token Factory | The key's base URL is `developer.amd.com.cn/radeon/api/v1`; the event FAQ names notebooks.amd.com/tokenfactory | Recorded per receipt (`host`); open question to the organisers |
| 2026-10-14 | Local Mac | The global `transformers` install fails to import (tokenizers version pin) | Kill test runs in an isolated `uv` extra (`uv sync --extra killtest`), never on the judged path |
