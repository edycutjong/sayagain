<div align="center">
  <img src="docs/icon.svg" alt="SayAgain icon" width="144">
  <h1>SayAgain 📻</h1>
  <p><em>Checks every radio read-back of a safety instruction, and sends the wrong ones to a human.</em></p>
  <img src="docs/screenshots/replay-1440-2-hero.png" alt="The supervisor queue: the field tech read back two-oh-eight, the operator said two-oh-three, and the one differing word is red" width="1000">

  <br/>

  [![Built for AMD Developer Hackathon ACT III](https://img.shields.io/badge/lablab-AMD_Developer_Hackathon_ACT_III-ed1c24?style=for-the-badge)](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii)

  <br/>

  ![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
  ![uv](https://img.shields.io/badge/uv-261230?style=flat&logo=uv&logoColor=white)
  ![ruff](https://img.shields.io/badge/ruff-261230?style=flat&logo=ruff&logoColor=white)
  [![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
  [![CI](https://github.com/edycutjong/sayagain/actions/workflows/ci.yml/badge.svg)](https://github.com/edycutjong/sayagain/actions/workflows/ci.yml)

</div>

---

> **Status: under construction (AMD Developer Hackathon ACT III, Oct 2026).** Every result below comes from a file in
> `bench/results/`. No recorded speech has been measured yet: the rows that exist feed the written words of the recording
> cards to each extractor (`text·scripted`), and are labelled that way everywhere.

## 📸 See it in Action

The supervisor queue, replaying a stored run over the 145 recording cards. Red marks only the read-back word that differs from
the instruction; the card leaves the queue only through **Confirm** or **Dismiss** (a reason is required).

| The hero exchange | The refused one | End of shift |
|---|---|---|
| ![MISMATCH card: tag XV-203 ≠ XV-208](docs/screenshots/replay-1440-2-hero.png) | ![ABSTAIN card: XV-205 is not in the plant registry](docs/screenshots/replay-1440-6-refused.png) | ![End-of-shift list: the ten spot-checks](docs/screenshots/replay-1440-5-eos.png) |

Every card says where each stage ran (`speaker`, `radio`, `extraction`, `referee`) and lists the receipt of each model call
behind it. The words on these cards are the recording cards' scripts (`text·scripted`): there is no audio yet, and the card
says so where a waveform will go.

## 💡 The Problem & Solution

### The Problem

An operator radios *"Close valve two-oh-three, line Bravo."* The field technician answers *"Copy, closing two-oh-eight,
line Bravo."* Valve XV-208 is on a different line. The read-back exists so the operator catches this, and sometimes
nobody does.

### The Solution

SayAgain transcribes both turns and asks a language model to **point at the words** that carry the action, the
equipment tag and the line. **Plain code** then decides: the words must really be in the transcript, every safety word
must be accounted for, spoken digits and NATO letters become plant tags through alias tables, and a pure function returns
`VERIFIED`, `MISMATCH`, `NO-READBACK` or `ABSTAIN`. A shift supervisor confirms or dismisses every card. Nothing is ever
sent to the plant.

The referee half of that design exists and is tested today; the speech half waits for the recordings.

## 🏗️ Architecture & Tech Stack

```mermaid
flowchart LR
    W["one WAV per card"] --> V["VAD: split into turns"]
    V --> A["speech-to-text<br/>(waiting for recordings)"]
    T["card script<br/>(text·scripted rows)"] --> X
    A --> X["span extraction<br/>AMD Token Factory, JSON mode"]
    X --> G["grounding + coverage<br/>(plain code)"]
    G --> R["referee: aliases, registry, pairing"]
    R --> Q["supervisor queue<br/>FastAPI + SSE"]
    X -. "a receipt per call" .-> C[("bench/results/receipts")]
    Q --> D["Confirm / Dismiss"]
```

| Layer | Choice |
|---|---|
| Language, runtime | Python 3.11, `uv` |
| Server | FastAPI, server-sent events, SQLite for reviews |
| Extraction | AMD Token Factory (OpenAI-compatible), `json_object` mode, validated locally, fails closed to `NOT_AUDITED` |
| Referee | pure Python; the verdict module imports no model code (`tests/test_verdict.py`) |
| Front end | one static page, no framework: `web/` (live) and `docs/replay/` (static export of the same page) |
| Quality | ruff, pytest, pip-audit, gitleaks; GitHub Actions workflows and CodeQL |

### Build status (2026-10-14)

| Part | Where | State |
|---|---|---|
| Eval protocol, pre-registered before any model call on recorded data | `bench/PROTOCOL.md`, git tag `prereg-v1` | done |
| Regex extractor (rung 0, no model) | `referee/regex_extractor.py` | frozen at `prereg-v1` |
| Alias tables (English + Bahasa digits, NATO, tag nouns, actions, negators) | `referee/aliases.py` | from `prereg-v1`, plus the hardening listed under Deviations in `bench/PROTOCOL.md` |
| Referee: grounding, coverage + retraction cues, registry precedence, pairing (say-again, operator correction), verdict | `referee/` | done |
| Plant registry (fictional, 120 tags) and 145 recording cards | `data/`, `scripts/seed.py` | done |
| Radio-path simulation and SNR mixer | `scripts/radioize.py`, `scripts/mix.py` | done |
| AMD Token Factory client with a receipt per call; access probe | `sayagain/amd_client.py`, `scripts/access_gate.py` | probe run, see below |
| Span extraction on the AMD Token Factory: JSON mode + local schema check, 429 back-off, fail closed to `NOT_AUDITED`, a receipt per call | `sayagain/extract.py` | done, run on text rows |
| Model ladder, grounding ablation, LLM-judge baseline, offline re-scoring of every stored model output | `scripts/eval.py` | run on the card scripts (`text·scripted`), see below |
| Speech path: one WAV per card → VAD turns → speech-to-text (local Whisper, labelled off-AMD; AMD notebook backend switched off until a notebook GPU is confirmed) → disagreement gate | `sayagain/ingest.py`, `sayagain/asr.py` | unit tests done; Whisper has not yet run on a human recording (only on one `tts·fixture` clip, in an opt-in test) |
| Local kill test (Whisper-large-v3 on a laptop, off-AMD) | `scripts/kill_test_local.py` | harness ready, waiting for recordings |
| Supervisor queue: read-back strip card, confirm / dismiss (reason required), "Try the refused one", end-of-shift list (spot-checks, caught on channel, not audited), receipts per model call | `sayagain/server.py`, `sayagain/shift.py`, `web/` | done, over the stored text runs |
| Static replay of stored runs (same page, GitHub Pages-ready, not published) | `scripts/export_run.py` → `docs/replay/` | done |
| Human recordings, audio eval rows | — | not built yet |

## 🏆 AMD Token Factory Integration

Span extraction is the one model job in the pipeline, and it runs on the AMD Token Factory: `sayagain/amd_client.py` is the
client (a receipt per call with request id, model served, host, latency and cost; a 429 is reported as not audited),
`sayagain/extract.py` the extraction stage (JSON mode, 429 back-off, schema checked in code, fail closed).

How to read the rows below: a **rung** is one extractor tried in turn, from the plain regex (rung 0, no model) to the larger
models; the **ladder rule** (pre-registered in `bench/PROTOCOL.md`) picks the smallest rung that lets **0 wrong read-backs
through as VERIFIED** (*false-VERIFIED*) while bothering the supervisor on no more than a pre-registered threshold of the correct ones (*false-alarm
load*); `text·scripted` marks rows run on the cards' written words, with no audio; a Wilson interval is the range a
rate could plausibly have, given so few cards.

### First measurements: the card scripts, no audio yet (2026-10-14)

These rows feed the **written words of the 145 recording cards** to each extractor (`text·scripted`): no microphone,
no speech-to-text. They are each rung's ceiling on clean words, not the audio result the protocol pre-registers
(bench/PROTOCOL.md). Scored on the 89 held-out cards: the hero and its secondary, plus the 87 A2 cards left after the
dev slice (the first 20 A2 cards and the 20 kill-test cards). Full tables with Wilson intervals: bench/results/text/RESULTS.md.

- **No wrong read-back was marked VERIFIED by any rung of the referee:** 0 false-VERIFIED for the regex rung and for all
  four models extracting spans on the AMD Token Factory (bench/results/text/ladder-r1-reaudit.json).
- **The plain regex rung already meets the ladder rule on clean words** (pre-registered for the audio row, applied here to text):
  41/41 seeded errors flagged, 0/48 false alarms (bench/results/text/ladder-r1-reaudit.json).
  By that rule, on scripted words no language model is needed. Whether that survives real speech, noise and other
  people's phrasing is what the audio rows will measure.
- **The only one of the four models that meets the rule:** Gemma-4-31B-IT,
  0 false-VERIFIED and 2/48 false alarms (bench/results/text/ladder-r1-reaudit.json), after one re-audit of the 118
  transcripts whose first call failed (108 answered with HTTP status 500 by the gateway, 10 time-outs); the as-run first pass is
  bench/results/text/ladder-r1.json. Qwen3.8-27B and DeepSeek-V4-Flash also passed no wrong read-back but raised too
  many false alarms for the shift budget. Qwen3.8-27B needed a re-audit as well
  (as run: 3 cards not audited, 39/41 errors flagged, bench/results/text/ladder-r1.json) and DeepSeek-V4-Flash one call.
- **Models that miss fail toward a human:** MiniCPM5-2B returns malformed or misplaced spans, and the referee abstains on
  all 48 correct read-backs too, 48/48 false alarms, none passed (bench/results/text/ladder-r1-reaudit.json).
- **Grounding ablation:** the same model returning normalised values instead of copied words raised false alarms
  from 2/48 to 6/48, with 1 correct card still not audited (bench/results/text/ablate-grounding-r1-reaudit.json).
  In 4 of the 6 it turned a spoken pump or motor-valve tag into an XV- valve tag the registry does not have.
- **LLM-as-judge baseline:** given the whole exchange but no plant registry, DeepSeek-V4.1-Flash passed
  2 wrong read-backs as VERIFIED (bench/results/text/llm-judge-r1-reaudit.json), both of which left out the action word.
  It also passed the refusal card, a valve that does not exist in the plant, as VERIFIED; 5 of its held-out cards were
  still not audited after the re-audit.

Every call above has a receipt in bench/results/receipts/, and `uv run python scripts/eval.py --rescore` re-judges
every stored model output offline and re-derives the per-rung rows, call metrics and rule outcome of every ladder, ablation, LLM-judge and dev-slice result file (it reports OK for each). A second live run of
DeepSeek-V4-Flash on the dev slice gave the same verdict on 40/40 cards (bench/results/text/repro-dev-DeepSeek-V4-Flash.json),
with every extraction identical but one turn's `kind` (same file) and one re-audit of 3 transcripts (bench/results/text/dev-r2-repro-reaudit.json);
temperature 0 gives no determinism guarantee at this gateway, so this is measured, not guaranteed.

The access probe (bench/results/access-gate.json) listed 11 instruct models on the AMD Token Factory; 8 of them returned a span
extraction of the hero read-back that the referee reads as `CLOSE · XV-208 · line B` (the probe answer of Gemma-4-31B-IT was not
valid JSON, and it passed in the ladder run). The model list has no speech-to-text model, so
transcription will run elsewhere, and every stage will be labelled with where it ran. Friction log:
[docs/DX-REPORT.md](docs/DX-REPORT.md).

## 📊 Engineering Rigor

501 tests (`uv run pytest`; one runs only with local Whisper weights), ruff lint and format, `pip-audit` and gitleaks over the
full history (`make ci`, `make secrets`), CI workflows for the same checks plus CodeQL (none has executed yet, see Testing & CI), and a readiness gate (`scripts/check_submission_readiness.py`) that fails the build on an unfilled
placeholder, a hardware or serving-stack name no receipt backs, or a number its cited file does not contain.

### Attacks defeated

| Control | Why it is load-bearing | Test |
|---|---|---|
| A model-invented span is ungrounded | The extractor may only point at words in the transcript | `tests/test_grounding_coverage.py:36` |
| A real but wrong word fails coverage | Returning "line Alpha" and dropping the spoken correction to Bravo must not pass | `tests/test_grounding_coverage.py:60` |
| A wide span cannot hide a safety word | A span that swallows a second command or a stray line is not accepted as one command | `tests/test_review_round2.py:70` |
| A span that drops part of the tag or the equipment word cannot verify | "valve two-oh-three-eight" and "pump" for "valve" must not read as a match | `tests/test_review_round4.py:34` |
| A negated command never verifies | "not closing", "will not close" or a negated order must reach a human | `tests/test_review_round1.py:31` |
| Swapped actions between two commands cannot verify | A model that reports the swapped actions must not turn the error into a pass | `tests/test_review_round1.py:52` |
| A retraction needs a spoken cue | A silent "correction" must not clear a wrong read-back | `tests/test_grounding_coverage.py:74` |
| A correction without a new read-back keeps the wrong one live | The operator saying "Negative" is not the technician fixing it | `tests/test_review_round1.py:91` |
| A `NOT_AUDITED` turn is never VERIFIED | A failed model call must not become a pass | `tests/test_verdict.py:196` |
| A referee exception fails closed | A bug must reach a human, not a log | `tests/test_verdict.py:203` |
| The verdict module imports no model code | The verdict cannot be moved into the model | `tests/test_verdict.py:215` |

### Honest limits (9)

1. No recorded speech has been measured. Every result is `text·scripted`, each rung's ceiling on clean words.
2. The held-out set is 89 cards, so the intervals are wide: the best model's 2/48 false alarms is a 4.17 % rate with a Wilson interval of 1.15 % to 13.98 % (bench/results/text/RESULTS.md).
3. On scripted words the plain regex rung meets the rule with no model, so what a language model adds is unproven until the audio rows exist.
4. The plant, its 120 tags and every exchange are fictional; the error mix is seeded, not sampled from a real site.
5. The first Gemma-4-31B-IT pass left 45 cards not audited; the headline row is a re-audit of those, and the as-run pass is kept next to it (bench/results/text/ladder-r1.json).
6. Temperature 0 does not make the gateway deterministic; the repeat run agreed on 40/40 cards (bench/results/text/repro-dev-DeepSeek-V4-Flash.json), which is evidence and not a guarantee.
7. The Token Factory is a hosted API: receipts name the model and host, never a GPU, so no page here claims one (`on_amd` in a receipt means only that the endpoint host is under amd.com or amd.com.cn; the receipts' host is developer.amd.com.cn).
8. The gateway ignores `logprobs` for the extraction models. Speech-to-text is planned on local Whisper, whose token probabilities can feed the confidence floor, with the disagreement gate between two transcription passes as the fallback; neither has run on a human recording (the Whisper token probabilities and the confidence floor ran only on one synthetic text-to-speech clip, `tts·fixture`, in an opt-in unit test, never in an eval row).
9. The static replay shows stored verdicts. The live queue calls the API only when `SAYAGAIN_LIVE_EXTRACT=1`, paced and capped per process.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- Optional, only to call the AMD Token Factory yourself: `AMD_TF_API_KEY` and `AMD_TF_BASE_URL` in
  `~/.config/amd-tokenfactory/env` (the names are in `.env.example`; a key never goes in this tree)

### Installation

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/seed.py --seed 1018          # registry + cards, byte-identical
uv run python scripts/check_submission_readiness.py
```

### The supervisor queue

```bash
uv run uvicorn sayagain.server:app --port 8018     # open http://127.0.0.1:8018/queue
uv run python scripts/export_run.py                # static replay of the stored runs -> docs/replay/
```

The queue replays a shift from a stored run: the referee judges each exchange from the model output the run
recorded, and every card shows the receipt of each model call (request id, model served, host, latency, cost).
`SAYAGAIN_LIVE_EXTRACT=1` lets the server call the AMD Token Factory for each turn instead of replaying the stored output
(paced, capped per process; a failed call is `NOT_AUDITED`).

## 🧪 Testing & CI

```bash
make ci            # ruff, pytest with coverage, pip-audit, readiness: the checks the CI workflow runs
make secrets       # gitleaks over the whole git history
uv run python scripts/eval.py --rescore   # re-judge every stored model output offline
```

Workflows in `.github/workflows/`: `ci` (quality + security jobs), `gitleaks` (full history), `codeql` (Python and
JavaScript), `release` (semantic version tag from conventional commits). Dependabot is grouped and monthly; major bumps are ignored for
Python dependencies. **None of these workflows has executed yet:** every run since the first push failed to start because of a GitHub
account billing block, not a test failure. The same checks pass locally with `make ci` and `make secrets`.

## 📁 Project Structure

```
referee/        pure-Python referee: aliases, grounding, coverage, pairing, registry, verdict
sayagain/       AMD Token Factory client, extraction, speech path, queue server, shift list
scripts/        seed, radio simulation, eval ladder, export, access probe, readiness gate
web/            the queue page (one static page, no framework)
docs/           static replay (docs/replay), screenshots, developer-experience report
bench/          pre-registered protocol and every stored result and receipt
data/           fictional plant registry and the 145 recording cards
tests/          501 tests, with regression tests from four review rounds (`test_review_round*.py`)
```

## 🗺️ Roadmap

- [x] Pre-registered protocol and frozen regex rung
- [x] Referee with grounding, coverage, pairing and fail-closed verdicts
- [x] Model ladder, grounding ablation and LLM-judge baseline on the AMD Token Factory (`text·scripted`)
- [x] Supervisor queue and static replay
- [ ] Human recordings, one WAV per card
- [ ] Audio rows: local kill test, speech-to-text, headline row and failure case
- [ ] Blind auditor set
- [ ] Demo video and pitch deck

## 📄 License

MIT, see [LICENSE](LICENSE).
