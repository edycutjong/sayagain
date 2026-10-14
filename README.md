# SayAgain

**Checks every radio read-back of a safety instruction, and sends the wrong ones to a human.**

> **Status: under construction (AMD Developer Hackathon ACT III, Oct 2026).** Nothing in this repository has been measured
> on recorded speech yet. This README says what exists in the code today; every result will come from a file in
> `bench/results/`.

## The problem

An operator radios *"Close valve two-oh-three, line Bravo."* The field technician answers *"Copy, closing two-oh-eight,
line Bravo."* Valve XV-208 is on a different line. The read-back exists so the operator catches this, and sometimes
nobody does.

## The design (being built)

SayAgain will transcribe both turns and ask a language model to **point at the words** that carry the action, the
equipment tag and the line. **Plain code** then decides: the words must really be in the transcript, every safety word
must be accounted for, spoken digits and NATO letters become plant tags through alias tables, and a pure function returns
`VERIFIED`, `MISMATCH`, `NO-READBACK` or `ABSTAIN`. A shift supervisor confirms or dismisses every card. Nothing is ever
sent to the plant.

The referee half of that design exists and is tested today; the speech half does not yet.

## Status (2026-10-14)

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
| Speech path: one WAV per card → VAD turns → speech-to-text (local Whisper, labelled off-AMD; AMD notebook backend switched off until a notebook GPU is confirmed) → disagreement gate | `sayagain/ingest.py`, `sayagain/asr.py` | code + tests done, waiting for recordings |
| Local kill test (Whisper-large-v3 on a laptop, off-AMD) | `scripts/kill_test_local.py` | harness ready, waiting for recordings |
| Supervisor queue: FastAPI + server-sent events, the read-back strip card, confirm / dismiss (reason required), "Try the refused one", end-of-shift list (spot-checks, caught on channel, not audited), receipts per model call | `sayagain/server.py`, `sayagain/shift.py`, `web/` | done, over the stored text runs |
| Static replay of stored runs (same page, GitHub Pages-ready, not published) | `scripts/export_run.py` → `docs/replay/` | done |
| Human recordings, audio eval rows | — | not built yet |

## First measurements: the card scripts, no audio yet (2026-10-14)

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
  many false alarms for the shift budget.
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
every stored model output offline and re-derives every table in the result files. A second live run of
DeepSeek-V4-Flash on the dev slice gave the same verdict on 40/40 cards (bench/results/text/repro-dev-DeepSeek-V4-Flash.json);
temperature 0 gives no determinism guarantee at this gateway, so this is measured, not guaranteed.

The access probe (bench/results/access-gate.json) found 8 instruct models on the AMD Token Factory whose span extraction
of the hero read-back the referee reads as `CLOSE · XV-208 · line B`. The model list has no speech-to-text model, so
transcription will run elsewhere, and every stage will be labelled with where it ran.

496 tests (`uv run pytest`; one runs only with local Whisper weights), lint with ruff, a CI workflow for every push. Friction log:
[docs/DX-REPORT.md](docs/DX-REPORT.md).

## Run it

```bash
uv sync
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
The words on these cards are the recording cards' scripts (`text·scripted`): there is no audio yet, and the card
says so where a waveform will go. `SAYAGAIN_LIVE_EXTRACT=1` lets the server call the AMD Token Factory for each
turn instead of replaying the stored output (paced, capped per process; a failed call is `NOT_AUDITED`).

## License

MIT, see [LICENSE](LICENSE).
