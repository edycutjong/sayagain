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
| Local kill test (Whisper-large-v3 on a laptop, off-AMD) | `scripts/kill_test_local.py` | harness ready, waiting for recordings |
| Human recordings, speech-to-text path, eval, queue UI | — | not built yet |

The access probe (bench/results/access-gate.json) found 8 instruct models on the AMD Token Factory whose span extraction
of the hero read-back the referee reads as `CLOSE · XV-208 · line B`. The model list has no speech-to-text model, so
transcription will run elsewhere, and every stage will be labelled with where it ran.

410 tests (`uv run pytest`), lint with ruff, a CI workflow for every push. Friction log:
[docs/DX-REPORT.md](docs/DX-REPORT.md).

## Run it

```bash
uv sync
uv run pytest -q
uv run python scripts/seed.py --seed 1018          # registry + cards, byte-identical
uv run python scripts/check_submission_readiness.py
```

## License

MIT, see [LICENSE](LICENSE).
