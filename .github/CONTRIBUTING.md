# Contributing

Thanks for your interest in improving SayAgain.

## Getting started
1. Fork the repo and branch from `main`: `git checkout -b feat/your-feature`
2. Install: `uv sync --locked`
3. Optional, only to call the AMD Token Factory yourself: put the two names from `.env.example` in
   `~/.config/amd-tokenfactory/env`. Never put a key in this tree.
4. Run the queue: `make queue`, then open http://127.0.0.1:8018/queue

## Before you open a PR
- `make ci` passes (ruff, pytest with coverage, pip-audit, the readiness check).
- Add or update tests for any behaviour change. A fix gets a test named after the defect.
- A number on any page must come from a file in `bench/results/`; the readiness check fails on one that does not.
- Keep commits conventional (`feat:`, `fix:`, `docs:`, `chore:`).

## Reporting bugs / requesting features
Open an issue with the provided templates. Include repro steps, expected vs. actual behaviour, and environment.
