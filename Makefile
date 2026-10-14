.PHONY: help install lint lint-fix test test-coverage audit readiness secrets ci seed queue replay

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## install locked dependencies
	uv sync --locked

# ── Code quality ────────────────────────────────────────────
lint: ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

lint-fix: ## ruff autofix + format
	uv run ruff check --fix . && uv run ruff format .

test: ## full test suite
	uv run pytest -q

test-coverage: ## tests with a line-coverage report
	uv run pytest -q --cov=referee --cov=sayagain --cov=scripts --cov-report=term-missing

# ── Security and claims ─────────────────────────────────────
audit: ## dependency CVEs (pip-audit)
	uv run pip-audit

secrets: ## secret scan over the whole git history
	gitleaks detect --no-banner --redact --log-opts="--all"

readiness: ## no placeholder, no claim without a receipt
	uv run python scripts/check_submission_readiness.py --stage build

ci: lint test-coverage audit readiness ## what the CI workflow runs

# ── The product ─────────────────────────────────────────────
seed: ## registry + 145 recording cards, byte-identical
	uv run python scripts/seed.py --seed 1018

queue: ## supervisor queue on http://127.0.0.1:8018/queue
	uv run uvicorn sayagain.server:app --port 8018

replay: ## static replay of the stored runs -> docs/replay/
	uv run python scripts/export_run.py
