"""Readiness gate for every judge-facing file: no placeholder, no claim without a receipt.

    uv run python scripts/check_submission_readiness.py [--stage build|submit] [--root .]

Checks the judged files (README.md, DEMO.md, ARCHITECTURE.md, docs/**/*.md|html):
  placeholder       unfilled slots: ⟨…⟩, TODO, TBD, FIXME, XXX, {{ }}, lorem ipsum
  results           every bench/results/… path a page cites exists; at `submit`, eval results + the access gate exist
  tests             a stated "N tests" equals what pytest collects (required at `submit`)
  hardware          "MI300X", "AMD Developer Cloud", "vLLM", "ROCm", "Instinct", "co-resident", "channels per GPU" only
                    if a receipt in bench/results/ contains that literal
  amd-stage         a sentence saying something ran on AMD maps to a receipt stage with on_amd = true (ASR sentences
                    to an ASR receipt, model/extraction sentences to an extraction receipt); "AMD GPU / hardware" needs
                    a receipt that names the GPU — a hosted API does not expose one
  hero              the hero is the tag error (XV-203 read back as XV-208); the old line-swap hero strings fail
  number            a percentage, p50/p95 or ms figure must cite bench/results/ or evidence/, or say target/threshold
Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PLACEHOLDER = re.compile(r"⟨[^⟩]*⟩|\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b|\{\{.*?\}\}|lorem ipsum", re.IGNORECASE)
HARDWARE_LITERALS = ("MI300X", "AMD Developer Cloud", "vLLM", "ROCm", "Instinct", "co-resident", "channels per GPU")
# a sentence that says something did NOT run on AMD is not a claim (heuristic: any negation word)
OFF_AMD = re.compile(r"off[- ]AMD|\b(?:no|not|cannot|can't|never|without|elsewhere|none)\b", re.IGNORECASE)
AMD_HW = re.compile(r"AMD (?:GPU|hardware|accelerator)s?", re.IGNORECASE)
ASR_WORDS = re.compile(r"\b(?:ASR|Whisper|transcri\w*|speech[- ]to[- ]text)\b", re.IGNORECASE)
LLM_WORDS = re.compile(r"\b(?:extract\w*|LLM|language model|Qwen|model ladder|span)\b", re.IGNORECASE)
NUMBER_CLAIM = re.compile(r"\d+(?:\.\d+)?\s?%|\bp(?:50|95)\b|\d+(?:\.\d+)?\s?ms\b", re.IGNORECASE)
NUMBER_OK = re.compile(r"bench/results/|evidence/|\btarget|\bthreshold|assum|≥|≤|\bpre-registered", re.IGNORECASE)
TESTS_STATED = re.compile(r"\b(\d+)\s+(?:unit\s+)?tests\b", re.IGNORECASE)


@dataclass
class Finding:
    check: str
    file: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"FAIL [{self.check}] {self.file}:{self.line}: {self.text}"


def judged_files(root: Path) -> list[Path]:
    files = [root / n for n in ("README.md", "DEMO.md", "ARCHITECTURE.md") if (root / n).exists()]
    docs = root / "docs"
    if docs.exists():
        files += sorted(p for p in docs.rglob("*") if p.suffix in (".md", ".html") and p.is_file())
    return files


def load_receipts(root: Path) -> list[dict]:
    out: list[dict] = []
    res = root / "bench" / "results"
    if not res.exists():
        return out
    for p in sorted(res.rglob("*")):
        if p.suffix == ".jsonl":
            out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif p.suffix == ".json":
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return out


def _values(obj) -> list[str]:
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _values(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in _values(x)]
    return [str(obj)] if obj is not None else []


def _sentences(text: str) -> list[tuple[int, str]]:
    """(line number, sentence) for prose; code blocks skipped."""
    out, in_code = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        out += [(n, s) for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return out


def check_files(root: Path, stage: str, collected_tests: int | None = None) -> list[Finding]:
    findings: list[Finding] = []
    receipts = load_receipts(root)
    receipt_text = "\n".join(_values(receipts)).lower()
    stage_amd = {
        "asr": any(r.get("stage") == "asr" and r.get("on_amd") and r.get("ok") for r in receipts),
        "extract": any(r.get("stage") == "extract" and r.get("on_amd") and r.get("ok") for r in receipts),
        "gpu": any(r.get("gpu") for r in receipts if isinstance(r, dict)),
    }
    any_amd = stage_amd["asr"] or stage_amd["extract"]

    for f in judged_files(root):
        rel = str(f.relative_to(root))
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        for n, line in enumerate(lines, 1):
            for m in PLACEHOLDER.finditer(line):
                findings.append(Finding("placeholder", rel, n, m.group(0)))
            for m in re.finditer(r"bench/results/[\w./-]+", line):
                path = m.group(0).rstrip(".,)")
                if not (root / path).exists():
                    findings.append(Finding("results", rel, n, f"cites missing {path}"))
            for lit in HARDWARE_LITERALS:
                if lit.lower() in line.lower() and lit.lower() not in receipt_text:
                    findings.append(Finding("hardware", rel, n, f"'{lit}' with no receipt containing it"))
            if NUMBER_CLAIM.search(line) and not NUMBER_OK.search(line):
                findings.append(Finding("number", rel, n, line.strip()[:100]))
        for n, s in _sentences(text):
            if "AMD" not in s or OFF_AMD.search(s):
                continue
            if AMD_HW.search(s) and not stage_amd["gpu"]:
                findings.append(Finding("amd-stage", rel, n, "claims AMD hardware; no receipt names a GPU"))
            elif ASR_WORDS.search(s) and not stage_amd["asr"]:
                findings.append(Finding("amd-stage", rel, n, "ASR on AMD with no AMD ASR receipt"))
            elif LLM_WORDS.search(s) and not stage_amd["extract"]:
                findings.append(Finding("amd-stage", rel, n, "extraction on AMD with no AMD extraction receipt"))
            elif not any_amd:
                findings.append(Finding("amd-stage", rel, n, "names AMD; no receipt has on_amd = true"))
        for k, para in enumerate(re.split(r"\n\s*\n", text)):
            if "hero" in para.lower() and ("B ≠ A" in para or "line: B" in para or "two-oh-three, line Alpha" in para):
                findings.append(Finding("hero", rel, 0, f"paragraph {k + 1} pairs the hero with the old line swap"))

    readme = root / "README.md"
    stated = TESTS_STATED.findall(readme.read_text(encoding="utf-8")) if readme.exists() else []
    if stage == "submit" and not stated:
        findings.append(Finding("tests", "README.md", 0, "no test count stated"))
    if stated and collected_tests is not None and int(stated[0]) != collected_tests:
        findings.append(
            Finding("tests", "README.md", 0, f"states {stated[0]} tests, pytest collects {collected_tests}")
        )
    if stage == "submit":
        res = root / "bench" / "results"
        if not list(res.glob("eval-*.json")):
            findings.append(Finding("results", "bench/results", 0, "no eval-*.json results"))
        if not (res / "access-gate.json").exists():
            findings.append(Finding("results", "bench/results", 0, "no access-gate.json"))
    return findings


def collect_tests(root: Path) -> int | None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=root, capture_output=True, text=True, check=False
    )
    m = re.search(r"(\d+) tests? collected", proc.stdout)
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("build", "submit"), default="build")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--no-collect", action="store_true", help="skip running pytest --collect-only")
    args = ap.parse_args(argv)
    n = None if args.no_collect else collect_tests(args.root)
    findings = check_files(args.root, args.stage, n)
    for f in findings:
        print(f)
    print(
        f"readiness ({args.stage}): {'FAIL' if findings else 'PASS'} · {len(findings)} finding(s)"
        + (f" · {n} tests collected" if n is not None else "")
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
