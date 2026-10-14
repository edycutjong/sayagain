"""Readiness gate for every judge-facing file: no placeholder, no claim without a receipt.

    uv run python scripts/check_submission_readiness.py [--stage build|submit] [--root .]

Checks every public page (all .md / .html outside tests/, .github/ and caches):
  placeholder       unfilled slots: ⟨…⟩, TODO, TBD, FIXME, XXX, {{ }}, lorem ipsum
  results           every bench/results/… path a page cites exists; at `submit`, eval results + the access gate exist
  tests             every stated "N tests" equals what pytest collects (README must state one at `submit`)
  hardware          MI-series / Radeon / Instinct / ROCm / vLLM / "AMD Developer Cloud" / "co-resident" /
                    "channels per GPU" only if a receipt's gpu / hardware / stack field names it
  amd-stage         a sentence saying something runs on / is served by AMD maps to a receipt stage with on_amd = true
                    (speech sentences to an ASR receipt, model/extraction sentences to an extraction receipt);
                    "AMD GPU / hardware" needs a receipt that names the GPU — a hosted API does not expose one
  hero              the hero is the tag error (XV-203 read back as XV-208); the old line-swap hero strings fail
  number            a percentage, fraction, latency, p50/p95/p99 or "Nx faster" must appear in the file the line cites,
                    or the line says target / threshold / assumption (bench/PROTOCOL.md: thresholds, exempt)
  submit            (--stage submit only) measured result files exist; README.md links the live replay over https (the
                    published site, not the repository folder); no page says "under construction" / "not published" or
                    carries a dated slot ("scheduled Oct 16")
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

PLACEHOLDER = re.compile(
    r"⟨[^⟩]*⟩|<(?:measured|placeholder|tbd|todo|fill[^>]*)>|\bTODO\b|\bTBD\b|\bTBA\b|\bFIXME\b|\bXXX\b|"
    r"\bxx\s?%|\?\?\?|\{\{.*?\}\}|lorem ipsum|coming soon",
    re.IGNORECASE,
)
# hardware / serving-stack names: backed only by a receipt field that names hardware or the stack
HARDWARE = re.compile(
    r"\bMI[- ]?\d{3}[A-Z]?\b|\bRadeon\b|\bInstinct\b|\bROCm\b|\bvLLM\b|\bRyzen\b|\bStrix\b|\bgfx\d+\b|\bHIP\b|"
    r"\bEPYC\b|AMD Developer Cloud|co-resident|channels per GPU",
    re.IGNORECASE,
)
HW_KEYS = {"gpu", "hardware", "device", "rocm", "vllm", "serving_stack", "stack"}
AMD_CLAIM = re.compile(
    r"\b(?:runs?|ran|running|serve[sd]?|serving|powered|hosted|hosts?|accelerated|built on|deployed|execut\w+|"
    r"inference|uses?|using|used)\b|\bon AMD\b",
    re.IGNORECASE,
)
OFF_AMD = re.compile(
    r"off[- ]AMD|(?:\bnot|\bnever|\bno longer|cannot|can't)\s+(?:\w+\s+){0,3}(?:on|by|from|with)\s+"
    r"(?:an?\s+|the\s+)?AMD|\bnot AMD\b|\bwithout AMD\b|\b(?:serves|lists|offers|has|returns)\s+no\b",
    re.IGNORECASE,
)
AMD_HW = re.compile(r"\bAMD(?:'s)?\s+(?:\w+\s+)?(?:GPU|hardware|accelerator|silicon|chip|card)s?\b", re.IGNORECASE)
ASR_WORDS = re.compile(
    r"\b(?:ASR|STT|Whisper|transcri\w*|speech[- ]to[- ]text|speech recognition|voice|audio)\b", re.IGNORECASE
)
LLM_WORDS = re.compile(
    r"\b(?:extract\w*|LLM|language model|instruct models?|Qwen|model ladder|rungs?|span)\b", re.IGNORECASE
)
NUMBER_CLAIM = re.compile(
    r"(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)|\b(0\.\d+)\b|"
    r"(\d+(?:\.\d+)?)\s?(?:ms|milliseconds?|s|sec|secs|seconds?)\b|\bp(?:50|90|95|99)\b|"
    r"(\d+(?:\.\d+)?)\s?(?:x|×|times)\s+(?:faster|slower|cheaper|more|less|lower|higher)|(\d+)[- ]fold\b|"
    r"\b(\d+)\s+(?:of|out of)\s+\d+\b|\b(\d+)/\d+\b|(\d+(?:\.\d+)?)\s?(?:/s|/sec|per second)\b|"
    r"\bsub-second\b|\breal[- ]time\b",
    re.IGNORECASE,
)
# a target / threshold / assumption word right before the number (same clause) marks it as not a result
NUMBER_OK = re.compile(
    r"(?:\btarget|\bthreshold|assum\w*|≥|≤|\bpre-registered|\bat most|\bat least|\bbudget)[^.;,|]{0,25}$", re.IGNORECASE
)
NUMBER_EXEMPT = {"bench/PROTOCOL.md"}  # pre-registered thresholds and formulas, by definition not results
CITATION = re.compile(r"(?<![\w/])((?:bench|evidence|scripts|referee|sayagain|data|docs|tests)/[\w./-]+)")
TESTS_STATED = re.compile(r"\b(\d+)\s+(?:[a-z-]+\s+){0,2}tests?(?:\s+cases)?\b|\btests?:\s*(\d+)", re.IGNORECASE)
SKIP_DIRS = {".git", ".venv", "node_modules", "tests", ".github", "graphify-out", ".ruff_cache", ".pytest_cache"}


@dataclass
class Finding:
    check: str
    file: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"FAIL [{self.check}] {self.file}:{self.line}: {self.text}"


# a promise with a date ("audio eval scheduled Oct 16"): stale by the time judges read it
FUTURE_SLOT = re.compile(
    r"\bscheduled\s+(?:for\s+|on\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", re.I
)


def judged_files(root: Path) -> list[Path]:
    """Every public page: Markdown and HTML anywhere in the repo except tests, CI config and tool caches."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (
            p.is_file()
            and p.suffix.lower() in (".md", ".mdx", ".markdown", ".html", ".htm", ".txt", ".rst")
            and not (set(rel.parts[:-1]) & SKIP_DIRS)
            and rel.parts[0] != "LICENSE"
        ):
            out.append(p)
    return out


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


def _hw_values(obj, under_hw: bool = False) -> list[str]:
    """String values found under hardware / stack keys only (a URL path that happens to say 'radeon' backs nothing),
    and only from receipts that did not fail."""
    if isinstance(obj, dict):
        if obj.get("ok") is False:
            return []
        return [v for k, x in obj.items() for v in _hw_values(x, under_hw or str(k).lower() in HW_KEYS)]
    if isinstance(obj, list):
        return [v for x in obj for v in _hw_values(x, under_hw)]
    return [str(obj)] if under_hw and obj not in (None, "", False) else []


def _strip_code(line: str) -> str:
    return re.sub(r"`[^`]*`", "", line)


def _sentences(text: str) -> list[tuple[int, str]]:
    """(line number, sentence) for prose; fenced code skipped."""
    out, in_code = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        out += [(n, s) for s in re.split(r"(?<=[.!?])\s+|\s\|\s", line) if s.strip()]
    return out


def _number_backed(value: str, percent: bool, cited: list[str]) -> bool:
    for text in cited:
        if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", text):
            return True
        if percent:
            target = float(value) / 100
            for m in re.finditer(r"\d+\.\d+", text):
                if abs(float(m.group(0)) - target) < 0.0005:
                    return True
    return False


def check_files(root: Path, stage: str, collected_tests: int | None = None) -> list[Finding]:
    findings: list[Finding] = []
    receipts = load_receipts(root)
    hw_text = "\n".join(_hw_values(receipts)).lower()
    stage_amd = {
        "asr": any(
            isinstance(r, dict) and r.get("stage") == "asr" and r.get("on_amd") is True and r.get("ok") is True
            for r in receipts
        ),
        "extract": any(
            isinstance(r, dict) and r.get("stage") == "extract" and r.get("on_amd") is True and r.get("ok") is True
            for r in receipts
        ),
        "gpu": bool(hw_text.strip()),
    }
    stated_counts: list[tuple[str, int, int]] = []

    for f in judged_files(root):
        rel = str(f.relative_to(root))
        text = f.read_text(encoding="utf-8")
        in_code = False
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            for m in PLACEHOLDER.finditer(line):
                findings.append(Finding("placeholder", rel, n, m.group(0)))
            cites = [c.rstrip(".,):;") for c in CITATION.findall(line)]
            for c in cites:
                if c.startswith("bench/results/") and not (root / c).exists():
                    findings.append(Finding("results", rel, n, f"cites missing {c}"))
            for m in HARDWARE.finditer(line):
                if m.group(0).lower() not in hw_text:
                    findings.append(
                        Finding("hardware", rel, n, f"'{m.group(0)}' with no hardware/stack receipt naming it")
                    )
            for m in TESTS_STATED.finditer(line):
                stated_counts.append((rel, n, int(m.group(1) or m.group(2))))
            if in_code or rel in NUMBER_EXEMPT:
                continue
            prose = _strip_code(line)
            claims = list(NUMBER_CLAIM.finditer(CITATION.sub("", prose)))
            if not claims:
                continue
            cited_texts = [
                (root / c).read_text(encoding="utf-8", errors="ignore") for c in cites if (root / c).is_file()
            ]
            uncited = CITATION.sub("", prose)
            for m in claims:
                value = next((g for g in m.groups() if g), None)
                is_pct = "%" in m.group(0) or "cent" in m.group(0).lower()
                if NUMBER_OK.search(uncited[: m.start()]):
                    continue  # "target ≥ 90 %": a threshold, not a result
                if cited_texts and value is not None:
                    if not _number_backed(value, is_pct, cited_texts):
                        findings.append(Finding("number", rel, n, f"{m.group(0).strip()} not found in the cited file"))
                else:
                    findings.append(
                        Finding("number", rel, n, f"{m.group(0).strip()} — cite the result file: {prose.strip()[:80]}")
                    )
        for n, s in _sentences(text):
            if not re.search(r"\bAMD\b", s):
                continue
            # judge each clause on its own: "lists no Whisper but extraction runs on AMD" is a claim
            clauses = re.split(r"[,;:]|\bbut\b|\band\b|\bwhile\b|—", s)
            claims = [c for c in clauses if re.search(r"\bAMD\b", c) and AMD_CLAIM.search(c) and not OFF_AMD.search(c)]
            if not claims:
                continue
            if any(AMD_HW.search(c) for c in claims) and not stage_amd["gpu"]:
                findings.append(Finding("amd-stage", rel, n, "claims AMD hardware; no receipt names a GPU"))
            elif ASR_WORDS.search(s) and not stage_amd["asr"]:
                findings.append(Finding("amd-stage", rel, n, "speech-to-text on AMD with no AMD ASR receipt"))
            elif LLM_WORDS.search(s):
                if not stage_amd["extract"]:
                    findings.append(Finding("amd-stage", rel, n, "extraction on AMD with no AMD extraction receipt"))
            elif not ASR_WORDS.search(s):
                findings.append(
                    Finding("amd-stage", rel, n, "says something runs on AMD without naming a receipted stage")
                )
        for k, para in enumerate(re.split(r"\n\s*\n", text)):
            if "hero" in para.lower() and ("B ≠ A" in para or "line: B" in para or "two-oh-three, line Alpha" in para):
                findings.append(Finding("hero", rel, 0, f"paragraph {k + 1} pairs the hero with the old line swap"))

    readme_counts = [c for c in stated_counts if c[0] == "README.md"]
    if stage == "submit" and not readme_counts:
        findings.append(Finding("tests", "README.md", 0, "no test count stated"))
    if collected_tests is not None:
        for rel, n, count in stated_counts:
            if count != collected_tests:
                findings.append(Finding("tests", rel, n, f"states {count} tests, pytest collects {collected_tests}"))
    if stage == "submit":
        res = root / "bench" / "results"
        # measured rows: audio rows are eval-*.json; the text·scripted rows (no recordings needed) are text/*.json
        if not list(res.glob("eval-*.json")) and not list((res / "text").glob("*.json")):
            findings.append(Finding("results", "bench/results", 0, "no measured results (eval-*.json or text/*.json)"))
        if not (res / "access-gate.json").exists():
            findings.append(Finding("results", "bench/results", 0, "no access-gate.json"))
        # the README a judge opens first links the live replay: the published site, not the repository folder
        readme = root / "README.md"
        body = readme.read_text(encoding="utf-8") if readme.exists() else ""
        if not re.search(r"https://(?!github\.com/)\S+/replay/", body):
            findings.append(Finding("submit", "README.md", 0, "no https link to the live replay (…/replay/)"))
        # no public page still calls itself unfinished or unpublished, or carries a dated promise judges read too late
        for f in judged_files(root):
            rel = str(f.relative_to(root))
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                for phrase in ("under construction", "not published"):
                    if phrase in line.lower():
                        findings.append(Finding("submit", rel, n, f"still says '{phrase}'"))
                for m in FUTURE_SLOT.finditer(line):
                    findings.append(Finding("submit", rel, n, f"dated slot '{m.group(0)}'"))
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
