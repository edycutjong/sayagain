"""Export stored runs as a static replay of the supervisor queue (GitHub Pages-ready; nothing is published here).

    uv run python scripts/export_run.py                                  # default runs -> docs/replay/
    uv run python scripts/export_run.py --runs ladder-r1-reaudit:regex,ladder-r1-reaudit:Gemma-4-31B-IT --out /tmp/r

The replay is the same page as the live queue (web/): it reads config.json (mode "replay") and one
data/<run>--<rung>.json per rung = run metadata + the event list the server would stream (sayagain/shift.py).
The referee runs here, at export, over each run's stored model output; every card keeps its verdict trace and the
receipt of every model call (request id, model served, host, latency, cost). Nothing runs a model in the browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sayagain import shift  # noqa: E402

WEB = ROOT / "web"
DEFAULT_OUT = ROOT / "docs" / "replay"
# the five ladder rungs after the re-audit, and Gemma's as-run first pass (its gateway failures show NOT_AUDITED)
DEFAULT_RUNS = (
    "ladder-r1-reaudit:Gemma-4-31B-IT",
    "ladder-r1-reaudit:regex",
    "ladder-r1-reaudit:DeepSeek-V4-Flash",
    "ladder-r1-reaudit:Qwen3.8-27B",
    "ladder-r1-reaudit:MiniCPM5-2B",
    "ladder-r1:Gemma-4-31B-IT",
)


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export(pairs: list[tuple[str, str]], out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB, out, ignore=shutil.ignore_patterns(".DS_Store"))
    (out / "data").mkdir()
    runs: dict[str, list[str]] = {}
    report = {}
    for run, rung in pairs:
        events = list(shift.shift_events(run, rung))
        judged = [e["card"] for e in events if e["type"] == "judged"]
        differs = [c["exchange_id"] for c in judged if not c.get("matches_stored")]
        if differs:  # the replay must show what the run measured: never ship a re-judged result that disagrees
            raise SystemExit(f"{run}/{rung}: re-judged label differs from the stored run on {differs[:5]}")
        payload = {"meta": shift.run_meta(run, rung), "events": events}
        (out / "data" / f"{run}--{rung}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        runs.setdefault(run, []).append(rung)
        report[f"{run}/{rung}"] = {"cards": len(judged), "live_cards": sum(c["live"] for c in judged)}
    first_run, first_rung = pairs[0]
    config = {
        "mode": "replay",
        "runs": [{"run": r, "rungs": rs, "n_cards": shift._summary(r).get("n_cards")} for r, rs in runs.items()],
        "default": {"run": first_run, "rung": first_rung},
        "live_extract": False,
        "refused": "R-01",
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": git_head(),
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="run:rung pairs, comma-separated; first = default")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    pairs = []
    for item in a.runs.split(","):
        run, _, rung = item.strip().partition(":")
        if not run or not rung:
            ap.error(f"bad run:rung {item!r}")
        pairs.append((run, rung))
    report = export(pairs, a.out)
    for k, v in report.items():
        print(f"{k}: {v['cards']} cards, {v['live_cards']} to the live queue")
    try:
        shown = a.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = a.out
    print(f"replay written to {shown} (open with any static file server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
