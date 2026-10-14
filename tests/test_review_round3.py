"""Regression tests for the P1 review round 3."""

import json
import sys
from pathlib import Path

import pytest
from conftest import judge_texts, make_turn

from referee.audit import audit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_submission_readiness as cr  # noqa: E402

HERO_I = "Close valve two-oh-three, line Bravo."
OK_RB = "Copy, closing two-oh-three, line Bravo."


def run(registry, *turns):
    return audit([make_turn(x[1], t=x[0], tid=f"u{i}") for i, x in enumerate(turns)], registry)


def test_operator_repeat_with_query_is_not_the_readback(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, "Close valve two-oh-three, line Bravo, do you copy?"),
        (6, "Copy, closing two-oh-eight, line Alpha."),
    )
    j = rep.judged[0]
    assert j.verdict.verdict == "MISMATCH" and "repeat" in j.exchange.events


def test_second_readback_is_judged_not_dropped(registry) -> None:
    rep = run(
        registry,
        (0, "Close valve two-oh-three."),
        (3, "Copy, closing two-oh-three, line Bravo."),
        (6, "Copy, closing two-oh-eight, line Alpha."),
    )
    j = rep.judged[0]
    assert j.verdict.verdict == "MISMATCH" and "second read-back" in j.exchange.events


def test_operator_prompt_without_command_extends_the_window(registry) -> None:
    rep = run(registry, (0, HERO_I), (5, "Field two, do you copy?"), (12, OK_RB))
    assert rep.judged[0].verdict.verdict == "VERIFIED"


def test_back_to_back_orders_are_two_exchanges(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Close valve two-oh-eight, line Alpha."), (6, OK_RB))
    assert [j.verdict.verdict for j in rep.judged] == ["NO_READBACK", "MISMATCH"]


@pytest.mark.parametrize(
    "instruction, readback",
    [
        ("Close valve no 203 and valve no 208.", "Copy, closing valve 208."),
        ("Tutup valve no dua nol tiga dan valve no dua nol delapan.", "Siap, menutup valve dua nol delapan."),
        (
            "Close valve two-oh-three, line Bravo. No leaks reported. Open valve two-oh-eight, line Alpha.",
            "Copy, opening two-oh-eight, line Alpha.",
        ),
        (HERO_I, "Copy, closing two-oh-eight. No problem, two-oh-three, line Bravo."),
        (HERO_I, "Siap, menutup valve dua nol delapan. Bukan masalah, dua nol tiga, jalur Be."),
    ],
)
def test_ordinary_no_never_retracts(registry, instruction: str, readback: str) -> None:
    assert judge_texts(registry, instruction, readback).verdict != "VERIFIED"


def test_punctuated_no_still_retracts(registry) -> None:
    v = judge_texts(registry, "Close valve two-oh-three, line Alpha, no, line Bravo.", OK_RB)
    assert v.verdict == "VERIFIED", v.trace


@pytest.mark.parametrize(
    "readback",
    [
        "Copy, closing two-oh-three for line Bravo.",
        "Copy, closing two-oh-three to line Bravo.",
        "Copy, closing two-oh-three, one moment, line Bravo.",
        "Copy, closing valve no. two-oh-three, line Bravo.",
    ],
)
def test_number_runs_stop_at_punctuation_and_trailing_homophones(registry, readback: str) -> None:
    assert judge_texts(registry, HERO_I, readback).verdict == "VERIFIED"


def test_instruction_with_purpose_clause(registry) -> None:
    v = judge_texts(registry, "Close valve two-oh-three for maintenance, line Bravo.", OK_RB)
    assert v.verdict == "VERIFIED", v.trace


def test_decimal_is_not_a_clock_time(registry) -> None:
    assert judge_texts(registry, HERO_I, "Copy, closing 20.3, line Bravo.").verdict != "VERIFIED"


def test_report_back_phrase_is_not_a_second_command(registry) -> None:
    from referee.audit import Exchange, judge
    from referee.turns import analyze

    text = "Roger, closing valve two-oh-three, line Bravo, I will confirm when closed."
    model = {
        "kind": "readback",
        "retracted_spans": [],
        "commands": [
            {"action_span": "closing", "tag_span": "valve two-oh-three", "state_span": None, "line_span": "line Bravo"}
        ],
    }
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn(text, t=3, extraction=model), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict == "VERIFIED"


def _site(tmp_path: Path, readme: str, receipts: list[dict]) -> Path:
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "bench" / "results" / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in receipts))
    return tmp_path


def test_checker_clauses_numbers_and_booleans(tmp_path: Path) -> None:
    ok_extract = {"stage": "extract", "on_amd": True, "ok": True}
    root = _site(
        tmp_path,
        "The referee has no state and runs on AMD.\n"
        "The API lists no Whisper but extraction runs on AMD.\n"
        "Target: catches 94% of errors; measured: catches 99% of errors.\n"
        "Verdicts in 120 milliseconds.\nIt caught 47 of 50 errors.\nRuns on an MI-300X.\nResult: <measured>.\n",
        [ok_extract, {"stage": "asr", "on_amd": "false", "ok": True}, {"ok": False, "gpu": "MI300X"}],
    )
    got = sorted(f.check for f in cr.check_files(root, "build"))
    # the referee claim names no stage; the Whisper sentence has only an on_amd = "false" ASR receipt
    assert got == ["amd-stage", "amd-stage", "hardware", "number", "number", "number", "placeholder"]
