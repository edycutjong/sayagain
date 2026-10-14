"""Regression tests for the P1 review round 2."""

import pytest
from conftest import judge_texts, make_turn

from referee.audit import Exchange, audit, judge
from referee.turns import analyze

HERO_I = "Close valve two-oh-three, line Bravo."
OK_RB = "Copy, closing two-oh-three, line Bravo."
WRONG_RB = "Copy, closing two-oh-eight, line Bravo."


def run(registry, *turns):
    ts = [make_turn(x[1], t=x[0], tid=f"u{i}") for i, x in enumerate(turns)]
    return audit(ts, registry)


@pytest.mark.parametrize(
    "readback",
    [
        "Copy, no closing two-oh-three, line Bravo.",
        "Copy, couldn't close two-oh-three, line Bravo.",
        "Copy, wouldn't close two-oh-three, line Bravo.",
        "Copy, haven't closed two-oh-three, line Bravo.",
        "Copy, failed to close two-oh-three, line Bravo.",
        "Copy, cancel closing two-oh-three, line Bravo.",
        "Copy, negative on closing two-oh-three, line Bravo.",
        "Copy, closing two-oh-three, line Bravo, negative.",
        "Negative, closing two-oh-three, line Bravo.",
        "Copy, ga bisa tutup dua nol tiga, line Bravo.",
        "Copy, tdk bisa tutup dua nol tiga, line Bravo.",
        "Tidak menutup valve dua nol tiga, jalur be.",
    ],
)
def test_more_negations_never_verify(registry, readback: str) -> None:
    assert judge_texts(registry, HERO_I, readback).verdict == "ABSTAIN"


@pytest.mark.parametrize(
    "instruction",
    ["Hold off closing valve two-oh-three, line Bravo.", "Cancel closing valve two-oh-three, line Bravo."],
)
def test_negated_instruction_never_verifies(registry, instruction: str) -> None:
    assert judge_texts(registry, instruction, OK_RB).verdict == "ABSTAIN"


def _model(*cmds):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {"kind": "readback", "retracted_spans": [], "commands": [dict(zip(keys, c, strict=True)) for c in cmds]}


@pytest.mark.parametrize(
    "text, cmd",
    [
        (
            "Copy, closing two-oh-eight and two-oh-three, line Bravo.",
            ("closing two-oh-eight", "two-oh-three", None, "line Bravo"),
        ),
        (
            "Copy, closing two-oh-three, line Bravo and opening two-oh-eight.",
            ("closing", "two-oh-three", None, "line Bravo and opening two-oh-eight"),
        ),
        (
            "Copy, closing line Alpha two-oh-three, line Bravo.",
            ("closing line Alpha", "two-oh-three", None, "line Bravo"),
        ),
    ],
)
def test_wide_spans_cannot_hide_safety_words(registry, text: str, cmd: tuple) -> None:
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn(text, t=3, extraction=_model(cmd)), registry)
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "ABSTAIN" and v.reason == "coverage"


@pytest.mark.parametrize(
    "instruction",
    [
        "Okay, close valve two-oh-three, line Bravo.",
        "Roger, close valve two-oh-three, line Bravo.",
        "Baik, tutup valve dua nol tiga, jalur Be.",
    ],
)
def test_instruction_opening_with_a_cue_is_not_an_orphan(registry, instruction: str) -> None:
    rep = run(registry, (0, instruction), (3, WRONG_RB))
    assert [j.verdict.verdict for j in rep.judged] == ["MISMATCH"] and rep.live_queue


@pytest.mark.parametrize(
    "chatter",
    [
        "I closed valve two-oh-three earlier, line Bravo.",
        "Pump one-forty stopped by itself, over.",
        "Valve two-oh-three was closed this morning.",
        "Valve dua nol delapan sudah ditutup dari tadi.",
    ],
)
def test_status_talk_opens_no_window(registry, chatter: str) -> None:
    rep = run(registry, (0, chatter))
    assert rep.judged == [] and rep.live_queue == []


def test_bahasa_passive_order_opens_a_window(registry) -> None:
    rep = run(registry, (0, "Tolong valve dua nol tiga ditutup, jalur Be."), (3, WRONG_RB))
    assert rep.judged and rep.judged[0].verdict.verdict == "MISMATCH"


def test_stray_negative_then_chatter_is_not_a_phantom_card(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, OK_RB), (6, "Negative."), (8, "Where are you?"))
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED"  # the correct read-back stands; chatter is not a read-back


def test_negative_then_unrelated_order_keeps_the_error_live(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, WRONG_RB),
        (6, "Negative."),
        (8, "Open valve two-oh-eight, line Alpha."),
        (11, "Copy, opening two-oh-eight, line Alpha."),
    )
    labels = [j.verdict.label for j in rep.judged]
    assert labels == ["MISMATCH · tag: XV-203 ≠ XV-208", "VERIFIED"]
    assert rep.judged[0].live


def test_not_audited_window_does_not_swallow_the_next_instruction(registry) -> None:
    ts = [
        make_turn("Where are you?", tid="a", not_audited="rate limited"),
        make_turn(HERO_I, t=3, tid="b"),
        make_turn(WRONG_RB, t=6, tid="c"),
    ]
    rep = audit(ts, registry)
    assert [j.verdict.verdict for j in rep.judged] == ["NOT_AUDITED", "MISMATCH"]
    assert rep.judged[1].live


def test_stand_by_extends_the_window(registry) -> None:
    rep = run(registry, (0, HERO_I), (4, "Stand by."), (11, OK_RB))
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED" and "stand by" in j.exchange.events
