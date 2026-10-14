"""Regression tests for the P1 review round 4 (the loop cap: verified by these tests and the gates)."""

import pytest
from conftest import make_turn

from referee.audit import Exchange, audit, judge
from referee.turns import analyze

HERO_I = "Close valve two-oh-three, line Bravo."
OK_RB = "Copy, closing two-oh-three, line Bravo."


def run(registry, *turns, **kw):
    return audit([make_turn(x[1], t=x[0], tid=f"u{i}") for i, x in enumerate(turns)], registry, **kw)


def _model(*cmds):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {"kind": "readback", "retracted_spans": [], "commands": [dict(zip(keys, c, strict=True)) for c in cmds]}


@pytest.mark.parametrize(
    "instruction, readback, cmd",
    [
        (
            HERO_I,
            "Copy, closing valve two-oh-three-eight, line Bravo.",
            ("closing", "valve two-oh-three", None, "line Bravo"),
        ),
        (HERO_I, "Copy, closing pump two-oh-three, line Bravo.", ("closing", "two-oh-three", None, "line Bravo")),
        ("Stop pump one-oh-three.", "Copy, stopping valve one-oh-three.", ("stopping", "one-oh-three", None, None)),
    ],
)
def test_shortened_tag_span_cannot_verify(registry, instruction: str, readback: str, cmd: tuple) -> None:
    ins = analyze(make_turn(instruction), registry)
    rb = analyze(make_turn(readback, t=3, extraction=_model(cmd)), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict != "VERIFIED"


@pytest.mark.parametrize("middle", ["and, once done, opening", "and after that opening"])
def test_report_back_word_cannot_hide_a_real_action(registry, middle: str) -> None:
    ins = analyze(make_turn("Close valve two-oh-three and valve two-oh-eight."), registry)
    text = f"Copy, closing valve two-oh-three {middle} valve two-oh-eight."
    model = _model(("closing", "valve two-oh-three", None, None), (None, "valve two-oh-eight", None, None))
    rb = analyze(make_turn(text, t=3, extraction=model), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict != "VERIFIED"


def test_report_back_past_participle_is_still_allowed(registry) -> None:
    text = "Roger, closing valve two-oh-three, line Bravo, I will confirm when closed."
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(
        make_turn(text, t=3, extraction=_model(("closing", "valve two-oh-three", None, "line Bravo"))), registry
    )
    assert judge(Exchange("CH-07", ins, rb)).verdict == "VERIFIED"


@pytest.mark.parametrize("opener", ["No problem, open", "Negative, open"])
def test_unrelated_order_does_not_demote_a_wrong_readback(registry, opener: str) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, "Copy, closing valve two-oh-eight, line Bravo."),
        (6, f"{opener} valve two-oh-seven, line Alpha."),
        (9, "Copy, opening valve two-oh-seven, line Alpha."),
    )
    first = rep.judged[0]
    assert first.verdict.label == "MISMATCH · tag: XV-203 ≠ XV-208" and first.live and first.caught == []


@pytest.mark.parametrize("status", ["Valve two-oh-three is now closed.", "Pump one-forty is running."])
def test_status_talk_after_a_correct_readback_is_no_card(registry, status: str) -> None:
    rep = run(registry, (0, HERO_I), (3, OK_RB), (5, status))
    assert [j.verdict.verdict for j in rep.judged] == ["VERIFIED"] and rep.live_queue == []


def test_correct_then_new_order_is_a_new_exchange(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, OK_RB),
        (5, "Correct, proceed. Then open valve two-oh-eight, line Alpha."),
        (8, "Copy, opening valve two-oh-eight, line Alpha."),
    )
    assert [j.verdict.verdict for j in rep.judged] == ["VERIFIED", "VERIFIED"]


def test_roger_then_readback_is_judged_on_the_readback(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Roger."), (4.5, "Closing valve two-oh-three, line Bravo."))
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED" and "ack without read-back" in j.exchange.events


def test_roger_without_readback_is_a_card(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Roger."))
    assert rep.judged[0].verdict.verdict == "NO_READBACK" and rep.live_queue


def test_negative_spot_check_count_is_clamped(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, OK_RB), spot_checks=-1)
    assert rep.end_of_shift["spot_checks"] == []
