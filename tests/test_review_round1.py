"""Regression tests for the P1 review round 1: every path that could pass a real error, or make a phantom card."""

import pytest
from conftest import judge_texts, make_turn

from referee.audit import Exchange, audit, judge
from referee.registry import Registry, Row
from referee.turns import analyze
from referee.verdict import CanonCommand, validate_instruction

HERO_I = "Close valve two-oh-three, line Bravo."
OK_RB = "Copy, closing two-oh-three, line Bravo."


def run(registry, *turns):
    ts = [make_turn(x[1], t=x[0], tid=f"u{i}") for i, x in enumerate(turns)]
    return audit(ts, registry)


@pytest.mark.parametrize(
    "instruction, readback",
    [
        ("Do not open valve two-oh-three, line Bravo.", "Copy, opening valve two-oh-three, line Bravo."),
        ("Jangan tutup valve dua nol tiga.", "Siap, tutup valve dua nol tiga."),
        (HERO_I, "Copy, unable to close two-oh-three, line Bravo."),
        (HERO_I, "Copy, will not close two-oh-three, line Bravo."),
        (HERO_I, "Copy, not closing two-oh-three, line Bravo."),
        (HERO_I, "Copy, I can't close two-oh-three, line Bravo."),
    ],
)
def test_negated_command_never_verifies(registry, instruction: str, readback: str) -> None:
    v = judge_texts(registry, instruction, readback)
    assert v.verdict == "ABSTAIN" and v.reason == "coverage", v.trace


def test_operator_correction_cue_is_not_a_negation(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, "Copy, closing two-oh-eight, line Bravo."),
        (6, "Tidak, close valve two-oh-three, line Bravo."),
        (9, OK_RB),
    )
    assert rep.judged[0].verdict.verdict == "VERIFIED" and len(rep.judged[0].caught) == 1


def _model(*cmds, kind="readback"):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {"kind": kind, "retracted_spans": [], "commands": [dict(zip(keys, c, strict=True)) for c in cmds]}


def test_model_swapping_actions_between_commands_cannot_verify(registry) -> None:
    ins = analyze(make_turn("Close valve two-oh-three and open valve two-oh-eight."), registry)
    text = "Copy, opening two-oh-three and closing two-oh-eight."  # a real error: both actions swapped
    swapped = _model(("closing", "two-oh-three", None, None), ("opening", "two-oh-eight", None, None))
    rb = analyze(make_turn(text, t=3, extraction=swapped), registry)
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "ABSTAIN" and v.reason == "grounding"
    honest = analyze(make_turn(text, t=3), registry)  # the regex rung binds by position
    assert judge(Exchange("CH-07", ins, honest)).verdict == "MISMATCH"


def test_model_swapping_lines_between_commands_cannot_verify(registry) -> None:
    ins_text = "Close valve two-oh-three, line Bravo and close valve two-oh-eight, line Alpha."
    ins = analyze(make_turn(ins_text), registry)
    text = "Copy, closing two-oh-three, line Alpha and closing two-oh-eight, line Bravo."
    swapped = _model(("closing", "two-oh-three", None, "line Bravo"), ("closing", "two-oh-eight", None, "line Alpha"))
    rb = analyze(make_turn(text, t=3, extraction=swapped), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict == "ABSTAIN"


def test_lone_nato_letter_is_not_ignored(registry) -> None:
    v = judge_texts(registry, "Close valve two-oh-three.", "Copy, closing two-oh-three, Alpha.")
    assert v.verdict != "VERIFIED"


def test_restated_value_after_correction_grounds(registry) -> None:
    v = judge_texts(registry, "Close valve two-oh-three, line Bravo, correction, line Bravo.", OK_RB)
    assert v.verdict == "VERIFIED", v.trace


def test_ambiguous_bare_number_abstains() -> None:
    rows = [Row("XV-203", "valve", "B", "x", ("CLOSE",), ()), Row("FV-203", "valve", "A", "x", ("CLOSE",), ())]
    reg = Registry(rows)
    v = judge_texts(reg, HERO_I, OK_RB)
    assert v.verdict == "ABSTAIN" and v.reason == "ambiguous tag"
    v = judge_texts(reg, "Close two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.")
    assert v.verdict == "ABSTAIN" and v.reason == "ambiguous tag"


def test_correction_without_new_readback_keeps_the_wrong_readback_live(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Copy, closing two-oh-eight, line Bravo."), (6, "Negative."))
    j = rep.judged[0]
    assert j.verdict.label == "MISMATCH · tag: XV-203 ≠ XV-208" and j.live
    assert "correction not read back" in j.exchange.events and j.caught == []


def test_no_hold_on_is_not_an_operator_correction(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, OK_RB), (6, "No, hold on, stand by."))
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED" and j.caught == [] and rep.live_queue == []


@pytest.mark.parametrize("gap, paired", [(8.0, True), (8.05, False)])
def test_window_boundary(registry, gap: float, paired: bool) -> None:
    rep = run(registry, (0, HERO_I), (2.0 + gap, OK_RB))  # instruction ends at t = 2.0
    assert (rep.judged[0].verdict.verdict == "VERIFIED") is paired


def test_bare_unknown_number_still_makes_a_card(registry) -> None:
    rep = run(registry, (0, "Close two-oh-five, line Charlie."), (3, "Copy, closing two-oh-five, line Charlie."))
    assert rep.judged[0].verdict.label == "ABSTAIN · unknown tag"


def test_orphan_readback_is_not_a_live_card(registry) -> None:
    rep = run(registry, (0, OK_RB))
    assert rep.judged == [] and rep.live_queue == [] and len(rep.end_of_shift["orphan_readbacks"]) == 1


def test_disagreement_gate_compares_resolved_tags(registry) -> None:
    ins = analyze(make_turn(HERO_I, alt_transcript="Close valve 203, line Bravo."), registry)
    rb = analyze(
        make_turn("Copy, closing valve two-oh-three, line Bravo.", t=3, alt_transcript="Copy, closing 203, line B."),
        registry,
    )
    assert not ins.disagrees and not rb.disagrees
    assert judge(Exchange("CH-07", ins, rb)).verdict == "VERIFIED"


def test_instruction_without_action_abstains() -> None:
    assert validate_instruction([CanonCommand(None, "XV-203", "ok", "B")]) == "unmapped: no action"
