"""Regression tests from code review round 5 (architecture review of the whole referee)."""

import pytest

from referee.audit import audit
from referee.coverage import CoverageError, check
from referee.grounding import ground
from referee.registry import Registry
from referee.turns import Turn

REG = Registry.load()
INSTR = "Stop pump one-seven-four, line Charlie, correction, line Bravo."  # card A2-085
READBACK = "Roger, stopping pump one-seven-four, line Charlie."


def ex(*cmds, retracted=()):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {
        "kind": "instruction",
        "commands": [dict(zip(keys, c, strict=True)) for c in cmds],
        "retracted_spans": list(retracted),
    }


def test_a_retracted_verb_or_tag_needs_a_replacement_of_its_own_field() -> None:
    # an extractor that takes back the whole order before the cue keeps only the line: nothing replaces verb or tag
    g = ground(INSTR, ex((None, None, None, "line Bravo"), retracted=["Stop", "pump one-seven-four", "line Charlie"]))
    with pytest.raises(CoverageError) as e:
        check(INSTR, g)
    assert e.value.reason == "retraction cue missing"
    assert "nothing replaces the retracted action" in e.value.detail


def test_the_correct_reading_of_the_same_correction_still_passes() -> None:
    g = ground(INSTR, ex(("Stop", "pump one-seven-four", None, "line Bravo"), retracted=["line Charlie"]))
    check(INSTR, g)


def test_an_over_retracted_order_still_reaches_the_supervisor() -> None:
    """Before the fix this instruction opened no pairing window: the wrong line read back (C for B) left no card at all,
    only an orphan read-back on the end-of-shift list."""
    over = ex((None, None, None, "line Bravo"), retracted=["Stop", "pump one-seven-four", "line Charlie"])
    rb = {
        "kind": "readback",
        "commands": [
            {
                "action_span": "stopping",
                "tag_span": "pump one-seven-four",
                "state_span": None,
                "line_span": "line Charlie",
            }
        ],
        "retracted_spans": [],
    }
    turns = [Turn("i", "CH-01", 0.0, 2.5, INSTR, over), Turn("r", "CH-01", 3.5, 6.0, READBACK, rb)]
    rep = audit(turns, REG, spot_checks=0)
    assert [j.verdict.verdict for j in rep.judged] == ["ABSTAIN"]
    assert rep.live_queue and not rep.orphans


def test_a_backwards_retraction_cannot_verify_a_wrong_self_correction() -> None:
    """The technician corrects two-oh-three to the WRONG two-oh-eight; an extractor that takes back the later word keeps
    the right tag. Before the fix any later span after a cue counted as the replacement, and this read-back VERIFIED."""
    instr = "Close valve two-oh-three, line Bravo."
    rb_text = "Copy, closing two-oh-three, no, two-oh-eight, correction, line Bravo."
    ins = ex(("Close", "valve two-oh-three", None, "line Bravo"))
    rb = {**ex(("closing", "two-oh-three", None, "line Bravo"), retracted=["two-oh-eight"]), "kind": "readback"}
    turns = [Turn("i", "CH-01", 0.0, 2.5, instr, ins), Turn("r", "CH-01", 3.5, 6.0, rb_text, rb)]
    rep = audit(turns, REG, spot_checks=0)
    assert [j.verdict.label for j in rep.judged] == ["ABSTAIN · retraction cue missing"]
