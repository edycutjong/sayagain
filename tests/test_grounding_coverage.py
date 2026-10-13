"""Grounding (invented words) and coverage (wrong real words) — the model only points at words."""

import pytest

from referee.coverage import CoverageError, check
from referee.grounding import GroundingError, ground, occurrences

T = "Close valve two-oh-three and close valve two-one-six, line Bravo."


def ex(*cmds, retracted=()):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {
        "kind": "instruction",
        "commands": [dict(zip(keys, c, strict=True)) for c in cmds],
        "retracted_spans": list(retracted),
    }


def test_repeated_verb_multi_command() -> None:
    """'close … and close …' aligns the two verbs to two different words, left to right."""
    g = ground(T, ex(("close", "valve two-oh-three", None, None), ("close", "valve two-one-six", None, "line Bravo")))
    (a1, a2) = (g.commands[0]["action_span"], g.commands[1]["action_span"])
    assert T[a1[0] : a1[1]] == "Close" and T[a2[0] : a2[1]] == "close" and a1[0] < a2[0]


def test_shared_line_span_is_allowed() -> None:
    text = "Close valve two-oh-three and valve two-one-six, line Bravo."
    g = ground(
        text,
        ex(("Close", "valve two-oh-three", None, "line Bravo"), ("Close", "valve two-one-six", None, "line Bravo")),
    )
    assert g.commands[0]["line_span"] == g.commands[1]["line_span"]


def test_invented_span_is_ungrounded() -> None:
    with pytest.raises(GroundingError):
        ground(T, ex(("close", "valve two-oh-eight", None, None)))


def test_out_of_order_commands_are_ungrounded() -> None:
    with pytest.raises(GroundingError):
        ground(T, ex(("close", "valve two-one-six", None, None), ("close", "valve two-oh-three", None, None)))


def test_matching_ignores_case_punctuation_and_hyphens_but_not_partial_words() -> None:
    assert occurrences("Copy, closing two-oh-eight, line Bravo.", "Two oh eight") == [(14, 26)]
    assert occurrences("Copy, closing 2030.", "203") == []
    assert occurrences("closed", "close") == []


def test_uncovered_line_mention_fails_coverage() -> None:
    text = "Close valve two-oh-three, line Alpha, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Bravo")))
    with pytest.raises(CoverageError) as e:
        check(text, g)
    assert e.value.reason == "coverage"


def test_model_picking_the_wrong_real_word_fails_coverage() -> None:
    # the model returns "line Alpha" as operative and silently drops the correction to Bravo
    text = "Close valve two-oh-three, line Alpha, correction, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Alpha")))
    with pytest.raises(CoverageError):
        check(text, g)


def test_retraction_with_cue_is_accepted() -> None:
    text = "Close valve two-oh-three, line Alpha, correction, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Bravo"), retracted=["line Alpha"]))
    check(text, g)


def test_retraction_without_cue_is_refused() -> None:
    text = "Close valve two-oh-three, line Alpha, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Bravo"), retracted=["line Alpha"]))
    with pytest.raises(CoverageError) as e:
        check(text, g)
    assert e.value.reason == "retraction cue missing"


def test_say_again_is_not_a_retraction_cue() -> None:
    text = "Close valve two-oh-three, line Alpha, say again, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Bravo"), retracted=["line Alpha"]))
    with pytest.raises(CoverageError):
        check(text, g)


def test_quantities_and_times_need_no_coverage() -> None:
    text = "Close valve two-oh-three at 2:03, pressure is 150 psi, line Bravo."
    g = ground(text, ex(("Close", "valve two-oh-three", None, "line Bravo")))
    check(text, g)
