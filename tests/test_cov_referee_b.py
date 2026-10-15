"""Coverage for the last branches of referee.audit, grounding, pairing, turns and verdict."""

import math

import pytest
from conftest import make_turn

from referee import grounding
from referee.audit import Exchange, judge
from referee.pairing import pair
from referee.turns import Turn, analyze, analyze_canonical, span_confidence
from referee.verdict import ABSTAIN, NOT_AUDITED, R_UNKNOWN, CanonCommand, compare, validate_instruction

HERO_I = "Close valve two-oh-three, line Bravo."
HERO_RB = "Copy, closing two-oh-three, line Bravo."


def cmd(**spans) -> dict:
    base = {"action_span": None, "tag_span": None, "state_span": None, "line_span": None}
    return base | spans


def ext(*commands: dict, retracted: list[str] | None = None, kind: str = "instruction") -> dict:
    return {"kind": kind, "commands": list(commands), "retracted_spans": retracted or []}


# ------------------------------------------------------------------------------------------------ audit


def test_judge_unknown_audio_gate_fails_closed(registry) -> None:
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn(HERO_RB, t=3.0), registry)
    v = judge(Exchange("CH-07", ins, rb), audio_gate="bogus")
    assert v.verdict == NOT_AUDITED
    assert v.reason == "referee error"
    assert "audio_gate 'bogus'" in v.trace[0]


def test_missing_audio_input_with_invalid_instruction_keeps_instruction_reason(registry) -> None:
    ins = analyze(make_turn("Close valve nine-nine-nine."), registry)
    rb = analyze(make_turn("Copy, closing nine-nine-nine.", t=3.0), registry)
    assert ins.status == "ok" and rb.status == "ok"
    # a floor is configured but the turns carry no token probabilities: the gate input is missing
    v = judge(Exchange("CH-07", ins, rb), floor=0.5)
    assert v.verdict == ABSTAIN
    assert v.reason == R_UNKNOWN


# ------------------------------------------------------------------------------------------------ grounding


def test_ground_rejects_retracted_span_missing_from_transcript() -> None:
    with pytest.raises(grounding.GroundingError, match="retracted span 'open' is not in the transcript"):
        grounding.ground("Close valve two-oh-three.", ext(cmd(action_span="Close"), retracted=["open"]))


def test_ground_rejects_command_without_spans() -> None:
    with pytest.raises(grounding.GroundingError, match="a command with no spans"):
        grounding.ground("Close valve two-oh-three.", ext(cmd()))


# ------------------------------------------------------------------------------------------------ pairing


def test_repeated_say_again_while_awaiting_the_repeat_stays_in_one_exchange(registry) -> None:
    turns = [
        analyze(make_turn(HERO_I, t=0.0, tid="a"), registry),
        analyze(make_turn("Say again.", t=3.0, tid="b"), registry),
        analyze(make_turn("Say again.", t=5.0, tid="c"), registry),
        analyze(make_turn(HERO_I, t=7.0, tid="d"), registry),
    ]
    assert turns[1].say_again and turns[2].say_again
    out = pair(turns)
    assert len(out.exchanges) == 1
    ex = out.exchanges[0]
    assert ex.turn_ids == ["a", "b", "c", "d"]
    assert ex.events == ["say again", "say again", "repeat"]


# ------------------------------------------------------------------------------------------------ verdict


def test_validate_instruction_rejects_empty_list() -> None:
    assert validate_instruction([]) == R_UNKNOWN


def test_compare_reports_state_difference() -> None:
    a = CanonCommand("CLOSE", "XV-203", "ok", "B", "B", "OPEN")
    b = CanonCommand("CLOSE", "XV-203", "ok", "B", "B", "CLOSED")
    diffs = compare([a], [b])
    assert [(d.field, d.expected, d.got) for d in diffs] == [("state", "OPEN", "CLOSED")]


# ------------------------------------------------------------------------------------------------ turns


def test_analyze_without_extraction_is_not_audited(registry) -> None:
    t = Turn(id="x", channel="CH-07", t_start=0.0, t_end=2.0, transcript=HERO_I, extraction=None)
    a = analyze(t, registry)
    assert (a.status, a.reason) == ("not_audited", "no extraction")


def test_analyze_canonical_not_audited_paths(registry) -> None:
    failed = make_turn(HERO_I, extraction={"kind": "other", "commands": []}, not_audited="timeout")
    a = analyze_canonical(failed, registry)
    assert (a.status, a.reason) == ("not_audited", "timeout")
    t = Turn(id="x", channel="CH-07", t_start=0.0, t_end=2.0, transcript=HERO_I, extraction=None)
    a = analyze_canonical(t, registry)
    assert (a.status, a.reason) == ("not_audited", "no extraction")


def test_analyze_canonical_rejects_unknown_state_and_skips_empty_commands(registry) -> None:
    bad = {"kind": "instruction", "commands": [{"action": "CLOSE", "tag": "XV-203", "line": "B", "state": "AJAR"}]}
    a = analyze_canonical(make_turn(HERO_I, extraction=bad), registry)
    assert a.status == "abstain"
    assert a.detail == "state not in the canonical set"

    empty = {
        "kind": "instruction",
        "commands": [
            {"action": None, "tag": None, "line": None, "state": None},
            {"action": "CLOSE", "tag": "XV-203", "line": "B", "state": None},
        ],
    }
    a = analyze_canonical(make_turn(HERO_I, extraction=empty), registry)
    assert a.status == "ok"
    assert [c.tag for c in a.commands] == ["XV-203"]


def test_analyze_abstains_when_one_tag_span_holds_two_tags(registry) -> None:
    text = "Close valve two-oh-three and valve two-oh-eight."
    e = ext(cmd(action_span="Close", tag_span="valve two-oh-three and valve two-oh-eight"))
    a = analyze(make_turn(text, extraction=e), registry)
    assert a.status == "abstain"
    assert a.detail == "one tag span holds two tags"


def test_analyze_abstains_on_unmapped_line_span(registry) -> None:
    text = "Close valve two-oh-three, line banana."
    e = ext(cmd(action_span="Close", tag_span="valve two-oh-three", line_span="banana"))
    a = analyze(make_turn(text, extraction=e), registry)
    assert a.status == "abstain"
    assert a.detail == "line words not in the alias tables"


def test_span_confidence_skips_absent_spans(registry) -> None:
    text = "Close valve two-oh-three."
    e = ext(cmd(action_span="Close", tag_span="valve two-oh-three"))
    a = analyze(make_turn(text, extraction=e, tokens=[(0, len(text), math.log(0.9))]), registry)
    assert a.status == "ok" and a.spans[0]["line_span"] is None
    assert span_confidence([a]) == pytest.approx(0.9)
