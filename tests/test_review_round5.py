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


# ---------------------------------------------------------------------------------------------------------------------
# Property: a technician's self-corrections, read by an extractor that groups and retracts the turn's real safety words
# at random. The final reading is computable: a different value replaces the earlier one only when a cue was spoken
# after the earlier mention; with no cue in between, that field is ambiguous until a cue-led value settles it.
# The referee may abstain or flag at will, but it must never VERIFY a read-back whose final reading is wrong or open.

_VALUES = {
    "action_span": [("closing", "CLOSE"), ("opening", "OPEN")],
    "tag_span": [("two-oh-three", "XV-203"), ("two-oh-eight", "XV-208"), ("valve two-oh-three", "XV-203")],
    "line_span": [("line Bravo", "B"), ("line Alpha", "A")],
}
_FIELD = {"action": "action_span", "tag": "tag_span", "line": "line_span"}


def _self_corrected_readback(rng) -> tuple[str, dict, bool]:
    words, final, cue_since, open_field = [], {}, {}, {}
    fields = list(_VALUES)

    def say(f: str) -> None:
        w, v = rng.choice(_VALUES[f])
        if f in final and v != final[f]:
            open_field[f] = not cue_since[f]
        elif f in final and cue_since[f]:
            open_field[f] = False
        final[f], cue_since[f] = v, False
        words.append(w)

    for f in fields:
        say(f)
    for _ in range(rng.choice((1, 1, 2))):
        if rng.random() < 0.8:
            words.append(rng.choice(("correction", "no", "koreksi")))
            cue_since = dict.fromkeys(cue_since, True)
        say(rng.choice(fields))
    return "Copy, " + ", ".join(words) + ".", final, any(open_field.values())


def _random_extraction(text: str, rng) -> dict:
    from referee import aliases as al

    cmds, cur, retracted = [], {}, []
    for m in al.all_mentions(text):
        f = _FIELD.get(m.field)
        if f is None:
            continue
        if rng.random() < 0.35:
            retracted.append(text[m.start : m.end])
            continue
        if f in cur or rng.random() < 0.1:
            cmds.append(cur)
            cur = {}
        cur[f] = text[m.start : m.end]
    cmds.append(cur)
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {
        "kind": "readback",
        "commands": [{k: c.get(k) for k in keys} for c in cmds if c],
        "retracted_spans": retracted,
    }


def test_no_extraction_of_a_self_corrected_readback_verifies_a_wrong_final_reading() -> None:
    import random

    rng = random.Random(1018)
    instr = "Close valve two-oh-three, line Bravo."
    ins = ex(("Close", "valve two-oh-three", None, "line Bravo"))
    right = {"action_span": "CLOSE", "tag_span": "XV-203", "line_span": "B"}
    wrong = []
    for _ in range(3000):
        text, final, still_open = _self_corrected_readback(rng)
        rb = _random_extraction(text, rng)
        turns = [Turn("i", "CH-01", 0.0, 2.5, instr, ins), Turn("r", "CH-01", 3.5, 6.0, text, rb)]
        v = audit(turns, REG, spot_checks=0).judged[0].verdict.verdict
        if v == "VERIFIED" and (still_open or final != right):
            wrong.append((text, rb))
    assert wrong == []


# ---------------------------------------------------------------------------------------------------------------------
# Fail closed beyond judge(): a crash while reading one turn or pairing the turns is NOT_AUDITED, never a lost shift.


def _boom(*_a, **_k):
    raise RuntimeError("injected")


def _hero_card() -> dict:
    from sayagain import shift

    card = next(c for c in shift.load_cards() if c["exchange_id"] == "H-01")
    ext = shift.stored_extractor(shift.DEFAULT_RUN, shift.DEFAULT_RUNG)
    return shift.judge_card(card, ext, REG, shift.DEFAULT_RUNG)


def test_a_crash_while_reading_a_turn_is_not_audited(monkeypatch) -> None:
    from referee import coverage

    assert _hero_card()["verdict"]["verdict"] == "MISMATCH"
    monkeypatch.setattr(coverage, "check", _boom)  # before the fix the RuntimeError escaped the shift
    rec = _hero_card()
    assert rec["verdict"]["verdict"] == "NOT_AUDITED" and rec["verdict"]["reason"] == "referee error"


def test_a_crash_while_pairing_is_not_audited(monkeypatch) -> None:
    from referee import audit as audit_mod

    monkeypatch.setattr(audit_mod, "pair", _boom)
    rec = _hero_card()
    assert rec["verdict"]["verdict"] == "NOT_AUDITED"
    assert "referee error (pairing)" in rec["trace"][0]


# ---------------------------------------------------------------------------------------------------------------------
# An unreadable second read-back is judged (ABSTAIN), never swallowed as the acknowledgement after a good first one.


def test_an_unreadable_second_readback_is_judged_not_taken_as_the_ack() -> None:
    from referee.regex_extractor import extract as regex_extract
    from referee.turns import analyze

    texts = (
        "Close valve two-oh-three, line Bravo.",
        "Copy, closing valve two-oh-three, line Bravo.",
        "Copy, closing valve two-oh-eight, line Bravo.",
    )
    turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
    turns[2].extraction = {"_invalid": "not JSON"}  # the model failed on the last turn
    a = analyze(turns[2], REG)
    assert a.status == "abstain" and a.readback_shaped
    assert audit(turns[:2], REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED"
    # before the fix the unreadable turn closed the exchange as an "ack" and it stood VERIFIED
    assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "ABSTAIN"


def test_a_status_report_after_the_readback_still_closes_the_exchange() -> None:
    from referee.regex_extractor import extract as regex_extract

    texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", "Correct.")
    turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
    turns[2].extraction = {"_invalid": "not JSON"}  # unreadable, but not read-back shaped: an acknowledgement
    assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED"


# ---------------------------------------------------------------------------------------------------------------------
# The referee routes an orphan read-back (no order heard) to the end-of-shift list; the queue must carry it there.


def test_an_orphan_readback_reaches_the_end_of_shift_list() -> None:
    from sayagain import shift
    from sayagain.extract import regex_extraction

    card = {
        "exchange_id": "T-ORPHAN",
        "set": "C",
        "dev_slice": False,
        "lang": "en",
        "error_type": None,
        "turns": [{"role": "tech", "text": "Copy, closing valve two-oh-three, line Bravo."}],
    }
    events = list(shift.shift_events(cards=[card], extract=regex_extraction, check_stored=False))
    assert events[-2]["card"]["verdict"] is None and events[-2]["card"]["orphans"]  # no card, but not nothing
    assert events[-1]["orphan_readbacks"] == ["T-ORPHAN"]  # before the fix the queue dropped it
