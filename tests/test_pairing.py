"""Pairing by turn order + 8 s window; say-again; the operator's third leg; chatter; routing."""

from conftest import make_turn

from referee.audit import audit
from referee.pairing import pair
from referee.turns import analyze, is_imperative, is_say_again

HERO_I = "Close valve two-oh-three, line Bravo."
OK_RB = "Copy, closing two-oh-three, line Bravo."


def run(registry, *turns, **kw):
    """turns: (t_start, text) or (t_start, text, channel)."""
    ts = [
        make_turn(x[1], t=x[0], channel=x[2] if len(x) > 2 else "CH-07", tid=f"u{i}", **kw) for i, x in enumerate(turns)
    ]
    return audit(ts, registry)


def test_chatter_opens_no_window(registry) -> None:
    rep = run(
        registry,
        (0, "Where are you now?"),
        (3, "At the boiler house, shift handover at 2:03."),
        (6, "Pump one-forty is running fine."),
        (9, "Roger, standing by."),
    )
    assert rep.judged == [] and len(rep.other) == 4


def test_instruction_then_readback_pairs(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Copy, closing two-oh-eight, line Bravo."))
    assert len(rep.judged) == 1 and rep.judged[0].exchange.turn_ids == ["u0", "u1"]
    assert rep.judged[0].verdict.label == "MISMATCH · tag: XV-203 ≠ XV-208"
    assert rep.live_queue == rep.judged


def test_verbatim_readback_pairs_without_cue(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, HERO_I))
    assert rep.judged[0].verdict.verdict == "VERIFIED"


def test_no_readback_inside_window(registry) -> None:
    rep = run(registry, (0, HERO_I), (11, OK_RB))  # 9 s after the instruction ended
    v = [j.verdict.verdict for j in rep.judged]
    assert v[0] == "NO_READBACK"


def test_readback_on_another_channel_does_not_pair(registry) -> None:
    rep = run(registry, (0, HERO_I, "CH-07"), (3, OK_RB, "CH-03"))
    assert rep.judged[0].verdict.verdict == "NO_READBACK"


def test_say_again_extends_and_repeat_replaces(registry) -> None:
    rep = run(
        registry,
        (0, "Close valve two-oh-three, line Alpha."),
        (3, "Say again?"),
        (5, "I say again, close valve two-oh-three, line Bravo."),
        (8, OK_RB),
    )
    j = rep.judged[0]
    assert j.exchange.events == ["say again", "repeat"]
    assert j.exchange.instruction.id == "u2"
    assert j.verdict.verdict == "VERIFIED"


def test_operator_repeat_is_never_the_readback(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, "Ulangi?"), (5, HERO_I), (8, "Copy, closing two-oh-eight, line Bravo."))
    j = rep.judged[0]
    assert j.exchange.readback.id == "u3"
    assert j.verdict.verdict == "MISMATCH"


def test_operator_correction_caught_on_channel(registry) -> None:
    """Third leg: the operator says 'negative' and repeats; the wrong read-back goes to the end-of-shift list."""
    rep = run(
        registry,
        (0, HERO_I),
        (3, "Copy, closing two-oh-eight, line Bravo."),
        (6, "Negative, close valve two-oh-three, line Bravo."),
        (9, OK_RB),
        (12, "Correct."),
    )
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED"
    assert [c.label for c in j.caught] == ["MISMATCH · tag: XV-203 ≠ XV-208"]
    assert rep.live_queue == [] and rep.end_of_shift["caught_on_channel"] == [j]
    assert j.exchange.turn_ids == ["u0", "u1", "u2", "u3", "u4"]


def test_operator_correction_with_repeat_in_next_turn(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, "Copy, closing two-oh-three, line Alpha."),
        (6, "Koreksi."),
        (8, "Tutup valve dua nol tiga, line Bravo."),
        (11, OK_RB),
    )
    j = rep.judged[0]
    assert j.verdict.verdict == "VERIFIED" and len(j.caught) == 1


def test_acknowledgement_closes_the_exchange(registry) -> None:
    rep = run(
        registry,
        (0, HERO_I),
        (3, OK_RB),
        (6, "Correct."),
        (9, "Stop pump one-one-four."),
        (12, "Copy, stopping pump one-one-four."),
    )
    assert [j.verdict.verdict for j in rep.judged] == ["VERIFIED", "VERIFIED"]


def test_back_to_back_instructions(registry) -> None:
    rep = run(registry, (0, HERO_I), (3, OK_RB), (6, "Stop pump one-one-four."), (9, "Copy, stopping pump one-forty."))
    assert [j.verdict.verdict for j in rep.judged] == ["VERIFIED", "MISMATCH"]


def test_refusal_opens_a_window_and_abstains(registry) -> None:
    rep = run(registry, (0, "Close valve two-oh-five, line Charlie."), (3, "Copy, closing two-oh-five, line Charlie."))
    assert rep.judged[0].verdict.label == "ABSTAIN · unknown tag"


def test_unreadable_instruction_still_makes_a_card(registry) -> None:
    """Fail closed: a turn naming a real tag + a verb that the model botched still opens a window."""
    bad = {
        "kind": "instruction",
        "retracted_spans": [],
        "commands": [{"action_span": "Open", "tag_span": "valve two-oh-three", "state_span": None, "line_span": None}],
    }
    ts = [make_turn(HERO_I, tid="u0", extraction=bad), make_turn(OK_RB, t=3, tid="u1")]
    rep = audit(ts, registry)
    assert rep.judged[0].verdict.verdict == "ABSTAIN" and rep.judged[0].verdict.reason == "grounding"


def test_not_audited_instruction_opens_a_window_and_goes_to_end_of_shift(registry) -> None:
    """An unreadable turn may be an instruction: its reply is paired to it (no phantom NO-READBACK card) and the
    exchange is NOT_AUDITED — end-of-shift list, never the live queue, never VERIFIED."""
    ts = [make_turn(HERO_I, tid="u0", not_audited="timeout"), make_turn(OK_RB, t=3, tid="u1")]
    rep = audit(ts, registry)
    assert [j.verdict.verdict for j in rep.judged] == ["NOT_AUDITED"]
    assert rep.live_queue == [] and rep.end_of_shift["not_audited"] == rep.judged
    assert rep.judged[0].exchange.turn_ids == ["u0", "u1"]


def test_spot_checks_sampled_from_verified(registry) -> None:
    turns = []
    for k in range(15):
        turns += [(k * 20, HERO_I), (k * 20 + 3, OK_RB)]
    rep = run(registry, *turns)
    assert len(rep.end_of_shift["spot_checks"]) == 10
    assert all(j.verdict.verdict == "VERIFIED" for j in rep.end_of_shift["spot_checks"])
    assert rep.live_queue == []


def test_code_kinds() -> None:
    assert is_say_again("Say again?") and is_say_again("Ulangi.") and not is_say_again("I say again, close valve 203")
    assert not is_say_again("Say again two-oh-three?")
    assert is_imperative("I say again, close valve two-oh-three") and is_imperative("Tolong tutup valve 203")
    assert is_imperative("Valve dua nol tiga tutup") and not is_imperative("Copy, closing two-oh-three")
    assert not is_imperative("Closing two-oh-three, line Bravo")


def test_pair_is_deterministic(registry) -> None:
    ts = [analyze(make_turn(x, t=i * 3, tid=f"u{i}"), registry) for i, x in enumerate([HERO_I, OK_RB, "Correct."])]
    a, b = pair(ts), pair(list(reversed(ts)))
    assert [e.turn_ids for e in a.exchanges] == [e.turn_ids for e in b.exchanges]
