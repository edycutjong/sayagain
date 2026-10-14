"""The referee truth table: precedence, field diff, fail-closed."""

import ast
from pathlib import Path

from conftest import judge_texts, make_turn

from referee.audit import Exchange, judge
from referee.turns import analyze
from referee.verdict import CanonCommand, verdict

HERO_I = "Close valve two-oh-three, line Bravo."


def test_hero_tag_mismatch(registry) -> None:
    """Synthetic half: the hero is always MISMATCH tag XV-203 ≠ XV-208 — never an unknown-tag ABSTAIN."""
    for rb in ("Copy, closing two-oh-eight, line Bravo.", "Copy, closing 208, line Bravo."):
        v = judge_texts(registry, HERO_I, rb)
        assert v.verdict == "MISMATCH", v.trace
        assert [str(d) for d in v.diffs] == ["tag: XV-203 ≠ XV-208"]
        assert v.diffs[0].note == "XV-208 is on line A"
        assert v.label == "MISMATCH · tag: XV-203 ≠ XV-208"


def test_hero_tag_mismatch_pure_tuples() -> None:
    i = [CanonCommand("CLOSE", "XV-203", "ok", "B", "B")]
    r = [CanonCommand("CLOSE", "XV-208", "ok", "A", "B")]
    v = verdict(i, r)
    assert v.verdict == "MISMATCH" and [d.field for d in v.diffs] == ["tag"]


def test_secondary_line_mismatch(registry) -> None:
    v = judge_texts(registry, HERO_I, "Copy, closing two-oh-three, line Alpha.")
    assert v.verdict == "MISMATCH"
    assert [str(d) for d in v.diffs] == ["line: B ≠ A"]


def test_clean_readback_verifies(registry) -> None:
    for rb in (
        "Copy, closing two-oh-three, line Bravo.",
        "Close valve two-oh-three, line Bravo.",
        "Siap, tutup dua nol tiga, jalur Be.",
    ):
        v = judge_texts(registry, HERO_I, rb)
        assert v.verdict == "VERIFIED", (rb, v.trace)


def test_code_switched_instruction_verifies(registry) -> None:
    v = judge_texts(registry, "Tolong tutup valve dua nol tiga, line Bravo.", "Copy, closing 203, line Bravo.")
    assert v.verdict == "VERIFIED", v.trace


def test_instruction_tag_wrong_line_abstains(registry) -> None:
    """The operator names a real tag on the wrong line: the instruction itself is unclear -> ABSTAIN(unknown tag)."""
    v = judge_texts(registry, "Close valve two-oh-three, line Alpha.", "Copy, closing two-oh-three, line Alpha.")
    assert v.verdict == "ABSTAIN" and v.reason == "unknown tag"


def test_refusal_unknown_tag(registry) -> None:
    """The guaranteed refusal: XV-205 is not in the registry -> ABSTAIN(unknown tag), on every run."""
    for rb in ("Copy, closing two-oh-five, line Charlie.", None):
        v = judge_texts(registry, "Close valve two-oh-five, line Charlie.", rb)
        assert v.verdict == "ABSTAIN" and v.reason == "unknown tag"


def test_readback_tag_not_in_registry_is_mismatch(registry) -> None:
    v = judge_texts(registry, HERO_I, "Copy, closing two-oh-five, line Bravo.")
    assert v.verdict == "MISMATCH" and str(v.diffs[0]) == "tag: 205 not in registry"


def test_wrong_action_is_mismatch(registry) -> None:
    v = judge_texts(registry, HERO_I, "Copy, opening two-oh-three, line Bravo.")
    assert [str(d) for d in v.diffs] == ["action: CLOSE ≠ OPEN"]


def test_omitted_line_is_mismatch(registry) -> None:
    v = judge_texts(registry, HERO_I, "Copy, closing two-oh-three.")
    assert v.verdict == "MISMATCH" and [str(d) for d in v.diffs] == ["line: missing (expected B)"]


def test_unstated_line_needs_no_readback_line(registry) -> None:
    v = judge_texts(registry, "Close valve two-oh-three.", "Copy, closing two-oh-three.")
    assert v.verdict == "VERIFIED"
    v = judge_texts(registry, "Close valve two-oh-three.", "Copy, closing two-oh-three, line Alpha.")
    assert [str(d) for d in v.diffs] == ["line: B ≠ A"]


def test_omitted_command_is_mismatch(registry) -> None:
    v = judge_texts(
        registry,
        "Close valve two-oh-three and close valve two-one-six, line Bravo.",
        "Copy, closing two-oh-three, line Bravo.",
    )
    assert v.verdict == "MISMATCH" and any(d.field == "command" and d.note == "omitted" for d in v.diffs)


def test_multi_command_readback_verifies(registry) -> None:
    v = judge_texts(
        registry, "Close valve two-oh-three and close valve two-one-six.", "Copy, closing two-oh-three and two-one-six."
    )
    assert v.verdict == "VERIFIED", v.trace


def test_corrected_instruction_compares_operative_value(registry) -> None:
    ins = "Close valve two-oh-three, line Alpha, correction, line Bravo."
    assert judge_texts(registry, ins, "Copy, closing two-oh-three, line Bravo.").verdict == "VERIFIED"
    v = judge_texts(registry, ins, "Copy, closing two-oh-three, line Alpha.")  # uncorrected retraction
    assert [str(d) for d in v.diffs] == ["line: B ≠ A"]


def test_lookalike_pairs(registry) -> None:
    v = judge_texts(registry, "Stop pump one-one-four.", "Copy, stopping pump one-forty.")
    assert [str(d) for d in v.diffs] == ["tag: P-114 ≠ P-140"]
    v = judge_texts(registry, "Close motor valve three-one Alpha.", "Copy, closing three-one Hotel.")
    assert [str(d) for d in v.diffs] == ["tag: MOV-31A ≠ MOV-31H"]


def test_no_readback(registry) -> None:
    assert judge_texts(registry, HERO_I, None).verdict == "NO_READBACK"


def test_audio_floor_abstains(registry) -> None:
    ins = analyze(make_turn(HERO_I, tokens=[(0, 99, -0.01)]), registry)
    rb = analyze(
        make_turn(
            "Copy, closing two-oh-three, line Bravo.", t=3, tokens=[(0, 13, -0.01), (14, 26, -2.0), (27, 39, -0.01)]
        ),
        registry,
    )
    v = judge(Exchange("CH-07", ins, rb), floor=0.5)
    assert v.verdict == "ABSTAIN" and v.reason == "audio"
    assert judge(Exchange("CH-07", ins, rb), floor=0.1).verdict == "VERIFIED"


def test_audio_gate_fails_closed_on_partial_token_data(registry) -> None:
    """With a floor set, a turn without token probabilities, or a span no token covers, abstains — never passes."""
    rb_text = "Copy, closing two-oh-three, line Bravo."
    ins = analyze(make_turn(HERO_I, tokens=[(0, 99, -0.01)]), registry)
    for tokens in (None, [], [(0, 13, -0.01)]):  # none · empty · the tag and line spans uncovered
        rb = analyze(make_turn(rb_text, t=3, tokens=tokens), registry)
        v = judge(Exchange("CH-07", ins, rb), floor=0.1)
        assert v.verdict == "ABSTAIN" and v.reason == "audio", tokens
    rb = analyze(make_turn(rb_text, t=3), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict == "VERIFIED"  # no floor configured: no audio gate


def test_disagreement_gate_abstains(registry) -> None:
    ins = analyze(make_turn(HERO_I, alt_transcript=HERO_I), registry)
    rb = analyze(
        make_turn(
            "Copy, closing two-oh-three, line Bravo.", t=3, alt_transcript="Copy, closing two-oh-eight, line Bravo."
        ),
        registry,
    )
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "ABSTAIN" and v.reason == "audio"


def test_ungrounded_model_output_abstains(registry) -> None:
    bad = {
        "kind": "readback",
        "retracted_spans": [],
        "commands": [
            {"action_span": "closing", "tag_span": "two-oh-three", "state_span": None, "line_span": "line Bravo"}
        ],
    }
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn("Copy, closing two-oh-eight, line Bravo.", t=3, extraction=bad), registry)
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "ABSTAIN" and v.reason == "grounding"


def test_malformed_extraction_abstains(registry) -> None:
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn("Copy, closing two-oh-three, line Bravo.", t=3, extraction={"kind": "readback"}), registry)
    assert judge(Exchange("CH-07", ins, rb)).reason == "grounding"


def test_unmapped_word_abstains(registry) -> None:
    odd = {
        "kind": "readback",
        "retracted_spans": [],
        "commands": [
            {"action_span": "copy", "tag_span": "two-oh-three", "state_span": None, "line_span": "line Bravo"}
        ],
    }
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn("Copy, closing two-oh-three, line Bravo.", t=3, extraction=odd), registry)
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "ABSTAIN" and v.reason == "coverage"  # "closing" is left uncovered
    odd["commands"][0]["action_span"] = "Copy, closing"
    rb = analyze(make_turn("Copy, closing two-oh-three, line Bravo.", t=3, extraction=odd), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict == "VERIFIED"


def test_not_audited_turn_is_never_verified(registry) -> None:
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn("Copy, closing two-oh-three, line Bravo.", t=3, not_audited="rate limited"), registry)
    v = judge(Exchange("CH-07", ins, rb))
    assert v.verdict == "NOT_AUDITED" and v.reason == "rate limited"


def test_referee_exception_fails_closed(registry, monkeypatch) -> None:
    import referee.audit as audit_mod

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(audit_mod, "verdict", boom)
    ins = analyze(make_turn(HERO_I), registry)
    rb = analyze(make_turn("Copy, closing two-oh-three, line Bravo.", t=3), registry)
    assert judge(Exchange("CH-07", ins, rb)).verdict == "NOT_AUDITED"


def test_verdict_module_imports_no_model_code() -> None:
    """Invariant 1: the decision imports nothing from ASR / extraction / network code."""
    src = Path(__file__).resolve().parent.parent / "referee" / "verdict.py"
    mods = {n.module for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "dataclasses"}, mods
