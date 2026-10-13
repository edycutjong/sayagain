"""Rung 0: the frozen regex + alias extractor (pre-registered, prereg-v1)."""

from referee import aliases as al
from referee.regex_extractor import extract

HERO_I = "Close valve two-oh-three, line Bravo."
HERO_R = "Copy, closing two-oh-eight, line Bravo."


def test_hero_instruction_spans() -> None:
    out = extract(HERO_I)
    assert out["kind"] == "instruction"
    assert out["commands"] == [
        {"action_span": "Close", "tag_span": "valve two-oh-three", "state_span": None, "line_span": "line Bravo"}
    ]
    assert out["retracted_spans"] == []


def test_hero_readback_spans() -> None:
    out = extract(HERO_R)
    assert out["kind"] == "readback"
    assert out["commands"][0]["tag_span"] == "two-oh-eight"
    assert out["commands"][0]["action_span"] == "closing"


def test_asr_digit_form() -> None:
    out = extract("Copy, closing 208, line Bravo.")
    assert out["commands"][0]["tag_span"] == "208"


def test_multi_command_split() -> None:
    out = extract("Close valve two-oh-three and open valve two-one-six, line Bravo.")
    assert [c["action_span"] for c in out["commands"]] == ["Close", "open"]
    assert [c["tag_span"] for c in out["commands"]] == ["valve two-oh-three", "valve two-one-six"]


def test_repeated_verb_multi_command_extract() -> None:
    out = extract("Close valve two-oh-three and close valve two-one-six.")
    assert [c["action_span"] for c in out["commands"]] == ["Close", "close"]


def test_correction_retracts_the_earlier_line() -> None:
    out = extract("Close valve two-oh-three, line Alpha, correction, line Bravo.")
    assert out["commands"][0]["line_span"] == "line Bravo"
    assert out["retracted_spans"] == ["line Alpha"]


def test_conflicting_line_without_cue_is_left_uncovered() -> None:
    out = extract("Close valve two-oh-three, line Alpha, line Bravo.")
    assert out["commands"][0]["line_span"] == "line Alpha"
    assert out["retracted_spans"] == []


def test_say_again_kind() -> None:
    assert extract("Say again?")["kind"] == "say_again"
    assert extract("Ulangi, ulangi.")["kind"] == "say_again"
    assert extract("I say again, close valve two-oh-three, line Bravo.")["kind"] == "instruction"


def test_chatter_has_no_commands() -> None:
    for text in ["Where are you now?", "At the boiler house, shift handover at 2:03.", "Roger, standing by."]:
        out = extract(text)
        assert out["commands"] == []
        assert out["kind"] == "other"


def test_bahasa_order_tag_before_action() -> None:
    out = extract("Valve dua nol tiga tutup, jalur Be.")
    assert out["commands"] == [
        {"action_span": "tutup", "tag_span": "Valve dua nol tiga", "state_span": None, "line_span": "jalur Be"}
    ]


def test_verbatim_readback_is_classified_by_code_not_trusted() -> None:
    # a word-for-word repeat looks like an instruction; pairing (not kind) decides it is the read-back
    assert extract("Close valve two-oh-three, line Bravo.")["kind"] == "instruction"
    assert extract("Siap, tutup dua nol tiga, line Bravo.")["kind"] == "readback"


def test_every_span_is_a_substring() -> None:
    for text in [HERO_I, HERO_R, "Tolong matikan pompa satu empat nol, jalur Alpha.", "Isolate MOV-31A line Charlie"]:
        out = extract(text)
        for c in out["commands"]:
            for v in c.values():
                assert v is None or v in text
        assert all(s in text for s in out["retracted_spans"])
        assert out["commands"], text


def test_extractor_uses_only_alias_tables() -> None:
    # nothing outside the alias tables becomes a field
    out = extract("Please handle the thing near the tank.")
    assert out["commands"] == [] and al.all_mentions("Please handle the thing near the tank.") == []


def test_whole_command_correction_retracts_previous_command() -> None:
    text = "Copy, closing two-oh-three. Correction, closing two-oh-eight."
    out = extract(text)
    assert [c["tag_span"] for c in out["commands"]] == ["two-oh-eight"]
    assert set(out["retracted_spans"]) == {"closing", "two-oh-three"}


def test_tag_correction_inside_a_command() -> None:
    out = extract("Close valve two-oh-three, correction, two-oh-eight, line Alpha.")
    assert out["commands"][0]["tag_span"] == "two-oh-eight"
    assert out["retracted_spans"] == ["valve two-oh-three"]
