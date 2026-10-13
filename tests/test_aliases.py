"""Alias tables: spoken radio words -> canonical plant vocabulary (pre-registered, prereg-v1)."""

import pytest

from referee import aliases as al


def tags(text: str) -> list[str]:
    return [m.value for m in al.all_mentions(text) if m.field == "tag"]


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("two-oh-three", "203"),
        ("two zero three", "203"),
        ("two hundred three", "203"),
        ("two hundred and three", "203"),
        ("to oh three", "203"),
        ("203", "203"),
        ("dua nol tiga", "203"),
        ("dua kosong tiga", "203"),
        ("dua ratus tiga", "203"),
        ("one-forty", "140"),
        ("one-one-four", "114"),
        ("seratus empat puluh", "140"),
        ("satu empat nol", "140"),
        ("thirty-one alpha", "31A"),
        ("three-one hotel", "31H"),
        ("31A", "31A"),
    ],
)
def test_spoken_numbers_to_suffix(spoken: str, expected: str) -> None:
    assert tags(spoken) == [expected]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Close valve two-oh-three", "XV-203"),
        ("XV-203", "XV-203"),
        ("XV203", "XV-203"),
        ("tutup katup dua nol tiga", "XV-203"),
        ("stop pump one-forty", "P-140"),
        ("matikan pompa satu satu empat", "P-114"),
        ("open motor valve thirty-one alpha", "MOV-31A"),
        ("MOV-31H", "MOV-31H"),
    ],
)
def test_noun_to_prefix(text: str, expected: str) -> None:
    assert tags(text) == [expected]


def test_time_format_counts_next_to_noun_or_verb() -> None:
    assert tags("valve 2:03") == ["XV-203"]
    assert tags("close 2.03") == ["203"]


def test_time_token_not_a_tag() -> None:
    """N10: a clock time in chatter is not a tag ("2:03" / "2.03" only next to a tag noun or action verb)."""
    assert tags("Shift handover at 2:03, see you there.") == []
    assert tags("Break until 2.03 okay") == []
    assert isinstance(al.canon_tag("2:03", preceding_word="at"), al.Unmapped)
    assert al.canon_tag("2:03", preceding_word="close").value == "203"


@pytest.mark.parametrize("text", ["pressure 150 psi", "wait 10 minutes", "temperature 120 degrees", "5 bar"])
def test_quantities_are_not_tags(text: str) -> None:
    assert tags(text) == []


@pytest.mark.parametrize(
    "span, letter",
    [("line Bravo", "B"), ("line B", "B"), ("jalur Be", "B"), ("line Alpha", "A"), ("lajur ce", "C"), ("Bravo", "B")],
)
def test_line_letters(span: str, letter: str) -> None:
    assert al.canon_line(span) == letter


def test_lima_is_a_letter_after_line_and_a_digit_elsewhere() -> None:
    assert al.canon_line("line Lima") == "L"
    assert tags("pompa satu lima nol") == ["P-150"]


@pytest.mark.parametrize(
    "word, action",
    [
        ("close", "CLOSE"),
        ("closing", "CLOSE"),
        ("shut", "CLOSE"),
        ("tutup", "CLOSE"),
        ("menutup", "CLOSE"),
        ("ditutup", "CLOSE"),
        ("open", "OPEN"),
        ("opening", "OPEN"),
        ("buka", "OPEN"),
        ("dibuka", "OPEN"),
        ("isolate", "ISOLATE"),
        ("isolasi", "ISOLATE"),
        ("lock out", "ISOLATE"),
        ("start", "START"),
        ("nyalakan", "START"),
        ("stop", "STOP"),
        ("stopping", "STOP"),
        ("matikan", "STOP"),
        ("shut down", "STOP"),
    ],
)
def test_action_lexicon_with_inflections(word: str, action: str) -> None:
    assert al.canon_action(word) == action


@pytest.mark.parametrize("span", ["turn", "do it", "close and open", ""])
def test_unmapped_action_is_never_guessed(span: str) -> None:
    assert isinstance(al.canon_action(span), al.Unmapped)


def test_unmapped_line_and_tag_are_never_guessed() -> None:
    assert isinstance(al.canon_line("line Brava"), al.Unmapped)
    assert isinstance(al.canon_tag("the big one"), al.Unmapped)
    assert isinstance(al.canon_tag("203 and 208"), al.Unmapped)


def test_oh_is_not_a_digit_outside_a_number() -> None:
    assert al.all_mentions("Oh, copy that.") == []


def test_mentions_carry_offsets_into_the_transcript() -> None:
    text = "Copy, closing two-oh-eight, line Bravo."
    for m in al.all_mentions(text):
        assert text[m.start : m.end].strip() == text[m.start : m.end]
    tag = [m for m in al.all_mentions(text) if m.field == "tag"][0]
    assert text[tag.start : tag.end] == "two-oh-eight"
