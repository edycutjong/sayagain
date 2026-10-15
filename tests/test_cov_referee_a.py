"""Coverage for referee.schema, referee.regex_extractor and referee.aliases number composition."""

import copy

import pytest

from referee import aliases as al
from referee import schema
from referee.regex_extractor import classify_kind, extract

# ------------------------------------------------------------------------------------------------ schema

GOOD = {
    "kind": "instruction",
    "commands": [{"action_span": "close", "tag_span": None, "state_span": None, "line_span": "line B"}],
    "retracted_spans": ["open"],
}


def test_validate_extraction_accepts_good_object() -> None:
    assert schema.validate_extraction(GOOD) is None
    assert schema.validate_extraction({"kind": "other", "commands": [], "retracted_spans": []}) is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda o: o.pop("retracted_spans"), "keys ['commands', 'kind']"),
        (lambda o: o.update(kind="shout"), "kind 'shout'"),
        (lambda o: o.update(commands="close"), "commands not a list"),
        (lambda o: o.update(commands=["close"]), "command keys"),
        (lambda o: o.update(commands=[{"action_span": "x"}]), "command keys"),
        (lambda o: o["commands"][0].update(tag_span=5), "span not a string"),
        (lambda o: o.update(retracted_spans="open"), "retracted_spans"),
        (lambda o: o.update(retracted_spans=["a", None]), "retracted_spans"),
    ],
)
def test_validate_extraction_rejects(mutate, reason: str) -> None:
    obj = copy.deepcopy(GOOD)
    mutate(obj)
    assert schema.validate_extraction(obj) == reason


def test_validate_extraction_rejects_non_object() -> None:
    assert schema.validate_extraction([GOOD]) == "not an object"
    assert schema.validate_extraction(None) == "not an object"


CANON_GOOD = {
    "kind": "instruction",
    "commands": [{"action": "CLOSE", "tag": "XV-203", "state": None, "line": "B"}],
}


def test_validate_canonical_accepts_good_object() -> None:
    assert schema.validate_canonical(CANON_GOOD) is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda o: o.update(extra=1), "keys ['commands', 'extra', 'kind']"),
        (lambda o: o.update(kind="x"), "kind 'x'"),
        (lambda o: o.update(commands={}), "commands not a list"),
        (lambda o: o.update(commands=[[]]), "command keys"),
        (lambda o: o.update(commands=[{"action": "CLOSE"}]), "command keys"),
        (lambda o: o["commands"][0].update(line=3), "value not a string"),
    ],
)
def test_validate_canonical_rejects(mutate, reason: str) -> None:
    obj = copy.deepcopy(CANON_GOOD)
    mutate(obj)
    assert schema.validate_canonical(obj) == reason


def test_validate_canonical_rejects_non_object() -> None:
    assert schema.validate_canonical("close") == "not an object"


def test_schema_constants_are_consistent() -> None:
    props = schema.SPAN_SCHEMA["properties"]["commands"]["items"]["properties"]
    assert tuple(props) == schema.FIELDS
    assert schema.SPAN_SCHEMA["properties"]["kind"]["enum"] == list(schema.KINDS)


# ------------------------------------------------------------------------------------- regex_extractor


def test_extract_empty_transcript_is_other() -> None:
    assert extract("") == {"kind": "other", "commands": [], "retracted_spans": []}
    assert classify_kind("", False, False) == "other"


def test_action_retracted_by_cue_before_tag() -> None:
    out = extract("Close, correction, open valve two-oh-three.")
    assert out["retracted_spans"] == ["Close"]
    assert out["commands"] == [
        {"action_span": "open", "tag_span": "valve two-oh-three", "state_span": None, "line_span": None}
    ]
    assert out["kind"] == "instruction"


# --------------------------------------------------------------------------------------- _compose


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        (["hundred"], None),
        (["puluh"], None),
        (["dua", "ratus", "hundred"], None),
        (["two", "hundred", "and"], "200"),
        (["two", "hundred", "twenty", "one"], "221"),
        (["two", "hundred", "twenty"], "220"),
        (["two", "hundred", "thirteen"], "213"),
        (["dua", "ratus", "sebelas"], "211"),
        (["dua", "ratus", "tiga", "puluh", "empat"], "234"),
        (["dua", "ratus", "tiga", "belas"], "213"),
        (["tiga", "puluh", "empat", "puluh"], "3040"),
        (["tiga", "puluh", "empat", "belas"], "3014"),
        (["tiga", "puluh"], "30"),
        (["sepuluh"], "10"),
        (["thirteen"], "13"),
        (["twenty", "two"], "22"),
        (["seratus"], "100"),
        (["seratus", "lima"], "105"),
        (["31a"], "31A"),
        (["tiga", "belas"], "13"),
        ([], None),
    ],
)
def test_compose(words: list[str], digits: str | None) -> None:
    assert al._compose(words) == digits


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("sebelas", "11"),
        ("sepuluh", "10"),
        ("nineteen", "19"),
        ("twenty zero", "200"),
        ("dua ratus sebelas", "211"),
        ("seratus empat puluh", "140"),
        ("tiga puluh dua", "32"),
        ("two hundred and five", "205"),
    ],
)
def test_number_runs_digits(text: str, digits: str) -> None:
    runs = al.number_runs(al.tokenize(text))
    assert [r.digits for r in runs] == [digits]


def test_number_runs_drops_malformed_run() -> None:
    assert al.number_runs(al.tokenize("hundred")) == []
    assert al.number_runs(al.tokenize("puluh then 5")) == [al.NumberRun(2, 3, "5", False)]


def test_number_runs_skips_unusable_multiplier_run() -> None:
    assert al.number_runs(al.tokenize("ten hundred")) == []


def test_number_runs_single_letter_suffix_is_upper_cased() -> None:
    assert [r.digits for r in al.number_runs(al.tokenize("thirty-one x"))] == ["31X"]
    assert [r.digits for r in al.number_runs(al.tokenize("thirty-one a"))] == ["31"]


def test_bare_correction_cue_is_not_a_negation() -> None:
    assert al.negation_mentions(al.tokenize("Negative."), "Negative.") == []


def test_report_back_unknown_mention_is_false() -> None:
    toks = al.tokenize("confirm closed")
    assert al.report_back(toks, al.Mention("action", 99, 105, "CLOSE")) is False
    real = al.action_mentions(toks)[0]
    assert al.report_back(toks, real) is True
