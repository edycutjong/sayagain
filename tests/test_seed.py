"""Seed data: deterministic, quotas met, and every card's script judged as its labels say."""

import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest

from referee.audit import audit
from referee.regex_extractor import extract
from referee.turns import Turn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import seed  # noqa: E402

CARDS = [json.loads(line) for line in (ROOT / "data" / "cards.jsonl").read_text(encoding="utf-8").splitlines()]


def test_seed_is_deterministic_and_matches_committed_files(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "seed.py"), "--seed", "1018", "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for name in ("registry.csv", "error_plan.csv", "cards.jsonl"):
        assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes(), name


def test_registry_invariants(registry) -> None:
    assert len(registry) == 120
    assert registry.suffixes_unique()
    assert registry.line_of("XV-203") == "B" and registry.line_of("XV-208") == "A"
    assert "XV-205" not in registry
    assert registry.resolve(None, "208").tag == "XV-208"
    assert registry.resolve(None, "205").status == "not_in_registry"


def test_a2_quotas() -> None:
    a2 = [c for c in CARDS if c["set"] in ("A2", "H")]
    errors = collections.Counter(c["error_type"] for c in a2)
    for err, n in seed.ERROR_PLAN:
        assert errors[err] == n, err
    assert sum(n for _, n in seed.ERROR_PLAN) == 52
    assert errors["none"] >= 50 and len(a2) >= 100
    phen = collections.Counter(p for c in a2 for p in c["phenomena"])
    assert phen["multi_command"] >= 8 and phen["correction"] >= 8 and phen["say_again"] >= 6
    assert 6 <= phen["caught_on_channel"] <= 8


def test_card_sets() -> None:
    sets = collections.Counter(c["set"] for c in CARDS)
    assert sets["H"] == 2 and sets["R"] == 1 and sets["KT"] == 20
    chatter_turns = sum(len(c["turns"]) for c in CARDS if c["set"] == "C")
    assert 20 <= chatter_turns <= 30
    assert sum("time_in_chatter" in c["phenomena"] for c in CARDS) == 3
    assert len({c["exchange_id"] for c in CARDS}) == len(CARDS)
    assert sum(c["dev_slice"] for c in CARDS if c["set"] == "A2") == 20
    hero = next(c for c in CARDS if c["exchange_id"] == "H-01")
    assert [t["text"] for t in hero["turns"]] == [
        "Close valve two-oh-three, line Bravo.",
        "Copy, closing two-oh-eight, line Bravo.",
    ]
    assert hero["expected"]["label"] == "MISMATCH · tag: XV-203 ≠ XV-208"


def _script_turns(card: dict) -> list[Turn]:
    t, out = 0.0, []
    for k, turn in enumerate(card["turns"]):
        if turn["role"] == "pause":
            t += turn["seconds"]
            continue
        out.append(Turn(f"{card['exchange_id']}-{k}", "CH", t, t + 2.5, turn["text"], extract(turn["text"])))
        t += 3.5
    return out


@pytest.mark.parametrize("card", CARDS, ids=[c["exchange_id"] for c in CARDS])
def test_card_script_is_judged_as_labelled(card: dict, registry) -> None:
    """The scripted words of every card, through the regex rung + the full referee, give the labelled verdict."""
    rep = audit(_script_turns(card), registry)
    got = [j.verdict.label for j in rep.judged]
    if card["expected"] is None:
        assert got == []  # chatter makes no card
    else:
        assert got == [card["expected"]["label"]]
    if card["error_type"] == "caught_on_channel":
        assert rep.judged[0].caught and rep.judged[0].caught[0].verdict == "MISMATCH"
