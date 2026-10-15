"""Coverage for scripts/seed.py: CLI main in-process, Markdown cards, phrasing helpers, verdict labels."""

import csv
import json
import random
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import seed  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return seed.build_registry(1018)


@pytest.fixture(scope="module")
def cards(rows):
    return seed.build_cards(1018, rows)


def test_spoken_suffix_variants() -> None:
    assert seed.spoken_suffix("203") == "two-oh-three"
    assert seed.spoken_suffix("203", "id") == "dua nol tiga"
    assert seed.spoken_suffix("31A") == "three-one Alpha"
    assert seed.spoken_suffix("31H", "id") == "tiga satu Hotel"


def test_registry_csv_roundtrip(rows) -> None:
    text = seed.registry_csv(rows)
    parsed = list(csv.DictReader(text.splitlines()))
    assert len(parsed) == len(rows) == 120
    xv = next(r for r in parsed if r["tag"] == "XV-203")
    assert xv["line"] == "B" and xv["allowed_actions"] == "OPEN|CLOSE|ISOLATE"
    assert xv["spoken_variants"].split("|")[0] == "203"
    mov = next(r for r in parsed if r["tag"] == "MOV-31A")
    assert mov["spoken_variants"].split("|")[0] == "three-one Alpha"  # no bare-digit variant for lettered suffix


def test_error_plan_csv() -> None:
    parsed = list(csv.reader(seed.error_plan_csv().splitlines()))
    assert parsed[0][0] == "error_type"
    assert {r[0]: int(r[1]) for r in parsed[1:]} == dict(seed.ERROR_PLAN)


def test_phraser_branches() -> None:
    class Fixed(random.Random):
        def __init__(self, val: float):
            super().__init__(0)
            self.val = val

        def random(self):
            return self.val

        def choice(self, seq):
            return seq[0]

    low, high = seed.Phraser(Fixed(0.0)), seed.Phraser(Fixed(0.99))
    assert low.noun("XV-203", "mix") == "katup"
    assert high.noun("XV-203", "mix") == "valve"
    assert high.noun("P-114", "mix") == "pompa" and high.noun("MOV-31A", "en") == "motor valve"
    assert low.line("B", "mix") == "jalur Bravo" and high.line("B", "mix") == "line Bravo"
    c = {"action": "CLOSE", "tag": "XV-203", "line": "B"}
    assert high.instruction_cmd(c, "en") == "Close valve two-oh-three, line Bravo"
    assert high.instruction_cmd(c, "en", first=False).startswith("close ")
    assert high.instruction_cmd(c, "mix") == "Tutup valve dua nol tiga, line Bravo"
    assert high.instruction_cmd({**c, "line": None}, "en") == "Close valve two-oh-three"
    assert low.instruction([c], "mix").startswith("Tolong tutup")
    two = high.instruction([c, {"action": "OPEN", "tag": "P-114", "line": None}], "en")
    assert two == "Close valve two-oh-three, line Bravo and open pump one-one-four."
    assert "dan" in high.instruction([c, c], "mix")
    # readback: with and without noun, missing action / tag / line, verbatim
    assert low.readback_cmd(c, "en", True) == "closing valve two-oh-three, line Bravo"
    assert high.readback_cmd(c, "en", False) == "closing two-oh-three, line Bravo"
    assert high.readback_cmd({"action": None, "tag": "XV-203", "line": "B"}, "en", False) == "two-oh-three, line Bravo"
    assert high.readback_cmd({"action": "OPEN", "tag": None, "line": None}, "mix", False) == "buka"
    assert low.readback([c], "en") == "Copy, closing valve two-oh-three, line Bravo."
    assert high.readback([c, c], "mix").startswith("Siap, tutup")
    assert high.readback([c], "en", verbatim=True) == high.instruction([c], "en")


def test_build_registry_deterministic(rows) -> None:
    assert seed.registry_csv(seed.build_registry(1018)) == seed.registry_csv(rows)
    other = seed.build_registry(7)
    assert {r.tag for r in other} != {r.tag for r in rows}
    assert "XV-205" not in {r.tag for r in other}


def test_cards_structure_and_quotas(cards) -> None:
    ids = [c["exchange_id"] for c in cards]
    assert len(ids) == len(set(ids))
    by_err = {}
    for c in cards:
        if c["set"] == "A2":
            by_err[c["error_type"]] = by_err.get(c["error_type"], 0) + 1
    assert by_err["wrong_line"] == 11 and by_err["lookalike_tag"] == 13
    assert by_err["no_readback"] == 4 and by_err["uncorrected_retraction"] == 4
    assert by_err["none"] == seed.N_CORRECT and by_err["caught_on_channel"] == seed.N_CAUGHT
    assert sum(1 for c in cards if c["set"] == "KT") == 20
    assert sum(1 for c in cards if c["dev_slice"] and c["set"] == "A2") == seed.DEV_SLICE_A2
    assert {c["fold"] for c in cards} == {1, 2}
    assert any(t["role"] == "pause" for c in cards for t in c["turns"])
    assert sum("say_again" in c["phenomena"] for c in cards if c["set"] == "A2") >= 4


def test_expected_verdicts(cards) -> None:
    by_id = {c["exchange_id"]: c for c in cards}
    assert by_id["H-01"]["expected"]["reason"] != "" and by_id["H-01"]["expected"]["label"]
    chatter = [c for c in cards if c["set"] == "C"]
    assert len(chatter) == 15 and all(c["expected"] is None for c in chatter)
    clean = [c for c in cards if c["error_type"] == "none"]
    assert {c["expected"]["verdict"] for c in clean} == {clean[0]["expected"]["verdict"]}
    assert by_id["H-01"]["expected"]["verdict"] != clean[0]["expected"]["verdict"]
    nb = next(c for c in cards if c["error_type"] == "no_readback")
    assert nb["readback"] is None and nb["expected"]["verdict"] != clean[0]["expected"]["verdict"]


def test_cards_markdown(cards) -> None:
    md = seed.cards_markdown(cards)
    assert md.startswith("# SayAgain recording cards")
    assert "## Kill test (one voice, both roles)" in md
    assert "### KT-01 · one voice" in md
    assert "### H-01 · operator S1 · tech S2" in md
    assert "`[silence 10 s]`" in md
    assert "1. **Operator:** Close valve two-oh-three, line Bravo." in md


def test_main_writes_files(tmp_path, capsys) -> None:
    out, md = tmp_path / "o" / "data", tmp_path / "cards.md"
    assert seed.main(["--seed", "1018", "--out", str(out), "--cards-md", str(md)]) == 0
    stdout = capsys.readouterr().out
    assert "registry: 120 tags" in stdout and "cards markdown ->" in stdout
    lines = (out / "cards.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 145 and json.loads(lines[0])["exchange_id"] == "H-01"
    assert (out / "registry.csv").read_text().startswith("tag,type,line")
    assert md.read_text().startswith("# SayAgain")
    out2 = tmp_path / "again"
    assert seed.main(["--out", str(out2)]) == 0
    assert "cards markdown" not in capsys.readouterr().out
    assert (out2 / "cards.jsonl").read_bytes() == (out / "cards.jsonl").read_bytes()


def test_run_as_script(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["seed.py", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as ei:
        runpy.run_path(str(ROOT / "scripts" / "seed.py"), run_name="__main__")
    assert ei.value.code == 0
    assert (tmp_path / "registry.csv").exists()
