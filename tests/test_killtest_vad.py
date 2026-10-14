"""Kill-test scoring (no Whisper in CI) and the energy VAD."""

import json
import sys
from pathlib import Path

import numpy as np

from sayagain.audio import SR
from sayagain.vad import segments

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import kill_test_local as kt  # noqa: E402

CARDS = {c["exchange_id"]: c for c in map(json.loads, (ROOT / "data" / "cards.jsonl").read_text().splitlines())}


def test_vad_finds_two_turns() -> None:
    t = np.arange(int(SR * 4.0)) / SR
    x = 0.3 * np.sin(2 * np.pi * 300 * t) * (((t > 0.3) & (t < 1.5)) | ((t > 2.5) & (t < 3.6)))
    x = x + 0.001 * np.random.default_rng(0).standard_normal(len(t))
    segs = segments(x.astype(np.float32))
    assert len(segs) == 2
    assert abs(segs[0][0] - 0.3) < 0.2 and abs(segs[1][1] - 3.6) < 0.2


def test_vad_silence_is_empty() -> None:
    assert segments(np.zeros(SR, dtype=np.float32)) == []


def test_score_card_exact_and_paired(registry) -> None:
    card = CARDS["KT-01"]
    i, r = (t["text"] for t in card["turns"])
    s = kt.score_card(card, [i, r], registry)
    assert s["paired_ok"] and all(t["tag_ok"] for t in s["turns"])
    assert s["verdict"] == s["expected"]


def test_score_card_counts_a_misheard_tag(registry) -> None:
    card = next(c for c in CARDS.values() if c["set"] == "KT" and c["error_type"] == "none")
    i, r = (t["text"] for t in card["turns"])
    s = kt.score_card(card, [i, "Copy, closing nine-nine-nine, line Bravo."], registry)
    assert s["turns"][0]["tag_ok"] and not s["turns"][1]["tag_ok"] and not s["paired_ok"]


def test_summary_and_thresholds(registry) -> None:
    kts = [c for c in CARDS.values() if c["set"] == "KT"]
    scored = [kt.score_card(c, [t["text"] for t in c["turns"]], registry) for c in kts]
    summ = kt.summarise(scored)
    assert summ["n_exchanges"] == 20 and summ["tag_exact"] == 1.0 and summ["paired"] == 1.0
    assert set(summ["by_lang"]) == {"en", "mix"}
    zero = kt.mix.snr_label("0")
    assert zero == "0dB"
    v = kt.verdict_of({"clean": summ, zero: {**summ, "tag_exact": 0.5}})
    p = v["pooled"]
    assert p["tag_clean"]["pass"] and p["tag_0db"]["pass"] is False and p["paired_clean"]["pass"]
    assert v["decision"].startswith("stop")
    assert kt.verdict_of({"clean": summ, zero: summ})["decision"] == "go"


def test_missing_0db_row_is_not_a_pass(registry) -> None:
    kts = [c for c in CARDS.values() if c["set"] == "KT"]
    summ = kt.summarise([kt.score_card(c, [t["text"] for t in c["turns"]], registry) for c in kts])
    v = kt.verdict_of({"clean": summ})
    assert v["pooled"]["tag_0db"]["pass"] is None and v["decision"] != "go"


def test_code_switched_failure_is_not_hidden_by_the_pooled_average() -> None:
    def summ(en: float, mixed: float) -> dict:
        return {
            "tag_exact": (en + mixed) / 2,
            "paired": (en + mixed) / 2,
            "by_lang": {"en": {"tag_exact": en, "paired": en}, "mix": {"tag_exact": mixed, "paired": mixed}},
        }

    v = kt.verdict_of({"clean": summ(1.0, 0.6), "0dB": summ(1.0, 0.6)})
    assert v["pooled"]["tag_clean"]["pass"]  # 0.80 pooled passes …
    assert v["decision"].startswith("pivot")  # … but the code-switched half fails: pivot


def test_no_clips_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "kt.json"
    assert kt.main(["--clips", str(tmp_path / "none"), "--out", str(out)]) == 3
    assert not out.exists()
