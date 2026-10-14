"""P2: Token Factory extraction (backoff, fail closed, receipts), the grounding-ablation path and the eval arithmetic.

Network calls go to an httpx MockTransport with SYNTHETIC responses (OpenAI shapes) — never a recorded AMD response.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

from referee.audit import audit
from referee.turns import Turn, analyze_canonical
from referee.verdict import ABSTAIN, MISMATCH, NO_READBACK, NOT_AUDITED, VERIFIED
from sayagain.amd_client import Config, TokenFactory
from sayagain.extract import REGEX, Cache, Pacer, Rung, TokenFactoryExtractor, regex_extraction

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval as ev  # noqa: E402

CFG = Config(api_key="sk-test-not-a-real-key", base_url="https://developer.example.amd.com.cn/radeon/api/v1")
RUNG = Rung("Test-Model", "Test-Model", "none", "off", 7.0)
SPANS = {
    "kind": "readback",
    "commands": [{"action_span": "closing", "tag_span": "two-oh-eight", "state_span": None, "line_span": "line Bravo"}],
    "retracted_spans": [],
}


def chat_json(content: str, *, cost: float = 2e-5, model: str = "self-dploy/Test-Model") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 250, "completion_tokens": 60, "total_tokens": 310, "cost": cost},
    }


def make(handler, tmp_path: Path | None = None) -> TokenFactory:
    rp = tmp_path / "r.jsonl" if tmp_path else None
    return TokenFactory(CFG, transport=httpx.MockTransport(handler), receipts_path=rp, thinking_off=None)


def test_request_shape_json_mode_reasoning_effort_no_registry(tmp_path: Path) -> None:
    seen: list = []

    def h(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=chat_json(json.dumps(SPANS)), headers={"x-request-id": "rq-1"})

    e = TokenFactoryExtractor(make(h, tmp_path), RUNG)("Copy, closing two-oh-eight, line Bravo.")
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "none" and "chat_template_kwargs" not in body
    assert body["temperature"] == 0.0
    prompt = json.dumps(body["messages"]).lower()
    assert "registry" not in prompt and "xv-2" not in prompt  # the registry lives in code, never in the prompt
    assert e.not_audited is None and e.obj == SPANS
    r = e.receipt
    assert r["llm_on_amd"] == 1 and r["cost_usd"] == 2e-5 and r["model_served"] == "self-dploy/Test-Model"
    assert r["request_id"] == "rq-1" and r["stage"] == "extract" and r["stage_timings"]["extract_s"] >= 0
    assert r["params"]["schema_valid"] is True and r["rung"] == "Test-Model"
    on_disk = (tmp_path / "r.jsonl").read_text()
    assert "sk-test" not in on_disk and json.loads(on_disk)["model_served"] == "self-dploy/Test-Model"


def test_429_backs_off_then_succeeds() -> None:
    calls, slept = [], []

    def h(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "7"})
        return httpx.Response(200, json=chat_json(json.dumps(SPANS)))

    e = TokenFactoryExtractor(make(h), RUNG, sleep=slept.append)("Copy, closing two-oh-eight, line Bravo.")
    assert e.not_audited is None and e.attempts == 3 and slept == [7.0, 7.0]


def test_429_exhausted_is_not_audited_never_a_verdict(registry) -> None:
    def h(req):
        return httpx.Response(429, json={"error": "slow down"})

    slept: list = []
    ext = TokenFactoryExtractor(make(h), RUNG, sleep=slept.append, max_retries=2, backoff_s=(1.0, 2.0))
    e = ext("Copy, closing two-oh-eight, line Bravo.")
    assert e.obj is None and e.not_audited == "rate limited" and e.attempts == 3 and slept == [1.0, 2.0]
    card = {
        "exchange_id": "T-1",
        "set": "A2",
        "dev_slice": False,
        "lang": "en",
        "error_type": "lookalike_tag",
        "expected": {"label": "MISMATCH · tag: XV-203 ≠ XV-208", "verdict": "MISMATCH"},
        "turns": [
            {"role": "operator", "text": "Close valve two-oh-three, line Bravo."},
            {"role": "tech", "text": "Copy, closing two-oh-eight, line Bravo."},
        ],
    }
    rec = ev.judge_card(card, lambda t: ext(t), registry)
    assert rec["verdict"] == NOT_AUDITED and rec["not_audited_turns"] == 2


@pytest.mark.parametrize("status", [500, 400])
def test_server_errors_fail_closed_without_retry(status: int) -> None:
    calls: list = []

    def h(req):
        calls.append(1)
        return httpx.Response(status, json={"error": "x"})

    e = TokenFactoryExtractor(make(h), RUNG, sleep=lambda s: None)("Copy.")
    assert e.obj is None and e.not_audited and len(calls) == 1


def test_invalid_json_is_kept_unrepaired_and_flagged(tmp_path: Path) -> None:
    def h(req):
        return httpx.Response(200, json=chat_json('{"commands": [{"{": "null"}]}'))

    e = TokenFactoryExtractor(make(h, tmp_path), RUNG)("Copy, closing two-oh-eight, line Bravo.")
    assert e.not_audited is None and e.obj == {"commands": [{"{": "null"}]}
    assert e.receipt["params"]["schema_valid"] is False and e.receipt["error"].startswith("schema:")


def test_think_block_and_fence_are_stripped() -> None:
    def h(req):
        return httpx.Response(200, json=chat_json("<think>hmm</think>```json\n" + json.dumps(SPANS) + "\n```"))

    e = TokenFactoryExtractor(make(h), RUNG)("Copy, closing two-oh-eight, line Bravo.")
    assert e.obj == SPANS and e.receipt["params"]["think_block_stripped"] is True


def test_rung0_makes_no_call_and_cache_dedupes() -> None:
    with pytest.raises(ValueError):
        TokenFactoryExtractor(make(lambda r: httpx.Response(500)), REGEX)
    seen: list = []
    c = Cache(lambda t: seen.append(t) or regex_extraction(t))
    c("Ulangi?")
    c("Ulangi?")
    assert seen == ["Ulangi?"] and c.hits == 1 and c("Ulangi?").receipt is None


def test_pacer_spaces_call_starts() -> None:
    import time

    p = Pacer(0.05)
    t0 = time.monotonic()
    for _ in range(4):
        p.wait()
    assert time.monotonic() - t0 >= 0.15


# ------------------------------------------------------------------------------------------- grounding ablation


def canon_turn(text: str, cmds: list[dict], t: float = 0.0, kind: str = "instruction") -> Turn:
    return Turn(f"t{t:g}", "CH", t, t + 2.5, text, {"kind": kind, "commands": cmds})


def test_ablation_trusts_canonical_values_and_resolves_in_code(registry) -> None:
    ins = canon_turn(
        "Close valve two-oh-three, line Bravo.", [{"action": "CLOSE", "tag": "XV-203", "line": "B", "state": None}]
    )
    rb = canon_turn(
        "Copy, closing two-oh-eight, line Bravo.",
        [{"action": "close", "tag": "xv 208", "line": "b", "state": None}],
        3.5,
        "readback",
    )
    rep = audit([ins, rb], registry, analyzer=analyze_canonical, spot_checks=0)
    assert [j.verdict.label for j in rep.judged] == ["MISMATCH · tag: XV-203 ≠ XV-208"]


def test_ablation_can_wave_through_what_grounding_catches(registry) -> None:
    """The reason the mechanism exists: a model that 'normalises' the read-back toward the instruction is believed."""
    ins = canon_turn(
        "Close valve two-oh-three, line Bravo.", [{"action": "CLOSE", "tag": "XV-203", "line": "B", "state": None}]
    )
    rb = canon_turn(
        "Copy, closing two-oh-eight, line Bravo.",
        [{"action": "CLOSE", "tag": "XV-203", "line": "B", "state": None}],
        3.5,
        "readback",
    )
    rep = audit([ins, rb], registry, analyzer=analyze_canonical, spot_checks=0)
    assert rep.judged[0].verdict.verdict == VERIFIED  # without grounding nothing checks the words were said


@pytest.mark.parametrize(
    "cmd, reason",
    [
        ({"action": "SHUT", "tag": "XV-203", "line": "B", "state": None}, "unmapped: SHUT"),
        ({"action": "CLOSE", "tag": "valve 203", "line": "B", "state": None}, "unmapped: VALVE 203"),
        ({"action": "CLOSE", "tag": "XV-203", "line": "Bravo", "state": None}, "unmapped: BRAVO"),
    ],
)
def test_ablation_rejects_values_outside_the_canonical_alphabet(registry, cmd: dict, reason: str) -> None:
    from referee.turns import analyze_canonical as ac

    a = ac(canon_turn("Close valve two-oh-three, line Bravo.", [cmd]), registry)
    assert a.status == "abstain" and a.reason == reason


def test_ablation_malformed_output_abstains(registry) -> None:
    t = Turn("t", "CH", 0, 2.5, "Close valve two-oh-three, line Bravo.", {"commands": []})
    a = analyze_canonical(t, registry)
    assert a.status == "abstain" and "malformed" in a.detail


# --------------------------------------------------------------------------------------------- eval arithmetic


def test_wilson_interval_known_values() -> None:
    w = ev.wilson(0, 50)
    assert w["rate"] == 0.0 and w["ci95"][0] == 0.0 and 0.07 < w["ci95"][1] < 0.072
    w = ev.wilson(45, 50)
    assert w["rate"] == 0.9 and 0.78 < w["ci95"][0] < 0.79 and 0.95 < w["ci95"][1] < 0.96
    assert ev.wilson(0, 0)["rate"] is None


def _rec(eid, expected_verdict, verdict, row="held_out", error_type="x"):
    return {
        "exchange_id": eid,
        "row": row,
        "error_type": error_type,
        "expected": expected_verdict,
        "expected_verdict": expected_verdict,
        "verdict": verdict,
        "label": verdict,
        "labels": [verdict] if verdict else [],
        "live_cards": int(verdict in (MISMATCH, NO_READBACK, ABSTAIN)),
        "caught": False,
        "turns": [],
    }


def test_row_metrics_follow_the_protocol_formulas() -> None:
    recs = [
        _rec("e1", MISMATCH, MISMATCH),
        _rec("e2", MISMATCH, ABSTAIN),
        _rec("e3", NO_READBACK, NO_READBACK),
        _rec("e4", MISMATCH, VERIFIED),  # false VERIFIED
        _rec("e5", MISMATCH, None),  # no card at all
        _rec("c1", VERIFIED, VERIFIED),
        _rec("c2", VERIFIED, ABSTAIN),  # false alarm
        _rec("c3", VERIFIED, NOT_AUDITED),  # end of shift, not a false alarm
        _rec("c4", VERIFIED, VERIFIED),
    ]
    m = ev.row_metrics(recs, model_rung=False)
    assert m["errors_flagged"]["k"] == 3 and m["errors_flagged"]["n"] == 5
    assert m["mismatch_only_recall"]["k"] == 1 and m["mismatch_only_recall"]["n"] == 4
    assert m["false_verified"]["count"] == 1 and m["false_verified"]["exchanges"] == ["e4"]
    assert m["false_verified"]["n"] == 5
    assert m["missed_no_card"] == {"count": 1, "exchanges": ["e5"]}
    assert m["false_alarm_load"]["k"] == 1 and m["false_alarm_load"]["n"] == 4
    assert m["not_audited"]["k"] == 1
    r, f = 3 / 5, 1 / 4
    assert m["precision_at_1pct_prevalence"] == round(r * 0.01 / (r * 0.01 + f * 0.99), 4)
    assert m["cards_per_shift_assumption"]["2_errors_per_180"] == round(f * 178 + r * 2, 2)
    s = {"rows": {"held_out": m}}
    assert ev.meets_rule(s) == {"literal": False, "strict": False, "fully_audited": False}


@pytest.mark.parametrize(
    "obj, verdict",
    [
        (
            {
                "instruction_present": False,
                "readback_present": False,
                "action_match": "n/a",
                "tag_match": "n/a",
                "line_match": "n/a",
            },
            None,
        ),
        (
            {
                "instruction_present": True,
                "readback_present": False,
                "action_match": "n/a",
                "tag_match": "n/a",
                "line_match": "n/a",
            },
            NO_READBACK,
        ),
        (
            {
                "instruction_present": True,
                "readback_present": True,
                "action_match": "yes",
                "tag_match": "no",
                "line_match": "yes",
            },
            MISMATCH,
        ),
        (
            {
                "instruction_present": True,
                "readback_present": True,
                "action_match": "yes",
                "tag_match": "yes",
                "line_match": "n/a",
            },
            VERIFIED,
        ),
        (
            {
                "instruction_present": True,
                "readback_present": True,
                "action_match": "maybe",
                "tag_match": "yes",
                "line_match": "yes",
            },
            ABSTAIN,
        ),
        ({"_invalid": "x"}, ABSTAIN),
    ],
)
def test_llm_judge_mapping(obj: dict, verdict) -> None:
    assert ev.parse_judge(obj)[0] == verdict


def test_regex_rung_text_rows_rescore_exactly(tmp_path: Path, registry, monkeypatch) -> None:
    """Rung 0 needs no network: run it, then re-judge from what was written — every row must match."""
    monkeypatch.setattr(ev, "OUT", tmp_path / "text")
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    cards = ev.load_cards()
    s = ev.run_extraction_rung("t", REGEX, cards, registry, Pacer(0), ev.Budget(0))
    h = s["rows"]["held_out"]
    assert h["false_verified"]["count"] == 0 and h["label_exact"]["rate"] == 1.0
    assert s["rows"]["refusal"][0]["got"] == "ABSTAIN · unknown tag" and s["rows"]["chatter"]["false_cards"] == 0
    again = ev.rescore_summary(s, cards, registry)
    assert ev._strip(again["rows"]) == ev._strip(s["rows"])


# ------------------------------------------------------------------------------------- review round 1 (P2) fixes


def test_a_row_nothing_audited_never_meets_the_rule(registry) -> None:
    """Round-1 high: a budget stop / outage made every card NOT_AUDITED and the rule still passed."""
    from sayagain.extract import Extraction

    cards = [c for c in ev.load_cards() if ev.row_of(c) == "held_out"]
    recs = [ev.judge_card(c, lambda t: Extraction(t, None, "budget guard"), registry) for c in cards]
    s = {"rows": {"held_out": ev.row_metrics(recs, model_rung=True)}}
    assert s["rows"]["held_out"]["false_verified"]["count"] == 0
    assert ev.meets_rule(s) == {"literal": False, "strict": False, "fully_audited": False}


@pytest.mark.parametrize("content", ["null", "[1, 2]", '"hi"', "42"])
def test_valid_json_that_is_not_an_object_abstains_and_never_crashes(registry, content: str) -> None:
    def h(req):
        return httpx.Response(200, json=chat_json(content))

    ext = TokenFactoryExtractor(make(h), RUNG)
    e = ext("Copy, closing two-oh-eight, line Bravo.")
    assert isinstance(e.obj, dict) and e.obj["_invalid"] == "not an object" and e.not_audited is None
    card = next(c for c in ev.load_cards() if c["exchange_id"] == "H-01")
    rec = ev.judge_card(card, ext, registry)
    assert rec["verdict"] == ABSTAIN


def test_receipts_on_disk_carry_rung_cost_and_llm_on_amd_even_for_429(tmp_path: Path) -> None:
    calls: list = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow"}, headers={"retry-after": "1"})
        return httpx.Response(200, json=chat_json(json.dumps(SPANS)))

    c = make(h, tmp_path)
    c.context = {"rung": "Test-Model", "run": "t"}
    TokenFactoryExtractor(c, RUNG, sleep=lambda s: None)("Copy, closing two-oh-eight, line Bravo.")
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert [r["http_status"] for r in lines] == [429, 200]
    for r in lines:
        assert r["rung"] == "Test-Model" and r["run"] == "t" and r["llm_on_amd"] == 1 and "stage_timings" in r
    assert lines[1]["cost_usd"] == 2e-5 and lines[0]["cost_usd"] is None


def test_retry_after_is_capped() -> None:
    slept: list = []
    calls: list = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={}, headers={"retry-after": "86400"})
        return httpx.Response(200, json=chat_json(json.dumps(SPANS)))

    TokenFactoryExtractor(make(h), RUNG, sleep=slept.append)("Copy.")
    assert slept == [60.0]


def test_garbage_usage_does_not_crash_the_receipt() -> None:
    def h(req):
        body = chat_json(json.dumps(SPANS))
        body["usage"] = "n/a"
        return httpx.Response(200, json=body)

    e = TokenFactoryExtractor(make(h), RUNG)("Copy.")
    assert e.not_audited is None and e.receipt["cost_usd"] is None


def test_reaudit_asks_again_only_for_not_audited_transcripts() -> None:
    from sayagain.extract import Extraction

    prior = [
        {"text": "a", "obj": SPANS, "not_audited": None, "request_id": "r1", "attempts": 1},
        {"text": "b", "obj": None, "not_audited": "server error", "request_id": None, "attempts": 1},
    ]
    asked: list = []
    fn = ev._reaudit(lambda t: asked.append(t) or Extraction(t, SPANS, None, {"request_id": "r2"}, 1), prior)
    assert fn("a").receipt["reused"] is True and fn("a").obj == SPANS
    assert fn("b").receipt == {"request_id": "r2"} and asked == ["b"]
