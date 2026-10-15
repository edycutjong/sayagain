"""Coverage for scripts/eval.py: arithmetic helpers, the budget guard, the ladder / ablation / judge runs, rescore,
report and compare. Every model call goes to an httpx MockTransport with SYNTHETIC responses; every file the script
writes lands under tmp_path (ROOT, OUT and RECEIPTS are redirected)."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from referee.verdict import ABSTAIN, MISMATCH, NO_READBACK, NOT_AUDITED, VERIFIED
from sayagain.amd_client import Config, MissingCredentials, NotAuditing, TokenFactory
from sayagain.extract import RUNGS, Extraction, Pacer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval as ev  # noqa: E402

CFG = Config(api_key="sk-test-not-a-real-key", base_url="https://developer.example.amd.com.cn/radeon/api/v1")
SPANS = {
    "kind": "readback",
    "commands": [{"action_span": "closing", "tag_span": "two-oh-eight", "state_span": None, "line_span": "line Bravo"}],
    "retracted_spans": [],
}
JUDGE_MISMATCH = {
    "instruction_present": True,
    "readback_present": True,
    "action_match": "yes",
    "tag_match": "no",
    "line_match": "yes",
}
WANTED = ("H-01", "H-02", "R-01", "C-01", "C-02", "A2-001", "A2-021")


ALL_CARDS = ev.load_cards()


def small_cards() -> list[dict]:
    got = [c for c in ALL_CARDS if c["exchange_id"] in WANTED]
    assert {c["exchange_id"] for c in got} == set(WANTED)
    return got


def chat_json(content: str, *, cost: float | None = 1e-5) -> dict:
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "chatcmpl-1", "model": "self-dploy/X", "choices": [{"message": {"content": content}}], "usage": usage}


class Server:
    """A fake gateway. `chat_status` may be a list consumed one call at a time (then 200)."""

    def __init__(self, used: float | None = 0.1, judge_out: object = JUDGE_MISMATCH, chat_status=None):
        self.used, self.judge_out, self.chat_status = used, judge_out, list(chat_status or [])
        self.chat_calls = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/usage"):
            body = {} if self.used is None else {"daily_cost_used_usd": self.used, "daily_cost_limit_usd": 1.0}
            return httpx.Response(200, json=body)
        self.chat_calls += 1
        if self.chat_status:
            st = self.chat_status.pop(0)
            if st != 200:
                return httpx.Response(st, text="nope", headers={"Retry-After": "1"} if st == 429 else {})
        msgs = json.loads(req.content)["messages"]
        if msgs[0]["content"] == ev.JUDGE_PROMPT:
            out = self.judge_out
            return httpx.Response(200, json=chat_json(out if isinstance(out, str) else json.dumps(out)))
        return httpx.Response(200, json=chat_json(json.dumps(SPANS)))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "OUT", tmp_path / "bench" / "results" / "text")
    monkeypatch.setattr(ev, "RECEIPTS", tmp_path / "bench" / "results" / "receipts")
    cards = small_cards()
    monkeypatch.setattr(ev, "load_cards", lambda path=None: cards)
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)

    def install(server: Server) -> Server:
        def open_client(receipts_path, **ctx):
            c = TokenFactory(CFG, transport=httpx.MockTransport(server), receipts_path=receipts_path, thinking_off=None)
            c.context = dict(ctx)
            return c

        monkeypatch.setattr(ev, "open_client", open_client)
        return server

    return install


# ------------------------------------------------------------------------------------------------ pure helpers


def test_pct_interpolates_and_handles_empty() -> None:
    assert ev.pct([], 0.5) is None
    assert ev.pct([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert ev.pct([5.0], 0.95) == 5.0
    assert ev.pct([4.0, 1.0], 1.0) == 4.0


def test_receipt_metrics_counts_cost_from_usage_and_flags() -> None:
    rs = [
        {"ok": True, "latency_s": 1.0, "cost_usd": 0.5, "usage": {"prompt_tokens": 10, "completion_tokens": 2},
         "reasoning_tokens": 7, "model_served": "m1", "on_amd": True, "host": "h",
         "params": {"schema_valid": False}},
        {"ok": True, "latency_s": 3.0, "usage": {"cost": 0.25}, "model_served": "m1", "on_amd": True, "host": "h"},
        {"ok": True, "latency_s": 2.0, "usage": {"cost": True}},  # a bool is not a cost
        {"ok": False, "http_status": 429, "usage": "weird", "on_amd": False},
    ]
    m = ev.receipt_metrics(rs)
    assert (m["calls"], m["calls_ok"], m["calls_failed"]) == (4, 3, 1)
    assert m["cost_usd"] == 0.75
    assert m["schema_invalid"] == 1 and m["rate_limited_attempts"] == 1
    assert m["latency_s_p50"] == 2.0 and m["latency_s_p95"] == 2.9
    assert m["prompt_tokens"] == 10 and m["completion_tokens"] == 2
    assert m["reasoning_tokens_reported"] == 7
    assert m["model_served"] == ["m1"] and m["hosts"] == ["h"] and m["on_amd"] == [False, True]
    assert ev.receipt_metrics([])["reasoning_tokens_reported"] is None


def test_budget_and_guard() -> None:
    b = ev.Budget(0.01)
    assert not b.exceeded
    b.add(None)
    assert b.spent == 0.0
    b.add(0.006)
    b.add(0.006)
    assert b.exceeded

    seen = []
    fn = ev.guarded(lambda t: seen.append(t) or Extraction(t, {}, None, {"cost_usd": 0.004}), ev.Budget(0.01))
    assert fn("a").not_audited is None
    stopped = ev.Budget(0.0)
    assert ev.guarded(lambda t: pytest.fail("called"), stopped)("x").not_audited == "budget guard"

    b2 = ev.Budget(10.0)
    ev.guarded(lambda t: Extraction(t, {}, None, None), b2)("answered, no cost")
    assert b2.spent == ev.UNKNOWN_COST_USD
    ev.guarded(lambda t: Extraction(t, None, "timeout", None), b2)("failed call is free")
    assert b2.spent == ev.UNKNOWN_COST_USD


def test_git_head_clean_dirty_and_unavailable(monkeypatch) -> None:
    def fake(out_head, out_status):
        def run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=out_head if "rev-parse" in cmd else out_status)

        return run

    monkeypatch.setattr(ev.subprocess, "run", fake("abc123\n", ""))
    assert ev.git_head() == "abc123"
    monkeypatch.setattr(ev.subprocess, "run", fake("abc123\n", " M file\n"))
    assert ev.git_head() == "abc123+dirty"

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(ev.subprocess, "run", boom)
    assert ev.git_head() is None
    monkeypatch.setattr(ev.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
    assert ev.git_head() is None


def test_jsonl_and_json_roundtrip(tmp_path) -> None:
    assert ev.read_jsonl(tmp_path / "missing.jsonl") == []
    p = tmp_path / "deep" / "x.jsonl"
    ev.write_jsonl(p, [{"b": 1, "a": "é"}, {"c": 2}])
    assert p.read_text(encoding="utf-8") == '{"a": "é", "b": 1}\n{"c": 2}\n'
    p.write_text(p.read_text(encoding="utf-8") + "\n   \n", encoding="utf-8")
    assert ev.read_jsonl(p) == [{"a": "é", "b": 1}, {"c": 2}]
    j = tmp_path / "n" / "o.json"
    ev.write_json(j, {"z": 1, "a": [1]})
    assert json.loads(j.read_text(encoding="utf-8")) == {"a": [1], "z": 1}
    assert j.read_text(encoding="utf-8").endswith("}\n")


def test_open_client_builds_a_receipting_client(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ev, "load_config", lambda: CFG)
    c = ev.open_client(tmp_path / "r.jsonl", rung="r", run="x")
    try:
        assert c.config is CFG and c.receipts_path == tmp_path / "r.jsonl"
        assert c.context == {"rung": "r", "run": "x"} and c.thinking_off is None
    finally:
        c.close()


def test_check_daily_caps_at_guard_and_fails_closed(env) -> None:
    env(Server(used=0.5))
    allowed, snap = ev.check_daily(0.5)
    assert allowed == pytest.approx(0.3) and snap["daily_cost_used_usd"] == 0.5
    assert snap["daily_cost_limit_usd"] == 1.0
    env(Server(used=0.0))
    assert ev.check_daily(0.2)[0] == 0.2
    env(Server(used=None))
    allowed, snap = ev.check_daily(0.5)
    assert allowed == 0.0 and snap["daily_cost_used_usd"] is None and "no daily spend" in snap["error"]


def test_stored_extractor_and_reaudit_reuse_only_audited_rows() -> None:
    rows = [
        {"text": "t1", "obj": {"k": 1}, "not_audited": None, "attempts": 2, "request_id": "r", "client_id": "c",
         "latency_s": 1.5, "cost_usd": 0.1},
        {"text": "t2", "obj": None, "not_audited": "timeout"},
    ]
    s = ev.stored_extractor(rows)("t2")
    assert (s.obj, s.not_audited, s.attempts, s.receipt) == (None, "timeout", 0, None)
    assert ev.stored_extractor(rows)("t1").attempts == 2

    fresh_calls = []

    def fresh(t):
        fresh_calls.append(t)
        return Extraction(t, {"new": 1}, None)

    re = ev._reaudit(fresh, rows)
    kept = re("t1")
    assert kept.obj == {"k": 1} and kept.attempts == 2
    assert kept.receipt == {"request_id": "r", "client_id": "c", "latency_s": 1.5, "cost_usd": 0.1, "reused": True}
    assert re("t2").obj == {"new": 1} and re("unseen").obj == {"new": 1}
    assert fresh_calls == ["t2", "unseen"]


def test_select_cards_and_stop_on_referee_errors() -> None:
    cards = small_cards()
    dev = ev.select_cards(cards, "dev")
    assert {c["exchange_id"] for c in dev} == {"A2-001"}
    assert ev.select_cards(cards, "all") is cards
    ev.stop_on_referee_errors([{"exchange_id": "x"}], "w", "a")  # no error: returns quietly
    with pytest.raises(SystemExit, match=r"w: referee error on 1 card\(s\) \['y'\]\. adv"):
        ev.stop_on_referee_errors([{"exchange_id": "y", "referee_errors": ["boom"]}], "w", "adv")


def test_parse_judge_every_branch() -> None:
    ok = dict(JUDGE_MISMATCH)
    assert ev.parse_judge("text") == (ABSTAIN, "malformed judge output")
    assert ev.parse_judge({"instruction_present": True}) == (ABSTAIN, "malformed judge output")
    assert ev.parse_judge({**ok, "instruction_present": "yes"})[0] == ABSTAIN
    assert ev.parse_judge({**ok, "readback_present": 1})[0] == ABSTAIN
    assert ev.parse_judge({**ok, "line_match": "maybe"})[0] == ABSTAIN
    assert ev.parse_judge({**ok, "instruction_present": False}) == (None, None)
    assert ev.parse_judge({**ok, "readback_present": False}) == (NO_READBACK, None)
    assert ev.parse_judge(ok) == (MISMATCH, None)
    assert ev.parse_judge({**ok, "tag_match": "n/a"}) == (VERIFIED, None)


def test_metadata_carries_hashes_and_extra(monkeypatch) -> None:
    monkeypatch.setattr(ev, "git_head", lambda: "deadbee")
    m = ev.metadata("run-1", small_cards(), {"subset": "all"})
    assert m["run"] == "run-1" and m["git_head"] == "deadbee" and m["n_cards"] == len(WANTED)
    assert m["subset"] == "all" and m["provenance"] == ev.PROVENANCE
    assert m["cards_sha256"] == ev.sha256(ev.CARDS_PATH.read_bytes())
    assert ev.sha256("a") == ev.sha256(b"a") and len(m["judge_prompt_sha256"]) == 64
    assert "180 exchanges" in m["shift_assumption"]


# ------------------------------------------------------------------------------------------ extraction rungs


def test_model_rung_writes_extractions_cards_receipts_and_summary(env, registry) -> None:
    env(Server())
    rung = RUNGS["Qwen3.8-27B"]  # thinking on -> four workers
    s = ev.run_extraction_rung("r1", rung, small_cards(), registry, Pacer(0.0), ev.Budget(1.0))
    out = ev.OUT / "r1"
    exts = ev.read_jsonl(out / "Qwen3.8-27B.extractions.jsonl")
    assert len(exts) == s["distinct_transcripts"] and all(e["not_audited"] is None for e in exts)
    assert all(e["request_id"] and e["cost_usd"] == 1e-5 for e in exts)
    cards = ev.read_jsonl(out / "Qwen3.8-27B.cards.jsonl")
    assert sorted(c["exchange_id"] for c in cards) == sorted(WANTED)
    assert s["calls"]["calls_ok"] == len(exts) and s["calls"]["cost_usd"] == pytest.approx(1e-5 * len(exts))
    assert s["receipts"] == "bench/results/receipts/text-r1-Qwen3.8-27B.jsonl"
    assert (ev.RECEIPTS / "text-r1-Qwen3.8-27B.jsonl").exists()
    assert s["mode"] == "spans" and s["cache_hits"] >= 1  # "Ulangi?"-style repeats and shared words hit the cache
    assert set(s["meets_rule"]) == {"literal", "strict", "fully_audited"}
    assert "reaudit" not in s and s["rows"]["refusal"][0]["exchange_id"] == "R-01"
    # a second run with the same name refuses to mix receipts
    with pytest.raises(SystemExit, match="exists: pick a new --run name"):
        ev.run_extraction_rung("r1", rung, small_cards(), registry, Pacer(0.0), ev.Budget(1.0))


def test_regex_rung_makes_no_call_and_canonical_tag(env, registry) -> None:
    env(Server())
    s = ev.run_extraction_rung("r2", RUNGS["regex"], small_cards(), registry, Pacer(0.0), ev.Budget(0.0))
    assert s["calls"] is None and s["receipts"] is None
    held = s["rows"]["held_out"]
    assert held["false_verified"]["count"] == 0 and held["errors_flagged"]["k"] == 2  # H-01 and H-02 are caught
    assert (ev.OUT / "r2" / "regex.cards.jsonl").exists()
    c = ev.run_extraction_rung("r2", RUNGS["Gemma-4-31B-IT"], small_cards(), registry, Pacer(0.0), ev.Budget(1.0),
                               mode="canonical")
    assert c["mode"] == "canonical" and c["cards"].endswith("Gemma-4-31B-IT.canonical.cards.jsonl")
    assert (ev.RECEIPTS / "text-r2-Gemma-4-31B-IT.canonical.jsonl").exists()


def test_failed_calls_are_not_audited_and_reaudit_requests_only_those(env, registry) -> None:
    server = env(Server(chat_status=[500]))
    rung = RUNGS["Gemma-4-31B-IT"]
    first = ev.run_extraction_rung("a", rung, small_cards(), registry, Pacer(0.0), ev.Budget(1.0))
    assert first["rows"]["held_out"]["not_audited"]["k"] >= 1
    assert first["meets_rule"]["fully_audited"] is False and first["meets_rule"]["literal"] is False
    prior = ev.read_jsonl(ev.OUT / "a" / "Gemma-4-31B-IT.extractions.jsonl")
    bad = [p for p in prior if p["not_audited"]]
    assert len(bad) == 1 and bad[0]["not_audited"] == "server error"
    before = server.chat_calls
    second = ev.run_extraction_rung("b", rung, small_cards(), registry, Pacer(0.0), ev.Budget(1.0), prior=prior)
    assert server.chat_calls - before == 1  # only the failed transcript went out again
    assert second["reaudit"] == {"requested_again": 1, "still_not_audited": 0}
    assert second["rows"]["held_out"]["not_audited"]["k"] == 0


# ----------------------------------------------------------------------------------------------- llm judge


def test_judge_run_scores_verdicts_and_writes_cards(env) -> None:
    env(Server(judge_out="```json\n" + json.dumps(JUDGE_MISMATCH) + "\n```"))
    s = ev.run_llm_judge("j1", RUNGS["DeepSeek-V4.1-Flash"], small_cards(), Pacer(0.0), ev.Budget(1.0))
    cards = ev.read_jsonl(ev.OUT / "j1" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")
    assert {c["verdict"] for c in cards} == {MISMATCH} and all(c["request_id"] for c in cards)
    assert all(c["judge_output"] == JUDGE_MISMATCH for c in cards)
    assert s["calls"]["calls_ok"] == len(WANTED) and s["mode"].startswith("llm-judge")
    h = s["rows"]["held_out"]
    assert h["errors_flagged"]["k"] == 2  # both seeded errors flagged
    assert h["false_alarm_load"]["k"] == 1  # A2-021 (a correct card) is flagged too
    assert s["rows"]["chatter"]["false_cards"] == 2  # a judge that always says MISMATCH invents cards on chatter
    with pytest.raises(SystemExit, match="exists: pick a new --run name"):
        ev.run_llm_judge("j1", RUNGS["DeepSeek-V4.1-Flash"], small_cards(), Pacer(0.0), ev.Budget(1.0))


def test_judge_invalid_json_becomes_abstain(env) -> None:
    env(Server(judge_out="not json at all"))
    s = ev.run_llm_judge("j2", RUNGS["DeepSeek-V4.1-Flash"], small_cards()[:1], Pacer(0.0), ev.Budget(1.0))
    rec = ev.read_jsonl(ev.OUT / "j2" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")[0]
    assert rec["verdict"] == ABSTAIN and rec["problem"] == "malformed judge output"
    assert rec["judge_output"] == {"_invalid": "not json at all"}
    assert s["rows"]["refusal"] == [] and s["rows"]["held_out"]["abstain_rate"]["k"] == 1


def test_judge_retries_429_then_gives_up_and_budget_guard(env, monkeypatch) -> None:
    sleeps: list[float] = []
    env(Server(chat_status=[429, 429]))
    monkeypatch.setattr(ev.time, "sleep", sleeps.append)
    cards = small_cards()[:1]
    s = ev.run_llm_judge("j3", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(1.0))
    assert sleeps == [15.0, 30.0]  # linear back-off, then the third attempt answers
    assert s["calls"]["rate_limited_attempts"] == 2 and s["calls"]["calls_ok"] == 1

    env(Server(chat_status=[429, 429, 429, 429]))
    s = ev.run_llm_judge("j4", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(1.0))
    rec = ev.read_jsonl(ev.OUT / "j4" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")[0]
    assert rec["verdict"] == NOT_AUDITED and rec["problem"] == "rate limited"
    assert s["meets_rule"]["literal"] is False
    assert s["rows"]["held_out"]["not_audited"]["k"] == 1

    env(Server(chat_status=[500]))
    ev.run_llm_judge("j5", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(1.0))
    rec = ev.read_jsonl(ev.OUT / "j5" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")[0]
    assert rec["problem"] == "server error"

    server = env(Server())
    ev.run_llm_judge("j6", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(0.0))
    rec = ev.read_jsonl(ev.OUT / "j6" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")[0]
    assert rec["verdict"] == NOT_AUDITED and rec["problem"] == "budget guard" and server.chat_calls == 0


def test_judge_notauditing_without_receipt(env, monkeypatch) -> None:
    env(Server())

    class Boom:
        def chat(self, *a, **k):
            raise NotAuditing("timeout")

        def close(self):
            pass

    monkeypatch.setattr(ev, "open_client", lambda p, **c: Boom())
    # receipts file is never created by the stub, so run_llm_judge's final read_jsonl sees an empty list
    s = ev.run_llm_judge("j7", RUNGS["DeepSeek-V4.1-Flash"], small_cards()[:1], Pacer(0.0), ev.Budget(1.0))
    rec = ev.read_jsonl(ev.OUT / "j7" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")[0]
    assert rec["problem"] == "timeout" and rec["request_id"] is None and s["calls"]["calls"] == 0


def test_judge_reaudit_reuses_audited_cards(env) -> None:
    env(Server(chat_status=[500]))
    cards = small_cards()[:3]
    ev.run_llm_judge("k1", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(1.0))
    prior = ev.read_jsonl(ev.OUT / "k1" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")
    assert [p["verdict"] for p in prior].count(NOT_AUDITED) == 1
    server = env(Server())
    ev.run_llm_judge("k2", RUNGS["DeepSeek-V4.1-Flash"], cards, Pacer(0.0), ev.Budget(1.0), prior=prior)
    assert server.chat_calls == 1
    rec = ev.read_jsonl(ev.OUT / "k2" / "DeepSeek-V4.1-Flash.judge.cards.jsonl")
    assert sum(bool(r.get("reused_from_first_pass")) for r in rec) == 2
    assert NOT_AUDITED not in {r["verdict"] for r in rec}


# ------------------------------------------------------------------------------------------------ commands


def ladder(*extra: str, run: str = "L1") -> list[str]:
    return ["--ladder", "--run", run, "--pace", "0", *extra]


def result(name: str) -> dict:
    return json.loads((ev.OUT / f"{name}.json").read_text(encoding="utf-8"))


def test_ladder_regex_only_needs_no_credentials(env, capsys) -> None:
    env(Server(used=None))  # even an unreadable usage endpoint is irrelevant: nothing is paid
    assert ev.main(ladder("--rungs", "regex")) == 0
    doc = result("L1")
    assert doc["rungs"]["regex"]["calls"] is None and doc["ladder_order"] == ["regex"]
    assert doc["budget_usd"] == 0.0 and doc["usage_before"] == {} and doc["calls_cost_usd"] == 0.0
    assert doc["subset"] == "all" and doc["git_head"] is None  # tmp ROOT is not a repo
    assert doc["smallest_rung_meeting_rule"] == "regex"  # 0 false-VERIFIED, 0/1 false alarms, fully audited
    out = capsys.readouterr().out
    assert "wrote bench/results/text/L1.json" in out and '"held_out_label_exact"' in out


def test_ladder_default_rungs_orders_smallest_first_and_names_the_winner(env, capsys) -> None:
    env(Server(used=0.0))
    assert ev.main(ladder("--subset", "all", "--budget", "0.3", run="L2")) == 0
    doc = result("L2")
    assert doc["ladder_order"] == ["regex", "MiniCPM5-2B", "Qwen3.8-27B", "Gemma-4-31B-IT", "DeepSeek-V4-Flash"]
    assert doc["budget_usd"] == 0.3 and doc["usage_before"]["daily_cost_used_usd"] == 0.0
    assert doc["spent_usd"] > 0 and doc["calls_cost_usd"] == pytest.approx(
        sum(r["calls"]["cost_usd"] for r in doc["rungs"].values() if r["calls"])
    )
    winner = doc["smallest_rung_meeting_rule"]
    expected = next((n for n in doc["ladder_order"] if doc["rungs"][n]["meets_rule"]["literal"]), None)
    assert winner == expected
    assert f"smallest rung meeting rule: {winner}" in capsys.readouterr().out


def test_ladder_dev_subset_and_daily_guard(env, capsys) -> None:
    env(Server(used=0.0))
    assert ev.main(ladder("--rungs", "regex,Gemma-4-31B-IT", "--subset", "dev", run="L3")) == 0
    assert result("L3")["rungs"]["regex"]["rows"]["dev"]["n_cards"] == 1
    env(Server(used=0.95))
    assert ev.main(ladder("--rungs", "Gemma-4-31B-IT", run="L4")) == 2
    assert "daily spend guard" in capsys.readouterr().err
    assert not (ev.OUT / "L4.json").exists()


def test_ladder_default_run_name_is_timestamped(env) -> None:
    env(Server())
    assert ev.main(["--ladder", "--rungs", "regex", "--pace", "0", "--subset", "dev"]) == 0
    names = [p.name for p in ev.OUT.glob("*.json")]
    assert len(names) == 1 and names[0].startswith("dev-") and names[0].endswith("Z.json")


def test_ladder_reaudit_rerequests_only_failed_transcripts(env, capsys) -> None:
    env(Server(chat_status=[500]))
    assert ev.main(ladder("--rungs", "regex,Gemma-4-31B-IT", run="first")) == 0
    first = result("first")
    assert first["rungs"]["Gemma-4-31B-IT"]["meets_rule"]["fully_audited"] is False
    server = env(Server())
    first_path = ev.OUT / "first.json"
    assert ev.main(ladder("--reaudit", str(first_path), run="second")) == 0
    second = result("second")
    assert second["reaudit_of"] == "bench/results/text/first.json"
    assert second["ladder_order"] == ["regex", "Gemma-4-31B-IT"]
    assert second["rungs"]["Gemma-4-31B-IT"]["reaudit"] == {"requested_again": 1, "still_not_audited": 0}
    assert server.chat_calls == 1
    assert second["rungs"]["regex"]["calls"] is None

    # a first pass scored on other cards is refused
    doc = result("first")
    doc["cards_sha256"] = "0" * 64
    ev.write_json(first_path, doc)
    capsys.readouterr()
    assert ev.main(ladder("--reaudit", str(first_path), run="third")) == 2
    assert "other cards" in capsys.readouterr().err
    doc["cards_sha256"], doc["subset"] = ev.sha256(ev.CARDS_PATH.read_bytes()), "dev"
    ev.write_json(first_path, doc)
    assert ev.main(ladder("--reaudit", str(first_path), run="third")) == 2


def test_missing_credentials_exit_code_3(env, monkeypatch, capsys) -> None:
    def no_key(*a, **k):
        raise MissingCredentials("put a key in ~/.config/x")

    monkeypatch.setattr(ev, "open_client", no_key)
    assert ev.main(ladder("--rungs", "Gemma-4-31B-IT")) == 3
    assert "no Token Factory key" in capsys.readouterr().err


def test_ablate_grounding_with_and_without_reuse_and_reaudit(env, capsys) -> None:
    env(Server(used=0.0))
    assert ev.main(["--ablate", "grounding", "--model", "Gemma-4-31B-IT", "--run", "ab1", "--pace", "0"]) == 0
    ab = result("ab1")
    assert ab["kind"] == "ablate-grounding" and set(ab["results"]) == {"with_grounding", "without_grounding"}
    assert ab["results"]["with_grounding"]["mode"] == "spans"
    assert ab["results"]["without_grounding"]["mode"] == "canonical"
    assert ab["calls_cost_usd"] == pytest.approx(sum(r["calls"]["cost_usd"] for r in ab["results"].values()))

    # reuse: the with-grounding side is copied from a ladder run on the same cards, no second call
    assert ev.main(ladder("--rungs", "regex,Gemma-4-31B-IT", run="lad")) == 0
    assert ev.main(["--ablate", "grounding", "--model", "Gemma-4-31B-IT", "--run", "ab2", "--pace", "0",
                    "--reuse", str(ev.OUT / "lad.json")]) == 0
    ab2 = result("ab2")
    assert ab2["results"]["with_grounding"]["reused_from"] == "bench/results/text/lad.json"
    assert ab2["calls_cost_usd"] == pytest.approx(ab2["results"]["without_grounding"]["calls"]["cost_usd"])

    # reuse of a ladder run on other cards is refused
    bad = result("lad")
    bad["cards_sha256"] = "f" * 64
    ev.write_json(ev.OUT / "lad.json", bad)
    capsys.readouterr()
    assert ev.main(["--ablate", "grounding", "--run", "ab3", "--model", "Gemma-4-31B-IT", "--reuse",
                    str(ev.OUT / "lad.json")]) == 2
    assert "--reuse" in capsys.readouterr().err

    # reaudit of the ablation: only failed canonical transcripts are asked again
    env(Server(chat_status=[200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 500]))
    assert ev.main(["--ablate", "grounding", "--model", "Gemma-4-31B-IT", "--run", "ab4", "--pace", "0"]) == 0
    ab4 = result("ab4")
    assert ab4["results"]["without_grounding"]["rows"]["held_out"]["not_audited"]["k"] == 0
    server = env(Server())
    assert ev.main(["--ablate", "grounding", "--model", "Gemma-4-31B-IT", "--run", "ab5", "--pace", "0",
                    "--reaudit", str(ev.OUT / "ab4.json")]) == 0
    ab5 = result("ab5")
    assert ab5["reaudit_of"] == "bench/results/text/ab4.json"
    assert ab5["results"]["with_grounding"] == ab4["results"]["with_grounding"]
    assert "reaudit" in ab5["results"]["without_grounding"]
    assert server.chat_calls == ab5["results"]["without_grounding"]["reaudit"]["requested_again"] - (
        ab5["results"]["without_grounding"]["reaudit"]["requested_again"] - server.chat_calls
    )

    doc = result("ab4")
    doc["subset"] = "dev"
    ev.write_json(ev.OUT / "ab4.json", doc)
    capsys.readouterr()
    assert ev.main(["--ablate", "grounding", "--model", "Gemma-4-31B-IT", "--run", "ab6", "--reaudit",
                    str(ev.OUT / "ab4.json")]) == 2
    assert "other cards" in capsys.readouterr().err


def test_ablate_and_baseline_stop_at_the_daily_guard(env, capsys) -> None:
    env(Server(used=0.99))
    assert ev.main(["--ablate", "grounding", "--run", "g1"]) == 2
    assert ev.main(["--baseline", "llm-judge", "--model", "DeepSeek-V4.1-Flash", "--run", "g2"]) == 2
    assert capsys.readouterr().err.count("daily spend guard") == 2


def test_baseline_llm_judge_and_its_reaudit(env) -> None:
    env(Server(used=0.0, chat_status=[500]))
    argv = ["--baseline", "llm-judge", "--model", "DeepSeek-V4.1-Flash", "--pace", "0"]
    assert ev.main([*argv, "--run", "jb1"]) == 0
    jb1 = result("jb1")
    assert jb1["kind"] == "llm-judge" and jb1["results"]["llm_judge"]["rows"]["held_out"]["not_audited"]["k"] == 1
    server = env(Server(used=0.0))
    assert ev.main([*argv, "--run", "jb2", "--reaudit", str(ev.OUT / "jb1.json")]) == 0
    jb2 = result("jb2")
    assert jb2["reaudit_of"] == "bench/results/text/jb1.json" and server.chat_calls == 1
    assert jb2["results"]["llm_judge"]["rows"]["held_out"]["not_audited"]["k"] == 0
    assert jb2["calls_cost_usd"] == jb2["results"]["llm_judge"]["calls"]["cost_usd"]


# --------------------------------------------------------------------------------------- rescore/report/compare


def test_rescore_ok_diff_write_and_reused_skip(env, capsys) -> None:
    env(Server(used=0.0))
    assert ev.main(ladder("--rungs", "regex,Gemma-4-31B-IT", run="rs")) == 0
    assert ev.main(["--baseline", "llm-judge", "--model", "DeepSeek-V4.1-Flash", "--run", "rj", "--pace", "0"]) == 0
    capsys.readouterr()
    # default: every result file in OUT
    assert ev.main(["--rescore"]) == 0
    out = capsys.readouterr().out
    assert out.count("OK   ") == 3 and "DIFF" not in out
    assert "rs.json :: Gemma-4-31B-IT (rows, calls, meets_rule)" in out

    doc = result("rs")
    doc["rungs"]["regex"]["rows"]["held_out"]["n_cards"] = 999
    doc["rungs"]["Gemma-4-31B-IT"]["reused_from"] = "elsewhere"
    doc["calls_cost_usd"] = 123.0
    ev.write_json(ev.OUT / "rs.json", doc)
    jd = result("rj")
    jd["results"]["llm_judge"]["rows"]["held_out"]["n_cards"] = 999
    ev.write_json(ev.OUT / "rj.json", jd)

    assert ev.main(["--rescore", str(ev.OUT / "rs.json"), str(ev.OUT / "rj.json")]) == 1
    out = capsys.readouterr().out
    assert "DIFF rs.json :: regex (rows, calls, meets_rule) differs: ['rows']" in out
    assert "DIFF rj.json :: llm_judge (rows, calls, meets_rule) differs: ['rows']" in out
    assert "Gemma" not in out  # the reused copy is checked where it was produced

    assert ev.main(["--rescore", "--write", str(ev.OUT / "rs.json"), str(ev.OUT / "rj.json")]) == 0
    out = capsys.readouterr().out
    assert out.count("rewrote") == 2
    fixed = result("rs")
    assert fixed["rungs"]["regex"]["rows"]["held_out"]["n_cards"] != 999
    assert "rescored" in fixed["rungs"]["regex"] and fixed["calls_cost_usd"] == 0.0
    assert fixed["smallest_rung_meeting_rule"] in (None, "regex")
    assert result("rj")["results"]["llm_judge"]["rows"]["held_out"]["n_cards"] != 999
    assert ev.main(["--rescore", str(ev.OUT / "rs.json"), str(ev.OUT / "rj.json")]) == 0


def test_rescore_judge_remaps_stored_outputs(env) -> None:
    env(Server(used=0.0))
    assert ev.main(["--baseline", "llm-judge", "--run", "rjj", "--pace", "0"]) == 0
    summary = result("rjj")["results"]["llm_judge"]
    path = ev.ROOT / summary["cards"]
    recs = ev.read_jsonl(path)
    recs[0]["judge_output"] = {**JUDGE_MISMATCH, "tag_match": "yes"}  # was MISMATCH, now VERIFIED
    recs[1]["verdict"] = NOT_AUDITED  # stays unmeasured
    recs[1]["judge_output"] = None
    ev.write_jsonl(path, recs)
    new = ev.rescore_judge(summary)
    again = {r["exchange_id"]: r for r in ev.read_jsonl(path)}
    assert again[recs[0]["exchange_id"]]["verdict"] == MISMATCH  # the stored file is untouched
    assert new["rows"]["held_out"]["not_audited"]["k"] == 1
    assert new["calls"]["calls_ok"] == len(WANTED)


def test_rescore_stops_loud_on_a_referee_error(env, monkeypatch) -> None:
    env(Server(used=0.0))
    assert ev.main(ladder("--rungs", "regex", run="re")) == 0
    from referee import coverage

    monkeypatch.setattr(coverage, "check", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SystemExit, match=r"--rescore bench/results/text/re/regex\.extractions\.jsonl: referee error"):
        ev.main(["--rescore", str(ev.OUT / "re.json")])


def test_report_formats_rows_and_notes(env, capsys) -> None:
    env(Server(used=0.0, chat_status=[500]))
    assert ev.main(ladder("--rungs", "regex,Gemma-4-31B-IT", run="rp1")) == 0
    server = env(Server())
    assert ev.main(ladder("--reaudit", str(ev.OUT / "rp1.json"), run="rp2")) == 0
    assert server.chat_calls == 1
    assert ev.main(["--baseline", "llm-judge", "--run", "rp3", "--pace", "0"]) == 0
    capsys.readouterr()
    files = [str(ev.OUT / f"{n}.json") for n in ("rp1", "rp2", "rp3")]
    assert ev.main(["--report", *files]) == 0
    assert "wrote bench/results/text/RESULTS.md" in capsys.readouterr().out
    md = (ev.OUT / "RESULTS.md").read_text(encoding="utf-8")
    assert md.startswith("# Text-row results")
    assert "### rp1 — `text·scripted`, row `held_out`, 7 cards run" in md
    assert "Re-audit of bench/results/text/rp1.json" in md
    assert "Cards in the *Not audited* column" in md  # rp1 had a failed call
    assert "| regex |" in md and "| llm_judge |" in md
    regex_row = next(line for line in md.splitlines() if line.startswith("| regex |"))
    assert "| — |" in regex_row  # no calls, so no latency
    assert "ABSTAIN · unknown tag" in regex_row  # the refusal column prints what R-01 got


def test_report_helpers_edge_cases() -> None:
    assert ev._r(None) == "n/a" and ev._r({"n": 0}) == "n/a"
    assert ev._r({"k": 1, "n": 3, "rate": 0.3333}) == "1/3 (33.33 %)"
    assert ev._r({"k": 1, "n": 2, "rate": 0.5}) == "1/2 (50 %)"
    assert ev._ci(None) == "" and ev._ci({"ci95": None}) == ""
    assert ev._ci({"ci95": [0.1, 0.5]}) == " [10 % – 50 %]"
    assert ev._run_cost({"a": {"calls": {"cost_usd": 0.2}}, "b": {"calls": None}, "c": {"reused_from": "x",
            "calls": {"cost_usd": 9}}}) == 0.2


def test_report_dev_row_without_cost_key(env, capsys, tmp_path) -> None:
    env(Server())
    assert ev.main(ladder("--rungs", "regex", "--subset", "dev", run="rd")) == 0
    doc = result("rd")
    del doc["calls_cost_usd"]
    ev.write_json(ev.OUT / "rd.json", doc)
    lines = ev.report_lines(ev.OUT / "rd.json")
    assert "row `dev`" in lines[0] and "cost $0.0" in lines[0]


def test_compare_reports_label_and_output_differences(env, capsys) -> None:
    env(Server(used=0.0))
    for run in ("c1", "c2"):
        assert ev.main(ladder("--rungs", "Gemma-4-31B-IT", run=run)) == 0
    # make the second run disagree on one output and one label
    b = result("c2")["rungs"]["Gemma-4-31B-IT"]
    exts = ev.read_jsonl(ev.ROOT / b["extractions"])
    exts[0]["obj"] = {"kind": "instruction", "commands": [], "retracted_spans": []}
    ev.write_jsonl(ev.ROOT / b["extractions"], exts)
    cards = ev.read_jsonl(ev.ROOT / b["cards"])
    cards[0]["label"] = "changed"
    ev.write_jsonl(ev.ROOT / b["cards"], cards)
    capsys.readouterr()
    argv = ["--compare", "--model", "Gemma-4-31B-IT", str(ev.OUT / "c1.json"), str(ev.OUT / "c2.json")]
    assert ev.main(argv) == 0
    out = capsys.readouterr().out
    res = json.loads((ev.OUT / "repro-c2.json").read_text(encoding="utf-8"))
    assert res["cards_compared"] == len(WANTED) and res["cards_same_label"] == len(WANTED) - 1
    assert len(res["label_differences"]) == 1 and "'changed'" in res["label_differences"][0]
    assert res["outputs_identical"] == res["outputs_compared"] - 1
    assert res["output_differences"][0]["second"]["kind"] == "instruction"
    assert res["first"] == "bench/results/text/c1.json" and res["rung"] == "Gemma-4-31B-IT"
    assert f"{res['cards_same_label']}/{res['cards_compared']} cards same label" in out
    assert ev.main([*argv, "--run", "named"]) == 0
    assert (ev.OUT / "repro-named.json").exists()
