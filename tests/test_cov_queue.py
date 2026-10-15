"""Coverage for the supervisor queue: sayagain/server.py, sayagain/shift.py, scripts/export_run.py (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sayagain import server, shift
from sayagain.amd_client import Config, MissingCredentials, TokenFactory
from sayagain.extract import Extraction
from scripts import export_run

RUN, RUNG = shift.DEFAULT_RUN, shift.DEFAULT_RUNG
HDRS = {"host": "127.0.0.1"}


class FakeLive:
    """Stands in for LiveExtract: serves the stored extractions as if they were live answers."""

    def __init__(self, var, budget_usd=0.05):
        self.rungs = []

    def for_rung(self, rung_name):
        self.rungs.append(rung_name)
        stored = shift.stored_extractor(RUN, rung_name)
        return lambda text: stored(text)


@pytest.fixture()
def live_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAYAGAIN_LIVE_EXTRACT", "1")
    monkeypatch.setattr(server, "load_config", lambda: Config(api_key="k", base_url="https://x.invalid/v1"))
    monkeypatch.setattr(server, "LiveExtract", FakeLive)
    return TestClient(server.create_app(var=tmp_path), base_url="http://127.0.0.1")


@pytest.fixture()
def plain_client(tmp_path, monkeypatch):
    monkeypatch.delenv("SAYAGAIN_LIVE_EXTRACT", raising=False)
    return TestClient(server.create_app(var=tmp_path), base_url="http://127.0.0.1")


# ------------------------------------------------------------------------------------------------- live_allowed


def test_live_allowed_requires_flag_and_credentials(monkeypatch):
    monkeypatch.delenv("SAYAGAIN_LIVE_EXTRACT", raising=False)
    assert server.live_allowed() is False
    monkeypatch.setenv("SAYAGAIN_LIVE_EXTRACT", "1")

    def missing():
        raise MissingCredentials("none")

    monkeypatch.setattr(server, "load_config", missing)
    assert server.live_allowed() is False
    monkeypatch.setattr(server, "load_config", lambda: Config(api_key="k", base_url="https://x.invalid"))
    assert server.live_allowed() is True


# ------------------------------------------------------------------------------------------------- LiveExtract


def test_for_rung_refuses_rungs_without_a_model(tmp_path):
    live = server.LiveExtract(tmp_path)
    for name in ("regex", "no-such-rung"):
        with pytest.raises(HTTPException) as e:
            live.for_rung(name)
        assert e.value.status_code == 400


def test_for_rung_builds_one_cached_client_per_rung(tmp_path, monkeypatch):
    made = []

    class FakeClient(TokenFactory):
        def __init__(self, config, **kw):
            made.append((config, kw))
            super().__init__(config, **kw)

    monkeypatch.setattr(server, "TokenFactory", FakeClient)
    monkeypatch.setattr(server, "load_config", lambda: Config(api_key="k", base_url="https://x.invalid/v1"))
    live = server.LiveExtract(tmp_path)
    a = live.for_rung(RUNG)
    assert live.for_rung(RUNG) is a and len(made) == 1  # the second call reuses the process-wide cache
    assert isinstance(a, server._AuditedCache)
    assert (tmp_path / "receipts").is_dir()
    assert made[0][1]["receipts_path"].parent == tmp_path / "receipts"
    assert made[0][1]["receipts_path"].name.endswith(f"-{RUNG}.jsonl")
    assert live.for_rung("MiniCPM5-2B") is not a and len(made) == 2


def test_guarded_stops_at_the_budget_and_charges_reported_cost(tmp_path):
    live = server.LiveExtract(tmp_path, budget_usd=0.0015)
    calls = []

    def ext(text):
        calls.append(text)
        return Extraction(text, {}, None, {"request_id": "r", "cost_usd": 0.0004}, 1)

    call = live._guarded(ext)
    out = call("a")
    assert out.receipt["live"] is True and out.receipt["cost_usd"] == 0.0004
    assert live.spent == pytest.approx(0.0004)
    call("b")  # 0.0004 + 0.001 reserve <= 0.0015: allowed
    assert live.spent == pytest.approx(0.0008)
    call("c")  # 0.0008 + 0.001 > 0.0015: refused without calling the model
    refused = call("d")
    assert refused.not_audited == "live budget spent" and refused.obj is None and refused.receipt is None
    assert calls == ["a", "b"]


def test_guarded_keeps_a_receiptless_answer(tmp_path):
    live = server.LiveExtract(tmp_path)
    out = live._guarded(lambda text: Extraction(text, None, "timeout", None, 1))("x")
    assert out.not_audited == "timeout" and out.receipt is None
    assert live.spent == pytest.approx(live.RESERVE_USD)


def test_audited_cache_hit_counter():
    cache = server._AuditedCache(lambda t: Extraction(t, {}, None, None, 1))
    cache("x")
    cache("x")
    assert cache.hits == 1


# ------------------------------------------------------------------------------------------------- origin guard


def test_not_cross_site_rules(plain_client):
    body = {"run": RUN, "rung": RUNG}
    ok = plain_client.post("/api/try-refused", json=body, headers={"sec-fetch-site": "same-origin"})
    assert ok.status_code == 200
    none = plain_client.post("/api/try-refused", json=body, headers={"sec-fetch-site": "none"})
    assert none.status_code == 200
    for site in ("cross-site", "same-site"):
        r = plain_client.post("/api/try-refused", json=body, headers={"sec-fetch-site": site})
        assert r.status_code == 403
    bad = plain_client.post("/api/try-refused", json=body, headers={"origin": "http://evil.example"})
    assert bad.status_code == 403
    good = plain_client.post("/api/try-refused", json=body, headers={"origin": "http://127.0.0.1"})
    assert good.status_code == 200


def test_untrusted_host_header_is_400(plain_client):
    assert plain_client.get("/config.json", headers={"host": "evil.example"}).status_code == 400


def test_root_redirects_to_queue(plain_client):
    r = plain_client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/queue"


# ------------------------------------------------------------------------------------------------- stream


def _events(resp):
    return [json.loads(x[6:]) for x in resp.iter_lines() if x.startswith("data: ")]


def test_stream_rejects_unknown_extract_mode(plain_client):
    r = plain_client.get("/api/stream", params={"run": RUN, "rung": RUNG, "extract": "bogus"})
    assert r.status_code == 400


def test_stream_unknown_run_is_404(plain_client):
    assert plain_client.get("/api/stream", params={"run": "nope", "rung": RUNG}).status_code == 404


def test_stream_live_refused_when_off(plain_client):
    r = plain_client.get("/api/stream", params={"run": RUN, "rung": RUNG, "extract": "live"})
    assert r.status_code == 403


def test_stream_live_refuses_cross_site(live_client):
    r = live_client.get("/api/stream", params={"extract": "live"}, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403


def test_stream_live_uses_wall_clock_and_tags_events(live_client):
    with live_client.stream(
        "GET", "/api/stream", params={"run": RUN, "rung": RUNG, "extract": "live", "start": 143}
    ) as r:
        evs = _events(r)
    assert evs[-1] == {"type": "closed"}
    assert evs[-2]["type"] == "end_of_shift"
    body = [e for e in evs[:-1]]
    assert all(e["extract"] == "live" for e in body)
    assert sum(e["type"] == "judged" for e in body) == 2
    ats = [e["at"] for e in body]
    assert ats == sorted(ats) and ats[-1] < 30  # wall clock since the stream opened, not the replay clock


def test_stream_stored_paces_at_speed(plain_client, monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    with plain_client.stream("GET", "/api/stream", params={"run": RUN, "rung": RUNG, "speed": 64, "start": 143}) as r:
        evs = _events(r)
    assert evs[-1] == {"type": "closed"} and all(e["extract"] == "stored" for e in evs[:-1])
    assert slept and all(s >= 0 for s in slept)
    replay = [e["at"] for e in evs[:-1]]
    assert sum(slept) == pytest.approx((replay[-1] - replay[0]) / 64, rel=1e-6)


# ------------------------------------------------------------------------------------------------- try-refused


def test_try_refused_live_path(live_client):
    j = live_client.post("/api/try-refused", json={"run": RUN, "rung": RUNG, "extract": "live"}).json()
    assert j["extract"] == "live" and j["card"]["verdict"]["label"] == "ABSTAIN · unknown tag"
    assert j["card"]["at"] is None and j["card"]["baseline"]["verdict"] == "VERIFIED"


def test_try_refused_bad_mode_and_missing_card(plain_client, monkeypatch):
    r = plain_client.post("/api/try-refused", json={"run": RUN, "rung": RUNG, "extract": "x"})
    assert r.status_code == 400 and "stored or live" in r.json()["detail"]
    monkeypatch.setattr(shift, "run_cards", lambda run, rung: [])
    r = plain_client.post("/api/try-refused", json={"run": RUN, "rung": RUNG})
    assert r.status_code == 404 and shift.REFUSED_ID in r.json()["detail"]


def test_try_refused_live_off_is_403(plain_client):
    r = plain_client.post("/api/try-refused", json={"run": RUN, "rung": RUNG, "extract": "live"})
    assert r.status_code == 403


# ------------------------------------------------------------------------------------------------- reviews


def test_review_reason_length_and_trim(plain_client):
    base = {"run": RUN, "rung": RUNG, "exchange_id": "H-01", "decision": "dismissed"}
    too_long = plain_client.post("/api/review", json={**base, "reason": "x" * (server.MAX_REASON + 1)})
    assert too_long.status_code == 422
    ok = plain_client.post("/api/review", json={**base, "reason": "  wrong tag  "})
    assert ok.status_code == 201 and ok.json()["reason"] == "wrong tag"
    again = plain_client.post("/api/review", json={**base, "decision": "confirmed", "reason": ""})
    assert again.json()["reason"] is None and again.json()["id"] == ok.json()["id"] + 1
    rows = plain_client.get("/api/reviews", params={"run": RUN, "rung": RUNG})
    assert [r["decision"] for r in rows.json()] == ["dismissed", "confirmed"]
    assert rows.headers["cache-control"] == "no-store"
    assert plain_client.get("/api/reviews", params={"run": RUN, "rung": "regex"}).json() == []


# ------------------------------------------------------------------------------------------------- shift.py


def test_summary_and_rung_info_errors(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        shift._summary("no-such-run")
    with pytest.raises(KeyError):
        shift._summary("../x")
    with pytest.raises(KeyError):
        shift._rung_info(RUN, "no-such-rung")


def test_judge_baseline_without_model_in_summary(monkeypatch):
    real = shift._summary

    def broken(run):
        if run == shift.JUDGE_RUN:
            return {}
        return real(run)

    monkeypatch.setattr(shift, "_summary", broken)
    base = shift.judge_baseline()
    assert base and {b["model"] for b in base.values()} == {None}


def test_rung_stage_without_receipts_says_so(monkeypatch):
    monkeypatch.setattr(shift, "run_receipts", lambda run, rung: [])
    ex = shift.rung_stage(RUN, RUNG)["extract"]
    assert ex["where"] == "no receipt found for this run" and ex["on_amd"] is None
    assert ex["calls"] == 0 and ex["latency_s_p50"] is None


def test_rung_stage_mixed_hosts_are_not_claimed_as_amd(monkeypatch):
    recs = [
        {"host": "a.amd.com", "on_amd": True, "model_served": "m", "latency_s": 1.0, "run": RUN},
        {"host": "other.example", "on_amd": False, "model_served": "m", "latency_s": 3.0, "run": "other"},
    ]
    monkeypatch.setattr(shift, "run_receipts", lambda run, rung: recs)
    ex = shift.rung_stage(RUN, RUNG)["extract"]
    assert ex["where"] == "API host not AMD-operated / mixed" and ex["on_amd"] is None
    assert ex["latency_s_p50"] == 2.0 and ex["calls_own"] == 1 and ex["calls_reused"] == 1


# ------------------------------------------------------------------------------------------------- export_run


def test_git_head_failure_is_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(export_run.subprocess, "run", boom)
    assert export_run.git_head() is None


def test_git_head_returns_short_hash(monkeypatch):
    class R:
        stdout = "abc1234\n"

    monkeypatch.setattr(export_run.subprocess, "run", lambda *a, **k: R())
    assert export_run.git_head() == "abc1234"


def test_export_writes_replay_and_swaps_in(tmp_path, monkeypatch):
    monkeypatch.setattr(export_run, "git_head", lambda: "deadbee")
    out = tmp_path / "replay"
    out.mkdir()
    (out / "old.txt").write_text("stale")
    stale_tmp = tmp_path / "replay.tmp"
    stale_tmp.mkdir()
    rc = export_run.main(["--runs", f"{RUN}:regex", "--out", str(out)])
    assert rc == 0
    assert not (out / "old.txt").exists() and not stale_tmp.exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["mode"] == "replay" and cfg["default"] == {"run": RUN, "rung": "regex"}
    assert cfg["live_extract"] is False and cfg["git_head"] == "deadbee"
    assert cfg["runs"] == [{"run": RUN, "rungs": ["regex"], "n_cards": 145}]
    data = json.loads((out / "data" / f"{RUN}--regex.json").read_text())
    assert data["meta"]["run"] == RUN and data["events"][-1]["type"] == "end_of_shift"
    assert (out / "index.html").exists()


def test_main_prints_report_and_relative_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_run, "export", lambda pairs, out: {"r/x": {"cards": 3, "live_cards": 1}})
    inside = export_run.ROOT / "docs" / "replay-test-not-written"
    assert export_run.main(["--runs", "r:x", "--out", str(inside)]) == 0
    text = capsys.readouterr().out
    assert "r/x: 3 cards, 1 to the live queue" in text
    assert "replay written to docs/replay-test-not-written" in text
    export_run.main(["--runs", "r:x", "--out", str(tmp_path / "o")])
    assert f"replay written to {tmp_path / 'o'}" in capsys.readouterr().out


def test_main_rejects_malformed_pair(capsys):
    with pytest.raises(SystemExit) as e:
        export_run.main(["--runs", "just-a-run"])
    assert e.value.code == 2
    assert "bad run:rung" in capsys.readouterr().err


def _fake_events(label_ok: bool, n_cards: int = 1):
    cards = [{"type": "judged", "card": {"exchange_id": f"E-{i}", "matches_stored": label_ok, "live": False}}
             for i in range(n_cards)]  # fmt: skip
    return lambda run, rung: iter(cards)


def test_export_refuses_a_replay_that_disagrees_with_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(shift, "shift_events", _fake_events(False))
    out = tmp_path / "replay"
    with pytest.raises(SystemExit) as e:
        export_run.export([(RUN, RUNG)], out)
    assert "re-judged label differs" in str(e.value) and "E-0" in str(e.value)
    assert not out.exists() and not (tmp_path / "replay.tmp").exists()


def test_export_refuses_a_card_count_mismatch_and_keeps_old_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(shift, "shift_events", _fake_events(True, 1))
    monkeypatch.setattr(shift, "stored_labels", lambda run, rung: {"a": None, "b": None})
    out = tmp_path / "replay"
    out.mkdir()
    (out / "keep.txt").write_text("old replay")
    with pytest.raises(SystemExit) as e:
        export_run.export([(RUN, RUNG)], out)
    assert "1 cards replayed, the run stored 2" in str(e.value)
    assert (out / "keep.txt").read_text() == "old replay"
    assert not (tmp_path / "replay.tmp").exists()


def test_default_runs_are_run_rung_pairs():
    assert all(":" in r for r in export_run.DEFAULT_RUNS)
    assert isinstance(export_run.DEFAULT_OUT, Path)


def test_orphan_readback_is_listed_at_end_of_shift():
    from referee.regex_extractor import extract as regex_extract

    cards = [
        {
            "exchange_id": "O-01",
            "set": "H",
            "dev_slice": False,
            "lang": "en",
            "error_type": "orphan",
            "turns": [{"role": "tech", "text": "Copy, closing two-oh-eight, line Bravo."}],
        }  # fmt: skip
    ]
    evs = list(
        shift.shift_events(
            RUN, "regex",
            extract=lambda text: Extraction(text, regex_extract(text), None, None, 0),
            cards=cards, check_stored=False,
        )
    )  # fmt: skip
    eos = evs[-1]
    assert eos["type"] == "end_of_shift" and eos["orphan_readbacks"] == ["O-01"] and eos["cards"] == 1
    assert [e["type"] for e in evs[:-1]] == ["turn", "judged"]
    assert "matches_stored" not in evs[1]["card"]
