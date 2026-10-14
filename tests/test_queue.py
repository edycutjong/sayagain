"""The supervisor queue (sayagain/shift.py, sayagain/server.py) and the static replay (scripts/export_run.py)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from referee.registry import Registry
from sayagain import server, shift
from sayagain.amd_client import Config, TokenFactory
from sayagain.extract import RUNGS, TokenFactoryExtractor
from scripts import export_run

RUN, RUNG = shift.DEFAULT_RUN, shift.DEFAULT_RUNG


@pytest.fixture(scope="module")
def events():
    return list(shift.shift_events(RUN, RUNG))


def judged(evs):
    return {e["card"]["exchange_id"]: e["card"] for e in evs if e["type"] == "judged"}


@pytest.mark.parametrize("rung", ["regex", "Gemma-4-31B-IT", "DeepSeek-V4-Flash", "Qwen3.8-27B", "MiniCPM5-2B"])
def test_rejudged_labels_equal_the_stored_run(rung):
    cards = judged(shift.shift_events(RUN, rung))
    assert len(cards) == 145
    assert all(c["matches_stored"] for c in cards.values()), [k for k, c in cards.items() if not c["matches_stored"]]


def test_hero_card_marks_the_differing_span(events):
    c = judged(events)["H-01"]
    assert c["verdict"]["label"] == "MISMATCH · tag: XV-203 ≠ XV-208"
    assert c["live"] and c["strip"] == {"instruction": "H-01-0", "readback": "H-01-1"}
    rb = next(t for t in c["turns"] if t["id"] == "H-01-1")
    tag = next(s for s in rb["spans"] if s["field"] == "tag")
    assert rb["text"][tag["start"] : tag["end"]] == "two-oh-eight"
    assert rb["receipt"]["on_amd"] is True and rb["receipt"]["request_id"] and rb["receipt"]["receipt_found"]
    assert c["provenance"]["speaker"] == "text·scripted"


def test_refusal_abstains_and_the_judge_baseline_passed_it(events):
    c = judged(events)["R-01"]
    assert c["verdict"]["label"] == "ABSTAIN · unknown tag" and c["live"]
    assert c["baseline"]["verdict"] == "VERIFIED"


def test_end_of_shift_list(events):
    eos = events[-1]
    assert eos["type"] == "end_of_shift" and eos["cards"] == 145
    cards = judged(events)
    assert len(eos["spot_checks"]) == shift.SPOT_CHECKS
    assert all(cards[i]["verdict"]["verdict"] == "VERIFIED" for i in eos["spot_checks"])
    assert eos["caught_on_channel"] and all(cards[i]["caught"] for i in eos["caught_on_channel"])
    assert eos == list(shift.shift_events(RUN, RUNG))[-1]  # deterministic sample


def test_first_pass_lists_not_audited_never_verifies_them():
    evs = list(shift.shift_events("ladder-r1", RUNG))
    cards, eos = judged(evs), evs[-1]
    assert eos["not_audited"]
    for i in eos["not_audited"]:  # a judged pair with an unreadable turn is never VERIFIED
        c = cards[i]
        pair = [t for t in c["turns"] if c["strip"] and t["id"] in c["strip"].values()]
        if any(t["not_audited"] for t in pair):
            assert c["verdict"]["verdict"] == "NOT_AUDITED"


def test_clock_is_monotonic_and_resume_keeps_it(events):
    ats = [e["at"] for e in events]
    assert ats == sorted(ats)
    resumed = list(shift.shift_events(RUN, RUNG, start=50))
    first = next(e for e in resumed if e["type"] == "judged")
    assert first["card"]["exchange_id"] == shift.run_cards(RUN, RUNG)[50]["exchange_id"]
    orig = next(e for e in events if e["type"] == "judged" and e["card"]["exchange_id"] == first["card"]["exchange_id"])
    assert first["at"] == orig["at"] and resumed[-1] == events[-1]


def test_dev_run_replays_only_its_cards():
    evs = list(shift.shift_events("dev-r1", "regex"))
    assert evs[-1]["cards"] == 40 and evs[-1]["not_audited"] == []


def test_unknown_transcript_is_not_audited():
    ext = shift.stored_extractor(RUN, RUNG)("a sentence no run ever saw")
    assert ext.obj is None and ext.not_audited


# ------------------------------------------------------------------------------------------- server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("SAYAGAIN_LIVE_EXTRACT", raising=False)
    return TestClient(server.create_app(var=tmp_path))


def test_config_and_shift(client):
    cfg = client.get("/config.json").json()
    assert cfg["mode"] == "live" and cfg["live_extract"] is False
    assert any(r["run"] == RUN and RUNG in r["rungs"] for r in cfg["runs"])
    meta = client.get("/api/shift", params={"run": RUN, "rung": RUNG}).json()
    assert meta["stages"]["extract"]["on_amd"] is True and meta["stages"]["asr"]["on_amd"] is None
    assert client.get("/api/shift", params={"run": "nope", "rung": RUNG}).status_code == 404
    assert client.get("/api/shift", params={"run": "../../x", "rung": RUNG}).status_code == 404
    assert client.get("/queue").status_code == 200 and client.get("/app.js").status_code == 200


def test_stream_is_server_sent_events(client):
    with client.stream("GET", "/api/stream", params={"run": RUN, "rung": RUNG, "speed": 0, "start": 140}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        evs = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert evs[-1]["type"] == "closed" and evs[-2]["type"] == "end_of_shift"
    assert sum(e["type"] == "judged" for e in evs) == 5
    assert client.get("/api/stream", params={"extract": "live"}).status_code == 403


def test_review_needs_a_reason_to_dismiss(client):
    base = {"run": RUN, "rung": RUNG, "exchange_id": "H-01"}
    assert client.post("/api/review", json={**base, "decision": "dismissed"}).status_code == 422
    assert client.post("/api/review", json={**base, "decision": "dismissed", "reason": "  "}).status_code == 422
    assert client.post("/api/review", json={**base, "decision": "maybe"}).status_code == 422
    assert client.post("/api/review", json={**base, "exchange_id": "X-9", "decision": "confirmed"}).status_code == 404
    r = client.post("/api/review", json={**base, "decision": "confirmed"})
    assert r.status_code == 201 and r.json()["decision"] == "confirmed"
    rows = client.get("/api/reviews", params={"run": RUN, "rung": RUNG}).json()
    assert [x["exchange_id"] for x in rows] == ["H-01"]


def test_try_the_refused_one(client):
    j = client.post("/api/try-refused", json={"run": RUN, "rung": RUNG}).json()
    assert j["card"]["verdict"]["label"] == "ABSTAIN · unknown tag" and j["extract"] == "stored"
    assert j["card"]["referee_ms"] >= 0
    j0 = client.post("/api/try-refused", json={"run": RUN, "rung": "regex"}).json()
    assert j0["card"]["verdict"]["label"] == "ABSTAIN · unknown tag"
    assert client.post("/api/try-refused", json={"extract": "live"}).status_code == 403


def test_live_extraction_receipt_is_marked_live():
    body = {"commands": [{"action_span": "Close", "tag_span": "valve two-oh-five", "line_span": "line Charlie",
                          "state_span": None}], "kind": "instruction", "retracted_spans": []}  # fmt: skip

    def handler(req):
        usage = {"prompt_tokens": 5, "completion_tokens": 5, "cost": 1e-6}
        msg = {"message": {"content": json.dumps(body)}}
        served = "self-dploy/Gemma-4-31B-IT"
        return httpx.Response(200, json={"id": "req-1", "model": served, "choices": [msg], "usage": usage})

    tf = TokenFactory(Config("k", "https://developer.amd.com.cn/radeon/api/v1"), transport=httpx.MockTransport(handler))
    ext = TokenFactoryExtractor(tf, RUNGS[RUNG])
    out = ext("Close valve two-oh-five, line Charlie.")
    out.receipt = {**out.receipt, "live": True}
    view = shift._receipt_view(out, RUNG)
    assert view["live"] and view["on_amd"] is True and view["receipt_found"] is False
    card = next(c for c in shift.load_cards() if c["exchange_id"] == "R-01")
    rec = shift.judge_card(card, lambda t: out if t.startswith("Close") else ext(t), Registry.load(), RUNG)
    assert rec["verdict"]["verdict"] == "ABSTAIN"


# ------------------------------------------------------------------------------------------- export


def test_export_writes_a_self_contained_replay(tmp_path):
    out = tmp_path / "replay"
    rep = export_run.export([(RUN, RUNG), (RUN, "regex")], out)
    assert rep[f"{RUN}/{RUNG}"]["cards"] == 145
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["mode"] == "replay" and cfg["default"] == {"run": RUN, "rung": RUNG}
    for name in ("index.html", "app.js", "style.css", "favicon.svg", "fonts/LICENSE-OFL.txt"):
        assert (out / name).exists()
    data = json.loads((out / "data" / f"{RUN}--{RUNG}.json").read_text())
    assert data["meta"]["run"] == RUN and data["events"][-1]["type"] == "end_of_shift"
    blob = (out / "data" / f"{RUN}--{RUNG}.json").read_text()
    assert "/Users/" not in blob and "AMD_TF_API_KEY" not in blob and "Bearer" not in blob


def test_review_limits(client):
    base = {"run": RUN, "rung": RUNG, "decision": "dismissed"}
    assert client.post("/api/review", json={**base, "exchange_id": "H-01", "reason": "x" * 501}).status_code == 422
    dev = {"run": "dev-r1", "rung": "regex", "decision": "confirmed"}
    assert client.post("/api/review", json={**dev, "exchange_id": "H-01"}).status_code == 404  # H-01 is not in dev-r1


def test_refused_only_where_the_run_has_it(client):
    assert client.post("/api/try-refused", json={"run": "dev-r1", "rung": "regex"}).status_code == 404
    assert shift.run_meta("dev-r1", "regex")["refused"] is None
    assert shift.run_meta(RUN, RUNG)["refused"] == {"exchange_id": "R-01", "tag": "XV-205"}


def test_meta_numbers_come_from_the_run():
    m = shift.run_meta(RUN, RUNG)
    s = json.loads((shift.RESULTS / f"{RUN}.json").read_text())["rungs"][RUNG]["rows"]["held_out"]
    assert m["held_out"]["false_verified"] == {"k": s["false_verified"]["k"], "n": s["false_verified"]["n"]}
    assert m["held_out"]["errors_flagged"] == {"k": s["errors_flagged"]["k"], "n": s["errors_flagged"]["n"]}
    ex = m["stages"]["extract"]
    recs = shift.run_receipts(RUN, RUNG)
    assert ex["calls"] == len(recs) and ex["hosts"] == sorted({r["host"] for r in recs})
    assert ex["where"] == "AMD-hosted API" and ex["latency_s_p50"] > 0
    dev = shift.run_meta("dev-r1", "regex")["held_out"]
    assert dev["errors_flagged"] is None  # no held-out row in a dev run: nothing to show, never 0/0


def test_judge_baseline_names_its_model():
    assert {b["model"] for b in shift.judge_baseline().values()} == {"DeepSeek-V4.1-Flash"}
