"""Regression tests from code review round 5 (architecture review of the whole referee)."""

import pytest

from referee.audit import audit
from referee.coverage import CoverageError, check
from referee.grounding import ground
from referee.registry import Registry
from referee.turns import Turn

REG = Registry.load()
INSTR = "Stop pump one-seven-four, line Charlie, correction, line Bravo."  # card A2-085
READBACK = "Roger, stopping pump one-seven-four, line Charlie."


def ex(*cmds, retracted=()):
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {
        "kind": "instruction",
        "commands": [dict(zip(keys, c, strict=True)) for c in cmds],
        "retracted_spans": list(retracted),
    }


def test_a_retracted_verb_or_tag_needs_a_replacement_of_its_own_field() -> None:
    # an extractor that takes back the whole order before the cue keeps only the line: nothing replaces verb or tag
    g = ground(INSTR, ex((None, None, None, "line Bravo"), retracted=["Stop", "pump one-seven-four", "line Charlie"]))
    with pytest.raises(CoverageError) as e:
        check(INSTR, g)
    assert e.value.reason == "retraction cue missing"
    assert "nothing replaces the retracted action" in e.value.detail


def test_the_correct_reading_of_the_same_correction_still_passes() -> None:
    g = ground(INSTR, ex(("Stop", "pump one-seven-four", None, "line Bravo"), retracted=["line Charlie"]))
    check(INSTR, g)


def test_an_over_retracted_order_still_reaches_the_supervisor() -> None:
    """Before the fix this instruction opened no pairing window: the wrong line read back (C for B) left no card at all,
    only an orphan read-back on the end-of-shift list."""
    over = ex((None, None, None, "line Bravo"), retracted=["Stop", "pump one-seven-four", "line Charlie"])
    rb = {
        "kind": "readback",
        "commands": [
            {
                "action_span": "stopping",
                "tag_span": "pump one-seven-four",
                "state_span": None,
                "line_span": "line Charlie",
            }
        ],
        "retracted_spans": [],
    }
    turns = [Turn("i", "CH-01", 0.0, 2.5, INSTR, over), Turn("r", "CH-01", 3.5, 6.0, READBACK, rb)]
    rep = audit(turns, REG, spot_checks=0)
    assert [j.verdict.verdict for j in rep.judged] == ["ABSTAIN"]
    assert rep.live_queue and not rep.orphans


def test_a_backwards_retraction_cannot_verify_a_wrong_self_correction() -> None:
    """The technician corrects two-oh-three to the WRONG two-oh-eight; an extractor that takes back the later word keeps
    the right tag. Before the fix any later span after a cue counted as the replacement, and this read-back VERIFIED."""
    instr = "Close valve two-oh-three, line Bravo."
    rb_text = "Copy, closing two-oh-three, no, two-oh-eight, correction, line Bravo."
    ins = ex(("Close", "valve two-oh-three", None, "line Bravo"))
    rb = {**ex(("closing", "two-oh-three", None, "line Bravo"), retracted=["two-oh-eight"]), "kind": "readback"}
    turns = [Turn("i", "CH-01", 0.0, 2.5, instr, ins), Turn("r", "CH-01", 3.5, 6.0, rb_text, rb)]
    rep = audit(turns, REG, spot_checks=0)
    assert [j.verdict.label for j in rep.judged] == ["ABSTAIN · retraction cue missing"]


# ---------------------------------------------------------------------------------------------------------------------
# Property: a technician's self-corrections, read by an extractor that groups and retracts the turn's real safety words
# at random. The final reading is computable: a different value replaces the earlier one only when a cue was spoken
# after the earlier mention; with no cue in between, that field is ambiguous until a cue-led value settles it.
# The referee may abstain or flag at will, but it must never VERIFY a read-back whose final reading is wrong or open.

_VALUES = {
    "action_span": [("closing", "CLOSE"), ("opening", "OPEN")],
    "tag_span": [("two-oh-three", "XV-203"), ("two-oh-eight", "XV-208"), ("valve two-oh-three", "XV-203")],
    "line_span": [("line Bravo", "B"), ("line Alpha", "A")],
}
_FIELD = {"action": "action_span", "tag": "tag_span", "line": "line_span"}


def _self_corrected_readback(rng) -> tuple[str, dict, bool]:
    words, final, cue_since, open_field = [], {}, {}, {}
    fields = list(_VALUES)

    def say(f: str) -> None:
        w, v = rng.choice(_VALUES[f])
        if f in final and v != final[f]:
            open_field[f] = not cue_since[f]
        elif f in final and cue_since[f]:
            open_field[f] = False
        final[f], cue_since[f] = v, False
        words.append(w)

    for f in fields:
        say(f)
    for _ in range(rng.choice((1, 1, 2))):
        if rng.random() < 0.8:
            words.append(rng.choice(("correction", "no", "koreksi")))
            cue_since = dict.fromkeys(cue_since, True)
        say(rng.choice(fields))
    return "Copy, " + ", ".join(words) + ".", final, any(open_field.values())


def _random_extraction(text: str, rng) -> dict:
    from referee import aliases as al

    cmds, cur, retracted = [], {}, []
    for m in al.all_mentions(text):
        f = _FIELD.get(m.field)
        if f is None:
            continue
        if rng.random() < 0.35:
            retracted.append(text[m.start : m.end])
            continue
        if f in cur or rng.random() < 0.1:
            cmds.append(cur)
            cur = {}
        cur[f] = text[m.start : m.end]
    cmds.append(cur)
    keys = ("action_span", "tag_span", "state_span", "line_span")
    return {
        "kind": "readback",
        "commands": [{k: c.get(k) for k in keys} for c in cmds if c],
        "retracted_spans": retracted,
    }


def test_no_extraction_of_a_self_corrected_readback_verifies_a_wrong_final_reading() -> None:
    import random

    rng = random.Random(1018)
    instr = "Close valve two-oh-three, line Bravo."
    ins = ex(("Close", "valve two-oh-three", None, "line Bravo"))
    right = {"action_span": "CLOSE", "tag_span": "XV-203", "line_span": "B"}
    wrong = []
    for _ in range(3000):
        text, final, still_open = _self_corrected_readback(rng)
        rb = _random_extraction(text, rng)
        turns = [Turn("i", "CH-01", 0.0, 2.5, instr, ins), Turn("r", "CH-01", 3.5, 6.0, text, rb)]
        v = audit(turns, REG, spot_checks=0).judged[0].verdict.verdict
        if v == "VERIFIED" and (still_open or final != right):
            wrong.append((text, rb))
    assert wrong == []


# ---------------------------------------------------------------------------------------------------------------------
# Fail closed beyond judge(): a crash while reading one turn or pairing the turns is NOT_AUDITED, never a lost shift.


def _boom(*_a, **_k):
    raise RuntimeError("injected")


def _hero_card() -> dict:
    from sayagain import shift

    card = next(c for c in shift.load_cards() if c["exchange_id"] == "H-01")
    ext = shift.stored_extractor(shift.DEFAULT_RUN, shift.DEFAULT_RUNG)
    return shift.judge_card(card, ext, REG, shift.DEFAULT_RUNG)


def test_a_crash_while_reading_a_turn_is_not_audited(monkeypatch) -> None:
    from referee import coverage

    assert _hero_card()["verdict"]["verdict"] == "MISMATCH"
    monkeypatch.setattr(coverage, "check", _boom)  # before the fix the RuntimeError escaped the shift
    rec = _hero_card()
    assert rec["verdict"]["verdict"] == "NOT_AUDITED" and rec["verdict"]["reason"] == "referee error"


def test_a_crash_while_pairing_is_not_audited(monkeypatch) -> None:
    from referee import audit as audit_mod

    monkeypatch.setattr(audit_mod, "pair", _boom)
    rec = _hero_card()
    assert rec["verdict"]["verdict"] == "NOT_AUDITED"
    assert "referee error (pairing)" in rec["trace"][0]


# ---------------------------------------------------------------------------------------------------------------------
# An unreadable second read-back is judged (ABSTAIN), never swallowed as the acknowledgement after a good first one.


def test_an_unreadable_second_readback_is_judged_not_taken_as_the_ack() -> None:
    from referee.regex_extractor import extract as regex_extract
    from referee.turns import analyze

    texts = (
        "Close valve two-oh-three, line Bravo.",
        "Copy, closing valve two-oh-three, line Bravo.",
        "Copy, closing valve two-oh-eight, line Bravo.",
    )
    turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
    turns[2].extraction = {"_invalid": "not JSON"}  # the model failed on the last turn
    a = analyze(turns[2], REG)
    assert a.status == "abstain" and a.readback_shaped
    assert audit(turns[:2], REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED"
    # before the fix the unreadable turn closed the exchange as an "ack" and it stood VERIFIED
    assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "ABSTAIN"


def test_control_correct_after_the_readback_closes_the_exchange_before_and_after_the_fix() -> None:
    from referee.regex_extractor import extract as regex_extract

    texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", "Correct.")
    turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
    turns[2].extraction = {"_invalid": "not JSON"}  # unreadable, but not read-back shaped: an acknowledgement
    assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED"


# ---------------------------------------------------------------------------------------------------------------------
# The referee routes an orphan read-back (no order heard) to the end-of-shift list; the queue must carry it there.


def test_an_orphan_readback_reaches_the_end_of_shift_list() -> None:
    from sayagain import shift
    from sayagain.extract import regex_extraction

    card = {
        "exchange_id": "T-ORPHAN",
        "set": "C",
        "dev_slice": False,
        "lang": "en",
        "error_type": None,
        "turns": [{"role": "tech", "text": "Copy, closing valve two-oh-three, line Bravo."}],
    }
    events = list(shift.shift_events(cards=[card], extract=regex_extraction, check_stored=False))
    assert events[-2]["card"]["verdict"] is None and events[-2]["card"]["orphans"]  # no card, but not nothing
    assert events[-1]["orphan_readbacks"] == ["T-ORPHAN"]  # before the fix the queue dropped it


# ---------------------------------------------------------------------------------------------------------------------
# The local queue: a DNS-rebinding page and a cross-site page cannot drive the one write or the paid live calls.


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from sayagain import server

    return TestClient(server.create_app(var=tmp_path), base_url="http://127.0.0.1")


def test_the_queue_answers_only_to_a_localhost_host_header(tmp_path) -> None:
    c = _client(tmp_path)
    assert c.get("/config.json").status_code == 200
    assert c.get("/config.json", headers={"host": "evil.example"}).status_code == 400


def test_a_cross_site_page_cannot_dismiss_a_card(tmp_path) -> None:
    from sayagain import shift

    c = _client(tmp_path)
    body = {"run": shift.DEFAULT_RUN, "rung": shift.DEFAULT_RUNG, "exchange_id": "H-01", "decision": "dismissed"}
    body["reason"] = "not a real error"
    assert c.post("/api/review", json=body, headers={"origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/review", json=body, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert c.post("/api/review", json=body, headers={"origin": "http://127.0.0.1"}).status_code == 201


def test_a_cross_site_page_cannot_start_paid_live_calls(tmp_path) -> None:
    r = _client(tmp_path).get("/api/stream", params={"extract": "live"}, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403 and "another origin" in r.text


# ---------------------------------------------------------------------------------------------------------------------
# A NOT_AUDITED turn shows the receipt of the call that failed (found by its client call id), not just the word.


def test_a_not_audited_turn_shows_the_receipt_of_the_call_that_failed() -> None:
    from sayagain import shift

    ext = shift.stored_extractor("ladder-r1", "Gemma-4-31B-IT")  # the as-run first pass, with its gateway failures
    rows = shift.read_jsonl(shift.ROOT / "bench/results/text/ladder-r1/Gemma-4-31B-IT.extractions.jsonl")
    failed = [r for r in rows if r["not_audited"]]
    views = [shift._receipt_view(ext(r["text"]), "Gemma-4-31B-IT") for r in failed]
    assert failed and all(v and v["ok"] is False and v["receipt_found"] and v["client_id"] for v in views)
    assert {(v["http_status"], v["error"]) for v in views} <= {(500, "server error"), (None, "timeout")}
    ok = next(r for r in rows if r["request_id"])
    assert "ok" not in shift._receipt_view(ext(ok["text"]), "Gemma-4-31B-IT")  # answered calls render as before


# ---------------------------------------------------------------------------------------------------------------------
# The benchmark fails loud: a referee crash stops the run (after the paid extractions are written), never a NOT_AUDITED.


def _eval_module():
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import eval as ev

    return ev


def test_the_benchmark_stops_on_a_referee_error_after_writing_its_extractions(monkeypatch, tmp_path) -> None:
    from referee import coverage
    from sayagain.extract import REGEX, Pacer

    ev = _eval_module()
    card = next(c for c in ev.load_cards() if c["exchange_id"] == "H-01")
    assert "referee_errors" not in ev.judge_card(card, ev.regex_extraction, REG)
    monkeypatch.setattr(coverage, "check", _boom)
    assert ev.judge_card(card, ev.regex_extraction, REG)["referee_errors"]  # before the fix: a quiet NOT_AUDITED
    monkeypatch.setattr(ev, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="referee error.*new --run name"):
        ev.run_extraction_rung("t", REGEX, [card], REG, Pacer(0.0), ev.Budget(0.0))
    assert (tmp_path / "t" / "regex.extractions.jsonl").exists() and (tmp_path / "t" / "regex.cards.jsonl").exists()


def test_a_page_on_another_local_port_cannot_drive_the_queue(tmp_path) -> None:
    from sayagain import shift

    c = _client(tmp_path)
    same_site = {"sec-fetch-site": "same-site"}  # e.g. a dev server on localhost:3000
    r = c.get("/api/stream", params={"extract": "live"}, headers=same_site)
    assert r.status_code == 403 and "another origin" in r.text
    body = {"run": shift.DEFAULT_RUN, "rung": shift.DEFAULT_RUNG, "exchange_id": "H-01", "decision": "confirmed"}
    assert c.post("/api/review", json=body, headers=same_site).status_code == 403
    assert c.post("/api/review", json=body, headers={"sec-fetch-site": "same-origin"}).status_code == 201


# ---------------------------------------------------------------------------------------------------------------------
# Loop-QA round 1 of this review: nothing the referee cannot read may vanish, and no harmless turn may raise an alarm.


def test_a_retracted_span_holding_verb_and_tag_needs_both_replaced() -> None:
    text = "Stop pump one-seven-four, correction, start, line Charlie."
    g = ground(text, ex(("start", None, None, "line Charlie"), retracted=["Stop pump one-seven-four"]))
    with pytest.raises(CoverageError) as e:  # first-field-only checking accepted this: the tag was dropped silently
        check(text, g)
    assert "nothing replaces the retracted tag" in e.value.detail


def test_an_order_split_across_commands_reaches_the_supervisor() -> None:
    instr = "Close valve two-oh-eight, line Alpha, correction, open."
    split = ex((None, "valve two-oh-eight", None, "line Alpha"), ("open", None, None, None), retracted=["Close"])
    rb_text = "Copy, closing valve two-oh-eight, line Alpha."
    rb = {**ex(("closing", "valve two-oh-eight", None, "line Alpha")), "kind": "readback"}
    rep = audit(
        [Turn("i", "CH-01", 0.0, 2.5, instr, split), Turn("r", "CH-01", 3.5, 6.0, rb_text, rb)], REG, spot_checks=0
    )
    assert [j.verdict.verdict for j in rep.judged] == ["ABSTAIN"] and not rep.orphans  # before: no card, an orphan


def test_an_unreadable_readback_with_no_order_heard_is_an_orphan_not_chatter() -> None:
    text = "Copy, closing two-oh-three, no, two-oh-eight, correction, line Bravo."
    rb = {**ex(("closing", "two-oh-three", None, "line Bravo"), retracted=["two-oh-eight"]), "kind": "readback"}
    rep = audit([Turn("r", "CH-01", 0.0, 2.5, text, rb)], REG, spot_checks=0)
    assert [o.id for o in rep.orphans] == ["r"] and not rep.other  # before: dropped as chatter, on no list


def test_an_unreadable_bare_ack_after_the_readback_raises_no_alarm() -> None:
    from referee.regex_extractor import extract as regex_extract

    for ack in ("Roger.", "Copy, thanks.", "Siap."):
        texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", ack)
        turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
        turns[2].extraction = {"_invalid": "not JSON"}  # no action, no tag: it cannot hide a wrong read-back
        assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED", ack


def test_a_crash_judging_a_caught_readback_stops_the_benchmark(monkeypatch) -> None:
    from referee import audit as audit_mod

    ev = _eval_module()
    card = next(c for c in ev.load_cards() if c["exchange_id"] == "A2-021")  # a read-back corrected on the channel
    assert ev.judge_card(card, ev.regex_extraction, REG)["caught"]
    real, wrong = audit_mod._judge, card["turns"][1]["text"]

    def crash_on_the_wrong_readback(ins, rb, **kw):
        if rb is not None and rb.turn.transcript == wrong:
            raise RuntimeError("injected")
        return real(ins, rb, **kw)

    monkeypatch.setattr(audit_mod, "_judge", crash_on_the_wrong_readback)
    rec = ev.judge_card(card, ev.regex_extraction, REG)
    assert rec["referee_errors"]  # before: only the final verdict was read, so this crash was counted silently
    with pytest.raises(SystemExit, match="referee error"):
        ev.stop_on_referee_errors([rec], "A2-021", "fix it")


def test_a_pairing_crash_makes_every_turn_its_own_not_audited_exchange(monkeypatch) -> None:
    from referee import audit as audit_mod
    from referee.regex_extractor import extract as regex_extract

    monkeypatch.setattr(audit_mod, "pair", _boom)
    texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", "Correct.")
    turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
    rep = audit(turns, REG, spot_checks=0)
    assert [j.exchange.instruction.id for j in rep.judged] == ["t0", "t1", "t2"]
    assert all(j.verdict.verdict == "NOT_AUDITED" and j.referee_errors for j in rep.judged)
    assert rep.judged[0].referee_errors is not rep.judged[1].referee_errors  # one list per exchange


def test_try_the_refused_one_refuses_another_origin(tmp_path) -> None:
    r = _client(tmp_path).post("/api/try-refused", json={}, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403


def test_a_run_with_no_receipt_is_not_labelled_a_non_amd_host(monkeypatch) -> None:
    from sayagain import shift

    monkeypatch.setattr(shift, "run_receipts", lambda run, rung: [])
    stage = shift.rung_stage("ladder-r1-reaudit", "Gemma-4-31B-IT")["extract"]
    assert stage["where"] == "no receipt found for this run" and stage["on_amd"] is None


# ---------------------------------------------------------------------------------------------------------------------
# Loop-QA round 2: an unreadable turn naming a line still reaches the supervisor; chatter naming no work stays out.


def test_an_unreadable_line_or_letter_after_the_readback_is_judged() -> None:
    from referee.regex_extractor import extract as regex_extract

    for said in ("Roger, line Charlie.", "Roger, Bravo.", "Siap, Charlie."):
        texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", said)
        turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
        turns[2].extraction = {"kind": "other", "commands": [], "retracted_spans": []}  # the model saw nothing
        assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "ABSTAIN", said  # was VERIFIED


def test_unreadable_chatter_that_names_no_work_is_not_an_orphan() -> None:
    for said in (
        "Roger, tank level two-fifty.",
        "Okay, starting my round now.",
        "Copy, I will stop by the lab.",
        "Roger, the gate is open.",
    ):
        empty = {"kind": "other", "commands": [], "retracted_spans": []}
        rep = audit([Turn("c", "CH-01", 0.0, 2.5, said, empty)], REG, spot_checks=0)
        assert not rep.orphans and not rep.judged, said


# ---------------------------------------------------------------------------------------------------------------------
# Loop-QA round 3: a NATO name is a line only if the plant has that line; a named pump is work with an unknown verb.


def test_an_unreadable_ack_with_a_nato_name_raises_no_card() -> None:
    from referee.regex_extractor import extract as regex_extract

    for said in ("Roger, Mike.", "Affirmative, Victor.", "Siap, Hotel.", "Roger, see you in November."):
        texts = ("Close valve two-oh-three, line Bravo.", "Copy, closing valve two-oh-three, line Bravo.", said)
        turns = [Turn(f"t{k}", "CH-01", 3.5 * k, 3.5 * k + 2.5, x, regex_extract(x)) for k, x in enumerate(texts)]
        turns[2].extraction = {"kind": "other", "commands": [], "retracted_spans": []}
        assert audit(turns, REG, spot_checks=0).judged[0].verdict.verdict == "VERIFIED", said


def test_an_unreadable_readback_naming_a_pump_with_an_unknown_verb_is_an_orphan() -> None:
    text = "Copy, turning off pump one-forty, line Alpha."  # "turning off" is not in the alias tables
    rb = {**ex(("turning", "pump one-forty", None, "line Alpha")), "kind": "readback"}
    rep = audit([Turn("r", "CH-01", 0.0, 2.5, text, rb)], REG, spot_checks=0)
    assert [o.id for o in rep.orphans] == ["r"]  # it vanished as chatter in the previous round


def test_the_rescore_stop_says_to_rescore_again(monkeypatch) -> None:
    import json

    from referee import coverage

    ev = _eval_module()
    summary = json.loads((ev.OUT / "ladder-r1-reaudit.json").read_text(encoding="utf-8"))["rungs"]["regex"]
    monkeypatch.setattr(coverage, "check", _boom)
    with pytest.raises(SystemExit, match="run --rescore again"):
        ev.rescore_summary(summary, ev.load_cards(), REG)
