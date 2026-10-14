"""A shift replayed from a stored run: each card's turns + the stored extraction -> the full referee -> one card record.

    meta = run_meta("ladder-r1-reaudit", "Gemma-4-31B-IT")
    for ev in shift_events(meta["run"], meta["rung"]):   # turn / judged / end_of_shift events, in replay order
        ...

The queue UI (sayagain/server.py, live over SSE) and the static replay (scripts/export_run.py) both read these events,
so the two surfaces show the same thing. Rules:
  * the referee really runs here (referee.audit) over the model's stored output; nothing is read from the stored
    verdicts except to check that this re-run gives the same label (`stored_label`);
  * every number a card shows comes from a receipt (the extraction call's request id, latency, cost, host) or is
    measured while the record is built (`referee_ms`, labelled with where it was measured);
  * provenance is `text·scripted` for every card here: the transcript of a turn is the card's script, there is no
    audio and no speech-to-text — the card says so instead of drawing a waveform.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path

from referee.audit import _SEVERITY, LIVE, SPOT_CHECKS_PER_SHIFT, audit, judge
from referee.pairing import Exchange
from referee.registry import Registry
from referee.turns import Turn, TurnAnalysis, analyze
from referee.verdict import NOT_AUDITED, VERIFIED
from sayagain.extract import Extraction

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results" / "text"
RECEIPTS = ROOT / "bench" / "results" / "receipts"
CARDS_PATH = ROOT / "data" / "cards.jsonl"
JUDGE_RUN = "llm-judge-r1-reaudit"  # the LLM-as-judge baseline shown beside each card (no registry)
DEFAULT_RUN, DEFAULT_RUNG = "ladder-r1-reaudit", "Gemma-4-31B-IT"
CHANNEL = "CH-01"  # one channel: the cards are replayed one after another on a single radio channel
TURN_STEP_S, TURN_LEN_S, CARD_GAP_S = 3.5, 2.5, 4.0  # replay clock: turn timing as in scripts/eval.py
SPOT_CHECKS, SEED = SPOT_CHECKS_PER_SHIFT, 1018  # same sample size + seed as referee.audit
PROVENANCE = "text·scripted"
REFUSED_ID = "R-01"  # the guaranteed refusal: its tag is not in the registry
ROLE = {"operator": "operator", "tech": "field"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def load_cards(path: Path = CARDS_PATH) -> list[dict]:
    return read_jsonl(path)


def row_of(card: dict) -> str:
    if card["set"] == "C":
        return "chatter"
    if card["set"] == "R":
        return "refusal"
    return "dev" if card["dev_slice"] else "held_out"


def script_turns(card: dict) -> list[tuple[str, float, float, str, str]]:
    """(turn id, start s, end s, role, text): the card's script as turns, pauses kept (as in scripts/eval.py)."""
    t, out = 0.0, []
    for k, turn in enumerate(card["turns"]):
        if turn["role"] == "pause":
            t += turn["seconds"]
            continue
        out.append((f"{card['exchange_id']}-{k}", t, t + TURN_LEN_S, turn["role"], turn["text"]))
        t += TURN_STEP_S
    return out


# ------------------------------------------------------------------------------------------------- runs on disk


def list_runs() -> list[dict]:
    """Every stored span-extraction run with its rungs (the grounding ablation and the judge baseline are not
    replayable as a queue: the first uses a different analyzer, the second has no spans)."""
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rungs = [
            k
            for k, v in (d.get("rungs") or {}).items()
            if v.get("mode") == "spans" and v.get("extractions") and (ROOT / v["extractions"]).exists()
        ]
        if rungs:
            order = d.get("ladder_order") or rungs
            out.append({"run": d["run"], "rungs": [r for r in order if r in rungs], "n_cards": d.get("n_cards")})
    return out


def _summary(run: str) -> dict:
    p = RESULTS / f"{run}.json"
    if not p.exists() or p.parent != RESULTS:
        raise KeyError(run)
    return json.loads(p.read_text(encoding="utf-8"))


def _rung_info(run: str, rung: str) -> dict:
    info = (_summary(run).get("rungs") or {}).get(rung)
    if not info or info.get("mode") != "spans":
        raise KeyError(f"{run}/{rung}")
    return info


@lru_cache(maxsize=16)
def receipts_by_request(rung: str) -> dict[str, dict]:
    """Every extraction receipt of this rung (first passes and re-audits), by request id."""
    out: dict[str, dict] = {}
    for p in sorted(RECEIPTS.glob(f"text-*-{rung}.jsonl")):
        for r in read_jsonl(p):
            if r.get("request_id") and r.get("ok"):
                out[r["request_id"]] = r
    return out


def stored_extractor(run: str, rung: str) -> Callable[[str], Extraction]:
    rows = read_jsonl(ROOT / _rung_info(run, rung)["extractions"])
    by_text = {r["text"]: r for r in rows}

    def call(text: str) -> Extraction:
        r = by_text.get(text)
        if r is None:  # a transcript this run never saw: not audited, never a verdict
            return Extraction(text, None, "not in this run", None, 0)
        keep = {k: r.get(k) for k in ("request_id", "latency_s", "cost_usd", "attempts")}
        return Extraction(text, r["obj"], r["not_audited"], keep if r.get("request_id") else None, r.get("attempts", 0))

    return call


def stored_labels(run: str, rung: str) -> dict[str, str | None]:
    return {c["exchange_id"]: c["label"] for c in read_jsonl(ROOT / _rung_info(run, rung)["cards"])}


def run_cards(run: str, rung: str) -> list[dict]:
    """The cards this run judged (a dev-slice run holds 40 of the 145), in card order."""
    ids = set(stored_labels(run, rung))
    return [c for c in load_cards() if c["exchange_id"] in ids]


def judge_baseline() -> dict[str, dict]:
    p = RESULTS / JUDGE_RUN
    rows = []
    for f in sorted(p.glob("*.judge.cards.jsonl")):
        rows += read_jsonl(f)
    try:
        model = _summary(JUDGE_RUN)["results"]["llm_judge"]["rung"]["model"]
    except (KeyError, TypeError):
        model = None
    return {r["exchange_id"]: {"verdict": r["verdict"], "label": r["label"], "model": model} for r in rows}


# ------------------------------------------------------------------------------------------------- one card


def _receipt_view(ext: Extraction, rung: str) -> dict | None:
    """The receipt behind one extraction: what the card prints (request id, model served, host, latency, cost)."""
    if ext.receipt is None:
        return None
    rid = ext.receipt.get("request_id")
    full = receipts_by_request(rung).get(rid) if rid else None
    src = full or ext.receipt
    view = {
        "request_id": rid,
        "model_served": src.get("model_served"),
        "host": src.get("host"),
        "on_amd": src.get("on_amd"),
        "latency_s": src.get("latency_s"),
        "cost_usd": src.get("cost_usd"),
        "started_at": src.get("started_at"),
        "run": src.get("run"),
        "receipt_found": full is not None,
    }
    if "live" in ext.receipt:
        view["live"] = True
    return view


def _turn_view(a: TurnAnalysis, role: str, at: float, ext: Extraction, rung: str) -> dict:
    spans = []
    for k, cmd in enumerate(a.spans):
        for f, sp in cmd.items():
            if sp is not None:
                spans.append({"cmd": k, "field": f.removesuffix("_span"), "start": sp[0], "end": sp[1]})
    return {
        "id": a.id,
        "role": ROLE.get(role, role),
        "text": a.turn.transcript,
        "at": round(at, 2),
        "status": a.status,
        "reason": a.reason,
        "detail": a.detail,
        "spans": spans,
        "retracted": [list(s) for s in a.retracted],
        "commands": [
            {"action": c.action, "tag": c.tag, "line": c.line, "state": c.state, "tag_status": c.tag_status}
            for c in a.commands
        ],
        "extraction": ext.obj,
        "not_audited": ext.not_audited,
        "receipt": _receipt_view(ext, rung),
    }


def _standing_readback(j) -> TurnAnalysis | None:
    """Which read-back produced the verdict that stands (audit keeps the worst of several)."""
    ex = j.exchange
    best, best_v = ex.readback, None
    for rb in [ex.readback, *ex.more_readbacks]:
        if rb is None:
            continue
        v = judge(Exchange(ex.channel, ex.instruction, rb))
        if best_v is None or _SEVERITY[v.verdict] > _SEVERITY[best_v]:
            best, best_v = rb, v.verdict
    return best


def judge_card(card: dict, extract: Callable[[str], Extraction], registry: Registry, rung: str) -> dict:
    """One card through extraction + the full referee -> the record a queue card renders."""
    turns, exts, roles = [], [], {}
    for tid, t0, t1, role, text in script_turns(card):
        e = extract(text)
        exts.append(e)
        roles[tid] = (role, t0)
        turns.append(Turn(tid, CHANNEL, t0, t1, text, e.obj, not_audited=e.not_audited))
    t = time.perf_counter()
    rep = audit(turns, registry, spot_checks=0)
    referee_ms = (time.perf_counter() - t) * 1000.0
    analyses = {a.id: a for a in (analyze(tu, registry) for tu in turns)}
    ext_by_id = {tu.id: e for tu, e in zip(turns, exts, strict=True)}
    judged = sorted(rep.judged, key=lambda j: -_SEVERITY[j.verdict.verdict])
    top = judged[0] if judged else None
    strip = None
    if top is not None:
        rb = _standing_readback(top)
        strip = {"instruction": top.exchange.instruction.id, "readback": rb.id if rb is not None else None}
    caught = [
        {"label": v.label, "instruction": i.id, "readback": r.id}
        for j in rep.judged
        for v, (i, r) in zip(j.caught, j.exchange.caught, strict=True)
    ]
    exp = card.get("expected") or {}
    return {
        "exchange_id": card["exchange_id"],
        "set": card["set"],
        "row": row_of(card),
        "lang": card["lang"],
        "error_type": card["error_type"],
        "expected": exp.get("label"),
        "verdict": top.verdict.to_dict() if top else None,
        "trace": top.verdict.trace if top else [],
        "labels": [j.verdict.label for j in rep.judged],
        "live": bool(top and top.verdict.verdict in LIVE),
        "caught": caught,
        "strip": strip,
        "turns": [_turn_view(analyses[tid], roles[tid][0], roles[tid][1], ext_by_id[tid], rung) for tid in roles],
        "not_audited_turns": sum(e.not_audited is not None for e in exts),
        "orphans": [o.id for o in rep.orphans],
        "referee_ms": round(referee_ms, 3),
        "provenance": {"speaker": PROVENANCE, "radio": "none (no audio)"},
    }


# ------------------------------------------------------------------------------------------------- the shift


def _pct(values: list[float], q: float) -> float | None:
    """Linear interpolation, the same percentile as scripts/eval.py."""
    if not values:
        return None
    v = sorted(values)
    i = (len(v) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return round(v[lo] + (v[hi] - v[lo]) * (i - lo), 4)


def run_receipts(run: str, rung: str) -> list[dict]:
    """The receipts behind this run's stored extractions (re-audits reuse first-pass calls), one per distinct call."""
    rows = read_jsonl(ROOT / _rung_info(run, rung)["extractions"])
    by_id = receipts_by_request(rung)
    ids = sorted({r["request_id"] for r in rows if r.get("request_id")})
    return [by_id[i] for i in ids if i in by_id]


def rung_stage(run: str, rung: str) -> dict:
    """Where each stage ran for this run, from its own receipts (never from a fixed string)."""
    if rung == "regex":
        extract = {"name": "regex rung 0 (no model, no call)", "where": "local", "on_amd": None, "hosts": []}
    else:
        recs = run_receipts(run, rung)
        hosts = sorted({r.get("host") for r in recs if r.get("host")})
        served = sorted({r.get("model_served") for r in recs if r.get("model_served")})
        on_amd = sorted({bool(r.get("on_amd")) for r in recs})
        lat = [float(r["latency_s"]) for r in recs if isinstance(r.get("latency_s"), int | float)]
        reaudit_of = _summary(run).get("reaudit_of")
        # a first pass owns all its calls (older receipts carry no `run` field); a re-audit reuses the rest
        reaudit_of = _summary(run).get("reaudit_of")
        # a first pass owns all its calls (older receipts carry no `run` field); a re-audit reuses the rest
        own = sum(1 for r in recs if r.get("run") == run) if reaudit_of else len(recs) if reaudit_of else len(recs)
        extract = {
            "name": rung,
            "model_served": served,
            # on_amd = the API host is AMD-operated (amd.com / amd.com.cn): where the API is hosted, never a GPU claim
            "where": "AMD-hosted API" if on_amd == [True] else "API host not AMD-operated / mixed",
            "on_amd": True if on_amd == [True] else None,
            "hosts": hosts,
            "calls": len(recs),
            # behind this run's outputs: its own calls + calls reused from the pass it re-audits (bench result files
            # count only the run's own calls)
            "calls_own": own,
            "calls_reused": len(recs) - own,
            "reused_from": reaudit_of,
            "latency_s_p50": _pct(lat, 0.5),
            "latency_s_p95": _pct(lat, 0.95),
        }
    return {
        "asr": {"name": "none: the transcript is the card script", "where": "n/a", "on_amd": None},
        "extract": extract,
        "referee": {"name": "plain Python referee (referee/), no model", "where": "local", "on_amd": None},
    }


def run_meta(run: str = DEFAULT_RUN, rung: str = DEFAULT_RUNG) -> dict:
    s = _summary(run)
    info = _rung_info(run, rung)
    rows = info.get("rows") or {}
    held = rows.get("held_out") or {}

    def kn(m: str) -> dict | None:
        w = held.get(m)
        return {"k": w["k"], "n": w["n"]} if isinstance(w, dict) and "k" in w and w.get("n") else None

    refused = next((c for c in run_cards(run, rung) if c["exchange_id"] == REFUSED_ID), None)

    return {
        "run": run,
        "rung": rung,
        "result_file": f"bench/results/text/{run}.json",
        "provenance": s.get("provenance", PROVENANCE),
        "provenance_note": s.get("provenance_note"),
        "finished_at": s.get("finished_at"),
        "git_head": s.get("git_head"),
        "n_cards": s.get("n_cards"),
        "channel": CHANNEL,
        "stages": rung_stage(run, rung),
        "n_cards_run": len(run_cards(run, rung)),
        "reaudit_of": s.get("reaudit_of"),
        "subset": s.get("subset"),
        "held_out": {
            "n": held.get("n_cards"),
            "false_verified": kn("false_verified"),
            "errors_flagged": kn("errors_flagged"),
            "false_alarm_load": kn("false_alarm_load"),
            "not_audited": kn("not_audited"),
        },
        "cards_per_shift_assumption": held.get("cards_per_shift_assumption"),
        "shift_assumption": s.get("shift_assumption"),
        "refused": {"exchange_id": REFUSED_ID, "tag": refused["instruction"][0]["tag"]} if refused else None,
        "meets_rule": info.get("meets_rule"),
        "rule": s.get("rule"),
        "judge_run": f"bench/results/text/{JUDGE_RUN}.json",
        "replay_clock": {"turn_step_s": TURN_STEP_S, "card_gap_s": CARD_GAP_S},
        "spot_checks_per_shift": SPOT_CHECKS,
        "referee_measured_on": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }


def shift_events(
    run: str = DEFAULT_RUN,
    rung: str = DEFAULT_RUNG,
    *,
    extract: Callable[[str], Extraction] | None = None,
    cards: list[dict] | None = None,
    check_stored: bool = True,
    start: int = 0,
) -> Iterator[dict]:
    """The replay, in order: per card one `turn` event per turn, then `judged`; at the end `end_of_shift`.
    `at` is the replay clock in seconds from the start of the shift. `start` resumes at that card: earlier cards are
    judged (the end-of-shift list covers the whole shift) but not sent."""
    registry = Registry.load()
    cards = cards if cards is not None else run_cards(run, rung)
    extract = extract or stored_extractor(run, rung)
    stored = stored_labels(run, rung) if check_stored else {}
    baseline = judge_baseline()
    clock, verified, caught_ids, not_audited, records = 0.0, [], [], [], {}
    for idx, card in enumerate(cards):
        rec = judge_card(card, extract, registry, rung)
        rec["at"] = round(clock, 2)
        if check_stored:
            rec["stored_label"] = stored.get(card["exchange_id"])
            rec["matches_stored"] = rec["stored_label"] == (rec["verdict"] or {}).get("label")
        rec["baseline"] = baseline.get(card["exchange_id"])
        last = max((tv["at"] for tv in rec["turns"]), default=0.0)
        if idx >= start:
            for tv in rec["turns"]:
                yield {"type": "turn", "exchange_id": rec["exchange_id"], "at": round(clock + tv["at"], 2), "turn": tv}
            yield {"type": "judged", "at": round(clock + last + TURN_LEN_S, 2), "card": rec}
        records[rec["exchange_id"]] = rec
        v = (rec["verdict"] or {}).get("verdict")
        if v == VERIFIED:
            verified.append(rec["exchange_id"])
        if v == NOT_AUDITED or rec["not_audited_turns"]:
            not_audited.append(rec["exchange_id"])
        if rec["caught"]:
            caught_ids.append(rec["exchange_id"])
        clock += last + TURN_STEP_S + CARD_GAP_S
    spot = random.Random(SEED).sample(verified, min(SPOT_CHECKS, len(verified)))
    yield {
        "type": "end_of_shift",
        "at": round(clock, 2),
        "spot_checks": sorted(spot, key=list(records).index),
        "caught_on_channel": caught_ids,
        "not_audited": not_audited,
        "cards": len(records),
        "live_cards": sum(r["live"] for r in records.values()),
    }
