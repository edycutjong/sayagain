"""Text-row evaluation: the model ladder, the grounding ablation and the LLM-judge baseline on the scripted card words.

    uv run python scripts/eval.py --ladder [--rungs regex,MiniCPM5-2B,...] [--subset dev|all] [--budget 0.50]
    uv run python scripts/eval.py --ablate grounding --model DeepSeek-V4-Flash
    uv run python scripts/eval.py --baseline llm-judge --model DeepSeek-V4.1-Flash
    uv run python scripts/eval.py --rescore            # re-judge the stored extractions offline, compare every number

Provenance `text·scripted`: the transcript of every turn IS the card's script — no audio, no ASR. This is the ASR-free
ceiling of each rung, reported as its own row; it is never mixed with, or presented as, an audio row. The
pre-registered headline (bench/PROTOCOL.md §5) is the 0 dB audio row; it needs the human recordings.

Rows (cards from data/cards.jsonl, seed 1018):
  held_out  A2 + H minus the dev slice — the set every headline-style number is scored on
  dev       the dev slice (KT + A2-001 … A2-020): prompt tuning only, reported separately
  refusal   R-01 (XV-205 is not in the registry)
  chatter   C (no instruction: every card opened is a false card)

Writes bench/results/text/<run>.json (metrics + metadata), bench/results/text/<run>/<rung>.extractions.jsonl
(model output per distinct transcript) and <rung>.cards.jsonl (verdict per card); call receipts go to
bench/results/receipts/text-<run>-<rung>.jsonl. Spend comes from `usage.cost` and is capped per run (`--budget`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee.audit import _SEVERITY, LIVE, audit  # noqa: E402
from referee.registry import Registry  # noqa: E402
from referee.schema import CANON_PROMPT, SYSTEM_PROMPT  # noqa: E402
from referee.turns import Turn, analyze, analyze_canonical  # noqa: E402
from referee.verdict import ABSTAIN, MISMATCH, NO_READBACK, NOT_AUDITED, R_GROUNDING, VERIFIED  # noqa: E402
from sayagain.amd_client import (  # noqa: E402
    SCHEMA_IN_PROMPT,
    MissingCredentials,
    NotAuditing,
    TokenFactory,
    load_config,
)
from sayagain.extract import (  # noqa: E402
    REGEX,
    Cache,
    Extraction,
    Pacer,
    Rung,
    TokenFactoryExtractor,
    regex_extraction,
)

PROVENANCE = "text·scripted"
CARDS_PATH = ROOT / "data" / "cards.jsonl"
OUT = ROOT / "bench" / "results" / "text"
RECEIPTS = ROOT / "bench" / "results" / "receipts"
DAILY_CAP_GUARD = 0.80  # never let the account's daily spend pass this (the cap is $1/day)
UNKNOWN_COST_USD = 0.001  # charged to the run budget for a call whose cost the API did not report
PACE_S = 3.2  # 20 requests per minute per account -> one call start every 3.2 s across all threads
SHIFT_EXCHANGES = 180  # site assumption (bench/PROTOCOL.md §4): 6 channels x 30 instructions per shift
SHIFT_ERRORS = (2, 5)  # assumption: ~2-5 real errors per 180 exchanges

# The ladder (bench/PROTOCOL.md §5: at most 4 model rungs, smallest first). Parameter counts are not exposed by the
# API; they come from the model names, and a model whose name states none is placed last.
CANDIDATES: dict[str, Rung] = {
    "regex": REGEX,
    "MiniCPM5-2B": Rung("MiniCPM5-2B", "MiniCPM5-2B", None, "none", 2.0),
    "Qwen3.8-27B": Rung("Qwen3.8-27B", "Qwen3.8-27B", "low", "on", 27.0, max_tokens=6144),
    "Gemma-4-31B-IT": Rung("Gemma-4-31B-IT", "Gemma-4-31B-IT", None, "off", 31.0),
    "DeepSeek-V4-Flash": Rung("DeepSeek-V4-Flash", "DeepSeek-V4-Flash", "none", "off", None),
    "MiMo-V2.6-Flash": Rung("MiMo-V2.6-Flash", "MiMo-V2.6-Flash", "none", "off", None),
    # the LLM-judge baseline: the newest, highest-priced general model in the catalog, served as-is (no
    # reasoning parameter sent) — the baseline gets its best shot
    "DeepSeek-V4.1-Flash": Rung("DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash", None, "default", None, max_tokens=2048),
}
DEFAULT_LADDER = ("regex", "MiniCPM5-2B", "Qwen3.8-27B", "Gemma-4-31B-IT", "DeepSeek-V4-Flash")

JUDGE_PROMPT = (
    "You check radio read-backs on a process-plant channel. You get one exchange as numbered turns with the speaker "
    "(operator = control room, field = field technician). Decide whether the field technician's LAST read-back "
    "repeats the operator's instruction in force correctly, field by field.\n"
    "action_match / tag_match / line_match: 'yes' if the read-back states the same value as the instruction, 'no' "
    "if it states a different value or leaves out one the instruction gave, 'n/a' if the instruction gives none. "
    "Compare every command when there are several.\n"
    "instruction_present: is there an operator order to act on equipment? readback_present: did the field "
    "technician read it back?\n"
    'Return only one JSON object, no prose: {"instruction_present": true|false, "readback_present": true|false, '
    '"action_match": "yes"|"no"|"n/a", "tag_match": "yes"|"no"|"n/a", "line_match": "yes"|"no"|"n/a"}'
)
SPEAKER = {"operator": "operator", "tech": "field"}
JUDGE_KEYS = ("instruction_present", "readback_present", "action_match", "tag_match", "line_match")


# ------------------------------------------------------------------------------------------------------- cards


def load_cards(path: Path = CARDS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_of(card: dict) -> str:
    if card["set"] == "C":
        return "chatter"
    if card["set"] == "R":
        return "refusal"
    if card["dev_slice"]:
        return "dev"
    return "held_out"


def script_turns(card: dict) -> list[tuple[str, float, float, str, str]]:
    """(turn id, start s, end s, role, text) — the card's script as turns, pauses kept (same timing as the tests)."""
    t, out = 0.0, []
    for k, turn in enumerate(card["turns"]):
        if turn["role"] == "pause":
            t += turn["seconds"]
            continue
        out.append((f"{card['exchange_id']}-{k}", t, t + 2.5, turn["role"], turn["text"]))
        t += 3.5
    return out


def judge_card(card: dict, extract: Callable[[str], Extraction], registry: Registry, analyzer=analyze) -> dict:
    """One card through extraction + the full referee. Returns the per-card record the metrics are computed from."""
    turns, exts = [], []
    for tid, t0, t1, _role, text in script_turns(card):
        e = extract(text)
        exts.append(e)
        turns.append(Turn(tid, card["exchange_id"], t0, t1, text, e.obj, not_audited=e.not_audited))
    rep = audit(turns, registry, analyzer=analyzer, spot_checks=0)
    judged = sorted(rep.judged, key=lambda j: -_SEVERITY[j.verdict.verdict])
    analyses = [analyzer(t, registry) for t in turns]
    return {
        "exchange_id": card["exchange_id"],
        "set": card["set"],
        "row": row_of(card),
        "lang": card["lang"],
        "error_type": card["error_type"],
        "expected": (card["expected"] or {}).get("label"),
        "expected_verdict": (card["expected"] or {}).get("verdict"),
        "labels": [j.verdict.label for j in rep.judged],
        "verdict": judged[0].verdict.verdict if judged else None,
        "label": judged[0].verdict.label if judged else None,
        "live_cards": sum(j.live for j in rep.judged),
        "caught": any(j.caught for j in rep.judged),
        "trace": judged[0].verdict.trace if judged else [],
        "turns": [
            {
                "id": a.turn.id,
                "status": a.status,
                "reason": a.reason,
                "commands": len(a.commands),
                "had_commands": isinstance(e.obj, dict)
                and isinstance(e.obj.get("commands"), list)
                and bool(e.obj["commands"]),
            }
            for a, e in zip(analyses, exts, strict=True)
        ],
        "not_audited_turns": sum(e.not_audited is not None for e in exts),
        "orphans": len(rep.orphans),
    }


# ----------------------------------------------------------------------------------------------------- metrics


def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": round(p, 4), "ci95": [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]}


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return round(s[lo] + (s[hi] - s[lo]) * (i - lo), 4)


def row_metrics(cards: list[dict], *, model_rung: bool) -> dict:
    errors = [c for c in cards if c["expected_verdict"] in (MISMATCH, NO_READBACK)]
    correct = [c for c in cards if c["expected_verdict"] == VERIFIED]
    flagged = [c for c in errors if c["verdict"] in LIVE]
    mismatch_errors = [c for c in errors if c["expected_verdict"] == MISMATCH]
    false_verified = [c["exchange_id"] for c in errors if c["verdict"] == VERIFIED]
    missed_no_card = [c["exchange_id"] for c in errors if c["verdict"] is None]
    false_alarms = [c for c in correct if c["verdict"] in LIVE]
    labelled = [c for c in cards if c["expected"] is not None]
    r = len(flagged) / len(errors) if errors else None
    f = len(false_alarms) / len(correct) if correct else None
    out = {
        "n_cards": len(cards),
        "n_seeded_errors": len(errors),
        "n_correct": len(correct),
        "errors_flagged": wilson(len(flagged), len(errors)),
        "mismatch_only_recall": wilson(sum(c["verdict"] == MISMATCH for c in mismatch_errors), len(mismatch_errors)),
        "false_verified": {
            "count": len(false_verified),
            "exchanges": false_verified,
            **wilson(len(false_verified), len(errors)),
        },
        "missed_no_card": {"count": len(missed_no_card), "exchanges": missed_no_card},
        "false_alarm_load": wilson(len(false_alarms), len(correct)),
        "false_alarm_exchanges": [f"{c['exchange_id']}: {c['label']}" for c in false_alarms],
        "abstain_rate": wilson(sum(c["verdict"] == ABSTAIN for c in labelled), len(labelled)),
        "not_audited": wilson(sum(c["verdict"] == NOT_AUDITED for c in labelled), len(labelled)),
        "label_exact": wilson(sum(c["label"] == c["expected"] for c in labelled), len(labelled)),
        "label_misses": [
            f"{c['exchange_id']}: expected {c['expected']!r}, got {c['label']!r}"
            for c in labelled
            if c["label"] != c["expected"]
        ],
        "caught_on_channel_detected": wilson(
            sum(c["caught"] for c in cards if c["error_type"] == "caught_on_channel"),
            sum(c["error_type"] == "caught_on_channel" for c in cards),
        ),
    }
    if model_rung:
        turns = [t for c in cards for t in c["turns"]]
        with_cmds = [t for t in turns if t["had_commands"] or t["reason"] == R_GROUNDING]
        out["ungrounded_rate"] = wilson(sum(t["reason"] == R_GROUNDING for t in with_cmds), len(with_cmds))
    if r is not None and f is not None:
        prec = (r * 0.01) / (r * 0.01 + f * 0.99) if (r + f) > 0 else None
        out["precision_at_1pct_prevalence"] = round(prec, 4) if prec is not None else None
        tp, fp = len(flagged), len(false_alarms)
        out["precision_at_corpus_prevalence"] = round(tp / (tp + fp), 4) if tp + fp else None
        out["cards_per_shift_assumption"] = {
            f"{e}_errors_per_{SHIFT_EXCHANGES}": round(f * (SHIFT_EXCHANGES - e) + r * e, 2) for e in SHIFT_ERRORS
        }
    return out


def chatter_metrics(cards: list[dict]) -> dict:
    return {
        "n_cards": len(cards),
        "n_turns": sum(len(c["turns"]) for c in cards),
        "false_cards": sum(c["live_cards"] for c in cards),
        "false_card_exchanges": [f"{c['exchange_id']}: {c['labels']}" for c in cards if c["live_cards"]],
        "per_channel_hour": "not measurable on text rows (no audio duration); measured on the audio rows",
    }


def receipt_metrics(receipts: list[dict]) -> dict:
    final = [r for r in receipts if r.get("ok")]
    lat = [r["latency_s"] for r in final]

    def _cost(r: dict) -> float:  # older receipts carry the cost only inside `usage`
        c = r.get("cost_usd")
        if c is None and isinstance(r.get("usage"), dict):
            c = r["usage"].get("cost")
        return float(c) if isinstance(c, int | float) and not isinstance(c, bool) else 0.0

    cost = sum(_cost(r) for r in receipts)
    toks = [r.get("usage") or {} for r in final]
    rt = [r.get("reasoning_tokens") for r in final if r.get("reasoning_tokens") is not None]
    return {
        "calls": len(receipts),
        "calls_ok": len(final),
        "calls_failed": len(receipts) - len(final),
        "schema_invalid": sum(1 for r in final if (r.get("params") or {}).get("schema_valid") is False),
        "rate_limited_attempts": sum(1 for r in receipts if r.get("http_status") == 429),
        "latency_s_p50": pct(lat, 0.5),
        "latency_s_p95": pct(lat, 0.95),
        "cost_usd": round(cost, 6),
        "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in toks),
        "completion_tokens": sum(t.get("completion_tokens") or 0 for t in toks),
        "reasoning_tokens_reported": sum(rt) if rt else None,
        "model_served": sorted({r["model_served"] for r in final if r.get("model_served")}),
        "on_amd": sorted({bool(r.get("on_amd")) for r in receipts}),
        "hosts": sorted({r.get("host") for r in receipts if r.get("host")}),
    }


def summarise(records: list[dict], receipts: list[dict], *, model_rung: bool) -> dict:
    by_row: dict[str, list[dict]] = {}
    for c in records:
        by_row.setdefault(c["row"], []).append(c)
    rows = {k: row_metrics(by_row.get(k, []), model_rung=model_rung) for k in ("held_out", "dev")}
    rows["refusal"] = [
        {"exchange_id": c["exchange_id"], "expected": c["expected"], "got": c["label"]}
        for c in by_row.get("refusal", [])
    ]
    rows["chatter"] = chatter_metrics(by_row.get("chatter", []))
    return {"rows": rows, "calls": receipt_metrics(receipts) if model_rung else None}


def meets_rule(summary: dict) -> dict:
    """The pre-registered ladder rule (PROTOCOL §5), applied here to the text·scripted held-out row: 0 false-VERIFIED
    AND false-alarm load <= 5 %. Only a fully audited row can meet it: a NOT_AUDITED card (rate limit, timeout, budget
    stop) is unmeasured, not safe. `strict` also requires no seeded error that formed no card at all."""
    h = summary["rows"]["held_out"]
    fa = h["false_alarm_load"]["rate"]
    audited = h["not_audited"]["n"] > 0 and h["not_audited"]["k"] == 0
    literal = audited and h["false_verified"]["count"] == 0 and fa is not None and fa <= 0.05
    return {"literal": literal, "strict": literal and h["missed_no_card"]["count"] == 0, "fully_audited": audited}


# --------------------------------------------------------------------------------------------------- plumbing


class Budget:
    def __init__(self, limit_usd: float):
        self.limit, self.spent, self._lock = limit_usd, 0.0, threading.Lock()

    def add(self, usd: float | None) -> None:
        with self._lock:
            self.spent += usd or 0.0

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.limit


def guarded(fn: Callable[[str], Extraction], budget: Budget) -> Callable[[str], Extraction]:
    def call(text: str) -> Extraction:
        if budget.exceeded:
            return Extraction(text, None, "budget guard")
        e = fn(text)
        cost = (e.receipt or {}).get("cost_usd")
        if cost is None:  # an answered call without a cost: count a pessimistic one; a failed call (no usage) is free
            cost = UNKNOWN_COST_USD if e.not_audited is None else 0.0
        budget.add(cost)
        return e

    return call


def sha256(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def git_head() -> str | None:
    """Short HEAD, with "+dirty" when tracked files differ from it (the run's code is then not that commit)."""
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def read_jsonl(p: Path) -> list[dict]:
    return (
        [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []
    )


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def open_client(receipts_path: Path, **context) -> TokenFactory:
    c = TokenFactory(load_config(), timeout=90.0, receipts_path=receipts_path, thinking_off=None)
    c.context = dict(context)
    return c


def check_daily(budget_usd: float) -> tuple[float, dict]:
    """The account's spend today -> the budget this run may use (never past DAILY_CAP_GUARD)."""
    c = open_client(RECEIPTS / "usage-checks.jsonl")
    try:
        u, _ = c.usage()
    finally:
        c.close()
    used = u.get("daily_cost_used_usd")
    if not isinstance(used, int | float):  # unknown spend today: the guard cannot work, so nothing is spent
        return 0.0, {"daily_cost_used_usd": None, "error": "usage endpoint returned no daily spend"}
    allowed = min(budget_usd, DAILY_CAP_GUARD - float(used))
    snap = {k: u.get(k) for k in ("daily_cost_used_usd", "daily_cost_limit_usd", "rpm_limit", "daily_reset_at")}
    return allowed, snap


# ------------------------------------------------------------------------------------------------------ runs


def stored_extractor(rows: list[dict]) -> Callable[[str], Extraction]:
    by_text = {r["text"]: r for r in rows}

    def call(text: str) -> Extraction:
        r = by_text[text]
        return Extraction(text, r["obj"], r["not_audited"], None, r.get("attempts", 0))

    return call


def _reaudit(fresh: Callable[[str], Extraction], prior: list[dict]) -> Callable[[str], Extraction]:
    stored = {r["text"]: r for r in prior}

    def call(text: str) -> Extraction:
        r = stored.get(text)
        if r is not None and r["not_audited"] is None:
            keep = {k: r.get(k) for k in ("request_id", "client_id", "latency_s", "cost_usd")}
            return Extraction(text, r["obj"], None, {**keep, "reused": True}, r.get("attempts", 0))
        return fresh(text)

    return call


def run_extraction_rung(
    run: str,
    rung: Rung,
    cards: list[dict],
    registry: Registry,
    pacer: Pacer,
    budget: Budget,
    mode: str = "spans",
    prior: list[dict] | None = None,
) -> dict:
    """`prior` (re-audit): the stored extractions of an earlier run of this rung — every transcript it audited is
    reused as-is, only the NOT_AUDITED ones (server error, timeout, rate limit) are requested again."""
    tag = rung.name if mode == "spans" else f"{rung.name}.canonical"
    analyzer = analyze if mode == "spans" else analyze_canonical
    rpath = RECEIPTS / f"text-{run}-{tag}.jsonl"
    client = None
    if rung.model is None:
        fn: Callable[[str], Extraction] = regex_extraction
    else:
        if rpath.exists():
            raise SystemExit(f"{rpath.relative_to(ROOT)} exists: pick a new --run name (receipts never mix runs)")
        client = open_client(rpath, rung=tag, run=run)
        fn = guarded(TokenFactoryExtractor(client, rung, pacer=pacer, mode=mode), budget)
        if prior is not None:
            fn = _reaudit(fn, prior)
    cache = Cache(fn)
    t0 = time.time()
    try:
        if rung.model is not None:
            # extract every distinct transcript first (a thinking rung uses several workers: ~25 s per call), then
            # judge the cards from the cache; call starts stay paced across all rungs by the shared pacer
            texts = list(dict.fromkeys(t[4] for c in cards for t in script_turns(c)))
            workers = 4 if rung.thinking == "on" else 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for text, e in zip(texts, pool.map(fn, texts), strict=True):
                    cache.store[text] = e
        records = [judge_card(c, cache, registry, analyzer) for c in cards]
    finally:
        if client is not None:
            client.close()
    exts = [
        {
            "text": e.text,
            "obj": e.obj,
            "not_audited": e.not_audited,
            "attempts": e.attempts,
            "request_id": (e.receipt or {}).get("request_id"),
            "client_id": (e.receipt or {}).get("client_id"),
            "latency_s": (e.receipt or {}).get("latency_s"),
            "cost_usd": (e.receipt or {}).get("cost_usd"),
            "reused_from_first_pass": bool((e.receipt or {}).get("reused")),
        }
        for e in cache.store.values()
    ]
    write_jsonl(OUT / run / f"{tag}.extractions.jsonl", exts)
    write_jsonl(OUT / run / f"{tag}.cards.jsonl", records)
    # call metrics from every attempt on disk (429s included), so --rescore can recompute them
    summary = summarise(records, read_jsonl(rpath), model_rung=rung.model is not None)
    summary.update(
        rung=rung.describe(),
        mode=mode,
        distinct_transcripts=len(cache.store),
        cache_hits=cache.hits,
        wall_s=round(time.time() - t0, 1),
        receipts=str(rpath.relative_to(ROOT)) if rung.model is not None else None,
        extractions=str((OUT / run / f"{tag}.extractions.jsonl").relative_to(ROOT)),
        cards=str((OUT / run / f"{tag}.cards.jsonl").relative_to(ROOT)),
    )
    if prior is not None:
        summary["reaudit"] = {
            "requested_again": sum(not x["reused_from_first_pass"] for x in exts),
            "still_not_audited": sum(x["not_audited"] is not None for x in exts),
        }
    summary["meets_rule"] = meets_rule(summary)
    return summary


def parse_judge(obj: object) -> tuple[str | None, str | None]:
    """LLM-judge JSON -> (verdict or None for 'no instruction', problem)."""
    if not isinstance(obj, dict) or set(obj) != set(JUDGE_KEYS):
        return ABSTAIN, "malformed judge output"
    if not isinstance(obj["instruction_present"], bool) or not isinstance(obj["readback_present"], bool):
        return ABSTAIN, "malformed judge output"
    fields = [obj[k] for k in ("action_match", "tag_match", "line_match")]
    if any(f not in ("yes", "no", "n/a") for f in fields):
        return ABSTAIN, "malformed judge output"
    if not obj["instruction_present"]:
        return None, None
    if not obj["readback_present"]:
        return NO_READBACK, None
    return (MISMATCH if "no" in fields else VERIFIED), None


def run_llm_judge(
    run: str, rung: Rung, cards: list[dict], pacer: Pacer, budget: Budget, prior: list[dict] | None = None
) -> dict:
    """`prior` (re-audit): the card records of a first pass — only its NOT_AUDITED cards are asked again."""
    done = {r["exchange_id"]: r for r in (prior or []) if r["verdict"] != NOT_AUDITED}
    rpath = RECEIPTS / f"text-{run}-{rung.name}.judge.jsonl"
    if rpath.exists():
        raise SystemExit(f"{rpath.relative_to(ROOT)} exists: pick a new --run name (receipts never mix runs)")
    client = open_client(rpath, rung=rung.name, run=run)
    records, receipts = [], []
    t0 = time.time()
    try:
        for card in cards:
            if card["exchange_id"] in done:
                records.append({**done[card["exchange_id"]], "reused_from_first_pass": True})
                continue
            lines = [f"{i + 1}. {SPEAKER[role]}: {text}" for i, (_, _, _, role, text) in enumerate(script_turns(card))]
            msgs = [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": "\n".join(lines)}]
            obj, na, rec = None, None, None
            if budget.exceeded:
                na = "budget guard"
            else:
                for attempt in range(4):
                    pacer.wait()
                    try:
                        content, _, r = client.chat(
                            rung.model,  # type: ignore[arg-type]
                            msgs,
                            schema={},
                            response_mode="json_object",
                            max_tokens=rung.max_tokens,
                            stage="judge",
                            reasoning_effort=rung.reasoning_effort,
                        )
                        rec = r.to_dict()
                        try:
                            obj = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
                        except json.JSONDecodeError:
                            obj = {"_invalid": content[:200]}
                        break
                    except NotAuditing as e:
                        rec = e.receipt.to_dict() if e.receipt else None
                        if e.status == 429 and attempt < 3:
                            time.sleep(15.0 * (attempt + 1))
                            continue
                        na = e.reason
                        break
            if rec is not None:
                cost = rec.get("cost_usd")
                budget.add(cost if cost is not None else UNKNOWN_COST_USD)
                receipts.append(rec)
            if na is not None:
                verdict, problem = NOT_AUDITED, na
            else:
                verdict, problem = parse_judge(obj)
            records.append(
                {
                    "exchange_id": card["exchange_id"],
                    "set": card["set"],
                    "row": row_of(card),
                    "lang": card["lang"],
                    "error_type": card["error_type"],
                    "expected": (card["expected"] or {}).get("label"),
                    "expected_verdict": (card["expected"] or {}).get("verdict"),
                    "verdict": verdict,
                    "label": verdict,
                    "labels": [verdict] if verdict else [],
                    "live_cards": int(verdict in LIVE),
                    "caught": False,
                    "judge_output": obj,
                    "problem": problem,
                    "request_id": (rec or {}).get("request_id"),
                    "turns": [],
                }
            )
    finally:
        client.close()
    write_jsonl(OUT / run / f"{rung.name}.judge.cards.jsonl", records)
    # the judge returns a verdict only, so exact-label comparisons use the verdict alone
    for r in records:
        r["expected"] = r["expected_verdict"]
    summary = summarise(records, receipts, model_rung=False)
    summary["calls"] = receipt_metrics(read_jsonl(rpath))
    summary.update(
        rung=rung.describe(),
        mode="llm-judge (per-field yes/no; speaker roles given; no registry)",
        wall_s=round(time.time() - t0, 1),
        receipts=str(rpath.relative_to(ROOT)),
        cards=str((OUT / run / f"{rung.name}.judge.cards.jsonl").relative_to(ROOT)),
    )
    summary["meets_rule"] = meets_rule(summary)
    return summary


def metadata(run: str, cards: list[dict], extra: dict) -> dict:
    return {
        "run": run,
        "provenance": PROVENANCE,
        "provenance_note": "turn transcripts are the card scripts (no audio, no ASR): the ASR-free ceiling per rung",
        "headline_note": "the pre-registered headline is the 0 dB audio row (bench/PROTOCOL.md §5): not run",
        "git_head": git_head(),
        "cards_sha256": sha256(CARDS_PATH.read_bytes()),
        "span_prompt_sha256": sha256(SYSTEM_PROMPT + SCHEMA_IN_PROMPT),
        "canonical_prompt_sha256": sha256(CANON_PROMPT),
        "judge_prompt_sha256": sha256(JUDGE_PROMPT),
        "n_cards": len(cards),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "shift_assumption": f"{SHIFT_EXCHANGES} exchanges per shift, {SHIFT_ERRORS[0]}-{SHIFT_ERRORS[1]} real errors",
        **extra,
    }


def select_cards(cards: list[dict], subset: str) -> list[dict]:
    if subset == "dev":
        return [c for c in cards if c["dev_slice"]]
    return cards


def cmd_ladder(args) -> int:
    registry, cards = Registry.load(), select_cards(load_cards(), args.subset)
    names = args.rungs.split(",") if args.rungs else list(DEFAULT_LADDER)
    priors: dict[str, list[dict] | None] = {}
    first = None
    if args.reaudit:  # same rungs and cards as the first pass; only its NOT_AUDITED transcripts are asked again
        first_path = Path(args.reaudit).resolve()
        first = json.loads(first_path.read_text(encoding="utf-8"))
        if first["cards_sha256"] != sha256(CARDS_PATH.read_bytes()) or first["subset"] != args.subset:
            print("--reaudit: the first pass used other cards", file=sys.stderr)
            return 2
        names = list(first["rungs"])
        for n in names:
            ex = first["rungs"][n].get("extractions")
            priors[n] = read_jsonl(ROOT / ex) if CANDIDATES[n].model is not None and ex else None
    rungs = [CANDIDATES[n] for n in names]
    paid = any(r.model for r in rungs)
    allowed, snap = check_daily(args.budget) if paid else (0.0, {})
    if paid and allowed <= 0:
        print(f"daily spend guard: {snap}", file=sys.stderr)
        return 2
    budget, pacer = Budget(allowed), Pacer(args.pace)
    run = args.run or f"{args.subset}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    extra = {"subset": args.subset, "budget_usd": round(allowed, 4), "usage_before": snap}
    if first is not None:
        extra["reaudit_of"] = str(first_path.relative_to(ROOT))
    meta = metadata(run, cards, extra)
    with ThreadPoolExecutor(max_workers=max(1, len(rungs))) as pool:
        futs = {
            r.name: pool.submit(
                run_extraction_rung, run, r, cards, registry, pacer, budget, "spans", priors.get(r.name)
            )
            for r in rungs
        }
        results = {n: f.result() for n, f in futs.items()}
    order = sorted(rungs, key=lambda r: (r.params_b is None, r.params_b or 0.0))
    headline = next((r.name for r in order if results[r.name]["meets_rule"]["literal"]), None)
    out = {
        **meta,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spent_usd": round(budget.spent, 6),
        "ladder_order": [r.name for r in order],
        "rule": "smallest rung with 0 false-VERIFIED AND false-alarm load <= 5 % (held_out row, text·scripted)",
        "smallest_rung_meeting_rule": headline,
        "rungs": results,
    }
    out["calls_cost_usd"] = _run_cost(results)
    path = OUT / f"{run}.json"
    write_json(path, out)
    print(json.dumps({n: _brief(s) for n, s in results.items()}, indent=1))
    print(f"wrote {path.relative_to(ROOT)} · spent ${budget.spent:.4f} · smallest rung meeting rule: {headline}")
    return 0


def _brief(s: dict) -> dict:
    h = s["rows"]["held_out"]
    d = s["rows"]["dev"]
    return {
        "held_out_label_exact": h["label_exact"]["rate"],
        "held_out_false_verified": h["false_verified"]["count"],
        "held_out_fa": h["false_alarm_load"]["rate"],
        "held_out_flagged": h["errors_flagged"]["rate"],
        "dev_label_exact": d["label_exact"]["rate"],
        "dev_fa": d["false_alarm_load"]["rate"],
        "refusal": s["rows"]["refusal"],
        "chatter_false_cards": s["rows"]["chatter"]["false_cards"],
        "ungrounded": (h.get("ungrounded_rate") or {}).get("rate"),
        "p50": (s.get("calls") or {}).get("latency_s_p50"),
        "cost": (s.get("calls") or {}).get("cost_usd"),
    }


def cmd_single(args, kind: str) -> int:
    registry, cards = Registry.load(), select_cards(load_cards(), args.subset)
    rung = CANDIDATES[args.model]
    allowed, snap = check_daily(args.budget)
    if allowed <= 0:
        print(f"daily spend guard: {snap}", file=sys.stderr)
        return 2
    budget, pacer = Budget(allowed), Pacer(args.pace)
    run = args.run or f"{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    meta = metadata(run, cards, {"subset": args.subset, "budget_usd": round(allowed, 4), "usage_before": snap})
    if kind == "ablate-grounding" and args.reaudit:  # re-request only the first pass's NOT_AUDITED transcripts
        first_path = Path(args.reaudit).resolve()
        first = json.loads(first_path.read_text(encoding="utf-8"))
        if first["cards_sha256"] != meta["cards_sha256"] or first["subset"] != args.subset:
            print("--reaudit: the first pass used other cards", file=sys.stderr)
            return 2
        meta["reaudit_of"] = str(first_path.relative_to(ROOT))
        prior = read_jsonl(ROOT / first["results"]["without_grounding"]["extractions"])
        res = {
            "with_grounding": first["results"]["with_grounding"],
            "without_grounding": run_extraction_rung(
                run, rung, cards, registry, pacer, budget, "canonical", prior=prior
            ),
        }
    elif kind == "ablate-grounding":
        if args.reuse:  # "with grounding" = that model's rung in a ladder run on the same cards (no second call)
            reuse = Path(args.reuse).resolve()
            ladder = json.loads(reuse.read_text(encoding="utf-8"))
            if ladder["cards_sha256"] != meta["cards_sha256"] or ladder["subset"] != args.subset:
                print("--reuse: the ladder run used other cards", file=sys.stderr)
                return 2
            with_g = {**ladder["rungs"][rung.name], "reused_from": str(reuse.relative_to(ROOT))}
        else:
            with_g = run_extraction_rung(run, rung, cards, registry, pacer, budget, "spans")
        res = {
            "with_grounding": with_g,
            "without_grounding": run_extraction_rung(run, rung, cards, registry, pacer, budget, "canonical"),
        }
    else:
        prior = None
        if args.reaudit:
            first_path = Path(args.reaudit).resolve()
            first = json.loads(first_path.read_text(encoding="utf-8"))
            meta["reaudit_of"] = str(first_path.relative_to(ROOT))
            prior = read_jsonl(ROOT / first["results"]["llm_judge"]["cards"])
        res = {"llm_judge": run_llm_judge(run, rung, cards, pacer, budget, prior)}
    out = {
        **meta,
        "kind": kind,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spent_usd": round(budget.spent, 6),  # the budget guard's tally (placeholder for cost-less calls)
        "results": res,
        "calls_cost_usd": _run_cost(res),  # what the calls cost: sum of `usage.cost` over the receipts
    }
    path = OUT / f"{run}.json"
    write_json(path, out)
    print(json.dumps({k: _brief(v) for k, v in res.items()}, indent=1))
    print(f"wrote {path.relative_to(ROOT)} · spent ${budget.spent:.4f}")
    return 0


# ---------------------------------------------------------------------------------------------------- rescore

VOLATILE = {"wall_s", "started_at", "finished_at", "git_head", "usage_before", "budget_usd", "spent_usd"}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def rescore_summary(summary: dict, cards: list[dict], registry: Registry) -> dict:
    """Re-judge one rung's stored extractions with today's referee; returns the recomputed summary."""
    mode = summary.get("mode", "spans")
    analyzer = analyze_canonical if mode == "canonical" else analyze
    rows = read_jsonl(ROOT / summary["extractions"]) if summary.get("extractions") else []
    stored = {c["exchange_id"]: c for c in read_jsonl(ROOT / summary["cards"])}
    subset = [c for c in cards if c["exchange_id"] in stored]
    fn = regex_extraction if summary["rung"]["model"] is None else stored_extractor(rows)
    records = [judge_card(c, fn, registry, analyzer) for c in subset]
    receipts = read_jsonl(ROOT / summary["receipts"]) if summary.get("receipts") else []
    new = summarise(records, receipts, model_rung=summary["rung"]["model"] is not None)
    new["meets_rule"] = meets_rule(new)
    return new


def rescore_judge(summary: dict) -> dict:
    """Re-map the stored judge outputs to verdicts and recompute the summary."""
    records = read_jsonl(ROOT / summary["cards"])
    for r in records:
        if r["verdict"] != NOT_AUDITED:
            r["verdict"], _ = parse_judge(r["judge_output"])
        r["label"], r["labels"], r["live_cards"] = r["verdict"], [r["verdict"]] if r["verdict"] else [], 0
        r["live_cards"] = int(r["verdict"] in LIVE)
        r["expected"] = r["expected_verdict"]
    new = summarise(records, [], model_rung=False)
    new["calls"] = receipt_metrics(read_jsonl(ROOT / summary["receipts"]))
    new["meets_rule"] = meets_rule(new)
    return new


def cmd_rescore(args) -> int:
    registry, cards = Registry.load(), load_cards()
    paths = [Path(p) for p in args.files] if args.files else sorted(OUT.glob("*.json"))
    bad = 0
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        groups = doc.get("rungs") or doc.get("results") or {}
        changed = False
        for name, summary in groups.items():
            if summary.get("reused_from"):
                continue  # a copy of a ladder rung: checked where it was produced
            new = rescore_judge(summary) if name == "llm_judge" else rescore_summary(summary, cards, registry)
            keys = [k for k in ("rows", "calls", "meets_rule") if k in new]
            diff = [k for k in keys if _strip(new[k]) != _strip(summary.get(k))]
            bad += bool(diff)
            tail = f" differs: {diff}" if diff else ""
            print(f"{'DIFF' if diff else 'OK  '} {p.name} :: {name} ({', '.join(keys)}){tail}")
            if diff and args.write:  # offline re-derivation from the stored model outputs: no call is repeated
                summary.update({k: new[k] for k in keys})
                summary["rescored"] = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "git_head": git_head()}
                changed = True
        if args.write and groups and doc.get("calls_cost_usd") != _run_cost(groups):
            doc["calls_cost_usd"] = _run_cost(groups)
            changed = True
        if changed:
            if doc.get("rungs"):
                order = [n for n in doc.get("ladder_order", []) if n in groups]
                doc["smallest_rung_meeting_rule"] = next((n for n in order if groups[n]["meets_rule"]["literal"]), None)
            write_json(p, doc)
            print(f"     rewrote {p.name} from its stored extractions")
    return 1 if bad and not args.write else 0


# ----------------------------------------------------------------------------------------------------- report


def _r(w: dict | None) -> str:
    """k/n (rate %) — the percentage keeps the stored precision so it greps back to the JSON."""
    if not w or not w.get("n"):
        return "n/a"
    pct_s = f"{w['rate'] * 100:.2f}".rstrip("0").rstrip(".")
    return f"{w['k']}/{w['n']} ({pct_s} %)"


def _ci(w: dict | None) -> str:
    if not w or not w.get("ci95"):
        return ""
    lo, hi = (f"{x * 100:.2f}".rstrip("0").rstrip(".") for x in w["ci95"])
    return f" [{lo} % – {hi} %]"


def _run_cost(groups: dict) -> float:
    """What the run's own calls cost, from the receipts' `usage.cost` (`calls.cost_usd`); reused groups excluded.
    The result file's `spent_usd` is the budget guard's tally, which charges a placeholder for calls without a cost."""
    return round(
        sum((g.get("calls") or {}).get("cost_usd") or 0.0 for g in groups.values() if not g.get("reused_from")), 6
    )


def report_lines(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    row = "dev" if doc.get("subset") == "dev" else "held_out"
    rel = str(path.relative_to(ROOT))
    groups = doc.get("rungs") or doc.get("results") or {}
    out = [
        f"### {doc['run']} — `{doc['provenance']}`, row `{row}`, {doc['n_cards']} cards run, "
        f"this run's calls cost ${doc.get('calls_cost_usd', _run_cost(groups))} (sum of `usage.cost` over its "
        f"receipts; {rel})",
        "",
        *(
            [
                f"Re-audit of {doc['reaudit_of']}: only the transcripts that pass could not audit were requested "
                "again; latency and cost columns cover those calls only.",
                "",
            ]
            if doc.get("reaudit_of")
            else []
        ),
        "| Rung | Thinking | False-VERIFIED | No card | Not audited | Errors flagged [Wilson CI] | "
        "MISMATCH-only recall | "
        "False-alarm load [Wilson CI] | Label exact | Ungrounded | Refusal R-01 | Chatter false cards | "
        "Median / 95th-pct latency, s | Source |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in groups.items():
        rows = s["rows"]
        h = rows[row]
        calls = s.get("calls") or {}
        lat = f"{calls['latency_s_p50']} / {calls['latency_s_p95']}" if calls.get("latency_s_p50") is not None else "—"
        ref = "; ".join(f"{r['got']}" for r in rows["refusal"]) or "—"
        thinking = (s.get("rung") or {}).get("thinking", "")
        out.append(
            f"| {name} | {thinking} | {h['false_verified']['count']} | {h['missed_no_card']['count']} | "
            f"{h['not_audited']['k']} | "
            f"{_r(h['errors_flagged'])}{_ci(h['errors_flagged'])} | {_r(h['mismatch_only_recall'])} | "
            f"{_r(h['false_alarm_load'])}{_ci(h['false_alarm_load'])} | {_r(h['label_exact'])} | "
            f"{_r(h.get('ungrounded_rate')) if h.get('ungrounded_rate') else '—'} | {ref} | "
            f"{rows['chatter']['false_cards']} | {lat} | {rel} |"
        )
    if any(g["rows"][row]["not_audited"]["k"] for g in groups.values()):
        out.append("")
        out.append(
            "Cards in the *Not audited* column had a failed call (server error, time-out). They stay in the "
            "denominators but are never counted as flagged errors or as false alarms — conservative for errors "
            "flagged, optimistic for false-alarm load — and they keep the row from meeting the rule; a re-audit "
            "result, where one exists, is listed above."
        )
    out.append("")
    return out


def cmd_report(args) -> int:
    paths = [Path(p) for p in args.files]
    lines = [
        "# Text-row results (generated by `scripts/eval.py --report`; do not edit)",
        "",
        "Provenance **`text·scripted`**: every turn's transcript is the card's script — no audio, no speech-to-text. "
        "These rows are the ASR-free ceiling of each rung, not the pre-registered headline (the 0 dB audio row, "
        "bench/PROTOCOL.md). Row `held_out` = A2 + H minus the dev slice; `dev` = the dev slice. "
        "Every row cites its file.",
        "",
    ]
    for p in paths:
        lines += report_lines(p.resolve())
    out = OUT / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def cmd_compare(args) -> int:
    """Run-to-run reproducibility of one rung: two result files -> per-card label agreement + identical outputs."""
    a_path, b_path = (Path(f).resolve() for f in args.files)
    a, b = (json.loads(x.read_text(encoding="utf-8")) for x in (a_path, b_path))
    ra, rb = a["rungs"][args.model], b["rungs"][args.model]
    ca = {c["exchange_id"]: c for c in read_jsonl(ROOT / ra["cards"])}
    cb = {c["exchange_id"]: c for c in read_jsonl(ROOT / rb["cards"])}
    ea = {e["text"]: e["obj"] for e in read_jsonl(ROOT / ra["extractions"])}
    eb = {e["text"]: e["obj"] for e in read_jsonl(ROOT / rb["extractions"])}
    common = sorted(set(ca) & set(cb))
    label_diff = [f"{k}: {ca[k]['label']!r} vs {cb[k]['label']!r}" for k in common if ca[k]["label"] != cb[k]["label"]]
    out_diff = [t for t in ea if t in eb and ea[t] != eb[t]]
    res = {
        "kind": "reproducibility",
        "rung": args.model,
        "provenance": PROVENANCE,
        "first": str(a_path.relative_to(ROOT)),
        "second": str(b_path.relative_to(ROOT)),
        "cards_compared": len(common),
        "cards_same_label": len(common) - len(label_diff),
        "label_differences": label_diff,
        "outputs_compared": len(set(ea) & set(eb)),
        "outputs_identical": len(set(ea) & set(eb)) - len(out_diff),
        "output_differences": [{"text": t, "first": ea[t], "second": eb[t]} for t in out_diff],
        "note": "temperature 0 gives no determinism guarantee at this gateway: identical outputs are measured",
    }
    out = OUT / f"repro-{args.run or b['run']}.json"
    write_json(out, res)
    print(
        f"{res['cards_same_label']}/{res['cards_compared']} cards same label · "
        f"{res['outputs_identical']}/{res['outputs_compared']} outputs identical -> {out.relative_to(ROOT)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ladder", action="store_true")
    g.add_argument("--ablate", choices=["grounding"])
    g.add_argument("--baseline", choices=["llm-judge"])
    g.add_argument("--rescore", action="store_true")
    g.add_argument("--compare", action="store_true", help="reproducibility of --model between two result files")
    g.add_argument("--report", action="store_true", help="write bench/results/text/RESULTS.md from result files")
    ap.add_argument("--rungs", help="comma-separated rung names (default: the ladder)")
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--subset", choices=["dev", "all"], default="all")
    ap.add_argument(
        "--budget", type=float, default=0.50, help="USD this run may spend (also capped by the daily guard)"
    )
    ap.add_argument("--run", help="run name (default: subset + UTC time)")
    ap.add_argument("--pace", type=float, default=PACE_S, help="seconds between call starts (20 RPM per account)")
    ap.add_argument("--reaudit", help="--ladder: re-request only the NOT_AUDITED transcripts of this ladder result")
    ap.add_argument("--write", action="store_true", help="--rescore: rewrite results that differ (offline)")
    ap.add_argument("--reuse", help="--ablate: take the with-grounding side from this ladder result file")
    ap.add_argument("files", nargs="*", help="--rescore: result files (default: all of bench/results/text/*.json)")
    args = ap.parse_args(argv)
    try:
        if args.ladder:
            return cmd_ladder(args)
        if args.ablate:
            return cmd_single(args, "ablate-grounding")
        if args.baseline:
            return cmd_single(args, "llm-judge")
        if args.report:
            return cmd_report(args)
        if args.compare:
            return cmd_compare(args)
        return cmd_rescore(args)
    except MissingCredentials as e:
        print(f"no Token Factory key: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
