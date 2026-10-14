"""The supervisor's queue: FastAPI + server-sent events over a replayed shift.

    uv run uvicorn sayagain.server:app --port 8018          # then open http://127.0.0.1:8018/queue
    SAYAGAIN_LIVE_EXTRACT=1 uv run uvicorn sayagain.server:app --port 8018   # allow live AMD Token Factory calls

Routes:
  GET  /queue                 the page (web/index.html; the same files build the static replay, scripts/export_run.py)
  GET  /config.json           mode "live", the stored runs, whether live extraction is allowed
  GET  /api/shift             run metadata: stages from receipts, the rung's held-out numbers from its result file
  GET  /api/stream            SSE: turn / judged / end_of_shift events, replayed at `speed`x the replay clock
  POST /api/try-refused       judges the refused exchange (XV-205 is not in the registry) now
  POST /api/review            confirm / dismiss(reason) — the only write; an exchange leaves the queue only through it
  GET  /api/reviews           the supervisor's decisions

Extraction comes from the run's stored model output by default (`extract=stored`): the referee runs live, the model
call is replayed with its receipt. `extract=live` calls the AMD Token Factory for every turn (paced at the account's
20 requests per minute, capped at LIVE_BUDGET_USD per server process); any failed call is NOT_AUDITED, never a verdict.

The queue is a local tool: it answers only to a localhost Host header (a DNS-rebinding page gets 400), and the one
write and the paid live calls refuse a request a browser marks as cross-site.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from referee.registry import Registry
from sayagain import shift
from sayagain.amd_client import MissingCredentials, TokenFactory, load_config
from sayagain.extract import RUNGS, Cache, Extraction, Pacer, TokenFactoryExtractor

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
VAR = Path(os.environ.get("SAYAGAIN_VAR", ROOT / "var"))
REFUSED_ID = shift.REFUSED_ID
MAX_REASON = 500
DECISIONS = ("confirmed", "dismissed")
LIVE_BUDGET_USD = 0.05  # per server process; the account cap is $1/day
PACE_S = 3.2  # 20 requests per minute per account
MAX_SPEED = 64.0
LOCAL_HOSTS = ["127.0.0.1", "localhost", "testserver"]  # testserver = FastAPI's TestClient


def live_allowed() -> bool:
    if os.environ.get("SAYAGAIN_LIVE_EXTRACT") != "1":
        return False
    try:
        load_config()
    except MissingCredentials:
        return False
    return True


# ------------------------------------------------------------------------------------------------- reviews


class Reviews:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with closing(self._db()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, run TEXT, rung TEXT, exchange_id TEXT, "
                "decision TEXT NOT NULL, reason TEXT, at TEXT NOT NULL)"
            )

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, run: str, rung: str, exchange_id: str, decision: str, reason: str | None) -> dict:
        at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock, closing(self._db()) as db, db:
            cur = db.execute(
                "INSERT INTO reviews (run, rung, exchange_id, decision, reason, at) VALUES (?, ?, ?, ?, ?, ?)",
                (run, rung, exchange_id, decision, reason, at),
            )
            rid = cur.lastrowid
        return {"id": rid, "run": run, "rung": rung, "exchange_id": exchange_id, "decision": decision, "reason": reason,
                "at": at}  # fmt: skip

    def list(self, run: str, rung: str) -> list[dict]:
        with closing(self._db()) as db:
            rows = db.execute(
                "SELECT id, run, rung, exchange_id, decision, reason, at FROM reviews WHERE run = ? AND rung = ? "
                "ORDER BY id",
                (run, rung),
            ).fetchall()
        keys = ("id", "run", "rung", "exchange_id", "decision", "reason", "at")
        return [dict(zip(keys, r, strict=True)) for r in rows]


# ------------------------------------------------------------------------------------------------- live extraction


class LiveExtract:
    """Token Factory extraction for the live mode: one paced client per process, a hard spend cap, receipts in var/."""

    RESERVE_USD = 0.001  # held against the cap while a call is in flight; the receipt's real cost replaces it

    def __init__(self, var: Path, budget_usd: float = LIVE_BUDGET_USD):
        self.var = var
        self.budget_usd = budget_usd
        self.spent = 0.0
        self.pacer = Pacer(PACE_S)
        self._lock = threading.Lock()
        self._by_rung: dict[str, Cache] = {}

    def for_rung(self, rung_name: str) -> Cache:
        """One cached extractor per rung for the life of the process: a resumed stream never pays for a turn twice."""
        rung = RUNGS.get(rung_name)
        if rung is None or rung.model is None:
            raise HTTPException(400, f"no live extraction for rung {rung_name!r}")
        with self._lock:
            if rung.name in self._by_rung:
                return self._by_rung[rung.name]
            receipts = self.var / "receipts" / f"live-{datetime.now(UTC):%Y-%m-%d}-{rung.name}.jsonl"
            receipts.parent.mkdir(parents=True, exist_ok=True)
            client = TokenFactory(load_config(), timeout=90.0, receipts_path=receipts, thinking_off=None)
            client.context = {"run": "live", "rung": rung.name}
            ext = TokenFactoryExtractor(client, rung, pacer=self.pacer)
            self._by_rung[rung.name] = cache = _AuditedCache(self._guarded(ext))
            return cache

    def _guarded(self, ext):
        def call(text: str) -> Extraction:
            with self._lock:
                if self.spent + self.RESERVE_USD > self.budget_usd:
                    return Extraction(text, None, "live budget spent", None, 0)
                self.spent += self.RESERVE_USD
            charge = 0.0  # if the call raises, the reserve is released and nothing is charged
            try:
                out = ext(text)
                cost = (out.receipt or {}).get("cost_usd")
                # an answer without a reported cost keeps the reserve as its charge, so the cap still binds
                charge = float(cost) if isinstance(cost, int | float) else self.RESERVE_USD
            finally:
                with self._lock:
                    self.spent += charge - self.RESERVE_USD
            if out.receipt is not None:
                out.receipt = {**out.receipt, "live": True}
            return out

        return call


class _AuditedCache(Cache):
    """Keeps only answered calls: a failed call (NOT_AUDITED) is asked again next time, never replayed as a failure."""

    def __call__(self, text: str) -> Extraction:
        if text in self.store:
            self.hits += 1
            return self.store[text]
        out = self.fn(text)  # type: ignore[operator]
        if out.not_audited is None:
            self.store[text] = out
        return out


# ------------------------------------------------------------------------------------------------- app


class Review(BaseModel):
    run: str
    rung: str
    exchange_id: str
    decision: str
    reason: str | None = None


class Try(BaseModel):
    run: str = shift.DEFAULT_RUN
    rung: str = shift.DEFAULT_RUNG
    extract: str = "stored"


def _not_cross_site(request: Request) -> None:
    """The one write and the paid live calls: a page on another site cannot drive them from a browser."""
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "cross-site request refused")
    origin = request.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "cross-origin request refused")


def _check_run(run: str, rung: str) -> None:
    for r in shift.list_runs():
        if r["run"] == run and rung in r["rungs"]:
            return
    raise HTTPException(404, f"no stored run {run}/{rung}")


def create_app(var: Path = VAR) -> FastAPI:
    app = FastAPI(title="SayAgain queue", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)
    reviews = Reviews(var / "reviews.sqlite")
    live = LiveExtract(var)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/queue")

    @app.get("/queue", include_in_schema=False)
    def queue():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/config.json")
    def config():
        return {
            "mode": "live",
            "runs": shift.list_runs(),
            "default": {"run": shift.DEFAULT_RUN, "rung": shift.DEFAULT_RUNG},
            "live_extract": live_allowed(),
            "refused": REFUSED_ID,
        }

    @app.get("/api/shift")
    def get_shift(run: str = shift.DEFAULT_RUN, rung: str = shift.DEFAULT_RUNG):
        _check_run(run, rung)
        return shift.run_meta(run, rung)

    @app.get("/api/stream")
    def stream(  # sync: run discovery reads files, so FastAPI runs this in its threadpool, off the event loop
        request: Request,
        run: str = shift.DEFAULT_RUN,
        rung: str = shift.DEFAULT_RUNG,
        speed: float = Query(8.0, ge=0.0, le=MAX_SPEED),
        extract: str = "stored",
        start: int = Query(0, ge=0),
    ):
        _check_run(run, rung)
        if extract not in ("stored", "live"):
            raise HTTPException(400, "extract must be stored or live")
        if extract == "live":
            _not_cross_site(request)
            if not live_allowed():
                raise HTTPException(403, "live extraction is off (SAYAGAIN_LIVE_EXTRACT=1 and credentials needed)")
            # cards before `start` are judged again so the end-of-shift list stays whole: free when this process already
            # paid for them (the cache), paid now otherwise
            events = shift.shift_events(run, rung, extract=live.for_rung(rung), start=start, check_stored=False)
        else:
            events = shift.shift_events(run, rung, start=start)

        async def gen():
            prev_at, t0 = None, time.monotonic()
            yield ": stream open\n\n"
            async for ev in iterate_in_threadpool(events):
                ev = {**ev, "extract": extract}
                if extract == "stored" and speed > 0 and prev_at is not None:
                    await asyncio.sleep(max(0.0, (ev["at"] - prev_at) / speed))
                if extract == "live":  # wall clock since this stream opened
                    ev["at"] = round(time.monotonic() - t0, 2)
                prev_at = ev["at"]
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield 'data: {"type": "closed"}\n\n'

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.post("/api/try-refused", dependencies=[Depends(_not_cross_site)])
    def try_refused(body: Try):
        _check_run(body.run, body.rung)
        card = next((c for c in shift.run_cards(body.run, body.rung) if c["exchange_id"] == REFUSED_ID), None)
        if card is None:
            raise HTTPException(404, f"{REFUSED_ID} is not in run {body.run}")
        if body.extract == "live":
            if not live_allowed():
                raise HTTPException(403, "live extraction is off")
            fn = live.for_rung(body.rung)
        elif body.extract == "stored":
            fn = shift.stored_extractor(body.run, body.rung)
        else:
            raise HTTPException(400, "extract must be stored or live")
        rec = shift.judge_card(card, fn, Registry.load(), body.rung)
        rec["baseline"] = shift.judge_baseline().get(REFUSED_ID)
        rec["at"] = None
        return {"card": rec, "extract": body.extract, "measured_on": shift.run_meta(body.run, body.rung)[
            "referee_measured_on"]}  # fmt: skip

    @app.post("/api/review", status_code=201, dependencies=[Depends(_not_cross_site)])
    def post_review(body: Review):
        _check_run(body.run, body.rung)
        if body.decision not in DECISIONS:
            raise HTTPException(422, "decision must be confirmed or dismissed")
        reason = (body.reason or "").strip() or None
        if body.decision == "dismissed" and not reason:
            raise HTTPException(422, "a dismissal needs a reason")
        if reason and len(reason) > MAX_REASON:
            raise HTTPException(422, f"reason longer than {MAX_REASON} characters")
        if not any(c["exchange_id"] == body.exchange_id for c in shift.run_cards(body.run, body.rung)):
            raise HTTPException(404, f"no exchange {body.exchange_id} in run {body.run}")
        return reviews.add(body.run, body.rung, body.exchange_id, body.decision, reason)

    @app.get("/api/reviews")
    def get_reviews(run: str = shift.DEFAULT_RUN, rung: str = shift.DEFAULT_RUNG):
        return JSONResponse(reviews.list(run, rung), headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app


app = create_app()
