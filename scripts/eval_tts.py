"""EXPLORATORY synthetic-speech rows (badge `tts·synthetic`) — NOT the pre-registered audio row.

    uv run python scripts/tts_cards.py                                 # 1. one ElevenLabs WAV per card
    uv run python scripts/eval_tts.py --transcribe --snr clean -6 0 6  # 2. radio path + local Whisper
    uv run python scripts/eval_tts.py --score --snr clean --rungs regex,Gemma-4-31B-IT --run tts-r1-clean
    uv run python scripts/eval.py --rescore bench/results/tts/tts-r1-clean.json   # offline re-derivation

Logged as a deviation in bench/PROTOCOL.md (2026-10-15): synthetic voices, not pre-registered, reported separately,
never as or beside the human audio row; referee, aliases and thresholds frozen at the run's HEAD.

Speech path per card and SNR condition (the same code the audio row will use):
  clean TTS take -> mix.noisy_copy (MIMII pump / valve / fan at the SNR, measured against the speech's active level;
  `clean` adds none) -> radioize.radioize (300-3400 Hz, mu-law 8 kHz, squelch tail; badge radio·simulated)
  -> ingest.cut (energy VAD on the CLEAN take, samples from the radio copy) -> asr.transcribe_turn (local
  Whisper-large-v3, auto language, vocabulary prompt + an unprompted pass for the disagreement gate; off-AMD)
  -> extraction rung (regex, or a Token Factory model) -> the referee (scripts/eval.py judge_card).

Transcripts (text only, no audio) go to bench/results/tts/transcripts/<condition>.jsonl, ASR receipts to
bench/results/receipts/tts-asr-<condition>.jsonl; result files are bench/results/tts/<run>.json in the text rows'
schema, so `scripts/eval.py --rescore` re-derives them from the stored transcripts and extractions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval as ev  # noqa: E402
import mix  # noqa: E402

from referee.registry import Registry  # noqa: E402
from sayagain.extract import Pacer  # noqa: E402

PROVENANCE = "tts·synthetic"
RADIO = "radio·simulated"
TTS = ROOT / "data" / "tts" / "clean"
OUT = ROOT / "bench" / "results" / "tts"
TRANSCRIPTS = OUT / "transcripts"
AUDIO_GATE = "disagreement"  # the local route returns token log-probs, but the floor gate needs a fitted floor: not run
# found on these rows and NOT fixed (the referee is frozen for the exploratory run): printed under "Open issues"
OPEN_ISSUES = (
    "Seeded errors that form no card: when the ASR mangles the instruction's tag ('for three Bravo' for four-three "
    "Bravo, A2-032; 'tutu, hotel' for two-two Hotel, A2-097; bench/results/tts/transcripts/radio-clean.jsonl) no order "
    "is heard, no pairing window opens and the wrong read-back raises no live card. Not a false-VERIFIED; a fix (e.g. a"
    " card for an exchange in which speech is heard but nothing parses) needs a decision and a new pre-registration.",
    "Spoken digit runs that the ASR splits with commas ('two, four, three'; '3-1, Alpha') end the number run, so the "
    "tag is unknown or uncovered: the card abstains (a false alarm, fail closed).",
    "The disagreement gate fires when only the unprompted Whisper pass mishears a word ('Lain alfa', 'Kopi'): the card "
    "abstains (a false alarm, fail closed).",
    "Whisper writes text that was not said on short or noisy turns ('お団に。' for 'Ulangi?' on A2-001 at radio-clean, "
    "'We are at the top of the runway.' on A2-023 at -6 dB), which feeds the first issue.",
    "The energy VAD splits some synthetic takes at pauses inside a turn (more turns than the script on A2-001, A2-018, "
    "A2-023, A2-030, A2-037, A2-048, A2-052, A2-057, A2-069, A2-071, A2-093); A2-030's false MISMATCH at radio-clean "
    "(regex rung) follows such a split and is not yet explained.",
)


def condition_name(snr: str) -> str:
    return f"radio-{mix.snr_label(snr)}"


def transcript_path(snr: str) -> Path:
    return TRANSCRIPTS / f"{condition_name(snr)}.jsonl"


def load_transcripts(path: Path) -> dict[str, list[dict]]:
    return {r["exchange_id"]: r["turns"] for r in ev.read_jsonl(path)}


# ------------------------------------------------------------------------------------------------ transcribe


def cmd_transcribe(args) -> int:
    import radioize

    from sayagain.asr import VOCAB_PROMPT, LocalWhisper, transcribe_turn
    from sayagain.audio import load
    from sayagain.ingest import cut

    cards = ev.load_cards()
    if args.only:
        keep = set(args.only.split(","))
        cards = [c for c in cards if c["exchange_id"] in keep]
    missing = [c["exchange_id"] for c in cards if not (TTS / f"{c['exchange_id']}.wav").exists()]
    if missing:
        print(f"no TTS take for {missing[:5]} (run scripts/tts_cards.py)", file=sys.stderr)
        return 2
    pool = sorted((ROOT / "data" / "noise").rglob("*.wav"))
    asr = LocalWhisper()
    for snr in args.snr:
        if snr != "clean" and not pool:
            print("no noise WAVs: run scripts/fetch_mimii.sh", file=sys.stderr)
            return 2
        cond = condition_name(snr)
        tpath, rpath = transcript_path(snr), ROOT / "bench" / "results" / "receipts" / f"tts-asr-{cond}.jsonl"
        done = set(load_transcripts(tpath))  # resumable: a card already transcribed is kept as is
        t0 = time.time()
        for i, card in enumerate(c for c in cards if c["exchange_id"] not in done):
            eid = card["exchange_id"]
            clean = load(TTS / f"{eid}.wav")
            variant = radioize.radioize(mix.noisy_copy(clean, eid, snr, pool or None))
            turns, receipts = [], []
            for ta in cut(clean, eid, variant=variant):
                t, recs = transcribe_turn(asr, ta, language=None, prompt=VOCAB_PROMPT, unprompted=True)
                receipts += [{**r, "exchange_id": eid, "turn": ta.id, "condition": cond} for r in recs]
                turns.append(
                    {
                        "id": ta.id,
                        "t_start": round(ta.t_start, 3),
                        "t_end": round(ta.t_end, 3),
                        "text": t.transcript,
                        "alt_text": t.alt_transcript,
                        "not_audited": t.not_audited,
                        "language": t.meta.get("asr_language"),
                    }
                )
            row = {"exchange_id": eid, "condition": cond, "badge": [PROVENANCE, RADIO, "asr·off-AMD"], "turns": turns}
            for p, rows in ((tpath, [row]), (rpath, receipts)):
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
            if (i + 1) % 10 == 0:
                print(f"{cond}: {i + 1} cards · {time.time() - t0:.0f} s", flush=True)
        print(f"{cond}: done -> {tpath.relative_to(ROOT)}", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------- WER

_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "nol": "0", "kosong": "0", "satu": "1", "dua": "2", "tiga": "3",
    "empat": "4", "lima": "5", "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9",
}  # fmt: skip


def wer_words(text: str) -> list[str]:
    """Lower-case words, punctuation dropped, hyphens split; spoken digits and digit runs collapse to one number
    token ('two-oh-three' and '203' are both '203'), so WER counts word errors, not number formatting."""
    out: list[str] = []
    run = ""
    for w in re.findall(r"[a-z0-9']+", text.lower().replace("-", " ")):
        d = w if w.isdigit() else _DIGITS.get(w)
        if d is not None:
            run += d
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_rows(cards: list[dict], transcripts: dict[str, list[dict]]) -> dict:
    """Card-level WER: the card's whole script vs. all its ASR turns joined (VAD may split or merge turns)."""
    by: dict[str, list[int]] = {}
    for c in cards:
        if c["exchange_id"] not in transcripts:
            continue
        ref = wer_words(" ".join(t["text"] for t in c["turns"] if t["role"] != "pause"))
        hyp = wer_words(" ".join(t["text"] or "" for t in transcripts[c["exchange_id"]]))
        for key in (ev.row_of(c), f"set:{c['set']}", f"lang:{c['lang']}", "all"):
            s = by.setdefault(key, [0, 0, 0])
            s[0] += edit_distance(ref, hyp)
            s[1] += len(ref)
            s[2] += 1
    return {
        k: {"errors": e, "ref_words": n, "cards": m, "wer": round(e / n, 4) if n else None}
        for k, (e, n, m) in by.items()
    }


# --------------------------------------------------------------------------------------------------------- score


def cmd_score(args) -> int:
    registry, cards = Registry.load(), ev.load_cards()
    if len(args.snr) != 1:
        print("--score takes one --snr condition per run", file=sys.stderr)
        return 2
    snr = args.snr[0]
    tpath = transcript_path(snr)
    transcripts = load_transcripts(tpath)
    cards = [c for c in cards if c["exchange_id"] in transcripts]
    names = args.rungs.split(",")
    rungs = [ev.CANDIDATES[n] for n in names]
    paid = any(r.model for r in rungs)
    allowed, snap = ev.check_daily(args.budget) if paid else (0.0, {})
    if paid and allowed <= 0:
        print(f"daily spend guard: {snap}", file=sys.stderr)
        return 2
    budget, pacer = ev.Budget(allowed), Pacer(args.pace)
    run = args.run or f"tts-{condition_name(snr)}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    first = json.loads(Path(args.reaudit).read_text(encoding="utf-8")) if args.reaudit else None
    meta = ev.metadata(run, cards, {"subset": "all", "budget_usd": round(allowed, 4), "usage_before": snap})
    meta.update(
        provenance=PROVENANCE,
        provenance_note=(
            "EXPLORATORY: ElevenLabs text-to-speech of the card scripts (bench/results/tts/manifest.json) through the "
            "simulated radio path, local Whisper-large-v3 (off-AMD), extraction rungs as the text rows; not "
            "pre-registered, never the audio row (bench/PROTOCOL.md Deviations, 2026-10-15)"
        ),
        headline_note="the pre-registered headline is the 0 dB HUMAN audio row (bench/PROTOCOL.md §5): not run",
        badges=[PROVENANCE, RADIO, "asr·off-AMD"],
        condition=condition_name(snr),
        snr=mix.snr_label(snr),
        noise="none"
        if snr == "clean"
        else "MIMII pump / valve / fan (CC BY-SA 4.0, eval only), scripts/mix.py seed 1018",
        speech_path="TTS take -> mix.noisy_copy -> radioize -> VAD on the clean take -> Whisper, prompted + unprompted",
        asr={"backend": "local-whisper", "model": "openai/whisper-large-v3", "language": "auto", "on_amd": False},
        audio_gate=AUDIO_GATE,
        transcripts=str(tpath.relative_to(ROOT)),
        wer=wer_rows(cards, transcripts),
    )
    if first is not None:
        meta["reaudit_of"] = str(Path(args.reaudit).resolve().relative_to(ROOT))
    results = {}
    for r in rungs:
        prior = None
        if first is not None and r.model is not None:
            prior = ev.read_jsonl(ROOT / first["rungs"][r.name]["extractions"])
        results[r.name] = ev.run_extraction_rung(
            run,
            r,
            cards,
            registry,
            pacer,
            budget,
            "spans",
            prior,
            transcripts=tpath,
            audio_gate=AUDIO_GATE,
            out=OUT,
            prefix="tts",
        )
    order = sorted(rungs, key=lambda r: (r.params_b is None, r.params_b or 0.0))
    out = {
        **meta,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spent_usd": round(budget.spent, 6),
        "ladder_order": [r.name for r in order],
        "rule": f"0 false-VERIFIED AND false-alarm load <= 5 % (held_out row, {PROVENANCE}, exploratory)",
        "smallest_rung_meeting_rule": next((r.name for r in order if results[r.name]["meets_rule"]["literal"]), None),
        "rungs": results,
    }
    out["calls_cost_usd"] = ev._run_cost(results)
    path = OUT / f"{run}.json"
    ev.write_json(path, out)
    print(json.dumps({n: ev._brief(s) for n, s in results.items()}, indent=1))
    print(f"wrote {path.relative_to(ROOT)} · spent ${budget.spent:.4f} · WER all {out['wer']['all']['wer']}")
    return 0


def cmd_report(args) -> int:
    lines = [
        "# Synthetic-speech results — EXPLORATORY (generated by `scripts/eval_tts.py --report`; do not edit)",
        "",
        "Badge **`tts·synthetic`**: ElevenLabs text-to-speech of the card scripts (two voices, "
        "`bench/results/tts/manifest.json`) through the simulated radio path (`radio·simulated`), local "
        "Whisper-large-v3 off-AMD (`asr·off-AMD`), extraction rungs as in the text rows. Not pre-registered and not "
        "the audio row: no human recording has been evaluated (bench/PROTOCOL.md, Deviations, 2026-10-15). "
        f"Audio gate: `{AUDIO_GATE}`. Every row cites its file.",
        "",
    ]
    total = 0.0
    for p in args.files:
        path = Path(p).resolve()
        doc = json.loads(path.read_text(encoding="utf-8"))
        total += doc.get("calls_cost_usd") or 0.0
        body = ev.report_lines(path)
        body[0] = body[0].replace("### ", f"### {doc['condition']} · ")
        w = doc["wer"]
        wer = " · ".join(f"{k} {v['wer']} ({v['errors']}/{v['ref_words']} words)" for k, v in sorted(w.items()))
        lines += [
            *body[:-1],
            "",
            f"Whisper WER (card level, number formatting normalised; {path.relative_to(ROOT)}): {wer}",
            "",
        ]
    lines += ["## Open issues (found on these rows, not fixed: the referee is frozen for this run)", ""]
    lines += [f"{k}. {t}" for k, t in enumerate(OPEN_ISSUES, 1)] + [""]
    spend = OUT / "spend.json"  # the total, in a file the sentence can cite
    files = [str(Path(p).resolve().relative_to(ROOT)) for p in args.files]
    ev.write_json(spend, {"calls_cost_usd_total": round(total, 6), "files": files, "badge": PROVENANCE})
    lines += [
        f"Token Factory spend over the {len(files)} files above: ${round(total, 6)} "
        f"(sum of their `calls_cost_usd`; {spend.relative_to(ROOT)}).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {(OUT / 'RESULTS.md').relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--transcribe", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--snr", nargs="+", default=["clean"], help="clean / 6 / 0 / -6 (PROTOCOL §2)")
    ap.add_argument("--only", help="--transcribe: comma-separated exchange ids")
    ap.add_argument("--rungs", default="regex,Gemma-4-31B-IT")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--pace", type=float, default=ev.PACE_S)
    ap.add_argument("--run")
    ap.add_argument("--reaudit", help="--score: re-request only the NOT_AUDITED extractions of this result file")
    ap.add_argument("files", nargs="*", help="--report: result files")
    args = ap.parse_args(argv)
    if args.transcribe:
        return cmd_transcribe(args)
    if args.score:
        return cmd_score(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
