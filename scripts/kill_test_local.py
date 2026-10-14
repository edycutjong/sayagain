"""Local kill test: can Whisper-large-v3 + the frozen regex/alias extractor read tags and lines off our own voices?

    uv sync --extra killtest
    uv run python scripts/kill_test_local.py [--clips data/human/killtest] [--snr clean 0]

Runs on the Mac (transformers, MPS), never on the judged path. Thresholds are pre-registered in bench/PROTOCOL.md:
tag exact match >= 80 % clean and >= 60 % at 0 dB; paired proxy (both turns exact on tag + line) >= 65 % clean.
Each mode is reported: forced `en`, forced `id`, auto-detect. Writes bench/results/killtest.json.
Without human clips it writes nothing and exits 3: the clips are recorded by people, never synthesised.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mix  # noqa: E402

from referee.audit import audit  # noqa: E402
from referee.regex_extractor import extract  # noqa: E402
from referee.registry import Registry  # noqa: E402
from referee.turns import Turn, analyze  # noqa: E402

THRESHOLDS = {"tag_clean": 0.80, "tag_0db": 0.60, "paired_clean": 0.65}
MODES = {"en": "en", "id": "id", "auto": None}


def truth_turns(card: dict) -> list[tuple[str, dict | None]]:
    """(role, truth command) for the instruction and the read-back, in speaking order."""
    out: list[tuple[str, dict | None]] = [("instruction", card["instruction"][0])]
    if card["readback"]:
        out.append(("readback", card["readback"][0]))
    return out


def read_turn(text: str, registry: Registry) -> dict | None:
    """First command of a transcript through the frozen regex extractor + aliases + registry: {tag, line} or None."""
    a = analyze(Turn("k", "k", 0, 1, text, extract(text)), registry)
    if a.status != "ok" or not a.commands:
        return None
    c = a.commands[0]
    return {"tag": c.tag, "line": c.line}


def score_card(card: dict, transcripts: list[str], registry: Registry) -> dict:
    """Exact-match scoring of one card's instruction + read-back transcripts against its truth labels."""
    truth = truth_turns(card)
    spoken = [t for t in transcripts if t.strip()]
    # the instruction is the first turn, the read-back the last one (cards in the kill test have exactly two)
    picked = [spoken[0] if spoken else "", spoken[-1] if len(spoken) > 1 else ""][: len(truth)]
    per_turn = []
    for (role, tc), text in zip(truth, picked, strict=False):
        got = read_turn(text, registry)
        tag_ok = got is not None and tc is not None and got["tag"] == tc["tag"]
        line_ok = got is not None and tc is not None and (got["line"] or None) == (tc["line"] or None)
        per_turn.append({"role": role, "text": text, "got": got, "truth": tc, "tag_ok": tag_ok, "line_ok": line_ok})
    ts = [Turn(f"{card['exchange_id']}-{i}", "KT", 3.5 * i, 3.5 * i + 2.5, t, extract(t)) for i, t in enumerate(spoken)]
    verdicts = [j.verdict.label for j in audit(ts, registry).judged]
    return {
        "exchange_id": card["exchange_id"],
        "lang": card["lang"],
        "turns": per_turn,
        "paired_ok": all(t["tag_ok"] and t["line_ok"] for t in per_turn),
        "verdict": verdicts[0] if verdicts else None,
        "expected": card["expected"]["label"] if card["expected"] else None,
    }


def summarise(scored: list[dict]) -> dict:
    turns = [t for s in scored for t in s["turns"]]
    n_t, n_x = len(turns), len(scored)
    by_lang = {}
    for lang in sorted({s["lang"] for s in scored}):
        sub = [s for s in scored if s["lang"] == lang]
        st = [t for s in sub for t in s["turns"]]
        by_lang[lang] = {
            "n_exchanges": len(sub),
            "tag_exact": sum(t["tag_ok"] for t in st) / len(st) if st else None,
            "paired": sum(s["paired_ok"] for s in sub) / len(sub),
        }
    return {
        "n_exchanges": n_x,
        "n_turns": n_t,
        "tag_exact": sum(t["tag_ok"] for t in turns) / n_t if n_t else None,
        "paired": sum(s["paired_ok"] for s in scored) / n_x if n_x else None,
        "verdict_as_labelled": sum(s["verdict"] == s["expected"] for s in scored) / n_x if n_x else None,
        "by_lang": by_lang,
    }


def _checks(clean: dict, zero: dict) -> dict:
    vals = {"tag_clean": clean.get("tag_exact"), "tag_0db": zero.get("tag_exact"), "paired_clean": clean.get("paired")}
    return {
        k: {"value": v, "threshold": THRESHOLDS[k], "pass": None if v is None else v >= THRESHOLDS[k]}
        for k, v in vals.items()
    }


def verdict_of(rows: dict) -> dict:
    """Pass/fail per pre-registered threshold for one language mode, pooled AND per language.

    rows = {"clean": summary, "0dB": summary} (labels from mix.snr_label). A check that could not be evaluated is
    `pass: None` and the mode is not a pass. Pivot rule: English passes, code-switched fails -> "pivot".
    """
    clean, zero = rows.get("clean") or {}, rows.get(mix.snr_label("0")) or {}
    pooled = _checks(clean, zero)
    by_lang = {}
    for lang in sorted(set(clean.get("by_lang", {})) | set(zero.get("by_lang", {}))):
        c = (clean.get("by_lang") or {}).get(lang, {})
        z = (zero.get("by_lang") or {}).get(lang, {})
        by_lang[lang] = _checks(
            {"tag_exact": c.get("tag_exact"), "paired": c.get("paired")}, {"tag_exact": z.get("tag_exact")}
        )

    def ok(checks: dict) -> bool:
        return all(c["pass"] is True for c in checks.values())

    en, mixed = by_lang.get("en"), by_lang.get("mix")
    if ok(pooled) and all(ok(v) for v in by_lang.values()):
        decision = "go"
    elif en is not None and mixed is not None and ok(en) and not ok(mixed):
        decision = "pivot: English radio phraseology with Bahasa action words only, then re-test"
    else:
        decision = "stop: a human decides"
    return {"pooled": pooled, "by_lang": by_lang, "decision": decision}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=ROOT / "data" / "human" / "killtest")
    ap.add_argument("--cards", type=Path, default=ROOT / "data" / "cards.jsonl")
    ap.add_argument("--snr", nargs="+", default=["clean", "0"])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--noise", type=Path, default=ROOT / "data" / "noise")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "killtest.json")
    args = ap.parse_args(argv)

    clips = sorted(args.clips.glob("KT-*.wav")) if args.clips.exists() else []
    if not clips:
        print(
            f"SKIPPED: no human kill-test clips in {args.clips} (KT-01.wav … KT-20.wav). Nothing written.",
            file=sys.stderr,
        )
        return 3

    import numpy as np
    import torch
    from transformers import pipeline

    from sayagain.audio import SR, load
    from sayagain.vad import segments

    cards = {c["exchange_id"]: c for c in map(json.loads, args.cards.read_text(encoding="utf-8").splitlines())}
    registry = Registry.load()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    )
    noise_pool = sorted(args.noise.rglob("*.wav"))
    if any(x != "clean" for x in args.snr) and not noise_pool:
        print("no noise WAVs: run scripts/fetch_mimii.sh first", file=sys.stderr)
        return 2

    results: dict = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "device": device,
        "host": platform.platform(),
        "on_amd": False,
        "speaker_prov": "human·seeded·1-voice",
        "radio_path": "direct",
        "thresholds": THRESHOLDS,
        "modes": {},
    }
    for mode in args.modes:
        lang = MODES[mode]
        rows: dict = {}
        for snr in args.snr:
            label = mix.snr_label(snr)
            scored, timings = [], []
            for clip in clips:
                card = cards[clip.stem]
                clean = load(clip)
                x = mix.noisy_copy(clean, clip.stem, snr, noise_pool)
                texts = []
                for s, e in segments(clean):
                    t0 = time.perf_counter()
                    kw = {"task": "transcribe"} | ({"language": lang} if lang else {})
                    out = asr({"raw": x[int(s * SR) : int(e * SR)], "sampling_rate": SR}, generate_kwargs=kw)
                    timings.append(time.perf_counter() - t0)
                    texts.append(out["text"].strip())
                scored.append(score_card(card, texts, registry))
            rows[label] = {
                **summarise(scored),
                "p50_s_per_turn": float(np.median(timings)) if timings else None,
                "exchanges": scored,
            }
        results["modes"][mode] = {"rows": rows, "verdict": verdict_of(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for mode, r in results["modes"].items():
        print(mode, r["verdict"]["decision"], {k: (v["value"], v["pass"]) for k, v in r["verdict"]["pooled"].items()})
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
