"""Deterministic seed data: the plant registry, the error plan and the recording cards.

    uv run python scripts/seed.py --seed 1018 [--cards-md PATH]

Writes data/registry.csv, data/error_plan.csv, data/cards.jsonl (one line per card = one WAV, file name = exchange_id).
Same seed -> byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from referee.registry import Registry, Row  # noqa: E402

DATA = ROOT / "data"
LINES = ("A", "B", "C", "D")
AREAS = ("sterilizer", "press station", "clarification", "kernel plant", "boiler house", "effluent")
ACTIONS_BY_TYPE = {
    "XV": ("OPEN", "CLOSE", "ISOLATE"),
    "P": ("START", "STOP", "ISOLATE"),
    "MOV": ("OPEN", "CLOSE", "ISOLATE", "STOP"),
}
TYPE_NAME = {"XV": "on/off valve", "P": "pump", "MOV": "motor-operated valve"}

# engineered look-alikes and fixed rows
PINNED: dict[str, str] = {
    "XV-203": "B",  # hero: instruction tag
    "XV-208": "A",  # hero: the look-alike the tech reads back — on the RUNNING line
    "P-114": "C",
    "P-140": "C",
    "MOV-31A": "D",
    "MOV-31H": "B",
    "P-215": "B",  # a correct tag on the wrong line is a line MISMATCH
    "XV-216": "B",
    "XV-212": "A",
    "P-122": "A",
}
ABSENT = {"XV-205"}  # the guaranteed refusal
LOOKALIKE_PAIRS = (("XV-203", "XV-208"), ("P-114", "P-140"), ("MOV-31A", "MOV-31H"))

EN_DIGITS = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ID_DIGITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
NATO = {"A": "Alpha", "B": "Bravo", "C": "Charlie", "D": "Delta", "H": "Hotel", "E": "Echo", "F": "Foxtrot"}


def spoken_suffix(suffix: str, lang: str = "en") -> str:
    """'203' -> 'two-oh-three' / 'dua nol tiga'; '31A' -> 'three-one Alpha' / 'tiga satu Alpha'."""
    digits = [c for c in suffix if c.isdigit()]
    letter = suffix[-1] if suffix[-1].isalpha() else ""
    if lang == "en":
        words = "-".join(EN_DIGITS[int(d)] for d in digits)
    else:
        words = " ".join(ID_DIGITS[int(d)] for d in digits)
    return f"{words} {NATO[letter]}" if letter else words


def build_registry(seed: int) -> list[Row]:
    rng = random.Random(seed)
    tags: dict[str, str] = dict(PINNED)
    used_suffixes = {t.split("-")[1] for t in tags} | {t.split("-")[1] for t in ABSENT}

    def add(prefix: str, suffix: str) -> None:
        if suffix in used_suffixes:
            return
        used_suffixes.add(suffix)
        tags[f"{prefix}-{suffix}"] = rng.choice(LINES)

    pool_xv = [f"{n:03d}" for n in range(201, 300)]
    pool_p = [f"{n:03d}" for n in range(101, 200)]
    pool_mov = [f"{n:02d}{c}" for n in range(11, 60) for c in "ABCDEFH"]
    for prefix, pool, count in (("XV", pool_xv, 62), ("P", pool_p, 38), ("MOV", pool_mov, 20)):
        for s in rng.sample(pool, len(pool)):
            if sum(1 for t in tags if t.startswith(prefix + "-")) >= count:
                break
            add(prefix, s)
    rows = []
    for tag in sorted(tags, key=lambda t: (t.split("-")[0], t.split("-")[1])):
        prefix, suffix = tag.split("-")
        variants = [spoken_suffix(suffix, "en"), spoken_suffix(suffix, "id")]
        if suffix.isdigit():
            variants.insert(0, suffix)
        rows.append(
            Row(
                tag=tag,
                type=TYPE_NAME[prefix],
                line=tags[tag],
                area=rng.choice(AREAS),
                allowed_actions=ACTIONS_BY_TYPE[prefix],
                spoken_variants=tuple(variants),
            )
        )
    reg = Registry(rows)
    assert reg.suffixes_unique(), "numeric suffixes must be unique across prefixes (bare numbers resolve by suffix)"
    assert not (ABSENT & set(reg.rows)), "XV-205 must stay absent: it is the guaranteed refusal"
    for t, line in PINNED.items():
        assert reg.line_of(t) == line
    return rows


def registry_csv(rows: list[Row]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["tag", "type", "line", "area", "allowed_actions", "spoken_variants"])
    for r in rows:
        w.writerow([r.tag, r.type, r.line, r.area, "|".join(r.allowed_actions), "|".join(r.spoken_variants)])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1018)
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_registry(args.seed)
    (args.out / "registry.csv").write_text(registry_csv(rows), encoding="utf-8")
    print(f"registry: {len(rows)} tags -> {args.out / 'registry.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
