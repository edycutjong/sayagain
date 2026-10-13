"""Deterministic seed data: the plant registry, the error plan and the recording cards.

    uv run python scripts/seed.py --seed 1018 [--cards-md PATH]

Writes data/registry.csv, data/error_plan.csv, data/cards.jsonl (one line per card = one WAV, file name = exchange_id).
Same seed -> byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
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


# ------------------------------------------------------------------------------------------------- cards
#
# One card = one WAV, file name = exchange_id. Turns are read in order with ~1 s between them; a "pause" turn means stay
# silent for that long. Labels (truth commands, error type, expected verdict) live in cards.jsonl, never on the card.

ERROR_PLAN = (  # error type, A2 count (the two hero cards count inside wrong_line / lookalike_tag)
    ("wrong_line", 12),
    ("lookalike_tag", 14),
    ("wrong_action", 8),
    ("omitted_field", 6),
    ("no_readback", 4),
    ("omitted_command", 4),
    ("uncorrected_retraction", 4),
)
N_CORRECT, N_CAUGHT, N_SAY_AGAIN_CORRECT, N_SAY_AGAIN_ERR = 50, 7, 4, 2
N_MULTI_CORRECT, N_CORRECTION_CORRECT, N_VERBATIM = 5, 5, 6
DEV_SLICE_A2 = 20

EN_IMP = {"CLOSE": "Close", "OPEN": "Open", "ISOLATE": "Isolate", "START": "Start", "STOP": "Stop"}
EN_ING = {"CLOSE": "closing", "OPEN": "opening", "ISOLATE": "isolating", "START": "starting", "STOP": "stopping"}
ID_IMP = {"CLOSE": "tutup", "OPEN": "buka", "ISOLATE": "isolasi", "START": "nyalakan", "STOP": "matikan"}
EN_NOUN = {"XV": "valve", "P": "pump", "MOV": "motor valve"}
ID_NOUN = {"XV": "valve", "P": "pompa", "MOV": "motor valve"}
OTHER_ACTION = {"CLOSE": "OPEN", "OPEN": "CLOSE", "START": "STOP", "STOP": "START", "ISOLATE": "OPEN"}


def _cmd(action: str | None, tag: str | None, line: str | None) -> dict:
    return {"action": action, "tag": tag, "line": line}


class Phraser:
    """Natural radio phrasing in English or code-switched Bahasa Indonesia / English."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def num(self, tag: str, lang: str) -> str:
        return spoken_suffix(tag.split("-")[1], lang)

    def noun(self, tag: str, lang: str) -> str:
        prefix = tag.split("-")[0]
        if lang == "mix" and prefix == "XV" and self.rng.random() < 0.3:
            return "katup"
        return (ID_NOUN if lang == "mix" else EN_NOUN)[prefix]

    def line(self, letter: str, lang: str) -> str:
        word = "jalur" if lang == "mix" and self.rng.random() < 0.5 else "line"
        return f"{word} {NATO[letter]}"

    def instruction_cmd(self, c: dict, lang: str, first: bool = True) -> str:
        act = EN_IMP[c["action"]] if lang == "en" else ID_IMP[c["action"]]
        if not first:
            act = act.lower()
        elif lang == "mix":
            act = act.capitalize()
        out = f"{act} {self.noun(c['tag'], lang)} {self.num(c['tag'], lang)}"
        return out + (f", {self.line(c['line'], lang)}" if c["line"] else "")

    def instruction(self, cmds: list[dict], lang: str) -> str:
        parts = [self.instruction_cmd(c, lang, first=(k == 0)) for k, c in enumerate(cmds)]
        joiner = " and " if lang == "en" else " dan "
        text = joiner.join(parts) + "."
        if lang == "mix" and self.rng.random() < 0.4:
            text = "Tolong " + text[0].lower() + text[1:]
        return text

    def readback_cmd(self, c: dict, lang: str, with_noun: bool) -> str:
        bits = []
        if c["action"]:
            bits.append(EN_ING[c["action"]] if lang == "en" else ID_IMP[c["action"]])
        if c["tag"]:
            bits.append((self.noun(c["tag"], lang) + " " if with_noun else "") + self.num(c["tag"], lang))
        out = " ".join(bits)
        return out + (f", {self.line(c['line'], lang)}" if c["line"] else "")

    def readback(self, cmds: list[dict], lang: str, verbatim: bool = False) -> str:
        if verbatim:
            return self.instruction(cmds, lang)
        opener = self.rng.choice(["Copy", "Roger"] if lang == "en" else ["Siap", "Copy", "Siap"])
        with_noun = self.rng.random() < 0.35
        joiner = " and " if lang == "en" else " dan "
        return f"{opener}, " + joiner.join(self.readback_cmd(c, lang, with_noun) for c in cmds) + "."


def _card(
    eid: str,
    kind: str,
    turns: list[dict],
    instr: list[dict] | None,
    rb: list[dict] | None,
    error: str,
    lang: str,
    phenomena: list[str],
    operator: str,
) -> dict:
    return {
        "exchange_id": eid,
        "set": kind,
        "operator": operator,
        "tech": "S2" if operator == "S1" else "S1",
        "lang": lang,
        "turns": turns,
        "instruction": instr,
        "readback": rb,
        "error_type": error,
        "phenomena": phenomena,
    }


def _t(role: str, text: str) -> dict:
    return {"role": role, "text": text}


def build_cards(seed: int, rows: list[Row]) -> list[dict]:
    rng = random.Random(seed + 1)
    ph = Phraser(rng)
    reg = Registry(rows)
    pair_tags = {t for p in LOOKALIKE_PAIRS for t in p}
    ordinary = [r for r in rows if r.tag not in pair_tags]

    def pick(n: int = 1, *, exclude: set[str] | None = None) -> list[Row]:
        pool = [r for r in ordinary if not exclude or r.tag not in exclude]
        return rng.sample(pool, n)

    def simple(r: Row, with_line: bool = True) -> dict:
        return _cmd(rng.choice(r.allowed_actions), r.tag, r.line if with_line else None)

    cards: list[dict] = []

    # H — the hero pair, recorded first (S1 = operator)
    hero_i = [_cmd("CLOSE", "XV-203", "B")]
    cards.append(
        _card(
            "H-01",
            "H",
            [
                _t("operator", "Close valve two-oh-three, line Bravo."),
                _t("tech", "Copy, closing two-oh-eight, line Bravo."),
            ],
            hero_i,
            [_cmd("CLOSE", "XV-208", "B")],
            "lookalike_tag",
            "en",
            ["hero"],
            "S1",
        )
    )
    cards.append(
        _card(
            "H-02",
            "H",
            [
                _t("operator", "Close valve two-oh-three, line Bravo."),
                _t("tech", "Copy, closing two-oh-three, line Alpha."),
            ],
            hero_i,
            [_cmd("CLOSE", "XV-203", "A")],
            "wrong_line",
            "en",
            ["secondary"],
            "S1",
        )
    )
    cards.append(
        _card(
            "R-01",
            "R",
            [
                _t("operator", "Close valve two-oh-five, line Charlie."),
                _t("tech", "Copy, closing two-oh-five, line Charlie."),
            ],
            [_cmd("CLOSE", "XV-205", "C")],
            [_cmd("CLOSE", "XV-205", "C")],
            "refusal",
            "en",
            ["refusal"],
            "S1",
        )
    )

    a2: list[dict] = []
    langs = iter(rng.choices(["en", "mix"], weights=[6, 4], k=400))

    def add(turns, instr, rb, error, lang, phen):
        a2.append(_card("", "A2", turns, instr, rb, error, lang, phen, ""))

    # seeded errors (hero cards already fill one wrong_line and one lookalike_tag)
    quotas = dict(ERROR_PLAN)
    quotas["wrong_line"] -= 1
    quotas["lookalike_tag"] -= 1
    say_again_err = N_SAY_AGAIN_ERR
    for err, n in quotas.items():
        for k in range(n):
            lang = next(langs)
            r = pick()[0]
            ins = [simple(r)]
            phen: list[str] = []
            if err == "wrong_line":
                bad = rng.choice([x for x in LINES if x != r.line])
                rb = [_cmd(ins[0]["action"], r.tag, bad)]
            elif err == "lookalike_tag":
                a, b = rng.choice(LOOKALIKE_PAIRS)
                if rng.random() < 0.5:
                    a, b = b, a
                ra = reg.rows[a]
                ins = [_cmd(rng.choice(ra.allowed_actions), a, ra.line)]
                rb = [_cmd(ins[0]["action"], b, ra.line)]  # the tag changes, the line stays as instructed
            elif err == "wrong_action":
                act = ins[0]["action"]
                rb = [
                    _cmd(
                        OTHER_ACTION[act]
                        if OTHER_ACTION[act] in r.allowed_actions
                        else next(x for x in r.allowed_actions if x != act),
                        r.tag,
                        r.line,
                    )
                ]
            elif err == "omitted_field":
                rb = [_cmd(ins[0]["action"], r.tag, None)] if k < 4 else [_cmd(None, r.tag, r.line)]
            elif err == "no_readback":
                rb = None
            elif err == "omitted_command":
                r2 = pick(exclude={r.tag})[0]
                ins = [simple(r, with_line=False), simple(r2)]
                rb = [ins[0]]
                phen.append("multi_command")
            elif err == "uncorrected_retraction":
                phen.append("correction")
                bad_line = rng.choice([x for x in LINES if x != r.line])
                spoken = ph.instruction([ins[0]], lang).rstrip(".")
                cue = "correction" if lang == "en" else rng.choice(["koreksi", "ralat"])
                said = f"{spoken.rsplit(',', 1)[0]}, {ph.line(bad_line, lang)}, {cue}, {ph.line(r.line, lang)}."
                rb = [_cmd(ins[0]["action"], r.tag, bad_line)]
                add([_t("operator", said), _t("tech", ph.readback(rb, lang))], ins, rb, err, lang, phen)
                continue
            turns = [_t("operator", ph.instruction(ins, lang))]
            if rb is None:
                turns.append({"role": "pause", "seconds": 10})
            else:
                if say_again_err and err in ("wrong_line", "wrong_action"):
                    say_again_err -= 1
                    phen.append("say_again")
                    turns += [
                        _t("tech", "Say again?" if lang == "en" else "Ulangi?"),
                        _t(
                            "operator",
                            ("I say again, " if lang == "en" else "Saya ulangi, ")
                            + ph.instruction(ins, lang)[0].lower()
                            + ph.instruction(ins, lang)[1:],
                        ),
                    ]
                turns.append(_t("tech", ph.readback(rb, lang)))
            add(turns, ins, rb, err, lang, phen)

    # correct exchanges, with the phenomena the model is there for
    for k in range(N_CORRECT):
        lang = next(langs)
        r = pick()[0]
        ins = [simple(r)]
        phen = []
        if k < N_MULTI_CORRECT:
            r2 = pick(exclude={r.tag})[0]
            ins = [simple(r, with_line=False), simple(r2)]
            phen.append("multi_command")
            turns = [_t("operator", ph.instruction(ins, lang)), _t("tech", ph.readback(ins, lang))]
        elif k < N_MULTI_CORRECT + N_CORRECTION_CORRECT:
            phen.append("correction")
            bad_line = rng.choice([x for x in LINES if x != r.line])
            spoken = ph.instruction([ins[0]], lang).rstrip(".")
            cue = "correction" if lang == "en" else rng.choice(["koreksi", "ralat"])
            said = f"{spoken.rsplit(',', 1)[0]}, {ph.line(bad_line, lang)}, {cue}, {ph.line(r.line, lang)}."
            turns = [_t("operator", said), _t("tech", ph.readback(ins, lang))]
        elif k < N_MULTI_CORRECT + N_CORRECTION_CORRECT + N_SAY_AGAIN_CORRECT:
            phen.append("say_again")
            again = ph.instruction(ins, lang)
            turns = [
                _t("operator", again),
                _t("tech", "Say again?" if lang == "en" else "Ulangi?"),
                _t("operator", ("I say again, " if lang == "en" else "Saya ulangi, ") + again[0].lower() + again[1:]),
                _t("tech", ph.readback(ins, lang)),
            ]
        elif k < N_MULTI_CORRECT + N_CORRECTION_CORRECT + N_SAY_AGAIN_CORRECT + N_VERBATIM:
            phen.append("verbatim")
            turns = [_t("operator", ph.instruction(ins, lang)), _t("tech", ph.readback(ins, lang, verbatim=True))]
        else:
            turns = [_t("operator", ph.instruction(ins, lang)), _t("tech", ph.readback(ins, lang))]
        if rng.random() < 0.3:
            turns.append(_t("operator", "Correct." if lang == "en" else rng.choice(["Betul.", "Correct."])))
        add(turns, ins, ins, "none", lang, phen)

    # caught on channel: the operator hears the wrong read-back, says "negative" and repeats
    for k in range(N_CAUGHT):
        lang = next(langs)
        if k < 3:
            a, b = LOOKALIKE_PAIRS[k]
            ra = reg.rows[a]
            ins = [_cmd(rng.choice(ra.allowed_actions), a, ra.line)]
            wrong = [_cmd(ins[0]["action"], b, ra.line)]
        else:
            r = pick()[0]
            ins = [simple(r)]
            wrong = [_cmd(ins[0]["action"], r.tag, rng.choice([x for x in LINES if x != r.line]))]
        neg = "Negative" if lang == "en" else rng.choice(["Koreksi", "Bukan"])
        rep = ph.instruction(ins, lang)
        turns = [_t("operator", ph.instruction(ins, lang)), _t("tech", ph.readback(wrong, lang))]
        if k % 2 == 0:
            turns.append(_t("operator", f"{neg}, {rep[0].lower()}{rep[1:]}"))
        else:
            turns += [_t("operator", f"{neg}."), _t("operator", rep)]
        turns += [_t("tech", ph.readback(ins, lang)), _t("operator", "Correct." if lang == "en" else "Betul.")]
        add(turns, ins, ins, "caught_on_channel", lang, ["caught_on_channel"])

    rng.shuffle(a2)
    for n, c in enumerate(a2, 1):
        c["exchange_id"] = f"A2-{n:03d}"
        c["operator"] = "S1" if n % 2 else "S2"  # roles swapped for half the deck
        c["tech"] = "S2" if n % 2 else "S1"
        c["dev_slice"] = n <= DEV_SLICE_A2
    cards += a2

    # chatter: no instruction, so no card may open (3 contain clock times, 2 name a tag without a verb)
    chatter = [
        ("Field two, where are you now?", "At the boiler house, walking to the press station."),
        ("What time is the shift handover?", "Handover at two-oh-five, same as yesterday."),
        ("Radio check, how do you read?", "Loud and clear."),
        ("Pump one-forty is running normal, no vibration.", "Copy, thanks."),
        ("Steam pressure is twelve bar at the header.", "Noted."),
        ("Truck for the kernel arrives at two-fifteen.", "Roger, I will tell the weighbridge."),
        ("Lunch break until twelve-thirty.", "Okay."),
        ("Is the forklift free?", "No, it is at the warehouse."),
        ("Valve dua nol delapan masih bocor sedikit, nanti dicek ya.", "Siap, nanti saya lihat."),
        ("Ada orang di area klarifikasi?", "Ada dua orang, lagi bersih-bersih."),
        ("Temperature at the sterilizer is one-forty degrees.", "Copy."),
        ("Can you bring the gas detector to line Bravo?", "On my way."),
    ]
    for n, (a, b) in enumerate(chatter, 1):
        lang = "mix" if n in (9, 10) else "en"
        phen = ["time_in_chatter"] if n in (2, 6, 7) else (["tag_without_verb"] if n in (4, 9) else [])
        cards.append(
            _card(
                f"C-{n:02d}",
                "C",
                [_t("operator", a), _t("tech", b)],
                None,
                None,
                "chatter",
                lang,
                phen,
                "S1" if n % 2 else "S2",
            )
        )

    # kill test: 20 exchanges in one voice, 10 English + 10 code-switched, 5 with errors
    for n in range(1, 21):
        lang = "en" if n <= 10 else "mix"
        r = pick()[0]
        ins = [simple(r)]
        rb = ins
        err = "none"
        if n in (3, 8, 13, 18):
            a, b = LOOKALIKE_PAIRS[n % 3]
            ra = reg.rows[a]
            ins = [_cmd(rng.choice(ra.allowed_actions), a, ra.line)]
            rb, err = [_cmd(ins[0]["action"], b, ra.line)], "lookalike_tag"
        elif n == 16:
            rb, err = [_cmd(ins[0]["action"], r.tag, rng.choice([x for x in LINES if x != r.line]))], "wrong_line"
        cards.append(
            _card(
                f"KT-{n:02d}",
                "KT",
                [_t("operator", ph.instruction(ins, lang)), _t("tech", ph.readback(rb, lang))],
                ins,
                rb,
                err,
                lang,
                [],
                "S1",
            )
        )

    for c in cards:
        c.setdefault("dev_slice", c["set"] == "KT")
        c["fold"] = 1 + (sum(ord(ch) for ch in c["exchange_id"]) % 2)
        c["expected"] = expected_verdict(c, reg)
    return cards


def expected_verdict(card: dict, reg: Registry) -> dict | None:
    """The verdict the truth labels imply, through the same pure referee function the pipeline uses."""
    from referee.verdict import CanonCommand, verdict

    if card["instruction"] is None:
        return None

    def canon(cmds: list[dict]) -> list[CanonCommand]:
        out = []
        for c in cmds:
            res = reg.resolve(c["tag"].split("-")[0], c["tag"].split("-")[1]) if c["tag"] else None
            out.append(
                CanonCommand(
                    c["action"],
                    res.tag if res else None,
                    res.status if res else None,
                    res.line if res else None,
                    c["line"],
                )
            )
        return out

    # instructions without a spoken line: the truth label still carries the registry line, as spoken lines would
    v = verdict(canon(card["instruction"]), canon(card["readback"]) if card["readback"] is not None else None)
    return {"verdict": v.verdict, "reason": v.reason, "label": v.label}


def error_plan_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["error_type", "count_a2", "what changes in the read-back"])
    what = {
        "wrong_line": "the line letter",
        "lookalike_tag": "the tag only, to its look-alike (XV-203/208, P-114/140, MOV-31A/31H); the line stays",
        "wrong_action": "the action",
        "omitted_field": "the line (4) or the action (2) is left out",
        "no_readback": "nothing: 10 s of silence",
        "omitted_command": "one of two commands is left out",
        "uncorrected_retraction": "the read-back keeps the value the operator took back",
    }
    for err, n in ERROR_PLAN:
        w.writerow([err, n, what[err]])
    return buf.getvalue()


def cards_markdown(cards: list[dict]) -> str:
    head = [
        "# SayAgain recording cards",
        "",
        "One card = one WAV file, named exactly `<exchange_id>.wav` (e.g. `A2-017.wav`). Phone recording is fine.",
        "Read the turns in order, naturally, like on a radio: about 1 s between turns. `[silence N s]` = stay quiet.",
        "S1 / S2 = the two speakers; the operator line says who speaks. If you slip, re-record the card.",
        "Files go in `build/data/human/<set>/` (never committed): `hero/ refusal/ a2/ chatter/ killtest/`.",
        "",
        'Consent (each speaker, once): *"I agree my voice recordings are used in the open-source SayAgain hackathon '
        'corpus (MIT repo), processed by an AMD-hosted API during the hackathon."*',
        "",
    ]
    sections = [
        ("Hero + refusal (record first)", ("H", "R")),
        ("A2 deck", ("A2",)),
        ("Chatter", ("C",)),
        ("Kill test (one voice, both roles)", ("KT",)),
    ]
    out = head
    for title, sets in sections:
        out += [f"## {title}", ""]
        for c in (c for c in cards if c["set"] in sets):
            who = "one voice" if c["set"] == "KT" else f"operator {c['operator']} · tech {c['tech']}"
            out.append(f"### {c['exchange_id']} · {who}")
            for k, t in enumerate(c["turns"], 1):
                if t["role"] == "pause":
                    out.append(f"{k}. `[silence {t['seconds']} s]`")
                else:
                    out.append(f"{k}. **{t['role'].capitalize()}:** {t['text']}")
            out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1018)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--cards-md", type=Path, help="also print the cards as Markdown for the recording session")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_registry(args.seed)
    (args.out / "registry.csv").write_text(registry_csv(rows), encoding="utf-8")
    (args.out / "error_plan.csv").write_text(error_plan_csv(), encoding="utf-8")
    cards = build_cards(args.seed, rows)
    (args.out / "cards.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cards), encoding="utf-8"
    )
    print(f"registry: {len(rows)} tags · cards: {len(cards)} -> {args.out}")
    if args.cards_md:
        args.cards_md.write_text(cards_markdown(cards), encoding="utf-8")
        print(f"cards markdown -> {args.cards_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
