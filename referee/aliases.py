"""Deterministic alias tables: spoken radio words -> canonical plant vocabulary.

No model, no fuzzy matching, no guessing. A word that is not in a table is *unmapped*, and the referee
abstains on it. Everything here is part of the pre-registered extractor (bench/PROTOCOL.md).

Covers:
  * English digit speech ("two-oh-three", "two zero three", "two hundred three", "one-forty", "to oh three")
  * digit strings an ASR usually emits ("203", "XV-203", "31A") and time-shaped strings ("2:03", "2.03")
  * Bahasa Indonesia digits ("dua nol tiga", "dua kosong tiga", "dua ratus tiga", "seratus empat puluh")
  * NATO letters and Bahasa letter names (the latter only in line context)
  * tag nouns -> plant prefixes, the action lexicon with inflections, a small state lexicon
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------------------------- vocabulary

ACTIONS: dict[str, tuple[str, ...]] = {
    "CLOSE": (
        "close",
        "closes",
        "closing",
        "closed",
        "shut",
        "shuts",
        "shutting",
        "tutup",
        "menutup",
        "ditutup",
        "tutupkan",
        "tutupin",
    ),
    "OPEN": ("open", "opens", "opening", "opened", "buka", "membuka", "dibuka", "bukakan", "bukain"),
    "ISOLATE": (
        "isolate",
        "isolates",
        "isolating",
        "isolated",
        "isolasi",
        "mengisolasi",
        "diisolasi",
        "lockout",
        "lock-out",
    ),
    "START": (
        "start",
        "starts",
        "starting",
        "started",
        "nyalakan",
        "menyalakan",
        "dinyalakan",
        "hidupkan",
        "menghidupkan",
        "jalankan",
        "menjalankan",
    ),
    "STOP": ("stop", "stops", "stopping", "stopped", "matikan", "mematikan", "dimatikan", "hentikan", "menghentikan"),
}
# two-word actions, checked before single words ("lock out" = ISOLATE, "shut down" = STOP)
MULTIWORD_ACTIONS: dict[tuple[str, ...], str] = {
    ("lock", "out"): "ISOLATE",
    ("locking", "out"): "ISOLATE",
    ("locked", "out"): "ISOLATE",
    ("shut", "down"): "STOP",
    ("shutting", "down"): "STOP",
    ("switch", "off"): "STOP",
    ("switching", "off"): "STOP",
    ("switch", "on"): "START",
    ("switching", "on"): "START",
}
# bare imperative forms: the operator's repeat after a say-again opens with one of these
IMPERATIVE_ACTIONS: frozenset[str] = frozenset(
    {
        "close",
        "shut",
        "tutup",
        "tutupkan",
        "open",
        "buka",
        "bukakan",
        "isolate",
        "isolasi",
        "lock",
        "start",
        "nyalakan",
        "hidupkan",
        "jalankan",
        "stop",
        "matikan",
        "hentikan",
        "switch",
    }
)
ACTION_WORDS: dict[str, str] = {w: canon for canon, words in ACTIONS.items() for w in words}

STATES: dict[str, str] = {
    "running": "RUNNING",
    "online": "RUNNING",
    "jalan": "RUNNING",
    "beroperasi": "RUNNING",
    "offline": "STOPPED",
    "standby": "STANDBY",
    "siaga": "STANDBY",
}

# equipment nouns -> plant tag prefix (XV on/off valve, P pump, MOV motor-operated valve)
TAG_NOUNS: dict[str, str] = {
    "valve": "XV",
    "valves": "XV",
    "katup": "XV",
    "valf": "XV",
    "xv": "XV",
    "pump": "P",
    "pumps": "P",
    "pompa": "P",
    "p": "P",
    "mov": "MOV",
}
MULTIWORD_TAG_NOUNS: dict[tuple[str, ...], str] = {
    ("motor", "operated", "valve"): "MOV",
    ("motor", "valve"): "MOV",
    ("katup", "motor"): "MOV",
    ("valve", "motor"): "MOV",
    ("x", "ray", "victor"): "XV",
    ("x", "v"): "XV",
}
LINE_WORDS: frozenset[str] = frozenset({"line", "jalur", "lajur", "lane"})

NATO: dict[str, str] = {
    "alpha": "A",
    "alfa": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliet": "J",
    "juliett": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "whisky": "W",
    "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}
# Bahasa letter names — only ever read right after a line word ("jalur be")
BAHASA_LETTERS: dict[str, str] = {
    "a": "A",
    "be": "B",
    "bé": "B",
    "ce": "C",
    "se": "C",
    "de": "D",
    "dé": "D",
    "e": "E",
    "ef": "F",
    "ge": "G",
    "ha": "H",
}

# retraction cues inside one transmission. Never "say again": that is the technician asking for a repeat.
RETRACTION_CUES: frozenset[str] = frozenset({"correction", "koreksi", "ralat", "bukan", "no"})
# "no" / "bukan" are ordinary words too ("no leaks", "valve no 203", "bukan masalah"): the referee accepts them as a
# retraction cue only when punctuation follows ("line Alpha, no, line Bravo")
WEAK_RETRACTION_CUES: frozenset[str] = frozenset({"no", "bukan"})


def retraction_cues(tokens: list[Token]) -> list[Token]:
    out = []
    for i, t in enumerate(tokens):
        if t.norm not in RETRACTION_CUES:
            continue
        if t.norm in WEAK_RETRACTION_CUES and not (i + 1 < len(tokens) and tokens[i + 1].pre_punct):
            continue
        out.append(t)
    return out


# an operator turn that opens with one of these after a read-back is a correction of that read-back
OPERATOR_CORRECTION_CUES: frozenset[str] = frozenset(
    {"negative", "koreksi", "correction", "bukan", "salah", "ralat", "no", "tidak"}
)
# read-back openers (never imperative)
READBACK_CUES: frozenset[str] = frozenset(
    {
        "copy",
        "copied",
        "roger",
        "siap",
        "dimengerti",
        "wilco",
        "understood",
        "oke",
        "ok",
        "okay",
        "baik",
        "affirm",
        "affirmative",
    }
)
SAY_AGAIN_PHRASES: tuple[tuple[str, ...], ...] = (
    ("say", "again"),
    ("come", "again"),
    ("repeat",),
    ("ulangi",),
    ("ulang",),
    ("sekali", "lagi"),
)
# the operator's own repeat starts with these; they are not a say-again request
REPEAT_OPENERS: tuple[tuple[str, ...], ...] = (("i", "say", "again"), ("saya", "ulangi"), ("i", "repeat"))

# negators: a negated command ("do not open", "jangan tutup", "unable to close") is never read as the command
# A closed list: words outside it are a stated limitation, not a guess.
# fmt: off
NEGATORS: frozenset[str] = frozenset({
    # English
    "not", "never", "unable", "cannot", "dont", "wont", "cant", "didnt", "negative", "refuse", "refused", "refusing",
    "cancel", "cancelled", "canceled", "cancelling", "canceling", "avoid", "avoiding", "fail", "failed", "failing",
    "disregard", "abort", "aborted", "skip", "skipped", "ignore", "without", "neither", "nor", "none",
    # Bahasa Indonesia (incl. chat / radio short forms)
    "jangan", "tidak", "tak", "belum", "nggak", "ngga", "gak", "ga", "enggak", "engga", "ndak", "tdk", "gabisa",
    "gakbisa", "batal", "batalkan", "dibatalkan", "tunda", "tunggu",
})
NEGATOR_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("don", "t"), ("won", "t"), ("can", "t"), ("didn", "t"), ("isn", "t"), ("doesn", "t"), ("shouldn", "t"),
    ("couldn", "t"), ("wouldn", "t"), ("hasn", "t"), ("haven", "t"), ("hadn", "t"), ("aren", "t"), ("wasn", "t"),
    ("weren", "t"), ("mustn", "t"), ("needn", "t"), ("shan", "t"), ("ain", "t"), ("hold", "off"), ("call", "off"),
})
# fmt: on

# units after a number mean "a quantity", never a tag ("150 psi", "10 minutes")
UNIT_WORDS: frozenset[str] = frozenset(
    {
        "bar",
        "psi",
        "percent",
        "persen",
        "degree",
        "degrees",
        "derajat",
        "minute",
        "minutes",
        "menit",
        "second",
        "seconds",
        "detik",
        "hour",
        "hours",
        "jam",
        "rpm",
        "liter",
        "liters",
        "litre",
        "litres",
        "kg",
        "meter",
        "meters",
        "metres",
        "ton",
        "tons",
        "kilo",
        "celsius",
        "c",
        "am",
        "pm",
        "hrs",
        "kpa",
        "mpa",
    }
)

# ---------------------------------------------------------------------------------------------- numbers

EN_UNITS: dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "tree": 3,
    "four": 4,
    "five": 5,
    "fife": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "niner": 9,
}
EN_TEENS: dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
EN_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
ID_UNITS: dict[str, int] = {
    "nol": 0,
    "kosong": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
}
ID_SPECIAL: dict[str, int] = {"sepuluh": 10, "sebelas": 11}
HUNDRED_WORDS: frozenset[str] = frozenset({"hundred", "ratus"})
# homophones an ASR writes for digits; only read as digits inside a number run
HOMOPHONES: dict[str, int] = {"to": 2, "too": 2, "for": 4, "fore": 4}
ZERO_LIKE: frozenset[str] = frozenset({"oh", "o", "zero", "nol", "kosong"})

_TOKEN_RE = re.compile(r"\d+[:.]\d+|\d+(?:[A-Za-z](?![A-Za-z]))?|[^\W\d_]+", re.UNICODE)
_SUFFIX_RE = re.compile(r"^(\d{3}|\d{2}[A-Z])$")


@dataclass(frozen=True)
class Token:
    text: str  # as written in the transcript
    norm: str  # lower-cased
    start: int  # char offsets into the transcript
    end: int
    pre_punct: bool = False  # a comma / full stop / … (not a hyphen) separates it from the previous token


def tokenize(text: str) -> list[Token]:
    """Word / number tokens with char offsets. Hyphens and punctuation separate tokens ("two-oh-three" -> 3)."""
    out: list[Token] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(text):
        gap = text[prev_end : m.start()]
        out.append(Token(m.group(0), m.group(0).lower(), m.start(), m.end(), bool(_PUNCT.search(gap)) and bool(out)))
        prev_end = m.end()
    return out


_PUNCT = re.compile(r"[,.;:!?…—–()]")


def _is_digits(tok: Token) -> bool:
    return tok.norm[:1].isdigit() and ":" not in tok.norm and "." not in tok.norm


def _is_time(tok: Token) -> bool:
    """Clock-shaped: H:MM / H.MM only ("2:03", "14.30"); "20.3" or "203.5" is a decimal, never a tag."""
    return bool(re.fullmatch(r"\d{1,2}[:.]\d{2}", tok.norm))


def _number_word(tokens: list[Token], i: int, in_run: bool) -> bool:
    """Is tokens[i] part of a spoken number, given whether a number run is already open?"""
    w = tokens[i].norm
    if w in EN_UNITS:
        # "oh"/"o" are digits only inside a run ("two-oh-three"), never as an interjection ("oh, copy")
        return in_run if w in ("oh", "o") else True
    if w in EN_TEENS or w in EN_TENS or w in ID_UNITS or w in ID_SPECIAL or w == "seratus":
        return True
    if w in HUNDRED_WORDS or w in ("puluh", "belas"):
        return in_run
    if w == "and":
        return in_run and i > 0 and tokens[i - 1].norm == "hundred"
    if w in HOMOPHONES:
        # a homophone is a digit only between number words: "to oh three", "one for zero" — never the trailing
        # "for" of "two-oh-three for line Bravo"
        nxt_tok = tokens[i + 1] if i + 1 < len(tokens) else None
        nxt_is_number = nxt_tok is not None and not nxt_tok.pre_punct and _plain_number_word(nxt_tok.norm)
        if in_run:
            return nxt_is_number
        return w in ("to", "too") and nxt_is_number and nxt_tok is not None and nxt_tok.norm in ZERO_LIKE
    return False


def _plain_number_word(w: str) -> bool:
    return w in EN_UNITS or w in EN_TEENS or w in EN_TENS or w in ID_UNITS or w in ID_SPECIAL or w[:1].isdigit()


def _compose(words: list[str]) -> str | None:
    """Spoken number words -> digit string. None when the run is malformed (e.g. 'hundred' with no base)."""
    out: list[str] = []
    i, n = 0, len(words)

    def unit(w: str) -> int | None:
        if w in EN_UNITS:
            return EN_UNITS[w]
        if w in ID_UNITS:
            return ID_UNITS[w]
        if w in HOMOPHONES:
            return HOMOPHONES[w]
        return None

    def two_digit(j: int) -> tuple[int | None, int]:
        """Value 0-99 starting at j (after 'hundred'/'ratus'), and the next index."""
        if j < n and words[j] == "and":
            j += 1
        if j >= n:
            return None, j
        w = words[j]
        if w in EN_TENS:
            if j + 1 < n and unit(words[j + 1]) is not None:
                return EN_TENS[w] + unit(words[j + 1]), j + 2  # type: ignore[operator]
            return EN_TENS[w], j + 1
        if w in EN_TEENS:
            return EN_TEENS[w], j + 1
        if w in ID_SPECIAL:
            return ID_SPECIAL[w], j + 1
        u = unit(w)
        if u is not None:
            if j + 1 < n and words[j + 1] == "puluh":
                v = u * 10
                unit_next = j + 2 < n and unit(words[j + 2]) is not None
                if unit_next and (j + 3 >= n or words[j + 3] not in ("puluh", "belas")):
                    return v + unit(words[j + 2]), j + 3  # type: ignore[operator]
                return v, j + 2
            if j + 1 < n and words[j + 1] == "belas":
                return 10 + u, j + 2
            return u, j + 1
        return None, j

    while i < n:
        w = words[i]
        if w[:1].isdigit():
            out.append(w.upper())
            i += 1
            continue
        if w == "seratus":
            v, i = two_digit(i + 1)
            out.append(f"1{v:02d}" if v is not None else "100")
            continue
        if w in ID_SPECIAL:
            out.append(str(ID_SPECIAL[w]))
            i += 1
            continue
        if w in EN_TEENS:
            out.append(str(EN_TEENS[w]))
            i += 1
            continue
        if w in EN_TENS:
            if i + 1 < n and unit(words[i + 1]) is not None and words[i + 1] not in ZERO_LIKE:
                out.append(str(EN_TENS[w] + unit(words[i + 1])))  # type: ignore[operator]
                i += 2
            else:
                out.append(str(EN_TENS[w]))
                i += 1
            continue
        u = unit(w)
        if u is None:
            return None  # a multiplier with no base ("hundred", "puluh" first)
        nxt = words[i + 1] if i + 1 < n else ""
        if nxt in HUNDRED_WORDS:
            v, i = two_digit(i + 2)
            out.append(f"{u}{v:02d}" if v is not None else f"{u}00")
            continue
        if nxt == "puluh":
            v, i = two_digit(i)
            out.append(f"{v:02d}")
            continue
        if nxt == "belas":
            out.append(str(10 + u))
            i += 2
            continue
        out.append(str(u))
        i += 1
    return "".join(out) if out else None


@dataclass(frozen=True)
class NumberRun:
    start_idx: int  # token index range [start_idx, end_idx)
    end_idx: int
    digits: str  # e.g. "203", "31A"
    time_format: bool  # came from "2:03" / "2.03"


def number_runs(tokens: list[Token]) -> list[NumberRun]:
    """Maximal spoken/written number runs, with an optional trailing letter for 2-digit suffixes ('31 alpha')."""
    runs: list[NumberRun] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_time(tok):
            runs.append(NumberRun(i, i + 1, re.sub(r"[:.]", "", tok.norm), True))
            i += 1
            continue
        if _is_digits(tok) or _number_word(tokens, i, in_run=False):
            j = i
            words: list[str] = []
            # a run never crosses punctuation: "two-oh-three, two minutes" is a tag, then a quantity
            while (
                j < len(tokens)
                and (j == i or not tokens[j].pre_punct)
                and (_is_digits(tokens[j]) or _number_word(tokens, j, in_run=bool(words)))
            ):
                words.append(tokens[j].norm)
                j += 1
            digits = _compose(words)
            if digits is None:
                i = j
                continue
            # "thirty-one alpha" / "3 1 A": a letter right after a 2-digit run completes an MOV-style suffix
            if (
                re.fullmatch(r"\d{2}", digits)
                and j < len(tokens)
                and not tokens[j].pre_punct
                and _letter_word(tokens[j]) is not None
            ):
                digits += _letter_word(tokens[j])  # type: ignore[operator]
                j += 1
            runs.append(NumberRun(i, j, digits, False))
            i = j
            continue
        i += 1
    return runs


def _letter_word(tok: Token) -> str | None:
    """A letter completing a 2-digit suffix: a NATO word, or a single letter (lower-case a/i/o are words)."""
    w = tok.norm
    if w in NATO and w != "lima":
        return NATO[w]
    if len(tok.text) == 1 and tok.text.isalpha() and (tok.text.isupper() or w not in ("a", "i", "o")):
        return w.upper()
    return None


def line_letter(w: str, after_line_word: bool) -> str | None:
    """A line letter: NATO word, a single letter, or (only after a line word) a Bahasa letter name."""
    if w in NATO:
        return NATO[w]
    if len(w) == 1 and w.isalpha():
        return w.upper()
    if after_line_word and w in BAHASA_LETTERS:
        return BAHASA_LETTERS[w]
    return None


# --------------------------------------------------------------------------------------------- mentions


@dataclass(frozen=True)
class Mention:
    field: str  # "action" | "tag" | "line" | "state"
    start: int  # char offsets
    end: int
    value: str  # canonical: "CLOSE" | "XV-203" or "203" (bare) | "B" | "RUNNING"
    prefix: str | None = None  # tag only: "XV" / "P" / "MOV", None for a bare number
    suffix: str | None = None  # tag only


def _match_multi(tokens: list[Token], i: int, table: dict[tuple[str, ...], str]) -> tuple[str, int] | None:
    for words, canon in sorted(table.items(), key=lambda kv: -len(kv[0])):
        k = len(words)
        if tuple(t.norm for t in tokens[i : i + k]) == words:
            return canon, k
    return None


def action_mentions(tokens: list[Token]) -> list[Mention]:
    out: list[Mention] = []
    i = 0
    while i < len(tokens):
        mw = _match_multi(tokens, i, MULTIWORD_ACTIONS)
        if mw:
            canon, k = mw
            out.append(Mention("action", tokens[i].start, tokens[i + k - 1].end, canon))
            i += k
            continue
        if tokens[i].norm in ACTION_WORDS:
            out.append(Mention("action", tokens[i].start, tokens[i].end, ACTION_WORDS[tokens[i].norm]))
        i += 1
    return out


def _noun_before(tokens: list[Token], idx: int) -> tuple[str, int] | None:
    """Tag noun ending right before token idx -> (prefix, start token index)."""
    for words, prefix in sorted(MULTIWORD_TAG_NOUNS.items(), key=lambda kv: -len(kv[0])):
        k = len(words)
        if idx - k >= 0 and tuple(t.norm for t in tokens[idx - k : idx]) == words:
            return prefix, idx - k
    if idx - 1 >= 0 and tokens[idx - 1].norm in TAG_NOUNS:
        return TAG_NOUNS[tokens[idx - 1].norm], idx - 1
    # "valve no. 203" / "pompa nomor satu empat nol": a number word between noun and number
    if idx - 2 >= 0 and tokens[idx - 1].norm in NUMBER_FILLERS:
        inner = _noun_before(tokens, idx - 1)
        if inner is not None:
            return inner
    return None


NUMBER_FILLERS = frozenset({"no", "nomor", "number", "nr", "num"})


def _action_before(tokens: list[Token], idx: int) -> bool:
    if idx - 1 < 0:
        return False
    w = tokens[idx - 1].norm
    if w in ACTION_WORDS:
        return True
    return idx - 2 >= 0 and (tokens[idx - 2].norm, w) in MULTIWORD_ACTIONS


def is_suffix_format(digits: str) -> bool:
    """Registry suffix format: 3 digits, or 2 digits + a letter."""
    return bool(_SUFFIX_RE.match(digits))


def tag_mentions(tokens: list[Token], *, explicit: bool = False) -> list[Mention]:
    """Every token run that names a plant tag.

    A number counts as a tag when (a) a tag noun precedes it, or (b) it has registry suffix format and is not a
    quantity ("150 psi"). Time-shaped strings ("2:03") count only right after a tag noun or an action verb.
    `explicit=True` is used when the text is already known to be a tag span: any number run then counts.
    """
    out: list[Mention] = []
    for run in number_runs(tokens):
        noun = _noun_before(tokens, run.start_idx)
        nxt = tokens[run.end_idx].norm if run.end_idx < len(tokens) else ""
        start_tok = noun[1] if noun else run.start_idx
        span = (tokens[start_tok].start, tokens[run.end_idx - 1].end)
        if noun:
            prefix = noun[0]
            out.append(Mention("tag", *span, f"{prefix}-{run.digits}", prefix, run.digits))
            continue
        if run.time_format:
            if _action_before(tokens, run.start_idx) or explicit:
                out.append(Mention("tag", *span, run.digits, None, run.digits))
            continue
        if explicit or (is_suffix_format(run.digits) and nxt not in UNIT_WORDS):
            out.append(Mention("tag", *span, run.digits, None, run.digits))
    # a prefix spoken as letters right before a number is already covered by MULTIWORD_TAG_NOUNS ("x ray victor")
    return out


def line_mentions(tokens: list[Token], *, explicit: bool = False) -> list[Mention]:
    """'line Bravo', 'jalur be', 'line B'. With explicit=True a lone letter word ('Bravo') also counts."""
    out: list[Mention] = []
    for i, tok in enumerate(tokens):
        if tok.norm in LINE_WORDS and i + 1 < len(tokens):
            letter = line_letter(tokens[i + 1].norm, after_line_word=True)
            if letter:
                out.append(Mention("line", tok.start, tokens[i + 1].end, letter))
    if explicit and not out:
        for tok in tokens:
            letter = line_letter(tok.norm, after_line_word=True)
            if letter and tok.norm not in ("a",):
                out.append(Mention("line", tok.start, tok.end, letter))
    return out


def state_mentions(tokens: list[Token]) -> list[Mention]:
    return [Mention("state", t.start, t.end, STATES[t.norm]) for t in tokens if t.norm in STATES]


def bare_letter_mentions(tokens: list[Token]) -> list[Mention]:
    """NATO letters not right after a line word ("…two-oh-three, Alpha."), and not part of a tag ("thirty-one
    Alpha"). Only the coverage scan uses these: a stray letter in an exchange turn must be accounted for."""
    in_tags = [(m.start, m.end) for m in tag_mentions(tokens)]
    out = []
    for i, t in enumerate(tokens):
        if t.norm in NATO and t.norm != "lima" and not (i > 0 and tokens[i - 1].norm in LINE_WORDS):
            if not any(s <= t.start and t.end <= e for s, e in in_tags):
                out.append(Mention("line", t.start, t.end, NATO[t.norm]))
    return out


def negation_mentions(tokens: list[Token], text: str = "") -> list[Mention]:
    """Negators anywhere in the turn — plus "no" right before a verb ("no closing").

    One exemption: an operator-correction cue that opens the turn, followed by punctuation and then an imperative
    ("Tidak, close valve …", "Negative, close valve …"), or alone ("Negative."). "Negative, closing …" or
    "Tidak menutup …" are negations: the referee never reads them as the command.
    """
    out = []
    for i, t in enumerate(tokens):
        if i == 0 and t.norm in OPERATOR_CORRECTION_CUES and _cue_then_imperative(tokens, text):
            continue
        if t.norm in NEGATORS:
            out.append(Mention("negation", t.start, t.end, t.norm))
        elif t.norm == "no" and i + 1 < len(tokens) and tokens[i + 1].norm in ACTION_WORDS:
            out.append(Mention("negation", t.start, tokens[i + 1].end, "no"))
        elif i + 1 < len(tokens) and (t.norm, tokens[i + 1].norm) in NEGATOR_PAIRS:
            out.append(Mention("negation", t.start, tokens[i + 1].end, f"{t.norm} {tokens[i + 1].norm}"))
    return out


def _cue_then_imperative(tokens: list[Token], text: str) -> bool:
    if len(tokens) < 2:
        return True  # "Negative." alone: a bare correction cue, nothing to negate
    gap = text[tokens[0].end : tokens[1].start] if text else ""
    if not any(ch in gap for ch in ",.;:!—-"):
        return False
    nxt = tokens[1].norm
    return nxt in IMPERATIVE_ACTIONS or nxt in TAG_NOUNS or nxt in ("i", "saya", "tolong", "please")


# "I'll confirm when closed", "report closed", "lapor kalau sudah ditutup": an action word that reports back, not one
# that commands (not a bare imperative, right after one of these words)
REPORT_BACK_WORDS = frozenset(
    {
        "when",
        "once",
        "report",
        "confirm",
        "after",
        "until",
        "kalau",
        "setelah",
        "sesudah",
        "lapor",
        "konfirmasi",
        "sudah",
    }
)


def report_back(tokens: list[Token], m: Mention) -> bool:
    """A past participle / Bahasa di- passive right after a report-back word, with no tag after it in its clause."""
    idx = next((i for i, t in enumerate(tokens) if t.start == m.start), None)
    if idx is None:
        return False
    w = tokens[idx].norm
    if not (w.endswith("ed") or w.startswith("di")):
        return False  # "once done, opening valve 208" is a command, not a report-back
    if not any(t.norm in REPORT_BACK_WORDS for t in tokens[max(0, idx - 2) : idx]):
        return False
    clause_end = next((j for j in range(idx + 1, len(tokens)) if tokens[j].pre_punct), len(tokens))
    tags_after = [t for t in tag_mentions(tokens) if tokens[idx].end <= t.start < (tokens[clause_end - 1].end + 1)]
    return not tags_after


def all_mentions(text: str) -> list[Mention]:
    """Every safety-relevant mention in a transcript, sorted by position (the coverage scan)."""
    toks = tokenize(text)
    ms = action_mentions(toks) + tag_mentions(toks) + line_mentions(toks) + bare_letter_mentions(toks)
    return sorted(ms, key=lambda m: (m.start, m.end))


# --------------------------------------------------------------------------------------- canonicalising

# words a tag span may carry around the tag itself ("the valve number two-oh-three")
TAG_FILLERS: frozenset[str] = frozenset({"the", "number", "nomor", "no", "tag", "unit"})


@dataclass(frozen=True)
class Unmapped:
    """A span the alias tables cannot map. The referee abstains on it: `unmapped: <word>`."""

    word: str


def canon_action(span: str) -> str | Unmapped:
    toks = tokenize(span)
    found = {m.value for m in action_mentions(toks)}
    if len(found) == 1:
        return found.pop()
    return Unmapped(span.strip() or "<empty>")


def canon_tag(span: str, *, preceding_word: str | None = None) -> Mention | Unmapped:
    """A tag span -> one tag mention (prefix is None for a bare number).

    `preceding_word` is the transcript word right before the span. A time-shaped number ("2:03") is a tag only
    next to a tag noun (then it has a prefix) or an action verb, inside the span or right before it.
    """
    toks = tokenize(span)
    ms = tag_mentions(toks, explicit=True)
    if len(ms) != 1:
        return Unmapped(span.strip() or "<empty>")
    m = ms[0]
    stray = [t for t in toks if (t.end <= m.start or t.start >= m.end) and t.norm not in TAG_FILLERS]
    if stray:
        return Unmapped(stray[0].text)  # words the tables cannot place: never guess around them
    time_idx = next((i for i, t in enumerate(toks) if _is_time(t)), None)
    if m.prefix is None and time_idx is not None:
        verb_inside = _action_before(toks, time_idx)
        verb_before = time_idx == 0 and preceding_word is not None and _ends_with_action(preceding_word)
        if not (verb_inside or verb_before):
            return Unmapped(span.strip())
    return m


def _ends_with_action(text: str) -> bool:
    toks = tokenize(text)
    return bool(toks) and _action_before(toks, len(toks))


def canon_line(span: str) -> str | Unmapped:
    ms = line_mentions(tokenize(span), explicit=True)
    vals = {m.value for m in ms}
    if len(vals) == 1:
        return vals.pop()
    return Unmapped(span.strip() or "<empty>")


def canon_state(span: str) -> str | Unmapped:
    vals = {m.value for m in state_mentions(tokenize(span))}
    if len(vals) == 1:
        return vals.pop()
    return Unmapped(span.strip() or "<empty>")


def starts_with(tokens: list[Token], phrase: tuple[str, ...]) -> bool:
    return tuple(t.norm for t in tokens[: len(phrase)]) == phrase


def contains_phrase(tokens: list[Token], phrase: tuple[str, ...]) -> bool:
    norms = [t.norm for t in tokens]
    k = len(phrase)
    return any(tuple(norms[i : i + k]) == phrase for i in range(len(norms) - k + 1))
