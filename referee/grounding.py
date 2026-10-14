"""Grounding: every span the extractor returns must be words that are really in the transcript.

Spans are aligned in order and without overlap: command k's spans all sit after command k-1's first span, and no two
spans share words (exception: several commands may point at the same action or line words, "close 203 and 216, line
Bravo").
A repeated verb ("close … and close …") therefore aligns to two different words. When no consistent assignment exists
the turn is ungrounded and the referee abstains. Matching ignores case, punctuation and hyphens, and only matches whole
words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from referee.schema import FIELDS

Span = tuple[int, int]  # char offsets into the original transcript
# a verb or a line can distribute over a conjunction ("close 203 and 216, line Bravo"): two commands may point at the
# same words for these fields
SHAREABLE = frozenset({"action_span", "line_span"})
_WORD = re.compile(r"[^\W_]+(?:[:.]\d+)?", re.UNICODE)


@dataclass
class Grounded:
    commands: list[dict[str, Span | None]] = field(default_factory=list)  # field name -> offsets
    retracted: list[Span] = field(default_factory=list)


class GroundingError(Exception):
    """No consistent alignment of the extractor's spans to the transcript."""


def _words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def occurrences(transcript: str, span: str) -> list[Span]:
    """All whole-word matches of `span` in `transcript` (case / punctuation / hyphen insensitive)."""
    tw = _words(transcript)
    sw = [w for w, _, _ in _words(span)]
    if not sw:
        return []
    k = len(sw)
    return [(tw[i][1], tw[i + k - 1][2]) for i in range(len(tw) - k + 1) if [w for w, _, _ in tw[i : i + k]] == sw]


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def ground(transcript: str, extraction: dict) -> Grounded:
    # retracted words first, each at its earliest occurrence: a retraction always precedes the words that replace it
    # ("line Bravo, correction, line Bravo" must not hand the operative mention to the retraction)
    retracted: list[Span] = []
    for s in extraction.get("retracted_spans", []):
        occ = [sp for sp in occurrences(transcript, s) if not any(_overlaps(sp, t) for t in retracted)]
        if not occ:
            raise GroundingError(f"retracted span {s!r} is not in the transcript")
        retracted.append(occ[0])

    cands: list[list[tuple[str, list[Span]]]] = []
    for c in extraction["commands"]:
        fields = []
        for f in FIELDS:
            s = c.get(f)
            if s is None:
                continue
            occ = occurrences(transcript, s)
            if not occ:
                raise GroundingError(f"{f} {s!r} is not in the transcript")
            fields.append((f, occ))
        if not fields:
            raise GroundingError("a command with no spans")
        cands.append(fields)

    used: list[tuple[str, Span]] = [("retracted", r) for r in retracted]
    placed: list[dict[str, Span]] = []

    def free(f: str, sp: Span) -> bool:
        for uf, u in used:
            if _overlaps(sp, u) and not (f == uf and f in SHAREABLE and sp == u):
                return False
        return True

    def in_order(chosen: dict[str, Span]) -> bool:
        """Every field keeps the order of the commands: command k's verb / line / state come after command k-1's
        (or are the same shared words). A model cannot swap two commands' verbs or lines and still ground."""
        if not placed:
            return True
        prev = placed[-1]
        key = chosen["tag_span"][0] if "tag_span" in chosen else min(sp[0] for sp in chosen.values())
        prev_key = prev["tag_span"][0] if "tag_span" in prev else min(sp[0] for sp in prev.values())
        if key <= prev_key:
            return False
        for f, sp in chosen.items():
            p = prev.get(f)
            if p is not None and not (sp[0] > p[0] or (sp == p and f in SHAREABLE)):
                return False
        return True

    def assign(k: int) -> bool:
        if k == len(cands):
            return True
        fields = cands[k]

        def rec(j: int, chosen: dict[str, Span]) -> bool:
            if j == len(fields):
                if not in_order(chosen):
                    return False
                placed.append(dict(chosen))
                if assign(k + 1):
                    return True
                placed.pop()
                return False
            f, occ = fields[j]
            # fresh words first; pointing at words another command already uses is the fallback
            for sp in sorted(occ, key=lambda o: (any(o == u for _, u in used), o[0])):
                if not free(f, sp):
                    continue
                used.append((f, sp))
                chosen[f] = sp
                ok = rec(j + 1, chosen)
                used.pop()
                del chosen[f]
                if ok:
                    return True
            return False

        return rec(0, {})

    if not assign(0):
        raise GroundingError("no ordered, non-overlapping alignment of the command spans")
    commands: list[dict[str, Span | None]] = []
    for chosen in placed:
        full: dict[str, Span | None] = {f: None for f in FIELDS}
        full.update(chosen)
        commands.append(full)
    return Grounded(commands=commands, retracted=retracted)
