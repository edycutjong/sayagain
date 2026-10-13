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
    commands = extraction["commands"]
    # candidate occurrences per (command, field)
    cands: list[list[tuple[str, list[Span]]]] = []
    for c in commands:
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

    used: list[tuple[str, Span]] = []

    def free(f: str, sp: Span) -> bool:
        for uf, u in used:
            if _overlaps(sp, u) and not (f == uf and f in SHAREABLE and sp == u):
                return False
        return True

    def assign_command(k: int, after: int) -> list[dict[str, Span | None]] | None:
        if k == len(cands):
            return []
        fields = cands[k]

        def rec(j: int, chosen: dict[str, Span]) -> list[dict[str, Span | None]] | None:
            if j == len(fields):
                # commands are ordered by their tag (else their first span)
                first = chosen["tag_span"][0] if "tag_span" in chosen else min(sp[0] for sp in chosen.values())
                if first <= after and k > 0:
                    return None
                rest = assign_command(k + 1, first)
                if rest is None:
                    return None
                full: dict[str, Span | None] = {f: None for f in FIELDS}
                full.update(chosen)
                return [full, *rest]
            f, occ = fields[j]
            # fresh words first; pointing at words another command already uses is the fallback
            for sp in sorted(occ, key=lambda o: (any(o == u for _, u in used), o[0])):
                if not free(f, sp):
                    continue
                used.append((f, sp))
                chosen[f] = sp
                out = rec(j + 1, chosen)
                used.pop()
                del chosen[f]
                if out is not None:
                    return out
            return None

        # keep this command's spans in `used` while later commands are placed
        return rec(0, {})

    result = assign_command(0, -1)
    if result is None:
        raise GroundingError("no ordered, non-overlapping alignment of the command spans")

    taken = [sp for cmd in result for sp in cmd.values() if sp is not None]
    retracted: list[Span] = []
    for s in extraction.get("retracted_spans", []):
        occ = [sp for sp in occurrences(transcript, s) if not any(_overlaps(sp, t) for t in taken + retracted)]
        if not occ:
            raise GroundingError(f"retracted span {s!r} is not in the transcript")
        retracted.append(occ[0])
    return Grounded(commands=result, retracted=retracted)
