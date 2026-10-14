"""Coverage: every safety word in the transcript must be accounted for.

Grounding catches invented words; coverage catches the model pointing at the *wrong* real words. Code scans the whole
transcript with the alias tables (`aliases.all_mentions`: actions, tags, line letters after a line word) and every
mention must overlap a command span or a retraction that code accepted. A retraction is accepted only if a cue from the
closed list ("correction", "koreksi", "ralat", "bukan", "no") sits between the retracted words and the operative words
that replace them. A turn with an action and a negator ("do not open", "jangan tutup", "unable to close") is never
read as the command: the referee abstains.
"""

from __future__ import annotations

from referee import aliases as al
from referee.grounding import Grounded, Span


class CoverageError(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason  # "coverage" | "retraction cue missing"
        self.detail = detail


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _field_of(transcript: str, sp: Span) -> str | None:
    for m in al.all_mentions(transcript):
        if _overlaps((m.start, m.end), sp):
            return m.field
    return None


_SPAN_FIELD = {"action_span": "action", "tag_span": "tag", "line_span": "line", "state_span": "state"}


def check(transcript: str, g: Grounded) -> None:
    """Raise CoverageError when a retraction lacks its cue or a safety mention is not accounted for."""
    toks = al.tokenize(transcript)
    command_spans = [(f, sp) for cmd in g.commands for f, sp in cmd.items() if sp is not None]

    for r in g.retracted:
        rfield = _field_of(transcript, r)
        later = sorted((sp for f, sp in command_spans if sp[0] >= r[1]), key=lambda s: s[0])
        same = sorted((sp for f, sp in command_spans if sp[0] >= r[1] and _SPAN_FIELD[f] == rfield), key=lambda s: s[0])
        operative = same[0] if same else (later[0] if later else None)
        if operative is None:
            raise CoverageError("retraction cue missing", f"nothing replaces the retracted {transcript[r[0] : r[1]]!r}")
        if not any(t.norm in al.RETRACTION_CUES and t.start >= r[1] and t.end <= operative[0] for t in toks):
            raise CoverageError(
                "retraction cue missing",
                f"no correction word between {transcript[r[0] : r[1]]!r} and "
                f"{transcript[operative[0] : operative[1]]!r}",
            )

    if al.action_mentions(toks):
        for m in al.negation_mentions(toks):
            raise CoverageError(
                "coverage", f"negated command: {transcript[m.start : m.end]!r} — never read as the command"
            )

    accounted = [sp for _, sp in command_spans] + list(g.retracted)
    for m in al.all_mentions(transcript):
        if not any(_overlaps((m.start, m.end), sp) for sp in accounted):
            raise CoverageError("coverage", f"{m.field} {transcript[m.start : m.end]!r} is not covered by any span")
