"""One radio turn -> grounded, covered, canonical commands (or the reason it cannot be read).

Order per turn: schema check -> grounding -> coverage (+ retraction cues) -> aliases -> registry resolution.
Anything that fails here makes the turn unreadable: an exchange that contains it abstains, it never verifies.
The turn kinds that steer pairing (say-again, operator correction, imperative repeat) are decided by code from the
alias tables, never taken from the model's `kind`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from referee import aliases as al
from referee import coverage, grounding
from referee.regex_extractor import extract as regex_extract
from referee.registry import Registry
from referee.schema import validate_extraction
from referee.verdict import R_COVERAGE, R_GROUNDING, CanonCommand, r_unmapped

# fillers that may precede the verb of an operator's repeat ("tolong tutup …", "operator, close …")
_LEAD_FILLERS = frozenset({"tolong", "please", "operator", "control", "over", "uh", "um", "eh", "okay", "ok"})


@dataclass
class Turn:
    id: str
    channel: str
    t_start: float
    t_end: float
    transcript: str
    extraction: dict | None = None
    # (char start, char end, log-prob) per ASR token, aligned to `transcript`; None when the ASR route has none
    tokens: list[tuple[int, int, float]] | None = None
    alt_transcript: str | None = None  # unprompted ASR pass, for the disagreement gate
    not_audited: str | None = None  # "rate limited" / "timeout" / … when a model call failed
    meta: dict = field(default_factory=dict)


@dataclass
class TurnAnalysis:
    turn: Turn
    status: str  # ok | abstain | not_audited
    reason: str | None = None
    detail: str = ""
    commands: list[CanonCommand] = field(default_factory=list)
    spans: list[dict[str, grounding.Span | None]] = field(default_factory=list)
    retracted: list[grounding.Span] = field(default_factory=list)
    say_again: bool = False
    operator_correction: bool = False
    imperative: bool = False
    readback_shaped: bool = False  # opens with copy / roger / siap …: never opens a window of its own
    stand_by: bool = False  # "stand by" / "wait one" / "tunggu" with no command: the reply is still coming
    query: bool = False  # "… do you copy?" / "how copy" — the operator asking for the read-back
    opens_window: bool = False
    disagrees: bool = False  # prompted vs unprompted ASR read differently (only with an unprompted pass)

    @property
    def id(self) -> str:
        return self.turn.id


# ------------------------------------------------------------------------------------------ code kinds


def _strip_openers(toks: list[al.Token]) -> list[al.Token]:
    changed = True
    while changed and toks:
        changed = False
        for p in al.REPEAT_OPENERS:
            if al.starts_with(toks, p):
                toks, changed = toks[len(p) :], True
        if toks and (toks[0].norm in _LEAD_FILLERS or toks[0].norm in al.OPERATOR_CORRECTION_CUES):
            toks, changed = toks[1:], True
    return toks


def is_say_again(text: str) -> bool:
    """The technician asking for a repeat: a say-again phrase, no tag, not the operator's 'I say again …'."""
    toks = al.tokenize(text)
    if any(al.starts_with(toks, p) for p in al.REPEAT_OPENERS):
        return False
    if al.tag_mentions(toks):
        return False
    return any(al.contains_phrase(toks, p) for p in al.SAY_AGAIN_PHRASES)


def opens_with_correction(text: str) -> bool:
    """An operator turn starting 'negative / koreksi / bukan / correction …' right after a read-back."""
    toks = al.tokenize(text)
    return bool(toks) and toks[0].norm in al.OPERATOR_CORRECTION_CUES


def cue_only(text: str) -> bool:
    """'Negative.' / 'Koreksi, salah.' — nothing but correction cues and fillers ('No, hold on, stand by' is not)."""
    toks = al.tokenize(text)
    return bool(toks) and all(t.norm in al.OPERATOR_CORRECTION_CUES or t.norm in _LEAD_FILLERS for t in toks)


def is_stand_by(text: str) -> bool:
    """No command, and either a stand-by phrase or the operator prompting ("Field two, do you copy?")."""
    toks = al.tokenize(text)
    if al.tag_mentions(toks) or al.action_mentions(toks):
        return False
    return any(al.contains_phrase(toks, p) for p in _STAND_BY) or has_query(text)


_QUERY = (
    ("do", "you", "copy"),
    ("how", "copy"),
    ("how", "do", "you", "read"),
    ("come", "in"),
    ("dengar",),
    ("copy", "ga"),
    ("copy", "nggak"),
    ("konfirmasi",),
    ("acknowledge",),
)


def has_query(text: str) -> bool:
    toks = al.tokenize(text)
    return any(al.contains_phrase(toks, p) for p in _QUERY)


_STAND_BY = (("stand", "by"), ("standby",), ("wait", "one"), ("wait",), ("tunggu",), ("sebentar",), ("hold", "on"))


def readback_shaped(text: str) -> bool:
    """Opens with copy / roger / siap … and does NOT continue with an order ("Okay, close valve …" is an order)."""
    toks = al.tokenize(text)
    return bool(toks) and toks[0].norm in al.READBACK_CUES and not _imperative_core(_strip_openers(toks[1:]))


# verb forms that can carry an order: bare imperatives, plus the Bahasa passive used as a polite order
# ("tolong valve dua nol tiga ditutup") — unless a status marker says it already happened
_ORDER_PASSIVES = frozenset({"ditutup", "dibuka", "dimatikan", "dinyalakan", "diisolasi", "dihidupkan", "dijalankan"})
_STATUS_MARKERS = frozenset(
    {"sudah", "udah", "telah", "sedang", "already", "was", "were", "has", "have", "had", "been"}
)


def has_order_verb(text: str) -> bool:
    """An action word in a form that can give an order. 'I closed …', 'pump … stopped', 'valve … sudah ditutup'
    are status talk: they never open a pairing window."""
    toks = al.tokenize(text)
    for i, t in enumerate(toks):
        if t.norm in al.IMPERATIVE_ACTIONS or t.norm in _ORDER_PASSIVES:
            if not any(p.norm in _STATUS_MARKERS for p in toks[max(0, i - 2) : i]):
                return True
    return False


def canonical_tuples(text: str, registry: Registry) -> list[tuple[str, str, str]]:
    """(action, tag, line) per command as the frozen regex + alias extractor reads it, tags resolved through the
    registry; unreadable fields keep their words so two different unreadable readings still disagree."""
    out = []
    for c in regex_extract(text)["commands"]:
        act = al.canon_action(c["action_span"]) if c["action_span"] else None
        tag = al.canon_tag(c["tag_span"]) if c["tag_span"] else None
        line = al.canon_line(c["line_span"]) if c["line_span"] else None
        tag_s = registry.resolve(tag.prefix, tag.suffix or "").tag if isinstance(tag, al.Mention) else tag
        out.append(
            tuple(
                "-" if v is None else (f"?{v.word}" if isinstance(v, al.Unmapped) else str(v))
                for v in (act, tag_s, line)
            )
        )
    return out  # type: ignore[return-value]


def is_imperative(text: str) -> bool:
    """Opens with a bare imperative verb (after 'I say again', 'negative', 'tolong' …) — the operator's repeat.

    Bahasa order ("valve dua nol tiga tutup") counts when a tag noun opens and an imperative verb follows.
    Read-back openers ('copy', 'siap') and progressive / past forms ('closing') are not imperative.
    """
    toks = _strip_openers(al.tokenize(text))
    if not toks or toks[0].norm in al.READBACK_CUES:
        return False
    return _imperative_core(toks)


def _imperative_core(toks: list[al.Token]) -> bool:
    if not toks:
        return False
    if toks[0].norm in al.IMPERATIVE_ACTIONS:
        return True
    if toks[0].norm in al.TAG_NOUNS or toks[0].norm == "motor":
        return any(t.norm in al.IMPERATIVE_ACTIONS or t.norm in _ORDER_PASSIVES for t in toks[1:])
    return False


# ------------------------------------------------------------------------------------------- analysis


def _prev_word(text: str, start: int) -> str | None:
    toks = [t for t in al.tokenize(text[:start])]
    return toks[-1].text if toks else None


def analyze(turn: Turn, registry: Registry) -> TurnAnalysis:
    a = TurnAnalysis(turn=turn, status="ok")
    text = turn.transcript or ""
    a.say_again = is_say_again(text)
    correction_cue = opens_with_correction(text)
    a.imperative = is_imperative(text)
    a.readback_shaped = readback_shaped(text)
    a.query = has_query(text)
    a.stand_by = is_stand_by(text)
    if turn.alt_transcript is not None:
        a.disagrees = canonical_tuples(text, registry) != canonical_tuples(turn.alt_transcript, registry)
    if turn.not_audited:
        a.status, a.reason = "not_audited", turn.not_audited
        return a
    if turn.extraction is None:
        a.status, a.reason = "not_audited", "no extraction"
        return a

    def fail(reason: str, detail: str) -> TurnAnalysis:
        a.status, a.reason, a.detail = "abstain", reason, detail
        a.opens_window = not a.readback_shaped and has_order_verb(text) and _scan_opens_window(text, registry)
        # a correction must be cue-only ("Negative.") or carry the repeated command
        a.operator_correction = correction_cue and (cue_only(text) or a.opens_window)
        return a

    problem = validate_extraction(turn.extraction)
    if problem:
        return fail(R_GROUNDING, f"malformed extraction: {problem}")
    try:
        g = grounding.ground(text, turn.extraction)
    except grounding.GroundingError as e:
        return fail(R_GROUNDING, str(e))
    try:
        coverage.check(text, g)
    except coverage.CoverageError as e:
        return fail(e.reason if e.reason != "coverage" else R_COVERAGE, e.detail)
    a.spans, a.retracted = g.commands, g.retracted

    cmds: list[CanonCommand] = []
    prev_action: str | None = None
    for sp in g.commands:
        vals: dict = {}
        if sp["action_span"] is not None:
            act = al.canon_action(text[sp["action_span"][0] : sp["action_span"][1]])
            if isinstance(act, al.Unmapped):
                return fail(r_unmapped(act.word), "action words not in the alias tables")
            vals["action"] = act
        else:
            vals["action"] = prev_action  # "close 203 and 208": the verb distributes over the conjunction
        prev_action = vals["action"]
        if sp["tag_span"] is not None:
            s, e = sp["tag_span"]
            m = al.canon_tag(text[s:e], preceding_word=_prev_word(text, s))
            if isinstance(m, al.Unmapped):
                return fail(r_unmapped(m.word), "tag words not in the alias tables")
            res = registry.resolve(m.prefix, m.suffix or "")
            vals.update(tag=res.tag, tag_status=res.status, registry_line=res.line)
        if sp["line_span"] is not None:
            ln = al.canon_line(text[sp["line_span"][0] : sp["line_span"][1]])
            if isinstance(ln, al.Unmapped):
                return fail(r_unmapped(ln.word), "line words not in the alias tables")
            vals["line"] = ln
        if sp["state_span"] is not None:
            st = al.canon_state(text[sp["state_span"][0] : sp["state_span"][1]])
            if isinstance(st, al.Unmapped):
                return fail(r_unmapped(st.word), "state words not in the alias tables")
            vals["state"] = st
        cmds.append(CanonCommand(**vals))
    a.commands = cmds
    # a turn opens a pairing window only with >= 1 grounded command that has an action and a tag that resolves to a
    # full tag (spoken prefix, or a bare number the registry knows); chatter and status talk ("pump one-forty is
    # running") never open one
    a.opens_window = (
        not a.readback_shaped
        and has_order_verb(text)
        and any(c.action is not None and _full_tag(c.tag, c.tag_status) for c in cmds)
    )
    a.operator_correction = correction_cue and (cue_only(text) or a.opens_window)
    return a


def _full_tag(tag: str | None, status: str | None) -> bool:
    """A tag that names equipment: in the registry, ambiguous, spoken with its prefix (XV-205), or a bare number in
    registry suffix format (205) — so an unknown tag still opens a window and ends in ABSTAIN(unknown tag)."""
    if tag is None:
        return False
    return status in ("ok", "ambiguous") or "-" in tag or al.is_suffix_format(tag)


def _scan_opens_window(text: str, registry: Registry) -> bool:
    """For an unreadable turn: does code alone see an instruction-shaped turn (an action + a resolvable tag)?

    Fail closed: if the model could not be checked on a turn that names a real tag and a verb, the exchange still
    opens and ends in a card, instead of vanishing as chatter.
    """
    toks = al.tokenize(text)
    if not al.action_mentions(toks):
        return False
    for m in al.tag_mentions(toks):
        res = registry.resolve(m.prefix, m.suffix or "")
        if _full_tag(res.tag, res.status):
            return True
    return False


def span_confidence(analyses: list[TurnAnalysis]) -> float | None:
    """Minimum ASR token probability inside the action, tag and line spans of every given turn.

    None when it cannot be computed for every span — a turn without token probabilities, or a span no token overlaps.
    Never invented, never partial: with a floor configured, None makes the referee abstain (audio).
    """
    probs: list[float] = []
    for a in analyses:
        if not a.turn.tokens:
            return None
        for cmd in a.spans:
            for f in ("action_span", "tag_span", "line_span"):
                sp = cmd.get(f)
                if sp is None:
                    continue
                lps = [lp for s, e, lp in a.turn.tokens if s < sp[1] and sp[0] < e]
                if not lps or any(not math.isfinite(lp) or lp > 0 for lp in lps):
                    return None  # missing, NaN or impossible (p > 1): never a pass
                hit = [math.exp(lp) for lp in lps]
                probs.extend(hit)
    return min(probs) if probs else None
