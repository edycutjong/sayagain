"""Audit a shift: analyse turns -> pair -> judge each exchange -> route cards.

Routing (the shift budget in bench/PROTOCOL.md):
  * live queue (the supervisor's screen): MISMATCH, NO_READBACK, ABSTAIN;
  * end-of-shift list: caught-on-channel read-backs, NOT_AUDITED exchanges/turns, 10 random VERIFIED spot-checks;
  * VERIFIED otherwise: logged, no card.
Fail closed: any exception while judging an exchange makes it NOT_AUDITED, never VERIFIED.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from referee import aliases as al
from referee.pairing import Exchange, pair
from referee.regex_extractor import extract as regex_extract
from referee.registry import Registry
from referee.turns import Turn, TurnAnalysis, analyze, span_confidence
from referee.verdict import ABSTAIN, MISMATCH, NO_READBACK, NOT_AUDITED, VERIFIED, Verdict, verdict

SPOT_CHECKS_PER_SHIFT = 10
LIVE = (MISMATCH, NO_READBACK, ABSTAIN)


def canonical_tuples(text: str) -> list[tuple[str | None, str | None, str | None]]:
    """(action, tag, line) per command, read by the frozen regex + alias extractor — the disagreement gate's lens."""
    out = []
    ext = regex_extract(text)
    for c in ext["commands"]:
        act = al.canon_action(c["action_span"]) if c["action_span"] else None
        tag = al.canon_tag(c["tag_span"]) if c["tag_span"] else None
        line = al.canon_line(c["line_span"]) if c["line_span"] else None
        out.append(
            (
                act if isinstance(act, str) else None,
                tag.value if isinstance(tag, al.Mention) else None,
                line if isinstance(line, str) else None,
            )
        )
    return out


def disagrees(a: TurnAnalysis) -> bool:
    """Prompted vs unprompted ASR read differently on action / tag / line (only when an unprompted pass exists)."""
    alt = a.turn.alt_transcript
    return alt is not None and canonical_tuples(a.turn.transcript) != canonical_tuples(alt)


@dataclass
class Judged:
    exchange: Exchange
    verdict: Verdict
    caught: list[Verdict] = field(default_factory=list)  # verdicts of the read-backs corrected on the channel
    spot_check: bool = False

    @property
    def live(self) -> bool:
        return self.verdict.verdict in LIVE


def judge(ex: Exchange, *, floor: float | None = None) -> Verdict:
    try:
        return _judge(ex.instruction, ex.readback, floor=floor)
    except Exception as e:  # noqa: BLE001 — fail closed: a crashed auditor is "not auditing", never a pass
        return Verdict(NOT_AUDITED, "referee error", trace=[f"referee error: {type(e).__name__}: {e}"])


def _judge(ins: TurnAnalysis, rb: TurnAnalysis | None, *, floor: float | None) -> Verdict:
    trace: list[str] = []
    for role, a in (("instruction", ins), ("read-back", rb)):
        if a is None:
            continue
        if a.status == "not_audited":
            return Verdict(NOT_AUDITED, a.reason, trace=[f"{role}: not audited ({a.reason})"])
    if ins.status == "abstain":
        return Verdict(ABSTAIN, ins.reason, trace=[f"instruction: {ins.reason} — {ins.detail}"])
    trace.append(f"instruction: grounded + covered ({len(ins.commands)} command(s))")
    if rb is not None:
        if rb.status == "abstain":
            v = verdict(ins.commands, None)
            if v.verdict == ABSTAIN:  # the instruction itself is invalid: that reason wins
                return Verdict(ABSTAIN, v.reason, trace=trace + v.trace)
            return Verdict(ABSTAIN, rb.reason, trace=[*trace, f"read-back: {rb.reason} — {rb.detail}"])
        trace.append(f"read-back: grounded + covered ({len(rb.commands)} command(s))")
    pair_turns = [ins] + ([rb] if rb is not None else [])
    conf = span_confidence(pair_turns)
    has_alt = all(a.turn.alt_transcript is not None for a in pair_turns)
    disagree = any(disagrees(a) for a in pair_turns)
    if conf is not None:
        trace.append(f"audio: span confidence {conf:.3f}" + (f" (floor {floor:.3f})" if floor is not None else ""))
    elif has_alt:
        trace.append("audio: disagreement gate " + ("FIRED" if disagree else "agree"))
    else:
        trace.append("audio: no gate (no token probabilities, no unprompted pass)")
    v = verdict(
        ins.commands, rb.commands if rb is not None else None, span_conf=conf, floor=floor, disagreement=disagree
    )
    v.trace = trace + v.trace
    return v


@dataclass
class ShiftReport:
    judged: list[Judged]
    other: list[TurnAnalysis]
    not_audited_turns: list[TurnAnalysis]

    @property
    def live_queue(self) -> list[Judged]:
        return [j for j in self.judged if j.live]

    @property
    def end_of_shift(self) -> dict[str, list]:
        return {
            "caught_on_channel": [j for j in self.judged if j.caught],
            "not_audited": [j for j in self.judged if j.verdict.verdict == NOT_AUDITED] + list(self.not_audited_turns),
            "spot_checks": [j for j in self.judged if j.spot_check],
        }


def audit(
    turns: list[Turn],
    registry: Registry,
    *,
    floor: float | None = None,
    spot_checks: int = SPOT_CHECKS_PER_SHIFT,
    seed: int = 1018,
) -> ShiftReport:
    analyses = [analyze(t, registry) for t in turns]
    paired = pair(analyses)
    judged: list[Judged] = []
    for ex in paired.exchanges:
        caught = [judge(Exchange(ex.channel, i, r), floor=floor) for i, r in ex.caught]
        judged.append(Judged(ex, judge(ex, floor=floor), caught))
    verified = [j for j in judged if j.verdict.verdict == VERIFIED]
    for j in random.Random(seed).sample(verified, min(spot_checks, len(verified))):
        j.spot_check = True
    return ShiftReport(judged, paired.other, paired.not_audited)
