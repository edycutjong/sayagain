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

from referee.pairing import Exchange, pair
from referee.registry import Registry
from referee.turns import Turn, TurnAnalysis, analyze, span_confidence
from referee.verdict import ABSTAIN, MISMATCH, NO_READBACK, NOT_AUDITED, R_AUDIO, VERIFIED, Verdict, verdict

SPOT_CHECKS_PER_SHIFT = 10
_SEVERITY = {VERIFIED: 0, NO_READBACK: 1, NOT_AUDITED: 2, ABSTAIN: 3, MISMATCH: 4}
LIVE = (MISMATCH, NO_READBACK, ABSTAIN)


@dataclass
class Judged:
    exchange: Exchange
    verdict: Verdict
    caught: list[Verdict] = field(default_factory=list)  # verdicts of the read-backs corrected on the channel
    spot_check: bool = False

    @property
    def live(self) -> bool:
        return self.verdict.verdict in LIVE


AUDIO_GATES = ("floor", "disagreement", "none")


def judge(ex: Exchange, *, floor: float | None = None, audio_gate: str | None = None) -> Verdict:
    """audio_gate: "floor" (token probabilities vs `floor`), "disagreement" (prompted vs unprompted ASR), "none".
    Default: "floor" when a floor is given, else "disagreement" when every turn has an unprompted pass, else "none".
    A configured gate whose input is missing abstains (audio) — it is never skipped."""
    try:
        return _judge(ex.instruction, ex.readback, floor=floor, audio_gate=audio_gate)
    except Exception as e:  # noqa: BLE001 — fail closed: a crashed auditor is "not auditing", never a pass
        return Verdict(NOT_AUDITED, "referee error", trace=[f"referee error: {type(e).__name__}: {e}"])


def _judge(ins: TurnAnalysis, rb: TurnAnalysis | None, *, floor: float | None, audio_gate: str | None) -> Verdict:
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
    has_alt = all(a.turn.alt_transcript is not None for a in pair_turns)
    gate = audio_gate or ("floor" if floor is not None else "disagreement" if has_alt else "none")
    if gate not in AUDIO_GATES:
        raise ValueError(f"audio_gate {gate!r}")
    conf = span_confidence(pair_turns) if gate == "floor" else None
    disagree = gate == "disagreement" and any(a.disagrees for a in pair_turns)
    missing = rb is not None and (
        (gate == "floor" and (floor is None or conf is None)) or (gate == "disagreement" and not has_alt)
    )
    if missing:
        v = verdict(ins.commands, None)
        if v.verdict == ABSTAIN:
            return Verdict(ABSTAIN, v.reason, trace=trace + v.trace)
        return Verdict(ABSTAIN, R_AUDIO, trace=[*trace, f"audio: {gate} gate configured but its input is missing"])
    if gate == "floor" and conf is not None and floor is not None:
        trace.append(f"audio: span confidence {conf:.3f} (floor {floor:.3f})")
    elif gate == "disagreement":
        trace.append("audio: disagreement gate " + ("FIRED" if disagree else "agree"))
    else:
        trace.append("audio: no gate (no token probabilities, no unprompted pass)")
    v = verdict(
        ins.commands,
        rb.commands if rb is not None else None,
        span_conf=conf,
        floor=floor if gate == "floor" else None,
        disagreement=disagree,
    )
    v.trace = trace + v.trace
    return v


@dataclass
class ShiftReport:
    judged: list[Judged]
    other: list[TurnAnalysis]
    not_audited_turns: list[TurnAnalysis]
    orphans: list[TurnAnalysis]

    @property
    def live_queue(self) -> list[Judged]:
        return [j for j in self.judged if j.live]

    @property
    def end_of_shift(self) -> dict[str, list]:
        return {
            "caught_on_channel": [j for j in self.judged if j.caught],
            "not_audited": [j for j in self.judged if j.verdict.verdict == NOT_AUDITED] + list(self.not_audited_turns),
            "spot_checks": [j for j in self.judged if j.spot_check],
            "orphan_readbacks": list(self.orphans),
        }


def audit(
    turns: list[Turn],
    registry: Registry,
    *,
    floor: float | None = None,
    audio_gate: str | None = None,
    spot_checks: int = SPOT_CHECKS_PER_SHIFT,
    seed: int = 1018,
) -> ShiftReport:
    analyses = [analyze(t, registry) for t in turns]
    paired = pair(analyses)
    judged: list[Judged] = []
    for ex in paired.exchanges:
        caught = [judge(Exchange(ex.channel, i, r), floor=floor, audio_gate=audio_gate) for i, r in ex.caught]
        v = judge(ex, floor=floor, audio_gate=audio_gate)
        for extra in ex.more_readbacks:  # every read-back in the exchange is judged; the worst one stands
            v2 = judge(Exchange(ex.channel, ex.instruction, extra), floor=floor, audio_gate=audio_gate)
            if _SEVERITY[v2.verdict] > _SEVERITY[v.verdict]:
                v2.trace = [*v2.trace, f"second read-back {extra.id} (first was {v.label})"]
                v = v2
        judged.append(Judged(ex, v, caught))
    verified = [j for j in judged if j.verdict.verdict == VERIFIED]
    for j in random.Random(seed).sample(verified, min(spot_checks, len(verified))):
        j.spot_check = True
    return ShiftReport(judged, paired.other, paired.not_audited, paired.orphans)
