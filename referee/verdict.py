"""The referee's decision: a pure function of canonical commands and two audio-gate inputs.

Imports nothing from ASR, extraction or I/O. Registry facts arrive already resolved on the commands (`tag_status`,
`registry_line`), so the same inputs always give the same verdict.

Precedence:
  * the INSTRUCTION is validated against the registry: its tag must exist and a spoken line must equal the tag's
    registry line, else ABSTAIN(unknown tag) — the operator's own words are unclear;
  * the READ-BACK is compared field by field to the instruction, never re-validated first: any differing, missing or
    not-in-registry field is MISMATCH, never ABSTAIN — except ABSTAIN(ambiguous tag) for a bare number whose suffix the
    registry shares between tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERIFIED, MISMATCH, NO_READBACK, ABSTAIN, NOT_AUDITED = "VERIFIED", "MISMATCH", "NO_READBACK", "ABSTAIN", "NOT_AUDITED"
# ABSTAIN reasons — one enum in code and UI
R_AUDIO, R_GROUNDING, R_COVERAGE, R_RETRACTION = "audio", "grounding", "coverage", "retraction cue missing"
R_AMBIGUOUS, R_UNKNOWN = "ambiguous tag", "unknown tag"


def r_unmapped(word: str) -> str:
    return f"unmapped: {word}"


@dataclass(frozen=True)
class CanonCommand:
    action: str | None = None  # CLOSE | OPEN | ISOLATE | START | STOP
    tag: str | None = None  # "XV-203"; the bare suffix ("205") when nothing in the registry matches
    tag_status: str | None = None  # ok | not_in_registry | ambiguous (None when no tag was spoken)
    registry_line: str | None = None  # line of `tag` in the registry (status ok)
    line: str | None = None  # spoken line letter
    state: str | None = None


@dataclass(frozen=True)
class Diff:
    field: str
    expected: str | None
    got: str | None
    note: str = ""

    def __str__(self) -> str:
        if self.field == "command":
            return f"command: {self.note}"
        if self.got is None:
            return f"{self.field}: missing (expected {self.expected})"
        if self.note == "not in registry":
            return f"{self.field}: {self.got} not in registry"
        return f"{self.field}: {self.expected} ≠ {self.got}"


@dataclass
class Verdict:
    verdict: str
    reason: str | None = None
    diffs: list[Diff] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """UI label: 'MISMATCH · tag: XV-203 ≠ XV-208', 'ABSTAIN · unknown tag', 'NO-READBACK'."""
        name = self.verdict.replace("_", "-")
        if self.diffs:
            return f"{name} · " + "; ".join(str(d) for d in self.diffs)
        if self.reason:
            return f"{name} · {self.reason}"
        return name

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "label": self.label,
            "diffs": [
                {"field": d.field, "expected": d.expected, "got": d.got, "note": d.note, "text": str(d)}
                for d in self.diffs
            ],
            "trace": list(self.trace),
        }


def validate_instruction(cmds: list[CanonCommand]) -> str | None:
    """None if every instruction command names a registry tag on its registry line, else the ABSTAIN reason."""
    if not cmds:
        return R_UNKNOWN
    for c in cmds:
        if c.action is None:
            return r_unmapped("no action")
        if c.tag is None or c.tag_status == "not_in_registry":
            return R_UNKNOWN
        if c.tag_status == "ambiguous":
            return R_AMBIGUOUS
        if c.line is not None and c.line != c.registry_line:
            return R_UNKNOWN
    return None


def compare(instr: list[CanonCommand], readback: list[CanonCommand]) -> list[Diff]:
    """Field-by-field, by position. Assumes a validated instruction."""
    diffs: list[Diff] = []
    if len(readback) < len(instr):
        diffs.append(Diff("command", str(len(instr)), str(len(readback)), "omitted"))
    elif len(readback) > len(instr):
        diffs.append(Diff("command", str(len(instr)), str(len(readback)), "extra"))
    for i, r in zip(instr, readback, strict=False):
        if i.action is not None and r.action != i.action:
            diffs.append(Diff("action", i.action, r.action))
        if r.tag is None:
            diffs.append(Diff("tag", i.tag, None))
        elif r.tag_status == "not_in_registry":
            diffs.append(Diff("tag", i.tag, r.tag, "not in registry"))
        elif r.tag != i.tag:
            note = f"{r.tag} is on line {r.registry_line}" if r.registry_line else ""
            diffs.append(Diff("tag", i.tag, r.tag, note))
        expected_line = i.line or i.registry_line
        if r.line is None:
            if i.line is not None:
                diffs.append(Diff("line", i.line, None))
        elif r.line != expected_line:
            diffs.append(Diff("line", expected_line, r.line))
        if i.state is not None and r.state is not None and r.state != i.state:
            diffs.append(Diff("state", i.state, r.state))
    return diffs


def verdict(
    instr: list[CanonCommand],
    readback: list[CanonCommand] | None,
    *,
    span_conf: float | None = None,
    floor: float | None = None,
    disagreement: bool = False,
) -> Verdict:
    """VERIFIED · MISMATCH · NO_READBACK · ABSTAIN for one paired exchange (canonical inputs only)."""
    bad = validate_instruction(instr)
    if bad:
        return Verdict(ABSTAIN, bad, trace=[f"instruction: {bad}"])
    trace = ["instruction: " + ", ".join(f"{c.tag} line {c.line or c.registry_line}" for c in instr) + " (registry ok)"]
    if readback is None:
        return Verdict(NO_READBACK, trace=[*trace, "pairing: no read-back inside the window"])
    if any(r.tag_status == "ambiguous" for r in readback):
        return Verdict(ABSTAIN, R_AMBIGUOUS, trace=[*trace, "read-back: bare number shared by several tags"])
    if disagreement:
        return Verdict(ABSTAIN, R_AUDIO, trace=[*trace, "audio: prompted and unprompted transcripts disagree"])
    if floor is not None and span_conf is not None and span_conf < floor:
        return Verdict(ABSTAIN, R_AUDIO, trace=[*trace, f"audio: span confidence {span_conf:.3f} < floor {floor:.3f}"])
    diffs = compare(instr, readback)
    if diffs:
        return Verdict(MISMATCH, diffs=diffs, trace=[*trace, "compare: " + "; ".join(str(d) for d in diffs)])
    return Verdict(VERIFIED, trace=[*trace, "compare: every field equal"])
