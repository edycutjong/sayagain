"""Rung 0 of the model ladder: a deterministic regex + alias extractor. No model.

Returns exactly the structure the LLM is asked for (span schema, see `referee.schema`), so the same referee judges
both. It is frozen at the pre-registration tag `prereg-v1` (bench/PROTOCOL.md); later alias changes are allowed only
on the declared dev slice and apply to every rung alike.

Grouping rules (left to right over the alias mentions):
  * an action starts a new command, unless the open command has a tag but no action yet (Bahasa order:
    "valve dua nol tiga tutup");
  * a tag / line fills the open command; a second tag / line / action for the same command is a retraction when a
    cue ("correction", "koreksi", "ralat", "bukan", "no") sits between the two mentions, otherwise a second tag starts
    a new command and a second line is left uncovered (the coverage check then abstains — no guessing);
  * kind: say_again (repeat request, no tag), readback (opens with copy / roger / siap … or a non-imperative verb),
    instruction (has a command), other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from referee import aliases as al


@dataclass
class _Cmd:
    action: al.Mention | None = None
    tag: al.Mention | None = None
    line: al.Mention | None = None
    state: al.Mention | None = None
    extra: list[al.Mention] = field(default_factory=list)


def _cue_between(tokens: list[al.Token], a_end: int, b_start: int) -> bool:
    return any(t.norm in al.RETRACTION_CUES and t.start >= a_end and t.end <= b_start for t in tokens)


def classify_kind(text: str, has_commands: bool, has_tag: bool) -> str:
    toks = al.tokenize(text)
    if not toks:
        return "other"
    opens_repeat = any(al.starts_with(toks, p) for p in al.REPEAT_OPENERS)
    if not has_tag and not opens_repeat and any(al.contains_phrase(toks, p) for p in al.SAY_AGAIN_PHRASES):
        return "say_again"
    if toks[0].norm in al.READBACK_CUES:
        return "readback" if has_commands else "other"
    if not has_commands:
        return "other"
    first_action = next((t.norm for t in toks if t.norm in al.ACTION_WORDS or t.norm in al.IMPERATIVE_ACTIONS), None)
    if first_action is not None and first_action not in al.IMPERATIVE_ACTIONS:
        return "readback"  # "closing two-oh-eight" / "menutup …"
    return "instruction"


def extract(text: str) -> dict:
    """Transcript -> {"kind", "commands": [{action_span, tag_span, state_span, line_span}], "retracted_spans"}."""
    toks = al.tokenize(text)
    mentions = sorted(
        al.action_mentions(toks) + al.tag_mentions(toks) + al.line_mentions(toks) + al.state_mentions(toks),
        key=lambda m: (m.start, m.end),
    )
    cmds: list[_Cmd] = []
    retracted: list[al.Mention] = []
    cur: _Cmd | None = None

    def cue(a: al.Mention, b: al.Mention) -> bool:
        return _cue_between(toks, a.end, b.start)

    def last_end(c: _Cmd) -> int:
        return max(x.end for x in (c.action, c.tag, c.line, c.state) if x is not None)

    def new_command(m: al.Mention) -> _Cmd:
        # "… two-oh-three. Correction, closing two-oh-eight": a cue before a fresh command retracts the whole
        # previous command
        if cmds and _cue_between(toks, last_end(cmds[-1]), m.start):
            old = cmds.pop()
            retracted.extend(x for x in (old.action, old.tag, old.line, old.state) if x is not None)
        c = _Cmd(**{m.field: m})
        cmds.append(c)
        return c

    for m in mentions:
        if m.field == "action":
            if cur is not None and cur.action is not None and cur.tag is None and cue(cur.action, m):
                retracted.append(cur.action)  # "close — correction — open valve 203"
                cur.action = m
            elif cur is not None and cur.action is None:
                cur.action = m
            else:
                cur = new_command(m)
        elif m.field == "tag":
            if cur is None:
                cur = new_command(m)
            elif cur.tag is None:
                cur.tag = m
            elif cue(cur.tag, m) and (cur.line is None or cur.line.start < cur.tag.start):
                retracted.append(cur.tag)  # "valve two-oh-three, correction, two-oh-eight"
                cur.tag = m
            else:
                cur = new_command(m)  # "close valve 203 and valve 208": second command, no verb of its own
        elif m.field == "line":
            if cur is None:
                cur = new_command(m)
            elif cur.line is None:
                cur.line = m
            elif cue(cur.line, m):
                retracted.append(cur.line)
                cur.line = m
            else:
                cur.extra.append(m)  # conflicting line, no cue: leave it uncovered
        elif m.field == "state":
            if cur is not None and cur.state is None:
                cur.state = m

    def span(m: al.Mention | None) -> str | None:
        return text[m.start : m.end] if m is not None else None

    commands = [
        {"action_span": span(c.action), "tag_span": span(c.tag), "state_span": span(c.state), "line_span": span(c.line)}
        for c in cmds
        if c.action is not None or c.tag is not None
    ]
    has_tag = any(c.tag is not None for c in cmds)
    return {
        "kind": classify_kind(text, bool(commands), has_tag),
        "commands": commands,
        "retracted_spans": [text[m.start : m.end] for m in retracted],
    }
