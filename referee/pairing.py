"""Pairing: which turn is the read-back of which instruction, per channel, by turn order and an 8-second window.

Speaker identity on a replayed channel rests on turn order (production would use the radio's unit ID), so the rules are
written down and tested:
  * no window open: a turn opens one only if it has >= 1 grounded command whose tag resolves (turns.analyze); anything
    else is logged as `other` and makes no card;
  * window open: the next turn within 8 s is the read-back candidate whatever its verb form — a word-for-word repeat is
    the most natural read-back;
  * a say-again from the technician extends the window by 8 s, and the operator's repeat (an imperative turn) replaces
    the instruction; it is never taken as the read-back;
  * after a read-back, an operator turn opening with "negative / koreksi / bukan / correction" within 8 s is the third
    leg: the first read-back goes to the end-of-shift list as "caught on channel", the repeat replaces the instruction
    and the window re-opens. "Correct / betul" or silence closes the exchange;
  * no read-back inside the window -> NO_READBACK. Radio is half-duplex: no overlap model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from referee.turns import TurnAnalysis

WINDOW_S = 8.0


@dataclass
class Exchange:
    channel: str
    instruction: TurnAnalysis
    readback: TurnAnalysis | None = None
    turn_ids: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    # (instruction in force, the read-back the operator corrected on the channel)
    caught: list[tuple[TurnAnalysis, TurnAnalysis]] = field(default_factory=list)


@dataclass
class Paired:
    exchanges: list[Exchange] = field(default_factory=list)
    other: list[TurnAnalysis] = field(default_factory=list)  # chatter / status talk: no card
    not_audited: list[TurnAnalysis] = field(default_factory=list)  # unreadable turns outside any exchange


_IDLE, _AWAIT_RB, _AWAIT_REPEAT, _AWAIT_ACK = "idle", "await_readback", "await_repeat", "await_ack"


def pair(turns: list[TurnAnalysis], window_s: float = WINDOW_S) -> Paired:
    out = Paired()
    by_channel: dict[str, list[TurnAnalysis]] = {}
    for t in turns:
        by_channel.setdefault(t.turn.channel, []).append(t)
    for ch in sorted(by_channel):
        _pair_channel(sorted(by_channel[ch], key=lambda a: a.turn.t_start), window_s, out)
    out.exchanges.sort(key=lambda e: (e.instruction.turn.t_start, e.channel))
    return out


def _pair_channel(turns: list[TurnAnalysis], window_s: float, out: Paired) -> None:
    state = _IDLE
    ex: Exchange | None = None
    deadline = 0.0

    def close() -> None:
        nonlocal state, ex
        if ex is not None:
            out.exchanges.append(ex)
        state, ex = _IDLE, None

    def idle(t: TurnAnalysis) -> None:
        nonlocal state, ex, deadline
        if t.status == "not_audited":
            out.not_audited.append(t)
        elif t.opens_window:
            ex = Exchange(t.turn.channel, instruction=t, turn_ids=[t.id])
            state, deadline = _AWAIT_RB, t.turn.t_end + window_s
        else:
            out.other.append(t)

    for t in turns:
        if state != _IDLE and t.turn.t_start > deadline:
            if state == _AWAIT_RB or state == _AWAIT_REPEAT:
                assert ex is not None
                ex.events.append("window expired")
            close()  # NO_READBACK if no read-back was paired; otherwise the exchange stands as read back
        if state == _IDLE:
            idle(t)
            continue
        assert ex is not None
        if state == _AWAIT_RB:
            if t.status != "not_audited" and t.say_again:
                ex.turn_ids.append(t.id)
                ex.events.append("say again")
                state, deadline = _AWAIT_REPEAT, t.turn.t_end + window_s
            else:
                ex.readback = t
                ex.turn_ids.append(t.id)
                state, deadline = _AWAIT_ACK, t.turn.t_end + window_s
        elif state == _AWAIT_REPEAT:
            if t.status != "not_audited" and t.imperative and t.opens_window:
                ex.instruction = t  # the operator's repeat replaces the instruction
                ex.turn_ids.append(t.id)
                ex.events.append("repeat")
                state, deadline = _AWAIT_RB, t.turn.t_end + window_s
            elif t.status != "not_audited" and t.say_again:
                ex.turn_ids.append(t.id)
                ex.events.append("say again")
                deadline = t.turn.t_end + window_s
            else:
                ex.readback = t  # the technician read back without a repeat
                ex.turn_ids.append(t.id)
                state, deadline = _AWAIT_ACK, t.turn.t_end + window_s
        elif state == _AWAIT_ACK:
            if t.status != "not_audited" and t.operator_correction:
                assert ex.readback is not None
                ex.caught.append((ex.instruction, ex.readback))
                ex.events.append("caught on channel")
                ex.turn_ids.append(t.id)
                ex.readback = None
                if t.opens_window:
                    ex.instruction = t  # "Negative, close valve two-oh-three, line Bravo."
                    state = _AWAIT_RB
                else:
                    state = _AWAIT_REPEAT  # "Negative." — the repeat follows in the next turn
                deadline = t.turn.t_end + window_s
            elif t.status != "not_audited" and not t.opens_window and not t.say_again:
                ex.turn_ids.append(t.id)  # "Correct." / "Betul." — the operator's acknowledgement
                close()
            else:
                close()
                idle(t)
    if state != _IDLE:
        close()
