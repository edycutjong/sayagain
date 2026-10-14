"""Pairing: which turn is the read-back of which instruction, per channel, by turn order and an 8-second window.

Speaker identity on a replayed channel rests on turn order (production would use the radio's unit ID), so the rules are
written down and tested:
  * no window open: a turn opens one only if it has >= 1 grounded command with an action and a tag that names
    equipment (turns.analyze); a read-back-shaped turn ("Copy, closing …") never opens one (orphan, end-of-shift list);
    an unreadable (NOT_AUDITED) turn opens one so its reply cannot become a phantom card; anything else is `other`;
  * window open: the next turn within 8 s is the read-back candidate whatever its verb form — a word-for-word repeat is
    the most natural read-back;
  * a say-again from the technician extends the window by 8 s, and the operator's repeat (an imperative turn) replaces
    the instruction; it is never taken as the read-back;
  * after a read-back, an operator turn opening with "negative / koreksi / bukan / correction" within 8 s is the third
    leg: the first read-back goes to the end-of-shift list as "caught on channel", the repeat replaces the instruction
    and the window re-opens. The correction turn must be cue-only ("Negative.") or carry the repeat. If no new
    read-back follows, the wrong read-back stands as the live verdict. "Correct / betul" or silence closes the exchange;
  * "Stand by" / "wait one" / "tunggu" (no command) while a read-back is awaited extends the window; chatter while a
    repeat is awaited is ignored; after "Negative." an order for different tags starts a new exchange;
  * an unreadable (NOT_AUDITED) instruction's window closes when a new instruction-shaped turn arrives;
  * while a read-back is awaited, a cue-less order for OTHER tags starts a new exchange; the same order with a query
    ("…, do you copy?") is the operator repeating; a second read-back after the first is kept and judged, also
    when the referee cannot read it (it abstains);
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
    # further read-back-shaped turns after the first (each is judged; the worst verdict stands)
    more_readbacks: list[TurnAnalysis] = field(default_factory=list)


@dataclass
class Paired:
    exchanges: list[Exchange] = field(default_factory=list)
    other: list[TurnAnalysis] = field(default_factory=list)  # chatter / status talk: no card
    not_audited: list[TurnAnalysis] = field(default_factory=list)  # unreadable turns outside any exchange
    orphans: list[TurnAnalysis] = field(default_factory=list)  # read-back-shaped turns with no open window


_IDLE, _AWAIT_RB, _AWAIT_REPEAT, _AWAIT_ACK = "idle", "await_readback", "await_repeat", "await_ack"


def _tags(a: TurnAnalysis) -> set[str]:
    return {c.tag for c in a.commands if c.tag}


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
            if ex.readback is None and ex.caught:
                # the operator said "negative" but no new read-back came: the wrong read-back stands, live
                ex.instruction, ex.readback = ex.caught.pop()
                ex.events.append("correction not read back")
            out.exchanges.append(ex)
        state, ex = _IDLE, None

    def idle(t: TurnAnalysis) -> None:
        nonlocal state, ex, deadline
        if t.status == "not_audited" or t.opens_window:
            # an unreadable turn may be an instruction: it opens a window so its reply cannot become a phantom card;
            # the exchange is then NOT_AUDITED (end-of-shift list)
            ex = Exchange(t.turn.channel, instruction=t, turn_ids=[t.id])
            state, deadline = _AWAIT_RB, t.turn.t_end + window_s
        elif t.readback_shaped and ((t.status == "ok" and t.commands) or (t.status == "abstain" and t.safety_words)):
            out.orphans.append(t)  # a read-back with no instruction heard (readable or not): end-of-shift list
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
            if ex.instruction.status == "not_audited" and t.status != "not_audited" and t.opens_window:
                close()  # an unreadable turn must not swallow the next real instruction
                idle(t)
            elif (
                t.status != "not_audited"
                and t.imperative
                and t.opens_window
                and t.query
                and _tags(t) == _tags(ex.instruction)
            ):
                ex.instruction = t  # "Close valve two-oh-three, line Bravo, do you copy?" — the operator's own repeat
                ex.turn_ids.append(t.id)
                ex.events.append("repeat")
                deadline = t.turn.t_end + window_s
            elif t.status != "not_audited" and t.imperative and t.opens_window and _tags(t) != _tags(ex.instruction):
                close()  # a cue-less order for other equipment: a new exchange, not this one's read-back
                idle(t)
            elif t.status != "not_audited" and t.ack_only:
                ex.turn_ids.append(t.id)  # "Roger." — acknowledged, not read back: wait for the read-back
                ex.events.append("ack without read-back")
                deadline = t.turn.t_end + window_s
            elif t.status != "not_audited" and t.stand_by:
                ex.turn_ids.append(t.id)  # "Stand by." — the read-back is still coming
                ex.events.append("stand by")
                deadline = t.turn.t_end + window_s
            elif t.status != "not_audited" and t.say_again:
                ex.turn_ids.append(t.id)
                ex.events.append("say again")
                state, deadline = _AWAIT_REPEAT, t.turn.t_end + window_s
            else:
                ex.readback = t
                ex.turn_ids.append(t.id)
                state, deadline = _AWAIT_ACK, t.turn.t_end + window_s
        elif state == _AWAIT_REPEAT:
            if t.status != "not_audited" and t.imperative and t.opens_window:
                in_force = ex.caught[-1][0] if ex.caught else ex.instruction
                if ex.caught and _tags(t) != _tags(in_force):
                    close()  # after "Negative." a different order is a new exchange, not the repeat
                    idle(t)
                    continue
                ex.instruction = t  # the operator's repeat replaces the instruction
                ex.turn_ids.append(t.id)
                ex.events.append("repeat")
                state, deadline = _AWAIT_RB, t.turn.t_end + window_s
            elif t.status != "not_audited" and not t.commands and not t.say_again and t.status == "ok":
                out.other.append(t)  # chatter while the repeat is awaited: neither the repeat nor a read-back
            elif t.status != "not_audited" and t.say_again:
                ex.turn_ids.append(t.id)
                ex.events.append("say again")
                deadline = t.turn.t_end + window_s
            else:
                ex.readback = t  # the technician read back without a repeat
                ex.turn_ids.append(t.id)
                state, deadline = _AWAIT_ACK, t.turn.t_end + window_s
        elif state == _AWAIT_ACK:
            if t.status != "not_audited" and t.operator_correction and t.commands and _tags(t) != _tags(ex.instruction):
                close()  # "No problem, open valve two-oh-seven": a new order, not a correction of this read-back
                idle(t)
            elif t.status != "not_audited" and t.operator_correction:
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
            elif (
                t.status == "ok"
                and t.commands
                and (t.readback_shaped or t.verb_first)
                and not (t.imperative and t.opens_window and _tags(t) != _tags(ex.instruction))
            ):
                ex.more_readbacks.append(t)  # a second read-back: judged too, never dropped as an "ack"
                ex.turn_ids.append(t.id)
                ex.events.append("second read-back")
                deadline = t.turn.t_end + window_s
            elif (
                t.status == "abstain"
                and t.safety_words
                and (t.readback_shaped or t.verb_first)
                and not t.opens_window
                and not t.say_again
            ):
                # the same shape as above, but the referee could not read it: judged (it abstains), never taken as
                # the acknowledgement that closes the exchange on the first read-back
                ex.more_readbacks.append(t)
                ex.turn_ids.append(t.id)
                ex.events.append("unreadable second read-back")
                deadline = t.turn.t_end + window_s
            elif t.status != "not_audited" and not t.opens_window and not t.say_again:
                ex.turn_ids.append(t.id)  # "Correct." / "Valve 203 is now closed." — acknowledgement or status
                close()
            else:
                close()
                idle(t)
    if state != _IDLE:
        close()
