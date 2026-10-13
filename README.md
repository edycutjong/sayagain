# SayAgain

**Checks every radio read-back of a safety instruction, and sends the wrong ones to a human.**

> **Status: under construction (AMD Developer Hackathon ACT III, Oct 2026).** Nothing in this repository has been measured yet.
> This README states only what exists in the code today; every number will come from a file in `bench/results/`.

## What it does

An operator radios *"Close valve two-oh-three, line Bravo."* The field technician answers *"Copy, closing two-oh-eight, line Bravo."*
Valve XV-208 is on a different line. The read-back exists so the operator catches this, and sometimes nobody does.

SayAgain transcribes both turns, asks a language model to **point at the words** that carry the action, the equipment tag and the line,
and then lets **plain code** decide: the words must really be in the transcript, every safety word must be accounted for, spoken digits and
NATO letters become plant tags through alias tables, and a pure function returns `VERIFIED`, `MISMATCH`, `NO-READBACK` or `ABSTAIN`.
A shift supervisor confirms or dismisses every card. Nothing is ever sent to the plant.

## Status

Built so far: repository scaffold. See the commit history.

## License

MIT, see [LICENSE](LICENSE).
