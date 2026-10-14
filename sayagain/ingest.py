"""Ingest one card's WAV (one file per card, file name = exchange_id): find the turns with the VAD, cut them out.

The turn boundaries always come from the CLEAN take; a noisy or radio-simulated copy of the same take is cut at the
same boundaries (same speech, same timing), so the SNR rows measure ASR + extraction, not the segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sayagain.audio import SR, load
from sayagain.vad import segments


@dataclass
class TurnAudio:
    id: str  # "<exchange_id>-<k>"
    channel: str  # the card's exchange_id: one card = one channel in the eval
    t_start: float
    t_end: float
    samples: np.ndarray  # mono float32 at SR

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


def cut(clean: np.ndarray, exchange_id: str, *, variant: np.ndarray | None = None, sr: int = SR) -> list[TurnAudio]:
    """VAD on `clean`; the samples come from `variant` (a noisy / radio copy of the same take) when given."""
    src = clean if variant is None else variant
    if len(src) != len(clean):
        raise ValueError("a variant must be the same take as the clean file (same length)")
    out = []
    for k, (s, e) in enumerate(segments(clean, sr)):
        out.append(TurnAudio(f"{exchange_id}-{k}", exchange_id, s, e, src[int(s * sr) : int(e * sr)].copy()))
    return out


def ingest_card(path: Path | str, *, variant: np.ndarray | None = None) -> list[TurnAudio]:
    """One card WAV -> its turns, in time order."""
    p = Path(path)
    return cut(load(p), p.stem, variant=variant)
