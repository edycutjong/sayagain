"""Energy VAD for clean, one-card-per-file recordings: where are the turns?

Used to cut a card's WAV into turns on the CLEAN take; noisy mixes of the same take reuse those boundaries (same speech,
same timing), so the SNR rows measure ASR + extraction, not the segmenter. The live pipeline's VAD is separate.
"""

from __future__ import annotations

import numpy as np

from sayagain.audio import SR


def segments(
    x: np.ndarray,
    sr: int = SR,
    *,
    frame_s: float = 0.03,
    rel_db: float = -30.0,
    min_silence_s: float = 0.45,
    min_speech_s: float = 0.25,
    pad_s: float = 0.12,
) -> list[tuple[float, float]]:
    """Speech regions (start_s, end_s): frames within `rel_db` of the loudest frame, gaps < min_silence_s bridged."""
    n = max(1, int(frame_s * sr))
    m = len(x) // n
    if m == 0:
        return []
    e = np.sqrt(np.mean(x[: m * n].reshape(m, n).astype(np.float64) ** 2, axis=1))
    if e.max() == 0:
        return []
    floor = max(np.percentile(e, 10) * 3.0, e.max() * 10 ** (rel_db / 20))
    voiced = e > floor
    regions: list[list[int]] = []
    gap_frames = int(min_silence_s / frame_s)
    for i, v in enumerate(voiced):
        if not v:
            continue
        if regions and i - regions[-1][1] <= gap_frames:
            regions[-1][1] = i
        else:
            regions.append([i, i])
    out = []
    for a, b in regions:
        start, end = a * frame_s, (b + 1) * frame_s
        if end - start >= min_speech_s:
            out.append((max(0.0, start - pad_s), min(len(x) / sr, end + pad_s)))
    return out
