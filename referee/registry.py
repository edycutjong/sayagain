"""Plant tag registry: which tags exist and which line each one is on.

The registry lives in code and data, never in the model prompt. A bare spoken number ("two-oh-eight") resolves only by a
unique numeric suffix in the registry — never from the instruction being checked.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.csv"


@dataclass(frozen=True)
class Row:
    tag: str
    type: str
    line: str
    area: str
    allowed_actions: tuple[str, ...]
    spoken_variants: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    tag: str  # "XV-203", or the bare suffix ("205") when nothing matches
    status: str  # "ok" | "not_in_registry" | "ambiguous"
    line: str | None  # registry line when status == "ok"
    candidates: tuple[str, ...] = ()


class Registry:
    def __init__(self, rows: list[Row]):
        self.rows: dict[str, Row] = {r.tag: r for r in rows}
        self.by_suffix: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            self.by_suffix[r.tag.split("-", 1)[1]].append(r.tag)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> Registry:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                Row(
                    tag=d["tag"],
                    type=d["type"],
                    line=d["line"],
                    area=d["area"],
                    allowed_actions=tuple(a for a in d["allowed_actions"].split("|") if a),
                    spoken_variants=tuple(v for v in d["spoken_variants"].split("|") if v),
                )
                for d in csv.DictReader(f)
            ]
        return cls(rows)

    def __contains__(self, tag: str) -> bool:
        return tag in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def lines(self) -> frozenset[str]:
        """The plant's line letters ("A" … "D")."""
        return frozenset(r.line for r in self.rows.values())

    @property
    def tag_letters(self) -> frozenset[str]:
        """Letters that end a tag ("MOV-22H" → "H"): a bare NATO word may complete one."""
        return frozenset(t[-1] for t in self.rows if t[-1].isalpha())

    def line_of(self, tag: str) -> str | None:
        r = self.rows.get(tag)
        return r.line if r else None

    def resolve(self, prefix: str | None, suffix: str) -> Resolution:
        """A spoken tag -> registry tag. Prefix known: exact lookup. Bare number: unique suffix or nothing."""
        suffix = suffix.upper()
        if prefix is not None:
            tag = f"{prefix}-{suffix}"
            if tag in self.rows:
                return Resolution(tag, "ok", self.rows[tag].line)
            return Resolution(tag, "not_in_registry", None)
        cands = self.by_suffix.get(suffix, [])
        if len(cands) == 1:
            return Resolution(cands[0], "ok", self.rows[cands[0]].line)
        if len(cands) > 1:
            return Resolution(suffix, "ambiguous", None, tuple(sorted(cands)))
        return Resolution(suffix, "not_in_registry", None)

    def suffixes_unique(self) -> bool:
        return all(len(v) == 1 for v in self.by_suffix.values())
