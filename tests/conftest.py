import os
import tempfile

import pytest

from referee.audit import Exchange, judge
from referee.regex_extractor import extract
from referee.registry import Registry
from referee.turns import Turn, analyze

# sayagain.server builds its app, and a reviews database under SAYAGAIN_VAR, at import: keep that out of the repo
os.environ.setdefault("SAYAGAIN_VAR", tempfile.mkdtemp(prefix="sayagain-test-var-"))


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.load()


def make_turn(
    text: str,
    *,
    t: float = 0.0,
    dur: float = 2.0,
    channel: str = "CH-07",
    tid: str | None = None,
    extraction: dict | None = None,
    **kw,
) -> Turn:
    return Turn(
        id=tid or f"{channel}@{t:g}",
        channel=channel,
        t_start=t,
        t_end=t + dur,
        transcript=text,
        extraction=extraction if extraction is not None else extract(text),
        **kw,
    )


def judge_texts(registry: Registry, instruction: str, readback: str | None, **kw):
    """Judge one exchange with the regex extractor (rung 0) standing in for the model."""
    ins = analyze(make_turn(instruction), registry)
    rb = analyze(make_turn(readback, t=3.0), registry) if readback is not None else None
    return judge(Exchange("CH-07", ins, rb), **kw)
