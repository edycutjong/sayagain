"""The structured-output contract with the extractor (any rung): spans, never values."""

from __future__ import annotations

FIELDS = ("action_span", "tag_span", "state_span", "line_span")
KINDS = ("instruction", "readback", "say_again", "other")

SPAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "kind": {"enum": list(KINDS)},
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {f: {"type": ["string", "null"]} for f in FIELDS},
                "required": list(FIELDS),
                "additionalProperties": False,
            },
        },
        "retracted_spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "commands", "retracted_spans"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You read one radio transmission from a process-plant channel. Copy the exact words from the transcript. "
    "Do not correct, translate, or normalise. If a field is not spoken, return null.\n"
    "commands: one entry per safety command (an action on one piece of equipment). action_span = the verb words, "
    "tag_span = the equipment words incl. its number, line_span = the line words (e.g. 'line Bravo'), "
    "state_span = a stated equipment state.\n"
    "retracted_spans: words the speaker explicitly took back with a correction word.\n"
    "kind: instruction (an order), readback (repeating an order back), say_again (asking for a repeat), other."
)


def validate_extraction(obj: object) -> str | None:
    """None if `obj` follows SPAN_SCHEMA, else a short reason. Malformed output is never repaired."""
    if not isinstance(obj, dict):
        return "not an object"
    if set(obj) != {"kind", "commands", "retracted_spans"}:
        return f"keys {sorted(obj)}"
    if obj["kind"] not in KINDS:
        return f"kind {obj['kind']!r}"
    if not isinstance(obj["commands"], list):
        return "commands not a list"
    for c in obj["commands"]:
        if not isinstance(c, dict) or set(c) != set(FIELDS):
            return "command keys"
        if any(v is not None and not isinstance(v, str) for v in c.values()):
            return "span not a string"
    rs = obj["retracted_spans"]
    if not isinstance(rs, list) or any(not isinstance(s, str) for s in rs):
        return "retracted_spans"
    return None
