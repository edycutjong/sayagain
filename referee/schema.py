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


# ---------------------------------------------------------------------------------------------------------------------
# Grounding ablation only (`scripts/eval.py --ablate grounding`): the model returns NORMALISED values instead of copied
# words, so grounding and coverage cannot run. The vocabulary below is the alias tables' output alphabet (action verbs,
# tag-noun prefixes, line letters) — no registry content: which tags exist and their lines stay in code.

CANON_ACTIONS = ("CLOSE", "OPEN", "ISOLATE", "START", "STOP")
CANON_STATES = ("RUNNING", "STOPPED", "STANDBY")
CANON_FIELDS = ("action", "tag", "state", "line")

CANON_PROMPT = (
    "You read one radio transmission from a process-plant channel and normalise every safety command in it.\n"
    "commands: one entry per safety command (an action on one piece of equipment), with normalised values:\n"
    "  action: one of CLOSE, OPEN, ISOLATE, START, STOP (Bahasa verbs too: tutup = CLOSE, buka = OPEN, "
    "isolasi = ISOLATE, nyalakan / jalankan = START, matikan / hentikan = STOP), or null;\n"
    "  tag: equipment prefix + '-' + its number in digits, e.g. 'valve two-oh-three' -> 'XV-203', "
    "'motor valve four-one Delta' -> 'MOV-41D', 'pump one-forty' -> 'P-140' (valve / katup = XV, "
    "motor valve / motor-operated valve = MOV, pump / pompa = P); only the number in digits if no equipment word "
    "is spoken; null if no tag;\n"
    "  line: the line letter only ('line Bravo' / 'jalur Bravo' -> 'B'), or null;\n"
    "  state: one of RUNNING, STOPPED, STANDBY if an equipment state is stated, else null.\n"
    "If the speaker takes a word back with a correction word, give only the corrected value.\n"
    "kind: instruction (an order), readback (repeating an order back), say_again (asking for a repeat), other.\n"
    'Return only one JSON object, no prose: {"kind": ..., "commands": [{"action": ..., "tag": ..., "state": ..., '
    '"line": ...}]}'
)


def validate_canonical(obj: object) -> str | None:
    """None if `obj` is a well-formed canonical extraction (ablation mode), else a short reason. Never repaired."""
    if not isinstance(obj, dict):
        return "not an object"
    if set(obj) != {"kind", "commands"}:
        return f"keys {sorted(obj)}"
    if obj["kind"] not in KINDS:
        return f"kind {obj['kind']!r}"
    if not isinstance(obj["commands"], list):
        return "commands not a list"
    for c in obj["commands"]:
        if not isinstance(c, dict) or set(c) != set(CANON_FIELDS):
            return "command keys"
        if any(v is not None and not isinstance(v, str) for v in c.values()):
            return "value not a string"
    return None
