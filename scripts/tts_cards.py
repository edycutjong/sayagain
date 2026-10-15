"""EXPLORATORY ONLY — synthesise one WAV per card with ElevenLabs text-to-speech (badge `tts·synthetic`).

    uv run python scripts/tts_cards.py [--cards data/cards.jsonl] [--out data/tts/clean] [--only H-01,R-01,...]

This is NOT the pre-registered audio corpus (bench/PROTOCOL.md §2: the cards are recorded by people). It is the
deviation logged in bench/PROTOCOL.md "Deviations" (2026-10-15): synthetic voices through the simulated radio path,
reported as their own rows, never as or beside the human audio row.

Layout = what sayagain/ingest.py expects: one WAV per card, file name = exchange_id, 16 kHz mono, turns in card order
separated by silence (a card's `pause` turn adds its own seconds). Two fixed voices: one for every operator turn, one
for every field-technician turn. Audio stays under data/tts/ (git-ignored); the manifest with voice ids, model id,
date and character counts is written to bench/results/tts/manifest.json (no audio, no key, no key name).

Credentials: ~/.config/elevenlabs/credentials.json ({"keys": [{"name", "key"}, ...]}); a key that is out of quota or
rate limited is skipped for the next one. Per-turn MP3s are cached under data/tts/turns/ so a re-run bills nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sayagain.audio import SR, load, save  # noqa: E402

BADGE = "tts·synthetic"
API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128"
CREDS = Path.home() / ".config" / "elevenlabs" / "credentials.json"
MODEL = "eleven_multilingual_v2"  # multilingual: the code-switched Bahasa/English lines
VOICES = {  # premade ElevenLabs voices
    "operator": {"voice_id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel"},
    "tech": {"voice_id": "iP95p4xoKVk53GoZ742B", "name": "Chris"},
}
SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "speed": 1.0, "use_speaker_boost": True}
SEED = 1018
LEAD_S, GAP_S, TAIL_S = 0.4, 0.9, 0.4  # silences: before the first turn, between turns, after the last


def _keys() -> list[str]:
    data = json.loads(CREDS.read_text(encoding="utf-8"))
    return [k["key"] for k in data.get("keys", []) if k.get("key", "").startswith("sk_")]


class Synth:
    def __init__(self, cache: Path):
        self.keys, self.k, self.cache = _keys(), 0, cache
        self.chars_billed = 0

    def mp3(self, role: str, text: str) -> Path:
        voice = VOICES[role]["voice_id"]
        h = hashlib.sha256(f"{voice}|{MODEL}|{SEED}|{json.dumps(SETTINGS, sort_keys=True)}|{text}".encode()).hexdigest()
        p = self.cache / f"{h[:20]}.mp3"
        if p.exists():
            return p
        body = json.dumps({"text": text, "model_id": MODEL, "voice_settings": SETTINGS, "seed": SEED}).encode()
        while self.k < len(self.keys):
            req = urllib.request.Request(
                API.format(voice=voice),
                data=body,
                headers={"xi-api-key": self.keys[self.k], "Content-Type": "application/json", "Accept": "audio/mpeg"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    audio = r.read()
            except urllib.error.HTTPError as e:
                msg = e.read()[:200].decode("utf-8", "replace")
                if e.code in (401, 402, 403, 429) or "quota" in msg:
                    self.k += 1  # out of quota / rate limited / rejected: next key (its own pool)
                    continue
                if e.code >= 500:
                    time.sleep(10)
                    continue
                raise SystemExit(f"ElevenLabs HTTP {e.code}: {msg}") from e
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(audio)
            self.chars_billed += len(text)
            return p
        raise SystemExit("every ElevenLabs key is out of quota or rejected")


def build_card(card: dict, synth: Synth) -> tuple[np.ndarray, list[dict]]:
    parts = [np.zeros(int(LEAD_S * SR), dtype=np.float32)]
    turns = []
    first = True
    for t in card["turns"]:
        if t["role"] == "pause":
            parts.append(np.zeros(int(t["seconds"] * SR), dtype=np.float32))
            continue
        if not first:
            parts.append(np.zeros(int(GAP_S * SR), dtype=np.float32))
        first = False
        x = load(synth.mp3(t["role"], t["text"]))
        peak = float(np.max(np.abs(x))) or 1.0
        parts.append((x / peak * 0.7).astype(np.float32))
        turns.append({"role": t["role"], "chars": len(t["text"]), "seconds": round(len(x) / SR, 3)})
    parts.append(np.zeros(int(TAIL_S * SR), dtype=np.float32))
    return np.concatenate(parts), turns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=Path, default=ROOT / "data" / "cards.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tts" / "clean")
    ap.add_argument("--manifest", type=Path, default=ROOT / "bench" / "results" / "tts" / "manifest.json")
    ap.add_argument("--only", help="comma-separated exchange ids (default: every card)")
    args = ap.parse_args(argv)
    cards = [json.loads(line) for line in args.cards.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        keep = set(args.only.split(","))
        cards = [c for c in cards if c["exchange_id"] in keep]
    synth = Synth(ROOT / "data" / "tts" / "turns")
    rows = []
    for i, card in enumerate(cards, 1):
        x, turns = build_card(card, synth)
        path = args.out / f"{card['exchange_id']}.wav"
        save(path, x)
        rows.append(
            {
                "exchange_id": card["exchange_id"],
                "chars": sum(t["chars"] for t in turns),
                "seconds": round(len(x) / SR, 3),
                "turns": turns,
                "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if i % 20 == 0:
            print(f"{i}/{len(cards)} cards · {synth.chars_billed} chars billed this run", flush=True)
    manifest = {
        "badge": BADGE,
        "exploratory": True,
        "note": "synthetic speech, not the pre-registered human recordings (bench/PROTOCOL.md Deviations, 2026-10-15)",
        "provider": "ElevenLabs text-to-speech API",
        "model_id": MODEL,
        "voices": VOICES,
        "voice_settings": SETTINGS,
        "seed": SEED,
        "output_format": "mp3_44100_128, decoded and resampled to 16 kHz mono WAV",
        "layout": {"lead_s": LEAD_S, "gap_s": GAP_S, "tail_s": TAIL_S, "per_turn_peak": 0.7},
        "synthesised_on": datetime.now(UTC).date().isoformat(),
        "cards_sha256": hashlib.sha256(args.cards.read_bytes()).hexdigest(),
        "n_cards": len(rows),
        "chars_in_cards": sum(r["chars"] for r in rows),
        "chars_sent_this_run": synth.chars_billed,
        "note_chars": "a (voice, text) pair said twice is synthesised once; a re-run sends only uncached turns",
        "audio_committed": False,
        "cards": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} WAVs -> {args.out} · manifest {args.manifest} · {synth.chars_billed} chars billed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
