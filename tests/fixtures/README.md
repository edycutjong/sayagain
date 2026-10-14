# Test fixtures

`tokenfactory_synthetic.json` is a **synthetic fixture**: hand-written responses in the OpenAI chat-completions /
models / audio-transcriptions shape. It is **not** a recorded AMD Token Factory response — no key existed when the
client was written. Replace it with recorded responses (redacted) once the access gate has run.

`tts-fixture-hero-readback.wav` (+ `.json`) is badged **`tts·fixture`**: a macOS `say` voice reading the hero read-back.
It exists only for the opt-in local-Whisper unit test (`SAYAGAIN_RUN_WHISPER=1`) and never appears in an eval row —
every eval row uses human recordings (bench/PROTOCOL.md §6).
