#!/usr/bin/env bash
# Copy the MIMII noise subset (Hitachi, CC BY-SA 4.0 — eval only, never committed) into data/noise/.
# Source cache: ~/.cache/sayagain/mimii/ (6_dB_{pump,valve,fan}, normal/, ids 00/02/04/06 × 10 files, Zenodo 3384388).
# If the cache is missing, fetch the subset from https://zenodo.org/records/3384388 into it first.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cache="${MIMII_CACHE:-$HOME/.cache/sayagain/mimii}"
dest="$here/data/noise/mimii"
if [ ! -d "$cache" ]; then
  echo "MIMII cache not found at $cache — download the 6_dB pump/valve/fan subset from Zenodo 3384388 first." >&2
  exit 2
fi
mkdir -p "$dest"
for machine in pump valve fan; do
  [ -d "$cache/$machine" ] && rsync -a "$cache/$machine/" "$dest/$machine/"
done
cp "$cache/NOTICE.md" "$dest/NOTICE.md" 2>/dev/null || true
echo "MIMII: $(find "$dest" -name '*.wav' | wc -l | tr -d ' ') WAVs -> $dest (git-ignored)"
