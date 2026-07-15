#!/usr/bin/env bash
# crispz-studio - X/Y/Z grid CLI example (see README_CLI.md).
# Runs a small 2x2 comparison grid (Steps x Guidance) for a prompt, then prints
# the annotated contact-sheet path (saved in out/xyz_<timestamp>/).
#
# Usage:
#   ./xyz_example.sh                          -> demo prompt
#   ./xyz_example.sh "your prompt here"       -> your prompt
#
# Tips (edit the --xyz lines below):
#   - up to 3 axes (X, Y, Z), values comma-separated, quotes protect commas
#   - axes: Checkpoint, Sampler, Schedule, Steps, Guidance, Seed, ESRGAN model,
#     Factor, Denoise, Tile, Refine tile, LoRA weight, Performance, Prompt S/R
#   - Ctrl+C assembles a partial sheet with the cells rendered so far

set -e
cd "$(dirname "$0")"

PROMPT="${1:-a red fox in a snowy forest, cinematic light}"

# Python: .venv if present, else system
RUNPY=python3
[ -x ".venv/bin/python" ] && RUNPY=".venv/bin/python"

echo "[xyz example] prompt: $PROMPT"
echo "[xyz example] grid  : Steps=4,8  x  Guidance=0,3.5  (4 renders + 1 sheet)"
echo

"$RUNPY" app.py --cli --txt2img --prompt "$PROMPT" \
    --xyz "Steps=4,8" \
    --xyz "Guidance=0, 3.5" \
    --save-mode local --output-dir out

echo
echo "[xyz example] done. The last line above is the annotated sheet path."
