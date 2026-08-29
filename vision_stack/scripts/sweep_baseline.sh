#!/usr/bin/env bash
# sweep_baseline.sh — run one bench test with NO fixes active
#
# Usage:
#   ./vision_stack/scripts/sweep_baseline.sh <track> <gen> <paramtag>
#   ./vision_stack/scripts/sweep_baseline.sh T3 gen0 baseline
#
# No --fix flag is passed to run_pipeline.py, so Config() keeps every fix
# at its False default. combine_runs.py labels this combo "NONE" from the
# fix_roi_inset..fix_anchor_halves columns (see _derive_combo in
# combine_runs.py). Output goes in its own NONE/ directory, a sibling of
# the I/T/O/D/A dirs sweep_fix.sh writes and the combo dirs
# sweep_fix_combo.sh writes, nested under gen the same way both of those
# nest theirs:
#   vision_stack/logs/T3Logs/NONE/gen0/baseline__20260829-1512.log
#   vision_stack/logs/T3Logs/NONE/gen0/baseline__20260829-1512_frames.csv
#   vision_stack/logs/T3Logs/NONE/gen0/baseline__20260829-1512_contours.csv
#
# gen and paramtag follow the same conventions as sweep_fix.sh /
# sweep_fix_combo.sh — see FILENAME_LEGEND.md.

set -euo pipefail

# Self-locate the repo root, no matter where this script is invoked from.
# Script lives at <repo_root>/vision_stack/scripts/sweep_baseline.sh, so the
# repo root is two directories up from this file's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRACK="${1:?Usage: $0 <track> <gen> <paramtag>}"
GEN="${2:?Usage: $0 <track> <gen> <paramtag>}"
PARAMTAG="${3:?Usage: $0 <track> <gen> <paramtag>}"

TS="$(date +%Y%m%d-%H%M)"

NONE_DIR="vision_stack/logs/${TRACK}Logs/NONE/${GEN}"
mkdir -p "$NONE_DIR"
base="${NONE_DIR}/${PARAMTAG}__${TS}"
out="${base}.log"
echo "[running] no fixes -> ${out}  (+ ${base}_frames.csv / ${base}_contours.csv)"
python3 vision_stack/src/run_pipeline.py \
  --frames "vision_stack/frames/track${TRACK}" \
  --csv "$base" \
  > "$out" 2>&1

echo "Done. Log + CSV pair written to ${base}.{log,_frames.csv,_contours.csv}"