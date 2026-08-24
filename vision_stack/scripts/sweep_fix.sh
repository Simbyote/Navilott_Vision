#!/usr/bin/env bash
# run_fix_sweep.sh — run all 5 single-fix bench tests with one command
#
# Usage:
#   ./vision_stack/scripts/sweep_fix.sh <track> <prefix> <value>
#   ./vision_stack/scripts/sweep_fix.sh T3 MA 400
#
# Produces a matching existing naming convention such like:
#   vision_stack/logs/T3Logs/MA_T3R_400.log   (roi_inset)
#   vision_stack/logs/T3Logs/MA_T3T_400.log   (trapezoid_mask)
#   vision_stack/logs/T3Logs/MA_T3O_400.log   (orientation_filt)
#   vision_stack/logs/T3Logs/MA_T3D_400.log   (dashed_dilate)
#   vision_stack/logs/T3Logs/MA_T3A_400.log   (anchor_halves)

#!/usr/bin/env bash
set -euo pipefail

# Self-locate the repo root, no matter where this script is invoked from.
# Script lives at <repo_root>/vision_stack/scripts/sweep_fix.sh, so the
# repo root is two directories up from this file's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRACK="${1:?Usage: $0 <track> <prefix> <value>}"
PREFIX="${2:?Usage: $0 <track> <prefix> <value>}"
VALUE="${3:?Usage: $0 <track> <prefix> <value>}"

LOG_DIR="vision_stack/logs/${TRACK}Logs"
mkdir -p "$LOG_DIR"

declare -A FIX_LETTER=(
  [roi_inset]=R
  [trapezoid_mask]=T
  [orientation_filt]=O
  [dashed_dilate]=D
  [anchor_halves]=A
)

for fix in roi_inset trapezoid_mask orientation_filt dashed_dilate anchor_halves; do
  letter="${FIX_LETTER[$fix]}"
  out="${LOG_DIR}/${PREFIX}_${TRACK}${letter}_${VALUE}.log"
  echo "[running] --fix ${fix} -> ${out}"
  python3 vision_stack/src/run_pipeline.py \
    --frames "vision_stack/frames/track${TRACK}" \
    --fix "$fix" \
    > "$out" 2>&1
done

echo "Done. 5 logs written to ${LOG_DIR}/"