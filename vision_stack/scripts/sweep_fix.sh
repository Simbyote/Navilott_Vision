#!/usr/bin/env bash
# sweep_fix.sh — run all 5 single-fix bench tests with one command
#
# Usage:
#   ./vision_stack/scripts/sweep_fix.sh <track> <paramtag>
#   ./vision_stack/scripts/sweep_fix.sh T3 baseline_calib1
#
# Runs each of the 5 fixes in isolation (its own pipeline invocation),
# one directory per fix letter, so a rerun with a different paramtag never
# collides with an earlier attempt:
#   vision_stack/logs/T3Logs/I/baseline_calib1__20260829-1512.log   (roi_inset)
#   vision_stack/logs/T3Logs/T/baseline_calib1__20260829-1512.log   (trapezoid_mask)
#   vision_stack/logs/T3Logs/O/baseline_calib1__20260829-1512.log   (orientation_filt)
#   vision_stack/logs/T3Logs/D/baseline_calib1__20260829-1512.log   (dashed_dilate)
#   vision_stack/logs/T3Logs/A/baseline_calib1__20260829-1512.log   (anchor_halves)
#
# paramtag is freeform — describe what this isolated-fix pass is checking
# (e.g. a calibration round name). For testing how fixes interact with
# each other, use sweep_fix_combo.sh instead — that's the one that runs
# several fixes together in a single invocation.

set -euo pipefail

# Self-locate the repo root, no matter where this script is invoked from.
# Script lives at <repo_root>/vision_stack/scripts/sweep_fix.sh, so the
# repo root is two directories up from this file's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRACK="${1:?Usage: $0 <track> <paramtag>}"
PARAMTAG="${2:?Usage: $0 <track> <paramtag>}"

TS="$(date +%Y%m%d-%H%M)"

declare -A FIX_LETTER=(
  [roi_inset]=I
  [trapezoid_mask]=T
  [orientation_filt]=O
  [dashed_dilate]=D
  [anchor_halves]=A
)

for fix in roi_inset trapezoid_mask orientation_filt dashed_dilate anchor_halves; do
  letter="${FIX_LETTER[$fix]}"
  fix_dir="vision_stack/logs/${TRACK}Logs/${letter}"
  mkdir -p "$fix_dir"
  base="${fix_dir}/${PARAMTAG}__${TS}"
  out="${base}.log"
  echo "[running] --fix ${fix} -> ${out}  (+ ${base}_frames.csv / ${base}_contours.csv)"
  python3 vision_stack/src/run_pipeline.py \
    --frames "vision_stack/frames/track${TRACK}" \
    --fix "$fix" \
    --csv "$base" \
    > "$out" 2>&1
done

echo "Done. 5 logs + CSV pairs written under vision_stack/logs/${TRACK}Logs/"