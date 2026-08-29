#!/usr/bin/env bash
# sweep_fix_combo.sh — run one bench test with several fixes enabled together
#
# Usage:
#   ./vision_stack/scripts/sweep_fix_combo.sh <track> <paramtag> <fix1,fix2,...>
#   ./vision_stack/scripts/sweep_fix_combo.sh T3 oa65_dh12 orientation_filt,dashed_dilate
#
# Unlike sweep_fix.sh (which isolates each of the 5 fixes into its own log),
# this runs ONE pipeline invocation with all requested fixes turned on at
# once, so combinations like "does orientation_filt hold up once
# dashed_dilate is also active" can actually be tested.
#
# Naming convention (interaction-calibration oriented):
#   - Runs are grouped into a directory per active-fix combo (canonical
#     I T O D A letter order), since that's what you hold constant across
#     several calibration attempts:
#       vision_stack/logs/T3Logs/OD/
#   - Inside that, each attempt is named after the variable(s) you changed
#     plus a timestamp, so nothing overwrites when you circle back to the
#     same combo later:
#       oa65_dh12__20260829-1512.log
#       oa65_dh12__20260829-1512_frames.csv
#       oa65_dh12__20260829-1512_contours.csv
#   - paramtag is freeform — write whatever short tag actually describes
#     what's different about this run (suggested abbreviations: oa=
#     orientation angle, rt=reject_top_frac, dh=dilate kernel_h, dw=dilate
#     kernel_w, sm=side_margin_frac, tf=top_frac). It's for your own
#     scanning; the CSVs already carry fix_roi_inset..fix_anchor_halves as
#     real columns for analysis.

set -euo pipefail

# Self-locate the repo root, no matter where this script is invoked from.
# Script lives at <repo_root>/vision_stack/scripts/sweep_fix_combo.sh, so the
# repo root is two directories up from this file's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TRACK="${1:?Usage: $0 <track> <paramtag> <fix1,fix2,...>}"
PARAMTAG="${2:?Usage: $0 <track> <paramtag> <fix1,fix2,...>}"
FIXES_RAW="${3:?Usage: $0 <track> <paramtag> <fix1,fix2,...>}"

# Canonical order matches Config.FIX_NAMES / flags_str() in config.py.
CANONICAL_ORDER=(roi_inset trapezoid_mask orientation_filt dashed_dilate anchor_halves)
declare -A FIX_LETTER=(
  [roi_inset]=I
  [trapezoid_mask]=T
  [orientation_filt]=O
  [dashed_dilate]=D
  [anchor_halves]=A
)

# Split the comma-separated fix list, trim whitespace, validate each name.
declare -A REQUESTED=()
IFS=',' read -ra RAW_ARR <<< "$FIXES_RAW"
for raw in "${RAW_ARR[@]}"; do
  fix="$(echo "$raw" | xargs)"
  if [[ -z "${FIX_LETTER[$fix]+x}" ]]; then
    echo "Unknown fix: '$fix'" >&2
    echo "Valid fixes: ${CANONICAL_ORDER[*]}" >&2
    exit 1
  fi
  REQUESTED[$fix]=1
done

if [[ ${#REQUESTED[@]} -lt 2 ]]; then
  echo "sweep_fix_combo.sh is for 2+ fixes at once. For a single fix, use sweep_fix.sh instead." >&2
  exit 1
fi

# Walk the canonical order (not the CLI order) so the --fix args, the combo
# directory letters, and the log's own [I T O D A] tag all agree.
FIX_ARGS=()
LETTERS=""
for fix in "${CANONICAL_ORDER[@]}"; do
  if [[ -n "${REQUESTED[$fix]+x}" ]]; then
    FIX_ARGS+=(--fix "$fix")
    LETTERS+="${FIX_LETTER[$fix]}"
  fi
done

COMBO_DIR="vision_stack/logs/${TRACK}Logs/${LETTERS}"
mkdir -p "$COMBO_DIR"

TS="$(date +%Y%m%d-%H%M)"
base="${COMBO_DIR}/${PARAMTAG}__${TS}"
out="${base}.log"
echo "[running] fixes=(${!REQUESTED[*]}) -> ${out}  (+ ${base}_frames.csv / ${base}_contours.csv)"
python3 vision_stack/src/run_pipeline.py \
  --frames "vision_stack/frames/track${TRACK}" \
  "${FIX_ARGS[@]}" \
  --csv "$base" \
  > "$out" 2>&1

echo "Done. Log + CSV pair written to ${base}.{log,_frames.csv,_contours.csv}"