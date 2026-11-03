#!/usr/bin/env bash
# =============================================================================
# farmland_prep.sh - ESA WorldCover 2021 tiles -> cropland block files for the
# Flood Impact Model. Downloads each 3x3 degree tile from ESA's public AWS
# bucket, converts it, validates it, then deletes the download.
#
# Usage (from any folder, in the conda environment):
#   ./farmland_prep.sh                                  # every tile in the world
#   ./farmland_prep.sh --bbox -81 21 -75 24             # just tiles in a lon/lat box
#   ./farmland_prep.sh --local-tifs /path/ESA_tiles     # use tiles already on disk
#   ./farmland_prep.sh --workers 8                      # more tiles at once
#
# All options are passed straight to farmland_prep.py (`python farmland_prep.py
# --help` lists them). Output: <work-dir>/Farmland/<tile>.parquet
# Safe to rerun: finished tiles are skipped.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

mkdir -p "$SCRIPT_DIR/logs"
log="$SCRIPT_DIR/logs/farmland_$(date '+%Y%m%d_%H%M%S').log"
echo "Log: $log"

# On a Mac, `caffeinate -i` keeps the computer awake until the run finishes.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$PYTHON" "$SCRIPT_DIR/farmland_prep.py" "$@" 2>&1 | tee "$log"
else
    "$PYTHON" "$SCRIPT_DIR/farmland_prep.py" "$@" 2>&1 | tee "$log"
fi
