#!/usr/bin/env bash
# =============================================================================
# population_prep.sh - WorldPop global 1 km population raster -> population
# file for the Flood Impact Model. Downloads the raster if it isn't in
# <work-dir>/Population, converts it, validates it, then deletes the download.
#
# Usage (from any folder, in the flood_impact conda environment):
#   ./population_prep.sh                  # WorldPop 2024
#   ./population_prep.sh --year 2025      # another year
#   ./population_prep.sh --tif /path/global_pop_2025_CN_1km_R2025A_UA_v1.tif
#                                         # a raster you already have
#
# All options are passed straight to population_prep.py (`python
# population_prep.py --help` lists them).
# Output: <work-dir>/Population/worldpop_<year>_1km.parquet
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

mkdir -p "$SCRIPT_DIR/logs"
log="$SCRIPT_DIR/logs/population_$(date '+%Y%m%d_%H%M%S').log"
echo "Log: $log"

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$PYTHON" "$SCRIPT_DIR/population_prep.py" "$@" 2>&1 | tee "$log"
else
    "$PYTHON" "$SCRIPT_DIR/population_prep.py" "$@" 2>&1 | tee "$log"
fi
