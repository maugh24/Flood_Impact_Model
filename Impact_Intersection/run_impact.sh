#!/usr/bin/env bash
# =============================================================================
# run_impact.sh - sum flood-impact exposure (population, buildings, farmland,
# roads/rail, bridges, dams, weirs) over one area-of-interest set and write a
# single CSV for FEWS4All.
#
# Usage (from any folder, in the conda environment):
#   ./run_impact.sh h3                    # H3 cells (resolution in config.py)
#   ./run_impact.sh geoglows              # GEOGLOWS catchments
#   ./run_impact.sh hydrobasins --check   # plan only: show inputs and tasks
#   ./run_impact.sh h3 --layers roads dams --workers 8
#
# The first word is an AOI name from AOIS in config.py; anything after it is
# passed straight to run_impact.py (see `python run_impact.py --help`).
# Output: <OUTPUT_FOLDER>/<aoi>/impact_<aoi>_<YYYY-MM>.csv
# A log of everything printed is saved in logs/ next to this script.
# Safe to rerun: finished work is skipped.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# Ask config.py which AOI names exist, so the usage message is always current.
aoi_names="$(cd "$SCRIPT_DIR" && "$PYTHON" -c 'import config; print(" ".join(config.AOIS))')"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./run_impact.sh <aoi> [options]"
    echo "  AOIs (from config.py): $aoi_names"
    echo "  Options: --check, --layers ..., --workers N, --label NAME, --merge-only"
    exit 1
fi

aoi="$1"
shift            # drop the AOI name; "$@" is now just the extra options

# Check the name before starting, for a friendlier message than Python's.
if [[ " $aoi_names " != *" $aoi "* ]]; then
    echo "Unknown AOI '$aoi'. Choose one of: $aoi_names"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
log="$SCRIPT_DIR/logs/${aoi}_$(date '+%Y%m%d_%H%M%S').log"
echo "Log: $log"

# On a Mac, `caffeinate -i` keeps the computer awake until the run finishes.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$PYTHON" "$SCRIPT_DIR/run_impact.py" --aoi "$aoi" "$@" 2>&1 | tee "$log"
else
    "$PYTHON" "$SCRIPT_DIR/run_impact.py" --aoi "$aoi" "$@" 2>&1 | tee "$log"
fi
