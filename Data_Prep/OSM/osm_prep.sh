#!/usr/bin/env bash
# =============================================================================
# osm_prep.sh - turn Geofabrik OpenStreetMap extracts into the filtered,
# bbox-covered, spatially sorted GeoParquet files the Flood Impact Model reads.
#
# For each region:
#   1. download  <region>-latest.osm.pbf from Geofabrik and check its MD5
#   2. convert   one ogr2ogr call per layer (lines, multipolygons):
#                read the .pbf directly, keep only flood-impact features,
#                write GeoParquet with a bbox column, sorted by location
#   3. validate  read every row group of the new file (validate_parquet.py)
#   4. checksum  write <file>.sha256 so copies can be verified later
#
# Usage (from this folder, in the hydroinformatics environment):
#   ./osm_prep.sh                      # all regions in REGIONS below
#   ./osm_prep.sh central-america      # just one (or several) regions
#   DRY_RUN=1 ./osm_prep.sh asia       # print the commands, run nothing
#
# Safe to rerun: finished steps are skipped.
# =============================================================================

# Stop on the first error (-e), on use of an unset variable (-u), and if any
# command inside a pipe fails (-o pipefail). Without this, bash keeps going
# after a failed command, which is how bad files slip through.
set -euo pipefail

# ----------------------------------------------------------------------------
# Settings (edit these)
# ----------------------------------------------------------------------------
# Geofabrik region names, exactly as they appear in the download URLs.
REGIONS="africa antarctica asia australia-oceania central-america europe north-america south-america"

# Everything this script writes goes under WORK_DIR:
#   pbf/       downloaded .osm.pbf files (kept so a rerun needn't re-download)
#   Lines/     <region>-lines.parquet      <- point the H3 config's OSM_ROOT here
#   Polygons/  <region>-polygons.parquet
#   tmp/       GDAL scratch space (can reach tens of GB for europe/asia)
WORK_DIR="${WORK_DIR:-/Users/maugh24/Flood_Impact_Model/Files/OSM_Prep}"

# Memory (MB) GDAL may use for its node index before spilling to tmp/.
# Bigger = faster on large regions. Keep it well under your RAM.
OSM_TMP_MB=4000

GEOFABRIK_URL="https://download.geofabrik.de"
PYTHON="${PYTHON:-python}"   # use $PYTHON if already set, otherwise "python"

# Features kept - the same lists the H3 impact code counts.
HIGHWAYS="'motorway','trunk','primary','secondary','tertiary','residential','motorway_link','trunk_link','primary_link','secondary_link','tertiary_link','living_street','busway','footway','cycleway'"
RAILWAYS="'light_rail','monorail','rail','subway','tram'"
LINES_WHERE="highway IN (${HIGHWAYS}) OR railway IN (${RAILWAYS})"
POLYGONS_WHERE="building IS NOT NULL"

# ----------------------------------------------------------------------------
# Derived paths (no need to edit)
# ----------------------------------------------------------------------------
# The folder this script lives in, so it finds its helper files no matter
# which folder you run it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSM_CONFIG="$SCRIPT_DIR/osmconf_flood_impact.ini"
VALIDATOR="$SCRIPT_DIR/validate_parquet.py"

PBF_DIR="$WORK_DIR/pbf"
TMP_DIR="$WORK_DIR/tmp"
export CPL_TMPDIR="$TMP_DIR"   # tells GDAL where to put its scratch files

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# run CMD ARGS... : run a command, or just print it when DRY_RUN=1.
run() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '  DRY RUN:'; printf ' %q' "$@"; echo
    else
        "$@"
    fi
}

# md5_of FILE : print the file's MD5 hash (macOS has `md5`, Linux `md5sum`).
md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        md5sum "$1" | cut -d' ' -f1
    fi
}

# pbf_ok REGION : succeed if the region's .pbf exists and matches its .md5.
pbf_ok() {
    local pbf="$PBF_DIR/$1-latest.osm.pbf"
    [[ -f "$pbf" && -f "$pbf.md5" ]] || return 1
    local expected actual
    expected="$(cut -d' ' -f1 "$pbf.md5")"
    actual="$(md5_of "$pbf")"
    [[ "$expected" == "$actual" ]]
}

# ----------------------------------------------------------------------------
# Step 1: download
# ----------------------------------------------------------------------------
download_region() {
    local region="$1"
    local pbf="$PBF_DIR/$region-latest.osm.pbf"
    local url="$GEOFABRIK_URL/$region-latest.osm.pbf"

    if pbf_ok "$region"; then
        log "$region: download already done (MD5 matches), skipping"
        return 0
    fi
    if [[ -f "$pbf" && ! -f "$pbf.md5" ]]; then
        log "$region: using existing $pbf (no .md5 to check against)"
        return 0
    fi

    log "$region: downloading $url"
    # Fetch the checksum first: Geofabrik rebuilds extracts daily, and the
    # .md5 must belong to the same build as the .pbf we download right after.
    run curl --fail --location --silent --show-error -o "$pbf.md5" "$url.md5"
    # --continue-at - resumes a half-finished download instead of restarting.
    run curl --fail --location --retry 5 --continue-at - -o "$pbf.part" "$url"
    run mv "$pbf.part" "$pbf"

    if [[ "${DRY_RUN:-0}" != "1" ]] && ! pbf_ok "$region"; then
        log "$region: MD5 MISMATCH - the download is damaged or Geofabrik updated mid-download."
        log "$region: delete $pbf and rerun."
        return 1
    fi
}

# ----------------------------------------------------------------------------
# Steps 2-4: convert, validate, checksum one layer
# ----------------------------------------------------------------------------
# convert_layer REGION OSM_LAYER WHERE OUT_FOLDER OUT_SUFFIX
convert_layer() {
    local region="$1" layer="$2" where="$3" folder="$4" suffix="$5"
    local pbf="$PBF_DIR/$region-latest.osm.pbf"
    local out="$WORK_DIR/$folder/$region-$suffix.parquet"
    local partial="$out.partial"      # not *.parquet, so the H3 run never picks it up

    if [[ -f "$out" && -f "$out.sha256" ]]; then
        log "$region/$layer: already converted, skipping"
        return 0
    fi

    log "$region/$layer: converting -> $out"
    rm -f "$partial"
    # -oo  = options for READING the .pbf     -lco = options for WRITING the parquet
    run ogr2ogr -f Parquet "$partial" "$pbf" "$layer" \
        -oo CONFIG_FILE="$OSM_CONFIG" \
        -oo MAX_TMPFILE_SIZE="$OSM_TMP_MB" \
        -where "$where" \
        -lco GEOMETRY_NAME=geometry \
        -lco WRITE_COVERING_BBOX=YES \
        -lco SORT_BY_BBOX=YES \
        -lco ROW_GROUP_SIZE=100000 \
        -lco COMPRESSION=ZSTD \
        -progress

    log "$region/$layer: validating"
    run "$PYTHON" "$VALIDATOR" "$partial"     # exits non-zero -> set -e stops the script

    run mv "$partial" "$out"
    # cd into the folder first so the .sha256 stores just the file name;
    # that way `shasum -c` works wherever the folder is copied to.
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        ( cd "$(dirname "$out")" && shasum -a 256 "$(basename "$out")" > "$(basename "$out").sha256" )
    fi
    log "$region/$layer: done"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# Regions from the command line if given ("$*"), otherwise the REGIONS list.
regions="${*:-$REGIONS}"

for tool in ogr2ogr curl shasum "$PYTHON"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Missing required tool: $tool"; exit 1; }
done
[[ -f "$OSM_CONFIG" ]] || { echo "Missing $OSM_CONFIG"; exit 1; }

mkdir -p "$PBF_DIR" "$TMP_DIR" "$WORK_DIR/Lines" "$WORK_DIR/Polygons"

log "OSM prep -> $WORK_DIR"
log "regions: $regions"
for region in $regions; do
    download_region "$region"
    convert_layer "$region" lines         "$LINES_WHERE"    Lines    lines
    convert_layer "$region" multipolygons "$POLYGONS_WHERE" Polygons polygons
done
log "all done"
