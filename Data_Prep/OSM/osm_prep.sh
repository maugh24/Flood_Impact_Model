#!/usr/bin/env bash
# =============================================================================
# osm_prep.sh - turn Geofabrik OpenStreetMap extracts into the filtered,
# bbox-covered, spatially sorted GeoParquet files the Flood Impact Model reads.
#
# For each region:
#   1. download  <region>-latest.osm.pbf from Geofabrik and check its MD5
#                (if Geofabrik has no .md5 at that moment, download unchecked)
#   2. convert   one ogr2ogr call per layer (lines, multipolygons):
#                read the .pbf directly, keep only flood-impact features,
#                write GeoParquet with a bbox column, sorted by location
#   3. validate  read every row group of the new file (validate_parquet.py)
#   4. checksum  write <file>.sha256 so copies can be verified later
#
# Usage (from this folder, in the flood_impact environment):
#   ./osm_prep.sh                      # all regions in REGIONS below
#   ./osm_prep.sh central-america      # just one (or several) regions
#   ./osm_prep.sh /path/to/asia-261029.osm.pbf
#                                      # a .pbf you already have (no download);
#                                      # outputs are named after the file:
#                                      # asia-261029-lines.parquet, ...
#   DRY_RUN=1 ./osm_prep.sh asia       # print the commands, run nothing
#
# Each region runs as its own process: if one fails, the others still run,
# and the failed ones are listed at the end. Safe to rerun: finished steps
# are skipped.
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
#   Lines/     <region>-lines.parquet      <- point the H3 config's OSM_ROOT here
#   Polygons/  <region>-polygons.parquet
#   Points/    <region>-points.parquet     (dams/weirs mapped as single points)
#   pbf/       downloads, deleted once a region is finished (see KEEP_PBF)
#   tmp/       GDAL scratch space (can reach tens of GB for europe/asia)
# $HOME is your home folder (/Users/<your user name>), so this works on any Mac.
WORK_DIR="${WORK_DIR:-$HOME/Flood_Impact_Model/Files/OSM_Prep}"

# Memory (MB) GDAL may use for its node index before spilling to tmp/.
# Bigger = faster on large regions. Keep it well under your RAM.
OSM_TMP_MB=4000

# 0 = delete a region's downloaded .pbf (and .md5) once all its layers are
#     converted and validated, to save disk. 1 = keep it.
#     A .pbf you pass in as a file path is never deleted.
KEEP_PBF="${KEEP_PBF:-0}"

# YES = sort features by location (fast bbox reads for the catchment runs and
# QGIS; GDAL writes an uncompressed temporary copy, which needs a lot of disk
# for Asia/Europe). NO = no temporary copy; the H3 run gives identical results.
SORT_BY_BBOX="${SORT_BY_BBOX:-YES}"

GEOFABRIK_URL="${GEOFABRIK_URL:-https://download.geofabrik.de}"
PYTHON="${PYTHON:-python}"   # use $PYTHON if already set, otherwise "python"

# Features kept. Roads/rail are the same lists the H3 impact code counts;
# road/rail segments keep their bridge=* tag (bridge=yes, viaduct, ...).
HIGHWAYS="'motorway','trunk','primary','secondary','tertiary','residential','motorway_link','trunk_link','primary_link','secondary_link','tertiary_link','living_street','busway','footway','cycleway'"
RAILWAYS="'light_rail','monorail','rail','subway','tram'"
DAMS="'dam','weir','check_dam'"          # waterway=* values
LINES_WHERE="highway IN (${HIGHWAYS}) OR railway IN (${RAILWAYS}) OR waterway IN (${DAMS})"
POLYGONS_WHERE="building IS NOT NULL OR waterway IN (${DAMS})"
POINTS_WHERE="waterway IN (${DAMS})"

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
    # It is occasionally missing for a while (europe gave "404" once), so try
    # a few times, then carry on without it rather than skip the region.
    # (`if ! command` is how bash lets a command fail without set -e stopping.)
    local got_md5=0 try
    for try in 1 2 3; do
        if run curl --fail --location --silent --show-error -o "$pbf.md5" "$url.md5"; then
            got_md5=1
            break
        fi
        log "$region: checksum not available (attempt $try of 3)"
        if [[ "$try" -lt 3 ]]; then sleep "${MD5_RETRY_WAIT:-60}"; fi
    done
    if [[ "$got_md5" != "1" ]]; then
        rm -f "$pbf.md5"
        log "$region: downloading WITHOUT an MD5 check (the conversion and validation still catch a broken file)"
    fi
    # --continue-at - resumes a half-finished download instead of restarting.
    run curl --fail --location --retry 5 --continue-at - -o "$pbf.part" "$url"
    run mv "$pbf.part" "$pbf"

    if [[ "${DRY_RUN:-0}" != "1" && -f "$pbf.md5" ]] && ! pbf_ok "$region"; then
        log "$region: MD5 MISMATCH - the download is damaged or Geofabrik updated mid-download."
        log "$region: delete $pbf and rerun."
        return 1
    fi
}

# ----------------------------------------------------------------------------
# Steps 2-4: convert, validate, checksum one layer
# ----------------------------------------------------------------------------
# convert_layer NAME PBF OSM_LAYER WHERE OUT_FOLDER OUT_SUFFIX
#   NAME is used for log messages and the output file name.
convert_layer() {
    local region="$1" pbf="$2" layer="$3" where="$4" folder="$5" suffix="$6"
    local out="$WORK_DIR/$folder/$region-$suffix.parquet"
    local partial="$out.partial"      # not *.parquet, so the H3 run never picks it up

    if [[ -f "$out" && -f "$out.sha256" ]]; then
        log "$region/$layer: already converted, skipping"
        return 0
    fi

    log "$region/$layer: converting -> $out"
    # Clear leftovers from a failed earlier run, including GDAL's temporary
    # sort file and its SQLite journal (a stale journal can corrupt the new one).
    rm -f "$partial" "$partial".tmp.gpkg*
    # -oo  = options for READING the .pbf     -lco = options for WRITING the parquet
    run ogr2ogr -f Parquet "$partial" "$pbf" "$layer" \
        -oo CONFIG_FILE="$OSM_CONFIG" \
        -oo MAX_TMPFILE_SIZE="$OSM_TMP_MB" \
        -where "$where" \
        -lco GEOMETRY_NAME=geometry \
        -lco WRITE_COVERING_BBOX=YES \
        -lco SORT_BY_BBOX="$SORT_BY_BBOX" \
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

# GDAL must be able to READ OpenStreetMap and WRITE Parquet. The list is saved
# in a variable first (rather than piped into grep) because `set -o pipefail`
# would treat grep closing the pipe early as a failure.
formats="$(ogrinfo --formats)"
if [[ "$formats" != *"OSM -vector-"* || "$formats" != *"Parquet -vector-"* ]]; then
    echo "This GDAL ($(command -v ogr2ogr)) is missing a driver it needs:"
    echo "$formats" | grep -i -E "osm|parquet" || true
    echo "Need both an 'OSM -vector-' and a 'Parquet -vector-' line above."
    echo "Fix (conda):  conda install -c conda-forge libgdal-arrow-parquet"
    exit 1
fi

mkdir -p "$PBF_DIR" "$TMP_DIR" "$WORK_DIR/Lines" "$WORK_DIR/Polygons" "$WORK_DIR/Points"

# ----------------------------------------------------------------------------
# More than one region: run each in its own process, so a failure in one
# (a missing download, say) doesn't stop the rest. Inside each process
# `set -e` works normally, so a broken step still stops THAT region before
# anything bad is kept.
# ----------------------------------------------------------------------------
if [[ "${OSM_PREP_ONE_REGION:-0}" != "1" ]]; then
    log "OSM prep -> $WORK_DIR"
    log "regions: $regions"
    failed=""
    for item in $regions; do
        if ! OSM_PREP_ONE_REGION=1 bash "$SCRIPT_DIR/osm_prep.sh" "$item"; then
            failed="$failed $item"
            log "$item: FAILED - continuing with the next region"
        fi
    done
    if [[ -n "$failed" ]]; then
        log "finished, but these region(s) failed:$failed"
        log "rerun just those with:  ./osm_prep.sh$failed"
        exit 1
    fi
    log "all done"
    exit 0
fi

# ----------------------------------------------------------------------------
# One region (we are the per-region process started above)
# ----------------------------------------------------------------------------
for item in $regions; do
    if [[ "$item" == *.pbf ]]; then
        # A file path: skip the download and name outputs after the file,
        # e.g. /x/asia-261029.osm.pbf -> name "asia-261029".
        [[ -f "$item" ]] || { echo "File not found: $item"; exit 1; }
        pbf="$item"
        name="$(basename "$item")"
        name="${name%.pbf}"     # strip ".pbf" from the end
        name="${name%.osm}"     # then ".osm"
        log "$name: using $pbf"
    else
        name="$item"
        download_region "$name"
        pbf="$PBF_DIR/$name-latest.osm.pbf"
    fi
    convert_layer "$name" "$pbf" lines         "$LINES_WHERE"    Lines    lines
    convert_layer "$name" "$pbf" multipolygons "$POLYGONS_WHERE" Polygons polygons
    convert_layer "$name" "$pbf" points        "$POINTS_WHERE"   Points   points

    # Every layer of this region is converted and validated (set -e would
    # have stopped the script otherwise), so the download can go.
    if [[ "$KEEP_PBF" != "1" && "$item" != *.pbf ]]; then
        log "$name: deleting $pbf to save space (set KEEP_PBF=1 to keep it)"
        run rm -f "$pbf" "$pbf.md5"
    fi
    rm -rf "${TMP_DIR:?}"/*      # GDAL scratch from this region
    log "$name: done"
done
