"""
Run settings for the global H3 flood-impact (exposure) run.

Normally you only edit DATA_ROOT and OUTPUT_FOLDER. Every input path below is
built from DATA_ROOT, so moving to another machine or drive letter is a
one-line change. Each value can also be overridden on the command line, e.g.

    python run_h3_impact.py --data-root "E:\\Flood_Impact_Model\\Files" --output "D:\\H3_Impact_Results"
"""
import os

# ---------------------------------------------------------------------------
# Paths (edit these)
# ---------------------------------------------------------------------------
# Folder that holds the prepped (bbox-covered) inputs (here: the portable drive).
DATA_ROOT = r"/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files"

# Local copy of the sorted OSM parquets (Lines/ and Polygons/ subfolders).
OSM_ROOT = r"/Users/maugh24/Flood_Impact_Model/Files/From_Portable_Drive/Parquet_sorted"

# Where the CSV outputs (and the resumable _partials/ working folder) go.
OUTPUT_FOLDER = r"/Users/maugh24/Flood_Impact_Model/H3_Impact_Results"

# Each layer: "path" is a file or a folder (relative to DATA_ROOT, or absolute),
# "pattern" is the glob used when "path" is a folder ("**" = search subfolders).
# Forward slashes work on Windows, macOS and Linux.
INPUTS = {
    "population": {"path": "Population/population.parquet",   "pattern": "*.parquet"},
    "farmland":   {"path": "ESA/bbox",                        "pattern": "**/*.parquet"},
    # OSM copied to the local drive (absolute paths, so DATA_ROOT doesn't apply):
    "buildings":  {"path": OSM_ROOT + "/Polygons",            "pattern": "*.parquet"},
    "roads":      {"path": OSM_ROOT + "/Lines",               "pattern": "*.parquet"},
}

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
H3_RESOLUTION = 6               # FEWS4All grid
LAYERS = ["population", "farmland", "buildings", "roads"]
MAX_WORKERS = os.cpu_count()    # lower if RAM is tight (~1-2 GB per worker)
RUN_LABEL = None                # None -> current YYYY-MM, used in output file names

# Rows read per task (one task = a batch of parquet row groups). Smaller =
# less memory per worker and finer-grained resume; larger = less overhead.
TARGET_ROWS_PER_TASK = {
    "population": 2_000_000,
    "farmland":     500_000,
    "buildings":  2_000_000,
    "roads":        500_000,
}

# Farmland polygons whose bbox is wider/taller than this (degrees, ~550 m at
# the equator) are split exactly along H3 cell edges; smaller ones are
# assigned whole to the cell containing their bbox centre.
FARMLAND_SPLIT_EXTENT_DEG = 0.005

# Road/rail lines are densified so no segment is longer than this (degrees);
# each segment's length goes to the cell containing its midpoint.
ROAD_MAX_SEGMENT_DEG = 0.005

# Written to the metadata CSV so FEWS4All can attribute the numbers.
SOURCES = {
    "population":    "WorldPop 2025 constrained 1 km (UN-adjusted)",
    "buildings":     "OpenStreetMap (Geofabrik extracts)",
    "farmland_km2":  "ESA WorldCover 2021 v200 (class 40 cropland)",
    "transportation_km": "OpenStreetMap (Geofabrik extracts)",
}
