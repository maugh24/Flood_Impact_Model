"""
Settings for the flood-impact intersection run.

Normally you only edit DATA_ROOT and OUTPUT_FOLDER. Every input below - the
prepped OSM, population and farmland files, and the prepared AOI polygon
files - lives under DATA_ROOT on the computation Mac:

    DATA_ROOT/
        OSM_Prep/      Lines/  Polygons/  Points/   (Data_Prep/OSM/osm_prep.sh)
        Population/    population.parquet
        Farmland/      <tile>.parquet           (Data_Prep/Farmland/farmland_prep.sh)
        AOI/           geoglows_catchments.parquet, hydrobasins.parquet, ...
                       (Data_Prep/AOI/prepare_aoi.py)

Either can also be overridden on the command line:
    ./run_impact.sh h3 --data-root /other/Files --output /other/Results
"""
import os

# ---------------------------------------------------------------------------
# Paths (edit these)
# ---------------------------------------------------------------------------
DATA_ROOT = r"/Users/maugh24/Flood_Impact_Model/Files"

# Results go to OUTPUT_FOLDER/<aoi name>/ (plus a resumable _partials_* folder).
OUTPUT_FOLDER = r"/Users/maugh24/Flood_Impact_Model/Impact_Results"

# Exposure inputs. "path" is relative to DATA_ROOT (or absolute) and may be a
# file, a folder, or a list of folders; "pattern" is the glob used in folders
# ("**" = search subfolders). Forward slashes work on macOS, Linux and Windows.
INPUTS = {
    "population": {"path": "Population/population.parquet", "pattern": "*.parquet"},
    # Cropland blocks from farmland_prep.sh. The older polygon chunks also work:
    # {"path": "ESA/bbox", "pattern": "**/*.parquet"}
    "farmland":   {"path": "Farmland",                      "pattern": "*.parquet"},
    "buildings":  {"path": "OSM_Prep/Polygons",             "pattern": "*.parquet"},
    "roads":      {"path": "OSM_Prep/Lines",                "pattern": "*.parquet"},
    # Dams/weirs can be mapped as lines, areas or points, so read all three.
    "dams":       {"path": ["OSM_Prep/Lines", "OSM_Prep/Polygons", "OSM_Prep/Points"],
                   "pattern": "*.parquet"},
}

# ---------------------------------------------------------------------------
# Areas of interest: what the exposure is summed over. Pick one per run:
#     ./run_impact.sh <name>
# "h3"       -> H3 cells at a resolution (no file needed)
# "polygons" -> a file made by Data_Prep/AOI/prepare_aoi.py, with an integer
#               id column; its values become the first column of the CSV.
# Add a new polygon set by preparing it and adding an entry here.
# ---------------------------------------------------------------------------
AOIS = {
    "h3": {
        "type": "h3",
        "resolution": 6,                                    # FEWS4All grid
    },
    "geoglows": {
        "type": "polygons",
        "path": "AOI/geoglows_catchments.parquet",
        "id_column": "LINKNO",
        "description": "GEOGLOWS river reach id (LINKNO) of the catchment",
        "source": "GEOGLOWS catchments (all VPUs)",
    },
    "hydrobasins": {
        "type": "polygons",
        "path": "AOI/hydrobasins.parquet",
        "id_column": "HYBAS_ID",
        "description": "HydroBASINS basin id (HYBAS_ID)",
        "source": "HydroBASINS",
    },
}

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
LAYERS = ["population", "farmland", "buildings", "roads", "dams"]
MAX_WORKERS = os.cpu_count()    # lower if RAM is tight (~1-2 GB per worker; more
                                # for polygon AOIs with very detailed polygons)
RUN_LABEL = None                # None -> current YYYY-MM, used in output file names

# Rows read per task (one task = a batch of parquet row groups). Smaller =
# less memory per worker and finer-grained resume; larger = less overhead.
TARGET_ROWS_PER_TASK = {
    "population": 2_000_000,
    "farmland":   2_000_000,    # blocks are points, so cheap (use ~500_000 for old polygons)
    "buildings":  2_000_000,
    "roads":        500_000,
    "dams":       2_000_000,    # reads only the waterway tag + bbox, so cheap
}

# Farmland polygons whose bbox is wider/taller than this (degrees, ~550 m at
# the equator) are split exactly along AOI edges; smaller ones are assigned
# whole to the AOI containing their bbox centre.
FARMLAND_SPLIT_EXTENT_DEG = 0.005

# Road/rail lines are densified so no segment is longer than this (degrees);
# each segment's length goes to the AOI containing its midpoint.
ROAD_MAX_SEGMENT_DEG = 0.005

# Written to the metadata CSV so FEWS4All can attribute the numbers.
SOURCES = {
    "population":        "WorldPop 2025 constrained 1 km (UN-adjusted)",
    "buildings":         "OpenStreetMap (Geofabrik extracts)",
    "farmland_km2":      "ESA WorldCover 2021 v200 (class 40 cropland)",
    "transportation_km": "OpenStreetMap (Geofabrik extracts)",
    "bridges":           "OpenStreetMap (Geofabrik extracts)",
    "dams":              "OpenStreetMap (Geofabrik extracts)",
    "weirs":             "OpenStreetMap (Geofabrik extracts)",
}
