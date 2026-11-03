# Flood Impact Model

Exposure (population, buildings, farmland, roads/rail) for flood early warning, run globally.
Code only - all input data and results live outside this folder.

## Setup
```bash
conda env create -f environment.yml     # once per computer
conda activate flood_impact             # every new terminal
```
To update an existing environment after `environment.yml` changes:
`conda env update -f environment.yml --prune`

## Layout
| Folder | What it is |
|---|---|
| `Impact_Intersection/` | **The run.** `./run_impact.sh <aoi> [level]` sums exposure over H3 cells, GEOGLOWS catchments, HydroBASINS or any prepared polygon set -> one CSV for FEWS4All |
| `Data_Prep/OSM/` | `osm_prep.sh`: Geofabrik .pbf -> filtered, sorted GeoParquet (buildings, roads/rail, bridges, dams) |
| `Data_Prep/Farmland/` | `farmland_prep.sh`: ESA WorldCover tiles -> cropland blocks |
| `Data_Prep/Population/` | `population_prep.sh`: WorldPop 1 km raster -> population points |
| `Data_Prep/AOI/` | `aoi_prep.py` (GEOGLOWS, HydroBASINS; used automatically) and `prepare_aoi.py` (any other polygon set) |
| `Archive/` | Superseded code kept for reference: old global runs (per catchment, per HydroBASIN), single-extent and return-period runs, and the old data-prep scripts (`Data_Prep_Legacy/`: OSM bbox/sort, polygon farmland, pixel-polygon population, flood-extent rasters, Open Buildings) |

Each folder with a README explains how to run it.
