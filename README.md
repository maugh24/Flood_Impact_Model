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
| `Impact_Intersection/` | **Current run.** `./run_impact.sh <aoi>` sums exposure over H3 cells, GEOGLOWS catchments, HydroBASINS or any prepared polygon set -> one CSV for FEWS4All |
| `Single_extent/` | Impact inside one flood-extent polygon |
| `Multiple_Return_Periods/` | Impact for several areas x return-period flood extents |
| `Data_Prep/OSM/` | **Current OSM prep pipeline** (`osm_prep.sh`): Geofabrik .pbf -> filtered, sorted GeoParquet |
| `Data_Prep/AOI/` | `prepare_aoi.py`: any polygon set -> AOI file for `Impact_Intersection` |
| `Data_Prep/OSM/Legacy/` | Old OSM steps (bbox scripts, DuckDB Hilbert sort, schema check) replaced by `osm_prep.sh` |
| `Data_Prep/Population/` | **Current population prep** (`population_prep.sh`): downloads WorldPop 1 km raster if missing -> population points (old script in `Legacy/`) |
| `Data_Prep/Farmland/` | **Current farmland prep** (`farmland_prep.sh`): ESA WorldCover tiles -> cropland blocks (old polygon scripts in `Legacy/`) |
| `Data_Prep/Flood_Extent/` | Flood-extent rasters -> parquet |
| `Data_Prep/Open_Buildings/` | Google Open Buildings CSV -> parquet, flood intersect |
| `Archive/` | Superseded code kept for reference: the old per-catchment and per-HydroBASIN global runs, per-dataset scripts, profiling output |

Each folder with a README explains how to run it.
