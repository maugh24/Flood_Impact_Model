# Flood Impact Model

Exposure (population, buildings, farmland, roads/rail) for flood early warning, run globally.
Code only - all input data and results live outside this folder.

## Layout
| Folder | What it is |
|---|---|
| `Global_Run_H3_Cells/` | **Current global run.** Exposure per H3 res-6 cell, CSV output for FEWS4All |
| `Global_Run_GEOGLOWS_Catchment/` | Global run per GEOGLOWS catchment (VPU partitions) |
| `Global_Run_HydroBASIN/` | Global run per HydroBASIN / HUC12 basin, plus per-basin split scripts |
| `Single_extent/` | Impact inside one flood-extent polygon |
| `Multiple_Return_Periods/` | Impact for several areas x return-period flood extents |
| `Data_Prep/OSM/` | **Current OSM prep pipeline** (`osm_prep.sh`): Geofabrik .pbf -> filtered, sorted GeoParquet |
| `Data_Prep/OSM/Legacy/` | Old OSM steps (bbox scripts, DuckDB Hilbert sort, schema check) replaced by `osm_prep.sh` |
| `Data_Prep/Population/` | WorldPop raster -> parquet |
| `Data_Prep/Farmland/` | ESA WorldCover cropland raster -> parquet, simplify, clip |
| `Data_Prep/Flood_Extent/` | Flood-extent rasters -> parquet |
| `Data_Prep/Open_Buildings/` | Google Open Buildings CSV -> parquet, flood intersect |
| `Archive/` | Old per-dataset scripts and profiling output, kept for reference |

Each folder with a README explains how to run it.
