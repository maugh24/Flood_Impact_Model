# Impact_Intersection

Sums flood-impact exposure over one set of **areas of interest (AOIs)** and writes a single
CSV for FEWS4All. The AOI is chosen per run: H3 cells, GEOGLOWS catchments, HydroBASINS, or
any polygon set you prepare.

```
./run_impact.sh h3 6            # H3 cells, resolution 6 (0-15; default 6)
./run_impact.sh hydrobasins 8   # HydroBASINS level 8 (1-12; default 12)
./run_impact.sh geoglows        # GEOGLOWS catchments (single level)
./run_impact.sh hydrobasins 8 --check
```
The number after the name is the level; leave it off for the default. GEOGLOWS and HydroBASINS
polygons are downloaded and prepared the first time you use them (see `Data_Prep/AOI`).

## How it fits together
1. **Prep once** (on the computation Mac, all under `DATA_ROOT`):
   - `Data_Prep/OSM/osm_prep.sh` -> `OSM_Prep/Lines`, `Polygons`, `Points`
   - `Data_Prep/Farmland/farmland_prep.sh` -> `Farmland/` (ESA WorldCover cropland blocks)
   - `Data_Prep/Population/population_prep.sh` -> `Population/worldpop_<year>_1km.parquet`
     (set the same year as `POPULATION_YEAR` in `config.py`)
   - polygon AOIs -> `AOI/`: GEOGLOWS and HydroBASINS automatically on first use; any other
     set with `Data_Prep/AOI/prepare_aoi.py`
2. **Intersect** as often as needed: `./run_impact.sh <aoi>`. Changing the AOI never means
   re-prepping the exposure data.

## Setup
- `pip install -r requirements.txt` (h3 must be >= 4.1)
- In `config.py` set `DATA_ROOT` and `OUTPUT_FOLDER`. AOI presets are in `AOIS`.
- `chmod +x run_impact.sh` once.

## Adding another polygon AOI
```
python ../Data_Prep/AOI/prepare_aoi.py INPUT ID_COLUMN ~/Flood_Impact_Model/Files/AOI/<name>.parquet [--pattern GLOB]
```
INPUT is a file (GeoParquet, GPKG, shapefile, GeoJSON) or a folder of them; the id must be an
integer. Then add an entry to `AOIS` in `config.py` (copy `geoglows` and drop `"prep"`).

## Output
`OUTPUT_FOLDER/<aoi>_<level>/impact_<aoi>_<level>_<YYYY-MM>.csv` (e.g. `h3_6/impact_h3_6_2026-11.csv`;
`geoglows/impact_geoglows_2026-11.csv`), one row per AOI with any exposure
(a missing AOI means zero), sorted by id:

| Column | Unit | Meaning |
|---|---|---|
| `h3_id` / `LINKNO` / `HYBAS_ID` / ... | | AOI id (H3 as a lowercase hex string; polygon ids as integers) |
| `population` | people | WorldPop 1 km pixels, by pixel centre |
| `buildings` | count | OSM buildings, by footprint centre |
| `farmland_km2` | km2 | ESA WorldCover cropland, ~300 m blocks by block centre (exact area per block) |
| `transportation_km` | km | OSM roads + rail; lines densified to <= 0.005 deg, each segment to the AOI at its midpoint |
| `bridges` | count | road/rail segments tagged `bridge=*` (not `no`), by centre; a dual carriageway counts 2 |
| `dams` | count | `waterway=dam` (lines, areas, points), by centre |
| `weirs` | count | `waterway=weir` / `check_dam`, by centre |

Beside it: `_summary.csv` (totals) and `_metadata.csv` (units, descriptions, sources).
Features outside every polygon (e.g. offshore for catchments) are not counted.

## Resume and safety
- Each task writes a partial CSV under `OUTPUT_FOLDER/<aoi>/_partials_<id>/`; rerunning skips
  finished tasks. `<id>` changes when the AOI changes (H3 resolution, or a re-prepared polygon
  file), so results for an old AOI version are never mixed in.
- Only partials for the current input files are merged; a partial missing a newer column is redone.
  Each task id includes its input file's size and modification time, so rebuilding an input
  (e.g. fresher OSM for one region) redoes just the tasks that read it.
- Final CSVs are written under a temporary name and then renamed, so a crash never leaves a
  cut-off `impact_*.csv`. Ctrl+C or a crash only loses the tasks that were in progress.
- Read errors (e.g. a sleeping external drive) are retried twice per task.
- A log of each run is saved in `logs/`.

## Known limitations
- Geofabrik continent extracts overlap slightly at their borders, so OSM features there can be
  counted twice.
- A dam mapped twice in OSM (e.g. as a line and an area) counts twice.
- For polygon AOIs each task loads the polygons inside its own bounding box; this relies on the
  OSM files being spatially sorted (`SORT_BY_BBOX=YES`, the default in `osm_prep.sh`).
