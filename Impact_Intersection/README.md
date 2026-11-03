# Impact_Intersection

Sums flood-impact exposure over one set of **areas of interest (AOIs)** and writes a single
CSV for FEWS4All. The AOI is chosen per run: H3 cells, GEOGLOWS catchments, HydroBASINS, or
any polygon set you prepare.

```
./run_impact.sh h3            # H3 cells (resolution in config.py)
./run_impact.sh geoglows      # GEOGLOWS catchments
./run_impact.sh hydrobasins --check
```

## How it fits together
1. **Prep once** (on the computation Mac, all under `DATA_ROOT`):
   - `Data_Prep/OSM/osm_prep.sh` -> `OSM_Prep/Lines`, `Polygons`, `Points`
   - `Data_Prep/Farmland/farmland_prep.sh` -> `Farmland/` (ESA WorldCover cropland blocks)
   - `Data_Prep/Population/population_prep.sh` -> `Population/worldpop_<year>_1km.parquet`
     (set the same year as `POPULATION_YEAR` in `config.py`)
   - each polygon AOI set -> `AOI/<name>.parquet` with `Data_Prep/AOI/prepare_aoi.py`
2. **Intersect** as often as needed: `./run_impact.sh <aoi>`. Changing the AOI never means
   re-prepping the exposure data.

## Setup
- `pip install -r requirements.txt` (h3 must be >= 4.1)
- In `config.py` set `DATA_ROOT` and `OUTPUT_FOLDER`. AOI presets are in `AOIS`.
- `chmod +x run_impact.sh` once.

## Adding a polygon AOI
```
python ../Data_Prep/AOI/prepare_aoi.py INPUT ID_COLUMN $DATA_ROOT/AOI/<name>.parquet [--pattern GLOB]
```
INPUT is a file (GeoParquet, GPKG, shapefile, GeoJSON) or a folder of them; the id must be an
integer. For the GEOGLOWS VPU layout:
```
python ../Data_Prep/AOI/prepare_aoi.py /path/GEOGLOW_Catchments LINKNO \
    ~/Flood_Impact_Model/Files/AOI/geoglows_catchments.parquet --pattern "vpu=*/*.parquet"
```
Then add an entry to `AOIS` in `config.py` (copy the `geoglows` one).

## Output
`OUTPUT_FOLDER/<aoi>/impact_<aoi>_<YYYY-MM>.csv`, one row per AOI with any exposure
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
- Read errors (e.g. a sleeping external drive) are retried twice per task.
- A log of each run is saved in `logs/`.

## Known limitations
- Geofabrik continent extracts overlap slightly at their borders, so OSM features there can be
  counted twice.
- A dam mapped twice in OSM (e.g. as a line and an area) counts twice.
- For polygon AOIs each task loads the polygons inside its own bounding box; this relies on the
  OSM files being spatially sorted (`SORT_BY_BBOX=YES`, the default in `osm_prep.sh`).
