# Global_Run_H3_Cells

Global flood-impact exposure per H3 resolution-6 cell, formatted for FEWS4All.
Self-contained: copy this folder to any machine, `pip install -r requirements.txt`.

## Run
1. In `config.py` set `DATA_ROOT` (folder holding the prepped bbox parquets) and `OUTPUT_FOLDER`.
2. `python run_h3_impact.py --check`  -> confirms every layer finds its files and shows task counts.
3. `python run_h3_impact.py`          -> full run. Safe to stop and rerun; finished tasks are skipped.

Paths can also be passed directly: `--data-root`, `--output`, `--workers`, `--layers`, `--label`.
`--merge-only` rebuilds the final CSVs from `_partials/` without reprocessing.

## Outputs
| File | Contents |
|---|---|
| `impact_h3r6_<label>.csv` | `h3_id,population,buildings,farmland_km2,transportation_km` - one row per cell with any exposure, sorted by `h3_id`; a missing cell means zero |
| `impact_h3r6_<label>_summary.csv` | global totals + cell count |
| `impact_h3r6_<label>_metadata.csv` | unit, description, source per column |

`_partials/` is the resumable working folder and can be deleted after a successful run.

## Method (no AOI overlays)
- **Population**: each 1 km pixel's centre -> cell; sum `pop_value`.
- **Buildings**: OSM footprint bbox centre -> cell; count (same `building` values as the catchment run).
- **Farmland**: cropland polygons <= ~550 m are assigned whole by centre; larger ones are clipped exactly
  to the H3 cells they overlap. Areas use the WGS84 ellipsoid.
- **Transportation**: road and rail lines (same OSM values as the catchment run) densified to
  <= 0.005 deg segments; each segment's length goes to the cell holding its midpoint, reported as
  one combined `transportation_km`.

## Known limitation
Geofabrik continent extracts overlap slightly along their borders, so OSM features in those
zones can be counted twice (same as the catchment run).
