# Data_Prep/AOI

Polygon sets that `Impact_Intersection/run_impact.sh` can sum exposure over.

## Built-in: downloaded and prepared automatically
`run_impact.sh` prepares these the first time they are used; you can also prepare ahead of time:
```bash
python aoi_prep.py geoglows          # GEOGLOWS v2 catchments, all 125 VPUs (id LINKNO)
python aoi_prep.py hydrobasins 8     # HydroBASINS v1c standard, level 8 (1-12, default 12; id HYBAS_ID)
```
- GEOGLOWS: `catchments_<vpu>.parquet` from the public `geoglows-v2` bucket
  (`hydrography/vpu=<vpu>/`) -> `~/Flood_Impact_Model/Files/AOI/geoglows_catchments/vpu_<vpu>.parquet`
- HydroBASINS: `hybas_<region>_lev01-12_v1c.zip` for 9 regions from data.hydrosheds.org; only the
  requested level is read from each zip -> `.../AOI/hydrobasins_lev<NN>/<region>.parquet`
- Each download is deleted as soon as its part is prepared (`--keep-download` keeps them). A
  HydroBASINS zip holds all 12 levels, so preparing another level later downloads it again.
- `_complete.txt` marks a finished folder; reruns skip finished parts.
- `--source-dir DIR` uses files already on disk instead of downloading (never deleted).

## Any other polygon set
```bash
python prepare_aoi.py INPUT ID_COLUMN ~/Flood_Impact_Model/Files/AOI/<name>.parquet [--pattern GLOB]
```
then add an entry (without `"prep"`) to `AOIS` in `Impact_Intersection/config.py`.

Prepared files hold only the integer id + geometry, in lon/lat, repaired, Hilbert-sorted, with a
covering bbox so each intersection task loads only the polygons in its own area.
