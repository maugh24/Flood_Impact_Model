# Data_Prep/OSM

Turns Geofabrik OpenStreetMap extracts into the GeoParquet files the Flood Impact Model
reads: filtered to buildings / roads / rail, with a covering bbox column, spatially sorted.
Replaces the old QGIS -> GPKG -> ogr2ogr -> bbox script -> DuckDB sort chain with one
`ogr2ogr` call per layer.

## Files
| File | Role |
|---|---|
| `osm_prep.sh` | the pipeline: download -> convert -> validate -> checksum |
| `osmconf_flood_impact.ini` | tells GDAL which OSM tags become columns (GDAL default, trimmed) |
| `validate_parquet.py` | reads every row group of every column; fails on damaged files |

## Requirements
GDAL >= 3.9 with the Parquet driver (`ogrinfo --formats | grep -i parquet` must print a line),
`curl`, `shasum`, and Python with `pyarrow`. Tested with GDAL 3.13.

## Run
```bash
cd ~/Flood_Impact_Model/Flood_Impact_Code/Data_Prep/OSM
chmod +x osm_prep.sh                 # once: allow the file to be run as a program
DRY_RUN=1 ./osm_prep.sh antarctica   # print the commands only
./osm_prep.sh antarctica             # smallest region (~30 MB) as a first real test
./osm_prep.sh                        # all regions in REGIONS
./osm_prep.sh /path/to/asia-261029.osm.pbf   # a .pbf you already have (no download)
```
A file path skips the download and MD5 check; outputs are named after the file
(`asia-261029-lines.parquet`, `asia-261029-polygons.parquet`).
Each region runs as its own process: if one fails (e.g. a download error), the rest still
run, and the end of the log lists the failed ones with the command to rerun just those. If
Geofabrik's `.md5` checksum is missing at that moment, the region is downloaded without it
(conversion and validation still catch a broken file). Rerunning skips finished conversions. Output goes to `WORK_DIR`
(default `Files/OSM_Prep`, override with `WORK_DIR=/path ./osm_prep.sh ...`):
`Lines/`, `Polygons/`, `Points/`. Only these GeoParquet files are kept: each region's
downloaded `.pbf` is deleted once all three layers are converted and validated
(`KEEP_PBF=1` keeps it; a `.pbf` you pass as a file path is never deleted), and
GDAL's `tmp/` scratch is emptied after each region.

`SORT_BY_BBOX=NO ./osm_prep.sh ...` skips the spatial sort. Results of the H3 run are
identical either way; sorting only speeds up small-area reads (catchment runs, QGIS),
but it makes GDAL write a large uncompressed temporary copy (Asia buildings filled a
~70 GB disk).

## Output
| Layer | Kept | Columns |
|---|---|---|
| `Lines/<region>-lines.parquet` | highway in (motorway ... cycleway), railway in (rail, light_rail, subway, monorail, tram), waterway in (dam, weir, check_dam) | osm_id, name, highway, railway, bridge, waterway, z_order, geometry, geometry_bbox |
| `Polygons/<region>-polygons.parquet` | building is not null, or waterway in (dam, weir, check_dam) | osm_id, osm_way_id, name, building, amenity, waterway, geometry, geometry_bbox |
| `Points/<region>-points.parquet` | waterway in (dam, weir, check_dam) (dams mapped as a single point) | osm_id, name, waterway, geometry, geometry_bbox |

`bridge` holds OSM's bridge tag on road/rail segments (`yes`, `viaduct`, `cantilever`, ...);
it is empty for segments that are not bridges. Bridges and dams are kept whether or not
they cross a river. The road/rail lists match `Global_Run_H3_Cells/h3_impact.py`; change
both together. Dam rows have no `building`/`highway`/`railway` value, so the current H3
run ignores them.

## Checking copies
Each output has a `.sha256` beside it. After copying a folder anywhere:
```bash
cd <copied>/Lines && shasum -a 256 -c *.sha256
```
`validate_parquet.py` catches damage that breaks decompression (like the old Tokyo row
group); the checksum catches any byte change at all, so use it after every copy.

## Disk space
Europe is ~29 GB as `.pbf`; GDAL's `tmp/` can need a similar amount while converting
it, and `SORT_BY_BBOX` writes a temporary file next to the output.
