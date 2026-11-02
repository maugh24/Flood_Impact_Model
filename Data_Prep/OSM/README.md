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
cd /Users/maugh24/Flood_Impact_Model/Flood_Impact_Code/Data_Prep/OSM
chmod +x osm_prep.sh                 # once: allow the file to be run as a program
DRY_RUN=1 ./osm_prep.sh antarctica   # print the commands only
./osm_prep.sh antarctica             # smallest region (~30 MB) as a first real test
./osm_prep.sh                        # all regions in REGIONS
```
Rerunning skips finished downloads and conversions. Output goes to `WORK_DIR`
(default `Files/OSM_Prep`): `Lines/`, `Polygons/`, `pbf/`, `tmp/`.

## Output
| Layer | Kept | Columns |
|---|---|---|
| `Lines/<region>-lines.parquet` | highway in (motorway ... cycleway) or railway in (rail, light_rail, subway, monorail, tram) | osm_id, name, highway, railway, z_order, geometry, geometry_bbox |
| `Polygons/<region>-polygons.parquet` | building is not null | osm_id, osm_way_id, name, building, amenity, geometry, geometry_bbox |

The road/rail lists match `Global_Run_H3_Cells/h3_impact.py`; change both together.

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
