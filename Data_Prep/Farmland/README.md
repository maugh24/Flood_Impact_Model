# Data_Prep/Farmland

`farmland_prep.sh` turns ESA WorldCover 2021 (v200) into the cropland files the Flood Impact
Model reads. For each 3x3 degree tile it downloads the GeoTIFF from ESA's public AWS bucket
(`esa-worldcover`, no account needed), counts cropland pixels (class 40) in blocks of 15 x 15
pixels (~150 m), stores one point per block with the block's exact cropland area, validates the
file, writes a `.sha256`, and deletes the download.

## Run
```bash
cd ~/Flood_Impact_Model/Flood_Impact_Code/Data_Prep/Farmland
chmod +x farmland_prep.sh                       # once
./farmland_prep.sh --bbox -81 21 -75 24         # a few tiles first, as a test
./farmland_prep.sh                              # every tile (~2,650)
```
Options (`python farmland_prep.py --help`):
- `--work-dir` (default `~/Flood_Impact_Model/Files`): output to `Farmland/`,
  temporary downloads to `Farmland_download/`
- `--workers N` tiles at once (default 4, ~2 GB RAM each)
- `--block-px N` block size in pixels (default 15)
- `--bbox MINX MINY MAXX MAXY` or `--tiles N21W081 ...` to limit the run
- `--local-tifs DIR` use ESA tiles already on disk (e.g. unzipped Zenodo downloads) instead of
  downloading; those are never deleted
- `--keep-tif` keep downloads

Rerunning skips finished tiles. A log of each run is saved in `logs/`.

## Output
`Farmland/<tile>.parquet` (e.g. `N21W081.parquet`): `geometry` (block-centre point), `area_m2`
(cropland area in the block, WGS84 ellipsoid), `bbox`. Tiles without cropland get an empty file.

## Why blocks instead of polygons
The old process (`Legacy/`) traced cropland into polygons, simplified them by 10 m and dropped
any under 500 m^2. On tile N21W081 that lost 1.0% of the farmland; the blocks keep all of it
(they match an independent pixel count to 0.02 km^2 of 3,092 km^2) and take ~5 s per tile.
The cost: at an AOI border a block's area goes to the AOI holding its centre.

| Block | Per-catchment error vs per-pixel truth (median / max) | Global size (rough) |
|---|---|---|
| 30 px (~300 m) | 2.2% / 22% | ~8 GB |
| 15 px (~150 m, default) | 1.0% / 6.9% | ~20 GB |
| 10 px (~100 m) | 0.5% / 3.5% | ~35-40 GB |

`Legacy/` holds the old polygon scripts. `Impact_Intersection` still reads the old polygon
chunks too, if `INPUTS["farmland"]` in its `config.py` points at them.
