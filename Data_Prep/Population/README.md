# Data_Prep/Population

`population_prep.sh` turns WorldPop's global 1 km population raster (R2025A release, constrained,
UN-adjusted) into the population file the Flood Impact Model reads. If the raster isn't already in
`<work-dir>/Population/`, it is downloaded from WorldPop; it is then converted to one point per
populated pixel (pixel centre + `pop_value`), validated (file total must equal the raster total),
checksummed, and the download is deleted.

## Run
```bash
cd ~/Flood_Impact_Model/Flood_Impact_Code/Data_Prep/Population
chmod +x population_prep.sh          # once
./population_prep.sh                 # WorldPop 2024
./population_prep.sh --year 2025     # another year
```
Options (`python population_prep.py --help`): `--year`, `--work-dir` (default
`~/Flood_Impact_Model/Files`), `--url` (download from somewhere else), `--tif FILE` (use a raster
you already have; it is never deleted), `--keep-tif`.

If the download fails, the script prints where to put a manually downloaded file; rerun after.

## Output
`Population/worldpop_<year>_1km.parquet`: `geometry` (pixel-centre point), `pop_value`, `bbox`.
Set `POPULATION_YEAR` in `Impact_Intersection/config.py` to the same year.

Tested on the 2025 raster: 51.2 M populated pixels, 8,196,931,816 people, 612 MB, ~70 s.
The old pixel-polygon script is in `Archive/Data_Prep_Legacy/Population/`.
