"""
split_farmland_by_basin.py

Split the single global farmland_statistics.parquet into one small GeoParquet
per basin (HYBAS_ID), which loads far more nicely in QGIS than one multi-GB
global layer.

How it works
------------
The global file was written one row group per processing chunk, and every
basin's rows live entirely within a single row group (each basin is processed
in exactly one chunk). So we can read the file one row group at a time -
never loading the whole multi-GB file - group that row group's rows by
HYBAS_ID, and write each basin's polygons to its own file. Each output is a
valid GeoParquet (geometry + CRS + a covering bbox), so QGIS reads it directly.

Output
------
    OUTPUT_DIR/
        <HYBAS_ID>.parquet          (flat, default)
    or, if SUBFOLDER_DIGITS > 0, bucketed to keep directories small:
        OUTPUT_DIR/<first N digits>/<HYBAS_ID>.parquet

Only basins that actually contain cropland get a file (basins with no farmland
have no rows in the global file).

Resumable: a basin whose file already exists is skipped, so you can re-run
after an interruption.

Requirements: geopandas, pyarrow, tqdm  (already in the hydroinformatics env)
"""
import os
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq
import tqdm


# ----------------------------------------------------------------------------
# CONFIG - edit for your machine.
# ----------------------------------------------------------------------------
INPUT_PARQUET = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\farmland_statistics.parquet"
OUTPUT_DIR    = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\farmland_by_basin"

ID_COL = "HYBAS_ID"

# Adds a bounding-box covering column to each per-basin file so QGIS can spatial-
# filter. Cheap; leave on unless you want the smallest possible files.
WRITE_COVERING_BBOX = True

# 0  -> flat layout:  OUTPUT_DIR/<HYBAS_ID>.parquet
# N  -> bucket by the first N digits of HYBAS_ID into subfolders, so no single
#       directory holds hundreds of thousands of files (gentler on Windows
#       Explorer). 4 is a good value; files are then OUTPUT_DIR/<abcd>/<id>.parquet
SUBFOLDER_DIGITS = 0
# ----------------------------------------------------------------------------


def _out_path(out_dir: Path, hybas_id) -> Path:
    name = f"{hybas_id}.parquet"
    if SUBFOLDER_DIGITS > 0:
        bucket = str(hybas_id)[:SUBFOLDER_DIGITS]
        return out_dir / bucket / name
    return out_dir / name


def main():
    src = Path(INPUT_PARQUET)
    out_dir = Path(OUTPUT_DIR)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(src))
    n_rg = pf.metadata.num_row_groups
    print(f"Input: {src}")
    print(f"  rows: {pf.metadata.num_rows:,}   row groups: {n_rg}")
    print(f"Output: {out_dir}   (covering_bbox={WRITE_COVERING_BBOX}, "
          f"subfolder_digits={SUBFOLDER_DIGITS})")

    written = skipped = 0
    for rg in tqdm.tqdm(range(n_rg), desc="row groups", dynamic_ncols=True):
        gdf = gpd.GeoDataFrame.from_arrow(pf.read_row_group(rg))
        # Only rows that carry geometry (defensive; the global file already
        # dropped empty-basin rows).
        gdf = gdf[gdf.geometry.notna()]
        if len(gdf) == 0:
            continue
        for hybas_id, idx in gdf.groupby(ID_COL).groups.items():
            dst = _out_path(out_dir, hybas_id)
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            gdf.loc[idx].to_parquet(dst, write_covering_bbox=WRITE_COVERING_BBOX)
            written += 1

    print("=" * 60)
    print(f"done: {written:,} basin files written, {skipped:,} already existed")
    print(f"per-basin farmland in: {out_dir}")


if __name__ == "__main__":
    main()
