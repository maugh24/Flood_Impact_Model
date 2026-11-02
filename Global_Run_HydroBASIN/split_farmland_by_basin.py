"""
Split the global farmland_statistics.parquet into one small GeoParquet per basin
(HYBAS_ID), read one row group at a time so the whole file is never loaded.
Resumable: a basin whose file already exists is skipped.
"""
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq
import tqdm


INPUT_PARQUET = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\farmland_statistics.parquet"
OUTPUT_DIR    = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\farmland_by_basin"

ID_COL = "HYBAS_ID"
WRITE_COVERING_BBOX = True
SUBFOLDER_DIGITS = 0   # 0 = flat; N = bucket into subfolders by first N digits of HYBAS_ID


def _out_path(out_dir, hybas_id):
    name = f"{hybas_id}.parquet"
    if SUBFOLDER_DIGITS > 0:
        return out_dir / str(hybas_id)[:SUBFOLDER_DIGITS] / name
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

    written = skipped = 0
    for rg in tqdm.tqdm(range(n_rg), desc="row groups", dynamic_ncols=True):
        gdf = gpd.GeoDataFrame.from_arrow(pf.read_row_group(rg))
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
