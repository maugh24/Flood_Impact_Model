"""
Split geometry output into one GeoParquet per basin (HYBAS_ID), in parallel.

INPUT may be a folder of per-chunk fragments (building_by_chunk/,
transportation_by_chunk/) or a single combined parquet (farmland_statistics.parquet).
Each basin lives entirely in one fragment/row-group, so the pool hands each
worker a whole fragment and workers write disjoint basins - no collisions.
Resumable: a basin whose file already exists is skipped.
"""
from pathlib import Path
import multiprocessing as mp

import geopandas as gpd
import pyarrow.parquet as pq
import tqdm


INPUT      = r"/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Global_HUC12_Impact_Results/Statistics/building_by_chunk"
OUTPUT_DIR = r"/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Global_HUC12_Impact_Results/Statistics/building_by_basin"

ID_COL = "HYBAS_ID"
WRITE_COVERING_BBOX = True
SUBFOLDER_DIGITS = 0   # 0 = flat; N = bucket into subfolders by first N digits of HYBAS_ID
MAX_WORKERS = 8        # writing millions of tiny files is I/O-heavy; keep this modest


def _out_path(hybas_id):
    out_dir = Path(OUTPUT_DIR)
    name = f"{hybas_id}.parquet"
    if SUBFOLDER_DIGITS > 0:
        return out_dir / str(hybas_id)[:SUBFOLDER_DIGITS] / name
    return out_dir / name


def _read_item(item):
    if item[0] == 'file':
        return gpd.read_parquet(item[1])
    return gpd.GeoDataFrame.from_arrow(pq.ParquetFile(item[1]).read_row_group(item[2]))


def _process(item):
    gdf = _read_item(item)
    gdf = gdf[gdf.geometry.notna()]
    written = skipped = 0
    for hybas_id, idx in gdf.groupby(ID_COL).groups.items():
        dst = _out_path(hybas_id)
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdf.loc[idx].to_parquet(dst, write_covering_bbox=WRITE_COVERING_BBOX)
        written += 1
    return written, skipped


def _items():
    p = Path(INPUT)
    if p.is_dir():
        return [('file', str(f)) for f in sorted(p.glob("*.parquet"))]
    n = pq.ParquetFile(str(p)).metadata.num_row_groups
    return [('rg', str(p), i) for i in range(n)]


def main():
    if not Path(INPUT).exists():
        raise SystemExit(f"Input not found: {INPUT}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    items = _items()
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Units:  {len(items)}   Workers: {MAX_WORKERS}")

    written = skipped = 0
    with mp.Pool(processes=MAX_WORKERS) as pool:
        for w, s in tqdm.tqdm(pool.imap_unordered(_process, items),
                              total=len(items), desc="units", dynamic_ncols=True):
            written += w
            skipped += s
    print("=" * 60)
    print(f"done: {written:,} basin files written, {skipped:,} already existed")
    print(f"per-basin files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
