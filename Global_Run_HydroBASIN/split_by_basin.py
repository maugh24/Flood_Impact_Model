"""
Split geometry output into one GeoParquet per basin (HYBAS_ID), in parallel.

INPUT may be a folder of per-chunk fragments (building_by_chunk/,
transportation_by_chunk/) or a single combined parquet (farmland_statistics.parquet).
Each basin lives entirely in one fragment/row-group, so the pool hands each
worker a whole fragment and workers write disjoint basins - no collisions.

Resumable at two levels:
  - Unit-level: once a fragment/row-group has had ALL its basins written, a
    "_complete" marker is dropped for it. On restart, units with a marker are
    skipped entirely (no re-reading, no re-checking thousands of basin files).
  - Basin-level (fallback, e.g. if killed mid-unit): a basin whose output file
    already exists is skipped, so a partially finished unit can resume too.
  - Writes are atomic (write to .tmp then rename), so a basin file killed
    mid-write is never left behind as a corrupt "completed" file.
"""
from pathlib import Path
import multiprocessing as mp
import os

import geopandas as gpd
import pyarrow.parquet as pq
import tqdm


INPUT      = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\building_by_chunk"
OUTPUT_DIR = r"D:\Brian\Flood_Impact_Model\Global_HUC12_Impact_Results\Statistics\building_by_basin"

ID_COL = "HYBAS_ID"
WRITE_COVERING_BBOX = True
SUBFOLDER_DIGITS = 0   # 0 = flat; N = bucket into subfolders by first N digits of HYBAS_ID
MAX_WORKERS = 32        # writing millions of tiny files is I/O-heavy; keep this modest

COMPLETE_DIRNAME = "_complete"  # unit-level sentinels live under OUTPUT_DIR/_complete


def _out_path(hybas_id):
    out_dir = Path(OUTPUT_DIR)
    name = f"{hybas_id}.parquet"
    if SUBFOLDER_DIGITS > 0:
        return out_dir / str(hybas_id)[:SUBFOLDER_DIGITS] / name
    return out_dir / name


def _unit_id(item):
    """Stable, filesystem-safe identifier for a unit of work."""
    if item[0] == 'file':
        return Path(item[1]).stem
    return f"{Path(item[1]).stem}_rg{item[2]}"


def _marker_path(item):
    return Path(OUTPUT_DIR) / COMPLETE_DIRNAME / f"{_unit_id(item)}.done"


def _read_item(item):
    if item[0] == 'file':
        return gpd.read_parquet(item[1])
    return gpd.GeoDataFrame.from_arrow(pq.ParquetFile(item[1]).read_row_group(item[2]))


def _atomic_write_parquet(gdf, dst):
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    gdf.to_parquet(tmp, write_covering_bbox=WRITE_COVERING_BBOX)
    os.replace(tmp, dst)  # atomic on same filesystem, even on Windows


def _process(item):
    marker = _marker_path(item)
    if marker.exists():
        return 0, 0, True  # shouldn't normally happen (filtered out already), but safe

    gdf = _read_item(item)
    gdf = gdf[gdf.geometry.notna()]
    written = skipped = 0
    for hybas_id, idx in gdf.groupby(ID_COL).groups.items():
        dst = _out_path(hybas_id)
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(gdf.loc[idx], dst)
        written += 1

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return written, skipped, False


def _items():
    p = Path(INPUT)
    if p.is_dir():
        all_items = [('file', str(f)) for f in sorted(p.glob("*.parquet"))]
    else:
        n = pq.ParquetFile(str(p)).metadata.num_row_groups
        all_items = [('rg', str(p), i) for i in range(n)]

    remaining = [it for it in all_items if not _marker_path(it).exists()]
    return all_items, remaining


def main():
    if not Path(INPUT).exists():
        raise SystemExit(f"Input not found: {INPUT}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT_DIR) / COMPLETE_DIRNAME).mkdir(parents=True, exist_ok=True)

    all_items, items = _items()
    already_done = len(all_items) - len(items)
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Units:  {len(all_items)} total, {already_done} already complete, "
          f"{len(items)} remaining   Workers: {MAX_WORKERS}")

    if not items:
        print("Nothing to do - all units already completed.")
        return

    written = skipped = 0
    with mp.Pool(processes=MAX_WORKERS) as pool:
        for w, s, _ in tqdm.tqdm(pool.imap_unordered(_process, items),
                                  total=len(items), desc="units", dynamic_ncols=True):
            written += w
            skipped += s
    print("=" * 60)
    print(f"done: {written:,} basin files written, {skipped:,} already existed")
    print(f"per-basin files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
