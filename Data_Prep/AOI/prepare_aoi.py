"""
Prepare a polygon set (GEOGLOWS catchments, HydroBASINS, any basins/regions)
as an area-of-interest (AOI) file for Impact_Intersection/run_impact.sh.

    python prepare_aoi.py INPUT ID_COLUMN OUTPUT.parquet [--pattern GLOB]

INPUT is one file (GeoParquet, GeoPackage, shapefile, GeoJSON, ...) or a
folder of them; --pattern picks the files in a folder, e.g. for the GEOGLOWS
VPU layout:

    python prepare_aoi.py /path/GEOGLOW_Catchments LINKNO \\
        /Users/maugh24/Flood_Impact_Model/Files/AOI/geoglows_catchments.parquet \\
        --pattern "vpu=*/*.parquet"

The output is one GeoParquet that the intersection run can read quickly:
  - only two columns: the id (renamed to exactly ID_COLUMN, even if the input
    spells it differently, e.g. linkno -> LINKNO) and the geometry
  - integer ids (required; they become the first column of the impact CSV)
  - lon/lat (EPSG:4326), invalid polygons repaired, empty ones dropped
  - sorted along a Hilbert curve with small row groups and a covering bbox
    column, so each task can load just the polygons in its own area
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

ROW_GROUP_SIZE = 5_000     # small groups = tasks skip more of the file


def read_one(path, id_column):
    if path.suffix.lower() in (".parquet", ".geoparquet"):
        names = pq.read_schema(path).names
        match = next((n for n in names if n.lower() == id_column.lower()), None)
        if match is None:
            raise SystemExit(f"{path}: no column matching '{id_column}' (has {names})")
        geom = "geometry" if "geometry" in names else None
        gdf = gpd.read_parquet(path, columns=[match] + ([geom] if geom else []))
    else:
        gdf = gpd.read_file(path)
        match = next((n for n in gdf.columns if n.lower() == id_column.lower()), None)
        if match is None:
            raise SystemExit(f"{path}: no column matching '{id_column}' (has {list(gdf.columns)})")
        gdf = gdf[[match, gdf.geometry.name]]
    gdf = gdf.rename(columns={match: id_column})
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    if gdf.crs is None:
        print(f"  WARNING {path.name}: no CRS, assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("id_column")
    ap.add_argument("output")
    ap.add_argument("--pattern", default="*", help="glob for files when INPUT is a folder")
    args = ap.parse_args()

    src = Path(args.input)
    if src.is_dir():
        files = sorted(f for f in src.glob(args.pattern)
                       if f.is_file() and not f.name.startswith(".")
                       and f.suffix.lower() in (".parquet", ".geoparquet", ".gpkg", ".shp", ".geojson", ".json"))
    else:
        files = [src]
    if not files:
        print(f"No input files found: {src} / {args.pattern}")
        return 1
    print(f"{len(files)} input file(s)")

    parts = []
    for i, f in enumerate(files, 1):
        gdf = read_one(f, args.id_column)
        print(f"  [{i}/{len(files)}] {f.name}: {len(gdf):,} polygons")
        parts.append(gdf)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)

    # Integer ids only.
    ids = pd.to_numeric(gdf[args.id_column], errors="coerce")
    if ids.isna().any() or not np.all(np.mod(ids.to_numpy(), 1) == 0):
        print(f"'{args.id_column}' has non-integer values; the AOI id must be an integer.")
        return 1
    gdf[args.id_column] = ids.astype("int64")

    # Clean geometry: drop empty, repair invalid, keep polygonal parts only.
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    bad = ~gdf.geometry.is_valid
    if bad.any():
        print(f"  repairing {int(bad.sum()):,} invalid polygon(s)")
        fixed = shapely.make_valid(gdf.geometry[bad].to_numpy())
        # make_valid can return collections with stray lines: keep polygon parts
        fixed = np.array([shapely.union_all([p for p in shapely.get_parts(g)
                                             if p.geom_type in ("Polygon", "MultiPolygon")])
                          for g in fixed], dtype=object)
        gdf.loc[bad, "geometry"] = fixed
        gdf = gdf[~gdf.geometry.is_empty]

    dup = gdf[args.id_column].duplicated().sum()
    if dup:
        print(f"  note: {dup:,} id(s) appear more than once (e.g. split across files); "
              f"their exposure is summed together, which is usually what you want")

    # Sort along a Hilbert curve so nearby polygons share row groups.
    gdf = gdf.iloc[np.argsort(gdf.geometry.hilbert_distance().to_numpy(), kind="stable")]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    gdf.to_parquet(tmp, index=False, compression="zstd", write_covering_bbox=True,
                   row_group_size=ROW_GROUP_SIZE)
    tmp.replace(out)
    md = pq.read_metadata(out)
    print(f"wrote {out}: {md.num_rows:,} polygons, {md.num_row_groups} row groups, "
          f"{out.stat().st_size / 1e6:,.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
