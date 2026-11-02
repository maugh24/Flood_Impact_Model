
"""
Compare the schema / GeoParquet metadata of every polygon tile so a mixed
folder (some files written by GeoPandas, some by the DuckDB sort script)
can be verified before a full orchestrator run.

Checks per file, then flags anything that deviates from the majority:
  - columns Building_Impact needs: building, name, amenity, geometry
  - covering bbox column declared in the 'geo' metadata (the thing whose
    absence silently returns zero buildings)
  - bbox struct field names (xmin/ymin/xmax/ymax)
  - file-level bbox used by tile_loader._file_bbox()
  - arrow dtype of 'building' (isin filters behave differently on
    string vs large_string vs dictionary)
"""

import glob
import json
import os

import pyarrow.parquet as pq

FOLDER = r"C:\Users\maugh24\Flood_Impact_Model\Files\OSM\Parquet_Sorted\Polygons"
PATTERN = "*polygons*.parquet"

NEEDED = ("building", "name", "amenity", "geometry")


def geo_meta(path):
    kv = pq.read_metadata(path).metadata or {}
    if b"geo" not in kv:
        return {}
    try:
        return json.loads(kv[b"geo"])
    except Exception:
        return {}


def describe(path):
    schema = pq.read_schema(path)
    meta = pq.read_metadata(path)
    geo = geo_meta(path)

    primary = geo.get("primary_column", "geometry")
    colmeta = geo.get("columns", {}).get(primary, {})
    cov = colmeta.get("covering", {}).get("bbox")

    bbox_col = cov["xmin"][0] if cov else ("bbox" if "bbox" in schema.names else None)

    bbox_fields = None
    if bbox_col and bbox_col in schema.names:
        f = schema.field(bbox_col).type
        if hasattr(f, "num_fields"):
            bbox_fields = tuple(f.field(i).name for i in range(f.num_fields))

    return {
        "file": os.path.basename(path),
        "rows": meta.num_rows,
        "row_groups": meta.num_row_groups,
        "size_gb": os.path.getsize(path) / 1e9,
        "geo_version": geo.get("version"),
        "primary": primary,
        "encoding": colmeta.get("encoding"),
        "covering_declared": bool(cov),
        "bbox_col": bbox_col,
        "bbox_fields": bbox_fields,
        "file_bbox": tuple(colmeta["bbox"]) if colmeta.get("bbox") else None,
        "missing": [c for c in NEEDED if c not in schema.names],
        "building_type": str(schema.field("building").type) if "building" in schema.names else None,
        "n_cols": len(schema.names),
        "columns": list(schema.names),
    }


def main():
    files = sorted(glob.glob(os.path.join(FOLDER, PATTERN)))
    if not files:
        print(f"No files matched {FOLDER} / {PATTERN}")
        return

    rows = [describe(f) for f in files]

    hdr = f"{'file':<42} {'rows':>12} {'GB':>6} {'ver':<6} {'cov':<4} {'bbox_col':<10} {'building type':<14} {'cols':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:<42} {r['rows']:>12,} {r['size_gb']:>6.1f} "
              f"{str(r['geo_version']):<6} {'yes' if r['covering_declared'] else 'NO':<4} "
              f"{str(r['bbox_col']):<10} {str(r['building_type']):<14} {r['n_cols']:>5}")

    print()
    problems = []

    for r in rows:
        if r["missing"]:
            problems.append(f"{r['file']}: MISSING columns {r['missing']}")
        if not r["covering_declared"]:
            problems.append(f"{r['file']}: no covering bbox declared -> falls back to "
                            f"gpd.read_parquet(bbox=...), which will also fail")
        if r["bbox_fields"] and set(r["bbox_fields"]) != {"xmin", "ymin", "xmax", "ymax"}:
            problems.append(f"{r['file']}: bbox struct fields are {r['bbox_fields']}")
        if r["file_bbox"] is None:
            problems.append(f"{r['file']}: no file-level bbox -> tile_loader._file_bbox() "
                            f"returns None, so this tile is never skipped (slow, not broken)")

    # Column-set drift against the most common schema.
    from collections import Counter
    sig = Counter(tuple(sorted(r["columns"])) for r in rows)
    majority = sig.most_common(1)[0][0]
    for r in rows:
        s = tuple(sorted(r["columns"]))
        if s != majority:
            extra = set(s) - set(majority)
            gone = set(majority) - set(s)
            problems.append(f"{r['file']}: column drift  extra={sorted(extra)} missing={sorted(gone)}")

    # dtype drift on the filtered column.
    types = Counter(r["building_type"] for r in rows if r["building_type"])
    if len(types) > 1:
        problems.append(f"'building' dtype varies across files: {dict(types)}")

    if problems:
        print("PROBLEMS")
        for p in problems:
            print("  -", p)
    else:
        print("All files consistent -- safe to run together.")

    print()
    print("Common columns across all files:")
    common = set(rows[0]["columns"])
    for r in rows[1:]:
        common &= set(r["columns"])
    print(" ", sorted(common))


if __name__ == "__main__":
    main()

