"""
Check that GeoParquet files are complete and readable before they are used.

    python validate_parquet.py FILE_OR_FOLDER [FILE_OR_FOLDER ...]

For every .parquet file it:
  1. reads the footer and the GeoParquet 'geo' metadata
  2. checks there is a covering bbox (needed for fast bbox reads) and a
     lon/lat CRS
  3. decompresses EVERY column of EVERY row group - this is the step that
     catches damaged bytes (e.g. the Tokyo row group in the old asia file),
     which a viewer like QGIS only hits if you happen to zoom to that area.

Exit code 0 = all files OK, 1 = at least one problem (so the bash pipeline
stops instead of publishing a bad file).
"""
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


def check_file(path):
    problems = []
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        return [f"cannot open footer: {e}"]
    md = pf.metadata
    names = pf.schema_arrow.names

    geo = None
    if md.metadata and b"geo" in md.metadata:
        geo = json.loads(md.metadata[b"geo"])
    if geo is None:
        problems.append("no GeoParquet 'geo' metadata")
    else:
        col = geo.get("columns", {}).get(geo.get("primary_column", "geometry"), {})
        if not col.get("covering", {}).get("bbox"):
            problems.append("no covering bbox declared")
        crs = col.get("crs", "missing")        # missing key = OGC:CRS84 (lon/lat)
        if isinstance(crs, dict) and crs.get("type") not in ("GeographicCRS", None):
            problems.append(f"CRS is not lon/lat: {crs.get('name')}")
    if md.num_rows == 0:
        problems.append("file has 0 rows")

    for rg in range(md.num_row_groups):
        for c in names:
            try:
                pf.read_row_group(rg, columns=[c])
            except Exception as e:
                problems.append(f"row group {rg}, column '{c}': {str(e).splitlines()[0]}")
    return problems


def main(args):
    files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*.parquet") if not f.name.startswith("."))
        else:
            files.append(p)
    if not files:
        print("usage: python validate_parquet.py FILE_OR_FOLDER [...]")
        return 1

    bad = 0
    for f in files:
        t0 = time.time()
        problems = check_file(f)
        if problems:
            bad += 1
            print(f"FAIL  {f}")
            for p in problems:
                print(f"        - {p}")
        else:
            md = pq.read_metadata(f)
            print(f"OK    {f}  ({md.num_rows:,} rows, {md.num_row_groups} row groups, "
                  f"{time.time() - t0:.0f}s)")
    print(f"{len(files) - bad}/{len(files)} file(s) OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
