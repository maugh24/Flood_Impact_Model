"""
Spatially sort GeoParquet tiles with DuckDB (Hilbert curve order).

Edit the CONFIG block below, then just run the file -- no command-line args.

Why the second pass exists: DuckDB writes GeoParquet *1.0* metadata, which has
no "covering" declaration. A bbox column survives the COPY but nothing can find
it -- Building_Impact._covering_col() returns None and the tile silently yields
zero buildings. So this sorts in DuckDB, then rewrites the Parquet footer
(streaming, row group at a time -- never loads the file into RAM) to declare the
covering bbox as GeoParquet 1.1 and stamp the file-level bbox that
tile_loader._file_bbox() uses to skip non-overlapping tiles.

Outputs are written as <name>_sorted.parquet. Files that already have an output
are skipped, so an interrupted run picks up where it stopped.
"""

import glob
import json
import os
import time

import duckdb
import pyarrow.parquet as pq

# ============================== CONFIG ======================================

SRC_FOLDER = r"C:\Users\maugh24\Flood_Impact_Model\Files\OSM\bbox"
PATTERN    = "*.parquet"      # "*.parquet" for everything
OUT_FOLDER = None                      # None = write next to the inputs
SUFFIX     = "_sorted"

TEMP_DIR   = r"C:\Users\maugh24\Flood_Impact_Model\Files\OSM\Parquet_sorted\duckdb_tmp"   # sort spill space -- needs room
MEMORY     = "16GB"
THREADS    = max(1, (os.cpu_count() or 8) // 2)

COMPRESSION       = "brotli"
COMPRESSION_LEVEL = 3
ROW_GROUP_SIZE    = 100_000

OVERWRITE = False                      # True = redo files that already have output

# ============================================================================


def inspect(path):
    """(geometry column name, existing bbox column name or None)."""
    kv = pq.read_metadata(path).metadata or {}
    geo = {}
    if b"geo" in kv:
        try:
            geo = json.loads(kv[b"geo"])
        except Exception:
            geo = {}

    names = pq.read_schema(path).names
    geom_col = geo.get("primary_column", "geometry")
    if geom_col not in names:
        geom_col = "geometry" if "geometry" in names else None

    bbox_col = None
    cov = geo.get("columns", {}).get(geom_col, {}).get("covering", {}).get("bbox")
    if cov:
        bbox_col = cov["xmin"][0]
    elif "bbox" in names:
        bbox_col = "bbox"              # column is there, just not declared
    return geom_col, bbox_col


def connect():
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(f"SET memory_limit = '{MEMORY}';")
    con.execute(f"SET threads = {THREADS};")
    if TEMP_DIR:
        os.makedirs(TEMP_DIR, exist_ok=True)
        con.execute(f"SET temp_directory = '{TEMP_DIR}';")
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET enable_progress_bar = true;")
    return con


def dataset_extent(con, src, geom_col):
    """(minx, miny, maxx, maxy) for the whole file, or None if it has no rows."""
    row = con.execute(
        f"""
        SELECT ST_XMin(e), ST_YMin(e), ST_XMax(e), ST_YMax(e)
        FROM (SELECT ST_Extent_Agg("{geom_col}") AS e FROM read_parquet(?))
        """,
        [src],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return tuple(float(v) for v in row)


def sort_to_temp(con, src, tmp, geom_col, bbox_col, extent):
    """Hilbert-sort src into tmp. Cheap SNAPPY here; real codec applied later."""
    minx, miny, maxx, maxy = extent
    bounds = (f"ST_Extent(ST_MakeEnvelope({minx!r}, {miny!r}, "
              f"{maxx!r}, {maxy!r}))::BOX_2D")

    if bbox_col:
        # An existing per-row bbox stays valid: sorting reorders rows, it does
        # not change any row's own extent. Carry the column through untouched.
        projection = "*"
    else:
        projection = f"""*,
            struct_pack(
                xmin := ST_XMin("{geom_col}"),
                ymin := ST_YMin("{geom_col}"),
                xmax := ST_XMax("{geom_col}"),
                ymax := ST_YMax("{geom_col}")
            ) AS bbox"""

    con.execute(
        f"""
        COPY (
            SELECT {projection}
            FROM read_parquet('{src}')
            ORDER BY ST_Hilbert("{geom_col}", {bounds})
        )
        TO '{tmp}'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE});
        """
    )


def patch_metadata(tmp, out, geom_col, bbox_col, extent):
    """Stream tmp -> out, rewriting 'geo' so the covering bbox is declared."""
    pf = pq.ParquetFile(tmp)
    schema = pf.schema_arrow

    geo = {}
    if schema.metadata and b"geo" in schema.metadata:
        try:
            geo = json.loads(schema.metadata[b"geo"])
        except Exception:
            geo = {}

    geo["version"] = "1.1.0"
    geo["primary_column"] = geom_col
    col = geo.setdefault("columns", {}).setdefault(geom_col, {})
    col.setdefault("encoding", "WKB")
    col.setdefault("geometry_types", [])
    col["bbox"] = list(extent)
    col["covering"] = {
        "bbox": {
            "xmin": [bbox_col, "xmin"],
            "ymin": [bbox_col, "ymin"],
            "xmax": [bbox_col, "xmax"],
            "ymax": [bbox_col, "ymax"],
        }
    }

    meta = dict(schema.metadata or {})
    meta[b"geo"] = json.dumps(geo).encode("utf-8")
    schema = schema.with_metadata(meta)

    writer = pq.ParquetWriter(
        out, schema,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
    )
    try:
        for i in range(pf.num_row_groups):
            writer.write_table(pf.read_row_group(i))
    finally:
        writer.close()


def process(con, src, out):
    geom_col, bbox_col = inspect(src)
    if geom_col is None:
        print("  SKIP - no geometry column found")
        return False

    extent = dataset_extent(con, src, geom_col)
    if extent is None:
        print("  SKIP - file has no geometries")
        return False

    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        sort_to_temp(con, src, tmp, geom_col, bbox_col, extent)
        patch_metadata(tmp, out, geom_col, bbox_col or "bbox", extent)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def main():
    if os.path.isdir(SRC_FOLDER):
        files = sorted(glob.glob(os.path.join(SRC_FOLDER, PATTERN)))
    elif os.path.isfile(SRC_FOLDER):
        files = [SRC_FOLDER]
    else:
        files = sorted(glob.glob(SRC_FOLDER))

    # Don't re-sort our own outputs on a second pass over the same folder.
    files = [f for f in files if not os.path.splitext(f)[0].endswith(SUFFIX)]

    if not files:
        print(f"No files matched: {SRC_FOLDER} / {PATTERN}")
        return

    if OUT_FOLDER:
        os.makedirs(OUT_FOLDER, exist_ok=True)

    print(f"{len(files)} file(s) to process")
    print(f"compression={COMPRESSION}:{COMPRESSION_LEVEL}  "
          f"threads={THREADS}  memory={MEMORY}")
    print("=" * 70)

    con = connect()
    done = failed = skipped = 0

    for n, src in enumerate(files, 1):
        stem, ext = os.path.splitext(os.path.basename(src))
        out = os.path.join(OUT_FOLDER or os.path.dirname(src),
                           f"{stem}{SUFFIX}{ext}")

        print(f"[{n}/{len(files)}] {os.path.basename(src)}")
        if os.path.exists(out) and not OVERWRITE:
            print("  SKIP - output already exists")
            skipped += 1
            continue

        t0 = time.time()
        try:
            if process(con, src, out):
                print(f"  OK - {os.path.getsize(out) / 1e6:,.1f} MB "
                      f"in {time.time() - t0:.1f}s")
                done += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  FAILED - {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"sorted={done}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()