import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
import json

parquet_input  = r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet\europe-polygons.parquet"
parquet_output = r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet\bbox\europe-polygons-bbox.parquet"
temp_dir       = r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet\bbox\temp_chunks"

# Keep these + geometry_bbox (required for bbox spatial filtering)
KEEP_COLS = ["building", "name", "amenity", "geometry", "geometry_bbox"]

CHUNK_ROWS = 500_000  # safe size for Windows writer

Path(parquet_output).parent.mkdir(parents=True, exist_ok=True)
Path(temp_dir).mkdir(parents=True, exist_ok=True)

# ── Preserve geo metadata, scoped to kept columns only ──────────────────────
original_schema = pq.read_schema(parquet_input)
raw_meta        = original_schema.metadata
geo_meta        = json.loads(raw_meta[b"geo"])

# Strip out any column entries not in KEEP_COLS
geo_meta["columns"] = {
    k: v for k, v in geo_meta["columns"].items()
    if k in KEEP_COLS
}
updated_meta = {**raw_meta, b"geo": json.dumps(geo_meta).encode()}

# ── Stream, filter columns, write chunks ────────────────────────────────────
dataset     = ds.dataset(parquet_input, format="parquet")
chunk_paths = []
batch_buf   = []
buf_rows    = 0
chunk_idx   = 0

def flush(batch_buf, chunk_idx, schema):
    table      = pa.Table.from_batches(batch_buf, schema=schema)
    chunk_path = str(Path(temp_dir) / f"chunk_{chunk_idx:04d}.parquet")
    pq.write_table(table, chunk_path, compression="snappy")
    print(f"  chunk {chunk_idx:04d}  —  {table.num_rows:,} rows")
    return chunk_path

# Build output schema from first batch
first_batch  = next(dataset.to_batches(batch_size=1, columns=KEEP_COLS))
out_schema   = first_batch.schema.with_metadata(updated_meta)

for batch in dataset.to_batches(batch_size=50_000, columns=KEEP_COLS):
    batch_buf.append(batch)
    buf_rows += batch.num_rows

    if buf_rows >= CHUNK_ROWS:
        chunk_paths.append(flush(batch_buf, chunk_idx, out_schema))
        batch_buf, buf_rows, chunk_idx = [], 0, chunk_idx + 1

if batch_buf:
    chunk_paths.append(flush(batch_buf, chunk_idx, out_schema))

print(f"\n{len(chunk_paths)} chunks written")

# ── Merge into single file ───────────────────────────────────────────────────
print("Merging...")
merged = ds.dataset(chunk_paths, format="parquet")

with pq.ParquetWriter(
    parquet_output,
    schema=merged.schema.with_metadata(updated_meta),
    compression="snappy"
) as writer:
    for batch in merged.to_batches(batch_size=50_000):
        writer.write_table(pa.Table.from_batches([batch]))

print(f"Done → {parquet_output}")