import geopandas as gpd
import pyarrow.dataset as ds
from shapely.ops import unary_union
from shapely.geometry import box
from pathlib import Path

# ── Inputs ───────────────────────────────────────────────────────────────────
flood_parquet    = r"D:\Brian\Flood_Impact_Model\Files\Catchment\Chian_Rai\GEOGLOWS_dem_99.70939_19.79092_99.97134_19.99271_ARC_Flood_rp2extent.parquet"  # your small tile
farmland_parquet = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\n30e060cropland.parquet"   # large file
output_parquet   = r"D:\Brian\Flood_Impact_Model\Files\ESA\Chian_Rai_farmland_clipped.parquet"


Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load flood tile ───────────────────────────────────────────────────
print("Loading flood tile...")
flood     = gpd.read_parquet(flood_parquet)
clip_geom = unary_union(flood.geometry)
clip_bbox = flood.total_bounds  # [minx, miny, maxx, maxy]
print(f"  Clip bbox: {clip_bbox}")

# ── Step 2: Manual bbox filter using pyarrow pushdown on coordinate bounds ────
# Build a pyarrow filter that checks geometry_bbox struct fields if present,
# otherwise fall back to reading all and filtering in-memory by bbox polygon
print("Reading farmland schema...")
import pyarrow.parquet as pq
schema    = pq.read_schema(farmland_parquet)
col_names = schema.names
print(f"  Columns: {col_names}")

minx, miny, maxx, maxy = clip_bbox

# Build pyarrow filter depending on what columns exist
if "geometry_bbox" in col_names:
    # Struct column — access subfields
    bbox_filter = (
        (ds.field("geometry_bbox", "xmin") <= maxx) &
        (ds.field("geometry_bbox", "xmax") >= minx) &
        (ds.field("geometry_bbox", "ymin") <= maxy) &
        (ds.field("geometry_bbox", "ymax") >= miny)
    )
    print("  Using geometry_bbox struct filter")
elif all(c in col_names for c in ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]):
    # Flat bbox columns from earlier processing
    bbox_filter = (
        (ds.field("bbox_minx") <= maxx) &
        (ds.field("bbox_maxx") >= minx) &
        (ds.field("bbox_miny") <= maxy) &
        (ds.field("bbox_maxy") >= miny)
    )
    print("  Using flat bbox column filter")
else:
    # No bbox index — pyarrow will still do row-group stat pushdown on geometry
    # bounds if stats exist, otherwise full scan (unavoidable without bbox index)
    bbox_filter = None
    print("  WARNING: No bbox index found — full scan required")

# Combine with building validity filter if column exists
building_filter = ds.field("building").is_valid() if "building" in col_names else None

if bbox_filter is not None and building_filter is not None:
    combined_filter = bbox_filter & building_filter
elif bbox_filter is not None:
    combined_filter = bbox_filter
elif building_filter is not None:
    combined_filter = building_filter
else:
    combined_filter = None

# ── Step 3: Read with pyarrow filter, convert to GeoDataFrame ─────────────────
print("Applying filter and reading...")
keep_cols = [c for c in ["building", "name", "amenity", "geometry"]
             if c in col_names]

dataset    = ds.dataset(farmland_parquet, format="parquet")
table      = dataset.to_table(columns=keep_cols, filter=combined_filter)
print(f"  Rows after pyarrow filter: {table.num_rows:,}")

gdf = gpd.GeoDataFrame.from_arrow(table)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
if gdf.crs != flood.crs:
    gdf = gdf.to_crs(flood.crs)

# ── Step 4: In-memory bbox clip (fast), then exact polygon clip ───────────────
print("Applying bbox spatial filter...")
bbox_poly = box(minx, miny, maxx, maxy)
gdf       = gdf[gdf.geometry.intersects(bbox_poly)].reset_index(drop=True)
print(f"  Rows after bbox intersect: {len(gdf):,}")

print("Clipping to exact flood geometry...")
clipped = gpd.clip(gdf, clip_geom)
clipped = clipped[~clipped.geometry.is_empty].reset_index(drop=True)
print(f"  Rows after exact clip: {len(clipped):,}")

# ── Step 5: Write output ──────────────────────────────────────────────────────
if len(clipped) == 0:
    print("No features intersect the flood tile — nothing written.")
else:
    clipped.to_parquet(
        output_parquet,
        index=False,
        compression="snappy",
        write_covering_bbox=True
    )
    print(f"\nDone → {output_parquet}")
    print(f"  Final rows: {len(clipped):,}")
    print(clipped.dtypes)
    print(clipped.head())