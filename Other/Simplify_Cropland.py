import geopandas as gpd

# 1. Load the parquet file
input_path = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\n30e000cropland.parquet"
output_path = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\bbox\n30e000cropland.parquet"

gdf = gpd.read_parquet(input_path)

# 2. Simplify the geometries
# 'tolerance' defines how aggressively to simplify (higher = more simplified).
# 'preserve_topology=True' prevents polygons from collapsing into weird shapes.
tolerance_value = 10  # Adjust this based on your Coordinate Reference System (CRS)
gdf['geometry'] = gdf['geometry'].simplify(tolerance=tolerance_value, preserve_topology=True)

# 3. Filter out small polygons
# 'min_area' threshold depends heavily on your CRS (e.g., square meters vs. square degrees)
min_area_threshold = 500
gdf = gdf[gdf['geometry'].area >= min_area_threshold]

# 4. Write to parquet with your existing bbox configuration
gdf.to_parquet(
    output_path,
    index=False,
    compression='brotli',
    write_covering_bbox=True
)