import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box
import shapely.wkb as wkblib
import warnings
warnings.filterwarnings('ignore')

#@profile
def calculate_basin_farmland(basin_file, farmland_raster_folder, output_folder, farmland_value=40):
    """
    Calculate total farmland area across all basins from multiple ESA raster tiles.
    OPTIMIZED: Faster processing with reduced I/O and memory operations.

    Parameters:
    -----------
    basin_file : str
        Path to basin shapefile or parquet file
    farmland_raster_folder : str
        Path to folder containing ESA WorldCover .tif tiles
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    farmland_value : int
        Raster value representing farmland (default: 40 for ESA WorldCover)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "farmland_statistics.csv"
    output_raster = output_path / "farmland_affected.tif"

    basin_path = Path(basin_file)
    print(f"\nReading basin data from {basin_path.suffix} file...")

    # Read based on file extension
    if basin_path.suffix == '.parquet':
        # Read parquet file and convert WKB geometry to shapely geometries
        df = pd.read_parquet(basin_file, engine='fastparquet')
        # Convert WKB geometry column to shapely geometries
        df['geometry'] = df['geometry'].apply(lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x))
        basins = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    elif basin_path.suffix in ['.shp', '.geojson', '.gpkg']:
        basins = gpd.read_file(basin_file)
    else:
        raise ValueError(f"Unsupported file format: {basin_path.suffix}")

    print(f"Loaded {len(basins)} basins")

    # Find all .tif files in the folder
    raster_folder = Path(farmland_raster_folder)
    raster_files = list(raster_folder.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in {raster_folder}")

    print(f"Found {len(raster_files)} raster tiles to process")

    # Initialize counters
    total_farmland_cells = 0
    raster_crs = None

    # Dictionary to store raster data and transforms for merging
    raster_data_dict = {}

    # Get first raster's CRS and reproject basins once
    with rasterio.open(raster_files[0]) as first_src:
        raster_crs = first_src.crs
        if basins.crs != raster_crs:
            print("Reprojecting basins to match raster CRS...")
            basins = basins.to_crs(raster_crs)

    print("Processing raster tiles...")

    # Process each raster tile
    for raster_idx, raster_file in enumerate(raster_files, 1):
        print(f"  Tile {raster_idx}/{len(raster_files)}: {raster_file.name}", end='')

        with rasterio.open(raster_file) as src:
            # Quick check: does this tile intersect with ANY basin?
            raster_bounds_geom = box(*src.bounds)

            # Find intersecting basins (spatial index is used automatically by geopandas)
            intersecting_basins = basins[basins.intersects(raster_bounds_geom)]

            if len(intersecting_basins) == 0:
                print(" - no intersecting basins, skipped")
                continue

            try:
                # OPTIMIZATION: Mask with all basin geometries at once
                basin_geometries = intersecting_basins.geometry.tolist()

                # Use optimized masking parameters
                out_image, out_transform = mask(
                    src,
                    basin_geometries,
                    crop=True,
                    nodata=0,
                    filled=True,
                    all_touched=False,
                    indexes=1  # Only read band 1
                )

                # Get the data directly (already a 2D array from indexes=1)
                if out_image.ndim == 3:
                    data = out_image[0]
                else:
                    data = out_image

                # Count farmland cells efficiently
                farmland_mask = (data == farmland_value)
                farmland_cells = np.count_nonzero(farmland_mask)

                if farmland_cells > 0:
                    total_farmland_cells += farmland_cells
                    print(f" - found {farmland_cells:,} farmland cells")

                    # Store data for final raster output
                    raster_data_dict[raster_idx] = {
                        'data': farmland_mask.astype(np.uint8),
                        'transform': out_transform,
                        'shape': farmland_mask.shape
                    }
                else:
                    print(" - no farmland found")

            except Exception as e:
                print(f" - error: {e}")
                continue

    print("\nProcessing complete!")

    # Calculate area
    pixel_area_km2 = (10 * 10) / 1_000_000  # 0.0001 km² per pixel
    total_farmland_area_km2 = total_farmland_cells * pixel_area_km2

    # === WRITE FINAL RASTER OUTPUT (if farmland found) ===
    if raster_data_dict:
        print(f"\n1. Writing farmland raster...")

        # Get the largest raster dimensions from collected data
        max_height = max(d['shape'][0] for d in raster_data_dict.values())
        max_width = max(d['shape'][1] for d in raster_data_dict.values())

        # Create output array
        output_mosaic = np.zeros((max_height, max_width), dtype=np.uint8)

        # Fill in the farmland data from each tile
        for idx, tile_data in raster_data_dict.items():
            data = tile_data['data']
            h, w = data.shape
            output_mosaic[:h, :w] = np.maximum(output_mosaic[:h, :w], data)

        # Write merged output with optimized settings
        out_meta = {
            "driver": "GTiff",
            "height": output_mosaic.shape[0],
            "width": output_mosaic.shape[1],
            "count": 1,
            "dtype": np.uint8,
            "crs": raster_crs,
            "transform": rasterio.transform.Affine.identity(),
            "compress": "deflate",
            "nodata": 0,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        }

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(output_mosaic, 1)

        print(f"   ✓ {output_raster.name}")
        print(f"     Size: {output_mosaic.shape[1]} x {output_mosaic.shape[0]} pixels")
        print(f"     Farmland pixels: {np.count_nonzero(output_mosaic == 1):,}")

    else:
        print(f"\n⚠ No farmland found - skipping raster output")

    # === CREATE STATISTICS CSV ===
    print(f"\n2. Saving statistics CSV...")

    result = pd.DataFrame({
        'total_farmland_area_km2': [total_farmland_area_km2]
    })

    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FARMLAND STATISTICS (ESA WorldCover Value 40)")
    print("=" * 60)
    print(f"Raster tiles processed: {len(raster_files)}")
    print(f"Total farmland area: {total_farmland_area_km2:,.2f} km²")
    print(f"\nFiles created in: {output_path}")
    print("  1. farmland_affected.tif (10m raster: 1=farmland, 0=not)")
    print("  2. farmland_statistics.csv (area in km²)")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    farmland_raster_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Caribbean"
    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Farmland_Impact"

    results = calculate_basin_farmland(
        basin_file,
        farmland_raster_folder,
        output_folder,
        farmland_value=40
    )

    print("\nResult:")
    print(results)