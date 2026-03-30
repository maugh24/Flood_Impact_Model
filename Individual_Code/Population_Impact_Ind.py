from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
import shapely.wkb as wkblib
import warnings
warnings.filterwarnings('ignore')

#@profile
def calculate_basin_population(basin_file, population_raster, output_folder):
    """
    Calculate total population across all basins and create population density heatmap.
    OPTIMIZED: Processes all basins in a single pass.

    Parameters:
    -----------
    basin_file : str
        Path to basin shapefile or parquet file
    population_raster : str
        Path to population raster file
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "population_statistics.csv"
    output_raster = output_path / "population_affected.tif"

    basin_path = Path(basin_file)
    print(f"\nReading basin data from {basin_path.suffix} file...")

    # Read based on file extension
    if basin_path.suffix == '.parquet':
        # Read parquet file and convert WKB geometry to shapely geometries
        df = pd.read_parquet(basin_file, engine='fastparquet')
        # Convert WKB geometry column to shapely geometries
        df['geometry'] = df['geometry'].apply(lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x))
        basins = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    else:
        raise ValueError(f"Unsupported file format: {basin_path.suffix}")

    print(f"Loaded {len(basins)} basins")

    # Open the raster ONCE
    with rasterio.open(population_raster) as src:
        print(f"Raster CRS: {src.crs}")
        print(f"Basin CRS: {basins.crs}")
        print(f"Raster resolution: {src.res[0]:.2f}m x {src.res[1]:.2f}m")

        # Reproject basins if needed
        if basins.crs != src.crs:
            print("Reprojecting basins to match raster...")
            basins = basins.to_crs(src.crs)

        # === OPTIMIZED: Process ALL basins in ONE operation ===
        print("\nMasking raster with basin geometries...")

        # Get all basin geometries as a list
        all_geometries = basins.geometry.tolist()

        # Mask the raster ONCE with all basins
        out_image, out_transform = mask(
            src,
            all_geometries,
            crop=True,
            nodata=0,
            filled=True,
            all_touched=True  # Include pixels that touch basin boundaries
        )

        # Get the data (first band)
        data = out_image[0]

        print(f"Masked raster size: {data.shape[1]} x {data.shape[0]} pixels")

        # Calculate total population efficiently
        # Use nansum to automatically handle both NaN and 0 values
        total_population = np.nansum(data[data > 0]) if np.any(data > 0) else 0

        print(f"Total population: {total_population:,.0f}")

    # === CREATE OUTPUT RASTER ===
    if total_population > 0:
        print(f"\n1. Saving population raster...")

        # Write the masked raster
        out_meta = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": data.dtype,
            "crs": basins.crs,
            "transform": out_transform,
            "compress": "deflate",
            "nodata": 0,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256
        }

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(data, 1)

        print(f"     {output_raster.name}")
        print(f"     Size: {data.shape[1]} x {data.shape[0]} pixels")
        print(f"     Resolution: ~100m x 100m")
        print(f"     Total population in raster: {total_population:,.0f}")

    else:
        print(f"\n⚠ No population found - skipping raster output")

    # === CREATE STATISTICS CSV ===
    print(f"\n2. Saving statistics CSV...")

    # Create simple DataFrame with just the total
    result = pd.DataFrame({
        'total_population': [total_population]
    })

    # Save results
    result.to_csv(output_csv, index=False)
    print(f"     {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("POPULATION STATISTICS")
    print("=" * 60)
    print(f"Total population affected: {total_population:,.0f}")
    print(f"\nFiles created in: {output_path}")
    print("  1. population_affected.tif (100m population density heatmap)")
    print("  2. population_statistics.csv (total population)")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    population_raster = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\global_pop_2025_CN_1km_R2025A_UA_v1.tif"
    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Population_Impact"

    results = calculate_basin_population(
        basin_file,
        population_raster,
        output_folder
    )

    print("\nResult:")
    print(results)