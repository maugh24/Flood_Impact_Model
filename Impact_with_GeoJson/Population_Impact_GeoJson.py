from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import numpy as np
import pandas as pd
import tempfile
import os
import json

@profile
def calculate_basin_population(basin_file, population_raster, output_folder):
    """
    Calculate total population across all basins and create population density heatmap.
    Creates CSV statistics, population density raster, and GeoJSON.

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
    output_geojson = output_path / "population_affected.geojson"

    basin_path = Path(basin_file)
    print(f"\nReading basin data from {basin_path.suffix} file...")

    # Read based on file extension
    if basin_path.suffix == '.parquet':
        basins = gpd.read_parquet(basin_file)
    elif basin_path.suffix in ['.shp', '.geojson', '.gpkg']:
        basins = gpd.read_file(basin_file)
    else:
        raise ValueError(f"Unsupported file format: {basin_path.suffix}")

    print(f"Loaded {len(basins)} basins")

    # Open the raster
    with rasterio.open(population_raster) as src:
        print(f"Raster CRS: {src.crs}")
        print(f"Basin CRS: {basins.crs}")
        print(f"Original raster resolution: {src.res[0]:.2f}m x {src.res[1]:.2f}m")

        # Reproject basins if needed
        if basins.crs != src.crs:
            print("Reprojecting basins to match raster...")
            basins = basins.to_crs(src.crs)

        # Initialize total population counter
        total_population = 0

        # Create temporary directory for intermediate rasters
        temp_dir = tempfile.mkdtemp()
        temp_rasters = []

        # Process each basin and create masked rasters
        for idx, basin in basins.iterrows():
            print(f"Processing basin {idx + 1}/{len(basins)}...", end='\r')

            try:
                # Extract raster values within basin polygon
                out_image, out_transform = mask(
                    src,
                    [basin.geometry],
                    crop=True,
                    nodata=0,
                    filled=True
                )

                # Get the data (first band)
                data = out_image[0]

                # Filter out nodata and negative values for statistics
                valid_data = data[data > 0]
                valid_data = valid_data[~np.isnan(valid_data)]

                # Add to total population
                if len(valid_data) > 0:
                    basin_pop = np.sum(valid_data)
                    total_population += basin_pop

                    # Save this basin's population raster to temp file
                    temp_file = os.path.join(temp_dir, f"pop_basin_{idx}.tif")

                    with rasterio.open(
                            temp_file,
                            'w',
                            driver='GTiff',
                            height=data.shape[0],
                            width=data.shape[1],
                            count=1,
                            dtype=data.dtype,
                            crs=src.crs,
                            transform=out_transform,
                            nodata=0
                    ) as dst:
                        dst.write(data, 1)

                    temp_rasters.append(temp_file)

            except Exception as e:
                print(f"\nError processing basin {idx}: {e}")
                continue

        print("\n\nProcessing complete!")

    # === MERGE TEMPORARY RASTERS INTO FINAL OUTPUT ===
    if temp_rasters:
        print(f"\n1. Merging {len(temp_rasters)} basin population rasters...")

        # Open all temporary rasters
        src_files_to_mosaic = []
        for temp_file in temp_rasters:
            src = rasterio.open(temp_file)
            src_files_to_mosaic.append(src)

        # Merge them
        mosaic, out_transform = merge(src_files_to_mosaic, nodata=0)

        # Close source files
        for src in src_files_to_mosaic:
            src.close()

        # Get original raster metadata
        with rasterio.open(population_raster) as original:
            original_dtype = original.dtypes[0]

        # Write merged output
        out_meta = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": 1,
            "dtype": original_dtype,
            "crs": basins.crs,
            "transform": out_transform,
            "compress": "lzw",
            "nodata": 0
        }

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(mosaic.astype(original_dtype))

        print(f"   ✓ {output_raster.name}")
        print(f"     Size: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
        print(f"     Resolution: ~100m x 100m (original resolution)")
        print(f"     Total population in raster: {np.sum(mosaic[mosaic > 0]):,.0f}")

        # Clean up temporary files
        for temp_file in temp_rasters:
            try:
                os.remove(temp_file)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

    else:
        print(f"\n⚠ No population found - skipping raster output")

    # === CREATE GEOJSON ===
    if total_population > 0:
        print(f"\n2. Creating GeoJSON (basin boundaries)...")

        # Reproject basins to WGS84 for GeoJSON
        basins_wgs84 = basins.to_crs("EPSG:4326")

        # Create GeoJSON with basin boundaries
        population_summary = {
            "type": "FeatureCollection",
            "features": []
        }

        for idx, basin in basins_wgs84.iterrows():
            feature = {
                "type": "Feature",
                "properties": {
                    "basin_id": int(idx),
                    "note": "Basins with affected population"
                },
                "geometry": basin.geometry.__geo_interface__
            }
            population_summary["features"].append(feature)

        with open(output_geojson, 'w') as f:
            json.dump(population_summary, f)

        print(f"   ✓ {output_geojson.name}")
        print(f"     Features: {len(basins_wgs84)} basin boundaries")
    else:
        print(f"\n⚠ No population found - skipping GeoJSON output")

    # === CREATE STATISTICS CSV ===
    print(f"\n3. Saving statistics CSV...")

    # Create simple DataFrame with just the total
    result = pd.DataFrame({
        'total_population': [total_population]
    })

    # Save results
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("POPULATION STATISTICS")
    print("=" * 60)
    print(f"Total population affected: {total_population:,.0f}")
    print(f"\nFiles created in: {output_path}")
    print("  1. population_affected.tif (100m population density heatmap)")
    print("  2. population_affected.geojson (basin boundaries)")
    print("  3. population_statistics.csv (total population)")
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