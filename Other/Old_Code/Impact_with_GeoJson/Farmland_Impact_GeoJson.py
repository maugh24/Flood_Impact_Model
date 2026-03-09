import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box
import json
import tempfile
import os


def calculate_basin_farmland(basin_file, farmland_raster_folder, output_folder, farmland_value=40):
    """
    Calculate total farmland area across all basins from multiple ESA raster tiles.
    Creates output raster, GeoJSON, and statistics CSV.

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
    output_geojson = output_path / "farmland_affected.geojson"

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

    # Find all .tif files in the folder
    raster_folder = Path(farmland_raster_folder)
    raster_files = list(raster_folder.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in {raster_folder}")

    print(f"Found {len(raster_files)} raster tiles to process")

    # Initialize counters
    total_farmland_cells = 0
    total_cells = 0
    raster_crs = None

    # Create temporary directory for intermediate rasters
    temp_dir = tempfile.mkdtemp()
    temp_rasters = []

    # Process each raster tile
    for raster_idx, raster_file in enumerate(raster_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing raster tile {raster_idx}/{len(raster_files)}: {raster_file.name}")
        print(f"{'=' * 60}")

        with rasterio.open(raster_file) as src:
            print(f"Raster CRS: {src.crs}")
            print(f"Raster size: {src.width} x {src.height} = {src.width * src.height:,} cells")

            # Reproject basins if needed (only once, using first raster's CRS)
            if raster_idx == 1:
                raster_crs = src.crs
                if basins.crs != src.crs:
                    print("Reprojecting basins to match raster...")
                    basins = basins.to_crs(src.crs)

            # Check which basins intersect with this raster tile
            raster_bounds_geom = box(*src.bounds)
            intersecting_basins = basins[basins.intersects(raster_bounds_geom)]

            if len(intersecting_basins) == 0:
                print(f"No basins intersect with this tile - skipping")
                continue

            print(f"Processing {len(intersecting_basins)} basins that intersect this tile...")

            # Mask the raster by all intersecting basins
            try:
                basin_geometries = intersecting_basins.geometry.tolist()
                out_image, out_transform = mask(
                    src,
                    basin_geometries,
                    crop=True,
                    nodata=0,
                    filled=True
                )

                # Get the data (first band)
                data = out_image[0]

                # Count cells for statistics
                valid_mask = data != 0
                valid_data = data[valid_mask]

                if len(valid_data) > 0:
                    total_cells += len(valid_data)
                    farmland_cells = np.sum(valid_data == farmland_value)
                    total_farmland_cells += farmland_cells

                    print(f"  Found {farmland_cells:,} farmland cells in this tile")

                # Create binary farmland raster (1 = farmland, 0 = not farmland)
                farmland_binary = np.where(data == farmland_value, 1, 0).astype(np.uint8)

                # Only save if there's farmland
                if np.any(farmland_binary == 1):
                    # Save to temporary file
                    temp_file = os.path.join(temp_dir, f"farmland_tile_{raster_idx}.tif")

                    with rasterio.open(
                            temp_file,
                            'w',
                            driver='GTiff',
                            height=farmland_binary.shape[0],
                            width=farmland_binary.shape[1],
                            count=1,
                            dtype=np.uint8,
                            crs=src.crs,
                            transform=out_transform,
                            nodata=0
                    ) as dst:
                        dst.write(farmland_binary, 1)

                    temp_rasters.append(temp_file)
                    print(f"  Saved temporary farmland raster")

            except Exception as e:
                print(f"  Error processing tile: {e}")
                continue

    print("\n\nProcessing complete!")

    # Calculate area
    pixel_area_km2 = (10 * 10) / 1_000_000  # 0.0001 km² per pixel
    total_farmland_area_km2 = total_farmland_cells * pixel_area_km2

    # === MERGE TEMPORARY RASTERS INTO FINAL OUTPUT ===
    if temp_rasters:
        print(f"\n1. Merging {len(temp_rasters)} farmland tiles into final raster...")

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

        # Write merged output
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "nodata": 0
        })

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"   ✓ {output_raster.name}")
        print(f"     Size: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
        print(f"     Resolution: 10m x 10m")
        print(f"     Farmland pixels: {np.sum(mosaic == 1):,}")

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
        print(f"\n⚠ No farmland found - skipping raster output")

    # === CREATE GEOJSON ===
    if total_farmland_cells > 0:
        print(f"\n2. Creating GeoJSON (basin boundaries)...")

        # Reproject basins to WGS84 for GeoJSON
        basins_wgs84 = basins.to_crs("EPSG:4326")

        # Create GeoJSON with basin boundaries
        farmland_summary = {
            "type": "FeatureCollection",
            "features": []
        }

        for idx, basin in basins_wgs84.iterrows():
            feature = {
                "type": "Feature",
                "properties": {
                    "basin_id": int(idx),
                    "note": "Basins containing affected farmland"
                },
                "geometry": basin.geometry.__geo_interface__
            }
            farmland_summary["features"].append(feature)

        with open(output_geojson, 'w') as f:
            json.dump(farmland_summary, f)

        print(f"   ✓ {output_geojson.name}")
        print(f"     Features: {len(basins_wgs84)} basin boundaries")
    else:
        print(f"\n⚠ No farmland found - skipping GeoJSON output")

    # === CREATE STATISTICS CSV ===
    print(f"\n3. Saving statistics CSV...")

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
    print("  2. farmland_affected.geojson (basin boundaries)")
    print("  3. farmland_statistics.csv (area in km²)")
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