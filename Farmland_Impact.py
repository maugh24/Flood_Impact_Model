from operator import index

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import box
from rasterio.io import MemoryFile
from rasterio.merge import merge
from osgeo import gdal
import warnings
warnings.filterwarnings('ignore')

# @profile
def calculate_basin_farmland(basin_file, rivers, farmland_raster_folder, output_folder, index, farmland_value=40):

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output files
    output_csv = output_path / f"farmland_statistics_{index}.csv"
    output_raster = output_path / f"farmland_affected_{index}.tif"

    # ===== READ BASINS =====
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)])

    # Find all .tif files in the folder
    raster_folder = Path(farmland_raster_folder)
    raster_files = list(raster_folder.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in {raster_folder}")

    # Initialize counters
    total_farmland_cells = 0
    raster_crs = None

    # Dictionary to store raster data and transforms for merging
    raster_data_dict = {}

    # Get first raster's CRS and reproject basins once
    with rasterio.open(raster_files[0]) as first_src:
        raster_crs = first_src.crs
        if basins.crs != raster_crs:
            basins = basins.to_crs(raster_crs)

    # Process each raster tile
    tiles_to_combine = []
    for raster_file in raster_files:

        with rasterio.open(raster_file) as src:
            # Quick check: does this tile intersect with ANY basin?
            raster_bounds_geom = box(*src.bounds)

            # Find intersecting basins (spatial index is used automatically by geopandas)
            intersecting_basins = basins[basins.intersects(raster_bounds_geom)]

            if len(intersecting_basins) == 0:
                continue

            tiles_to_combine.append(raster_file)

    if not tiles_to_combine:
        raise FileNotFoundError(f"No tile files found in {raster_folder}")

    datasets = [rasterio.open(fp) for fp in tiles_to_combine]
    mosaic, transform = merge(datasets)

    meta = datasets[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    # Create in-memory dataset
    memfile = MemoryFile()
    with memfile.open(**meta) as dataset:
        dataset.write(mosaic)

    combined_dataset = memfile.open()

    # OPTIMIZATION: Mask with all basin geometries at once
    basin_geometries = intersecting_basins.geometry.tolist()

    # Use optimized masking parameters
    out_image, out_transform = mask(
        combined_dataset,
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

        # Store data for final raster output
        raster_data_dict ={
            'data': farmland_mask.astype(np.uint8),
            'transform': out_transform,
            'shape': farmland_mask.shape
        }

    # Calculate area
    pixel_area_km2 = (10 * 10) / 1_000_000  # 0.0001 km² per pixel
    total_farmland_area_km2 = total_farmland_cells * pixel_area_km2

    # === WRITE FINAL RASTER OUTPUT (if farmland found) ===
    mask_data = raster_data_dict["data"]
    transform = raster_data_dict["transform"]
    height, width = raster_data_dict["shape"]

    with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=mask_data.dtype,
            crs=combined_dataset.crs,  # preserve CRS
            transform=transform,
            nodata=0
    ) as dst:
        dst.write(mask_data, 1)

    # === CREATE STATISTICS CSV ===

    result = pd.DataFrame({
        'total_farmland_area_km2': [total_farmland_area_km2]
    })

    result.to_csv(output_csv, index=False)
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