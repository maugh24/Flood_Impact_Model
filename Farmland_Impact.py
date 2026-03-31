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
from rasterio.features import rasterize
import warnings
warnings.filterwarnings('ignore')

# @profile
def calculate_basin_farmland(basin_file, rivers, farmland_raster_folder, output_folder, index, farmland_value=40):

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output files
    output_csv = output_path / f"farmland_statistics_{index}.csv"
    

    # ===== READ BASINS =====
    # Get first raster's CRS and reproject basins once
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)]).to_crs(4326)

    # Find all .tif files in the folder
    raster_folder = Path(farmland_raster_folder)
    raster_files = list(raster_folder.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in {raster_folder}")

    # Initialize counters
    total_farmland_cells = 0
    
    # spatial index
    sindex = basins.sindex

    # Process each raster tile
    num_tiles_rasterized = 0
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            # Quick check: does this tile intersect with ANY basin?
            raster_bounds_geom = box(*src.bounds)

            # Find intersecting basins (spatial index is used automatically by geopandas)
            candidates = basins.iloc[list(sindex.intersection(raster_bounds_geom.bounds))]
            intersecting_basins: gpd.GeoDataFrame = candidates[candidates.intersects(raster_bounds_geom)]
            
            if intersecting_basins.empty:
                continue  # Skip this tile if it doesn't intersect any basin
            
            # mask_arr = rasterize(
            #     [(geom, 1) for geom in intersecting_basins.geometry],
            #     out_shape=(src.height, src.width),
            #     transform=src.transform,
            #     fill=0,
            #     dtype="uint8"
            # )
            farmland_mask, transform = mask(src, intersecting_basins.geometry, crop=True, all_touched=True, nodata=0)
            

            # # Read raster (still full read — see next improvement)
            # data = src.read(1)

            farmland_cells = np.count_nonzero(farmland_mask)

            if farmland_cells == 0:
                continue
            
            total_farmland_cells += farmland_cells
            output_raster = output_path / f"farmland_affected_{index}_{num_tiles_rasterized}.tif"
            num_tiles_rasterized += 1
            
            shape = farmland_mask.shape
            transform = src.transform
            
            with rasterio.open(
                    output_raster,
                    "w",
                    driver="GTiff",
                    height=shape[0],
                    width=shape[1],
                    count=1,
                    dtype=rasterio.uint8,
                    crs=src.crs,
                    transform=transform,
                    nodata=0,
                    compress='deflate'
            ) as dst:
                dst.write(farmland_mask.astype(np.uint8), 1)

    if num_tiles_rasterized == 0:
        raise FileNotFoundError(f"No valid tile files found in {raster_folder}")


    # Calculate area
    pixel_area_km2 = (10 * 10) / 1_000_000  # 0.0001 km² per pixel
    total_farmland_area_km2 = total_farmland_cells * pixel_area_km2

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