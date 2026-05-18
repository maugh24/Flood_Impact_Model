import rasterio
from rasterio.windows import Window
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape
import multiprocessing as mp
from pathlib import Path
import pandas as pd


def process_tile(args):
    raster_path, window, transform, crs = args

    with rasterio.open(raster_path) as src:
        data = src.read(1, window=window)
        mask = data > 0
        if not mask.any():
            return None

        win_transform = src.window_transform(window)
        shapes_gen = features.shapes(data, mask=mask, transform=win_transform)

        geoms = [{"geometry": shape(s), "pop_value": v} for s, v in shapes_gen]
        return gpd.GeoDataFrame(geoms, crs=crs)


def run_parallel_vectorization(raster_path, output_path, tile_size=4000):
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        windows = []
        for j in range(0, height, tile_size):
            for i in range(0, width, tile_size):
                window = Window(i, j,
                                min(tile_size, width - i),
                                min(tile_size, height - j))
                windows.append(window)

        args = [(str(raster_path), w, transform, crs) for w in windows]

    with mp.Pool(processes=mp.cpu_count() // 2) as pool:
        results = pool.map(process_tile, args)

    results = [r for r in results if r is not None]

    final_gdf = pd.concat(results, ignore_index=True)
    final_gdf = final_gdf.to_crs(epsg=4326)
    final_gdf.to_parquet(output_path, index=False, write_covering_bbox=True)


if __name__ == "__main__":
    input_tif = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\global_pop_2025_CN_1km_R2025A_UA_v1.tif"
    output_pq = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\population.parquet"

    if Path(input_tif).exists():
        run_parallel_vectorization(input_tif, output_pq)