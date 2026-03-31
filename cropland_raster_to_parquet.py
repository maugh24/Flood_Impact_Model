import glob
import rasterio
import pandas as pd
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape
import multiprocessing as mp
import tqdm

def vectorize(cropland_file: str, index):
    with rasterio.open(cropland_file) as src:
        data = src.read(1)
        
    cropland_mask = (data == 40)
    shapes = features.shapes(data, cropland_mask, transform=src.transform)
    shapes = [shape(geom) for geom, val in shapes if val == 40]
    gdf = gpd.GeoDataFrame(geometry=shapes, crs=4326)
    out_file = rf"C:\Users\ricky\Downloads\cropland_{index}.parquet"
    gdf.to_parquet(out_file)
    return out_file

def vectorize_wrapper(args):
    return vectorize(*args)

if __name__ == "__main__":
    cropland_files = glob.glob(r"C:\Users\ricky\Downloads\ESA_Caribbean\ESA_Caribbean\*.tif")
    # ~3.75 GB per core peak, ~7 mins for 28 files with 16 cores (60 GB peak memory)
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(vectorize_wrapper, [(file, idx) for idx, file in enumerate(cropland_files)]), total=len(cropland_files)))
    (
        gpd.GeoDataFrame(pd.concat([gpd.read_parquet(file) for file in results], ignore_index=True, copy=False))
        .to_parquet(r"C:\Users\ricky\Downloads\cropland.parquet")
    )
