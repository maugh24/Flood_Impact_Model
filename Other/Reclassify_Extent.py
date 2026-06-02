import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
from pathlib import Path

folder = Path(r'C:\C_Drive_Brians_Stuff\Python_Projects\Napal_Floods\Karnali')

for tif_path in folder.glob('*.tif'):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype('float32')
        transform = src.transform
        crs = src.crs

    data[data == 0] = np.nan

    mask = np.isfinite(data)
    geoms = [
        {"geometry": shape(geom), "value": val}
        for geom, val in shapes(data, mask=mask, transform=transform)
    ]

    gdf = gpd.GeoDataFrame(geoms, crs=crs)
    out_path = folder / (tif_path.stem + 'extent.parquet')
    gdf.to_parquet(out_path)
    print(f"Saved {out_path.name}")