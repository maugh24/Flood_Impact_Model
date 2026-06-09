import pandas as pd
import geopandas as gpd
from shapely import wkt

# 1. Load data
df = pd.read_csv(r"C:\Users\maugh24\Downloads\30d_buildings.csv")

# 2. Convert the WKT text strings into actual geometric shapes
df['geometry'] = df['geometry'].apply(wkt.loads)

# 3. Convert to a GeoDataFrame and save
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
gdf.to_parquet(r"C:\Users\maugh24\Downloads\30d_buildings.parquet")