import geopandas as gpd
import rasterio
import pandas as pd
from pathlib import Path



try:
    from rasterstats import zonal_stats
    print("rasterstats imported successfully")
except ImportError as e:
    print(f"Error: {e}")
    print("Try: pip install rasterstats")