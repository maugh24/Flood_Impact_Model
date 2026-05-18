import geopandas as gpd

gpd.read_parquet(r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines.parquet").to_parquet(r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines_bbox.parquet", index=False, compression="brotli", write_covering_bbox=True)