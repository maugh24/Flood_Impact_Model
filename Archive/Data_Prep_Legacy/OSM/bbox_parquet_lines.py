import geopandas as gpd
import pyarrow.dataset as ds

parquet_input =r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet\europe-lines.parquet"
parquet_output = r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet\bbox\europe-lines-bbox.parquet"

gpd.read_parquet(parquet_input).to_parquet(
    parquet_output,
    index=False,
    compression="brotli",
    write_covering_bbox=True
)
#gpd.read_parquet(
    #parquet_output,
    #columns=['building','name','amenity','geometry'],
    #filters=ds.field("building").is_valid())['building'].unique()
