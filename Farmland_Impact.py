import geopandas as gpd
import pandas as pd

_FARM_GDF: gpd.GeoDataFrame = None
def get_farmland_gdf(farmland_parquet):
    # This means each process only reads the farm parquet once
    global _FARM_GDF
    if _FARM_GDF is None:
        _FARM_GDF = gpd.read_parquet(farmland_parquet)
    return _FARM_GDF

# @profile
def calculate_basin_farmland(basin_file, rivers, farmland_parquet):
    # ===== READ BASINS =====
    # Get first raster's CRS and reproject basins once
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)]).to_crs({'proj': 'cea'})

    # Get farmland GeoDataFrame (cached in memory after first read)
    farm_gdf = get_farmland_gdf(farmland_parquet)
    
    # Filter farmland to only the basins of interest (spatial index is used automatically by geopandas and is fast)
    farm_gdf = (
        gpd.sjoin(farm_gdf, basins[['geometry']], how='inner', predicate='intersects')
        .drop(columns=['index_right'])
        .to_crs({'proj': 'cea'}) # This projection is correctly in meters, so area calculations will be accurate
    )
    
    # Find the area of farmland intersecting each basin
    intersections = gpd.overlay(basins, farm_gdf, how="intersection")
    intersections["area_m2"] = intersections.geometry.area
    result = (
        intersections
        .groupby("linkno")["area_m2"]
        .sum()
        .reset_index()
    )
    
    # Add dropped linknos with 0 area
    missing_linknos = set(rivers) - set(result['linkno'])
    if missing_linknos:
        result = pd.concat([result, pd.DataFrame({'linkno': list(missing_linknos), 'area_m2': 0})], ignore_index=True)
    
    result['area_km2'] = result['area_m2'] / 1e6

    return result[['linkno', 'area_km2']]

def calculate_basin_farmland_wrapper(args):
    return calculate_basin_farmland(*args)

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