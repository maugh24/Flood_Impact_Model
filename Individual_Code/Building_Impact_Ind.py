import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely.wkb as wkblib

#@profile
def calculate_basin_buildings_from_parquet(basin_file, building_parquet, output_folder):
    """
    Calculate building statistics from building parquet file (multipolygons only).
    Uses vectorized operations for fast processing.
    FULLY OPTIMIZED - NO LOOPS.

    Parameters:
    -----------
    basin_file : str
        Path to basin parquet file
    building_parquet : str
        Path to building parquet file with OSM multipolygon data
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "building_statistics.csv"
    output_gpkg = output_path / "buildings_affected.gpkg"

    # ===== READ BASINS =====
    print("\nReading basin data...")
    basin_path = Path(basin_file)
    if not basin_path.exists():
        raise FileNotFoundError(f"Basin file not found: {basin_path}")

    try:
        basins_df = pd.read_parquet(basin_file, engine='fastparquet')
        print("  Using fastparquet engine")
    except Exception as e:
        print(f"  ⚠ fastparquet failed: {str(e)}")
        print("  Falling back to pyarrow engine...")
        basins_df = pd.read_parquet(basin_file, engine='pyarrow')

    # Convert WKB geometry to shapely
    print("  Converting geometries...")
    if 'geometry' in basins_df.columns:
        first_geom = basins_df['geometry'].iloc[0]
        if isinstance(first_geom, (str, bytes)):
            basins_df['geometry'] = basins_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )

    basins = gpd.GeoDataFrame(basins_df, geometry='geometry', crs="EPSG:4326")
    print(f"Loaded {len(basins)} basins")

    # ===== READ BUILDING PARQUET =====
    print("\nReading building parquet...")
    building_path = Path(building_parquet)
    if not building_path.exists():
        raise FileNotFoundError(f"Building parquet file not found: {building_path}")

    try:
        building_df = pd.read_parquet(building_parquet, engine='fastparquet')
        print("  Using fastparquet engine")
    except Exception as e:
        print(f"  ⚠ fastparquet failed: {str(e)}")
        print("  Falling back to pyarrow engine...")
        building_df = pd.read_parquet(building_parquet, engine='pyarrow')

    # Convert WKB geometry to shapely if needed
    print("  Converting geometries...")
    if 'geometry' in building_df.columns:
        first_geom = building_df['geometry'].iloc[0]
        if isinstance(first_geom, (str, bytes)):
            building_df['geometry'] = building_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )

    buildings = gpd.GeoDataFrame(building_df, geometry='geometry', crs="EPSG:4326")
    print(f"  Loaded {len(buildings):,} building features")

    # Verify geometry types
    geom_types = buildings.geometry.type.value_counts()
    print(f"  Geometry types: {geom_types.to_dict()}")

    # ===== VECTORIZED FILTERING =====
    print("\nFiltering for building types...")

    # Building values we want
    building_values = [
        'yes', 'apartments', 'industrial', 'commercial', 'retail', 'residential',
        'civic', 'house', 'policlinic', 'hotel', 'stadium', 'church', 'government',
        'hospital', 'school', 'college', 'fire_station', 'university', 'monastery',
        'public', 'office', 'terminal', 'castle', 'ruins', 'garage', 'garages',
        'shed', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe', 'dormitory',
        'farm', 'ger', 'boathouse', 'semidetached_house', 'static_caravan',
        'stilt_house', 'terrace', 'tree_house', 'trullo', 'kiosk', 'supermarket',
        'warehouse', 'religious', 'cathedral', 'chapel', 'military', 'houseboat',
        'kingdom_hall', 'mosque', 'presbytery', 'shrine', 'synagogue', 'temple',
        'bakehouse', 'bridge', 'clock_tower', 'gatehouse', 'kindergarten', 'museum',
        'toilets', 'train_station', 'barn', 'conservatory', 'cowshed',
        'farm_auxiliary', 'greenhouse', 'slurry_tank', 'stable', 'sty', 'livestock',
        'grandstand', 'pavilion', 'riding_hall', 'sports_hall', 'sports_centre',
        'allotment_house', 'hangar', 'hut', 'carport', 'parking', 'digester',
        'service', 'tech_cab', 'transformer_tower', 'water_tower', 'storage_tank',
        'silo', 'beach_hut', 'bunker', 'construction', 'container', 'guardhouse',
        'outbuilding', 'pagoda', 'quonset_hut', 'roof', 'ship', 'tent', 'tower',
        'triumphal_arch', 'windmill'
    ]

    # Simple filter - just check building column
    if 'building' not in buildings.columns:
        raise ValueError("No 'building' column found in parquet file")

    # Vectorized filtering
    buildings_filtered = buildings[buildings['building'].isin(building_values)].copy()
    print(f"Filtered to {len(buildings_filtered):,} buildings")

    if len(buildings_filtered) == 0:
        print("\nWarning: No buildings found matching filter criteria!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'building_type': ['All Buildings'],
            'count': [0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # Add building_type column (same as building value)
    buildings_filtered['building_type'] = buildings_filtered['building']

    # ===== SPATIAL JOIN WITH BASINS =====
    print("\nIntersecting buildings with basins...")

    # Ensure same CRS
    if buildings_filtered.crs != basins.crs:
        buildings_filtered = buildings_filtered.to_crs(basins.crs)

    # Vectorized spatial join
    buildings_in_basins = gpd.sjoin(
        buildings_filtered,
        basins,
        how='inner',
        predicate='intersects'
    )

    print(f"Buildings within basins: {len(buildings_in_basins):,}")

    if len(buildings_in_basins) == 0:
        print("\nWarning: No buildings found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'building_type': ['All Buildings'],
            'count': [0]
        })
        total_buildings = 0
        type_counts = pd.DataFrame()
    else:
        # ===== EXPORT GEOPACKAGE =====
        print(f"\n1. Saving buildings to GeoPackage (as centroids)...")

        # Reproject to WGS84
        buildings_for_export = buildings_in_basins.to_crs("EPSG:4326")

        # Select columns
        export_columns = ['building_type', 'name', 'osm_id', 'geometry']

        # Handle missing columns gracefully
        available_cols = [col for col in export_columns if col in buildings_for_export.columns]
        if 'geometry' not in available_cols:
            available_cols.append('geometry')

        buildings_export = buildings_for_export[available_cols].copy()

        # Convert to centroids for point representation
        try:
            # Use projected CRS for accurate centroids
            buildings_projected = buildings_export.to_crs("EPSG:32616")  # UTM Zone 16N for Central America
            buildings_export['geometry'] = buildings_projected.geometry.centroid
            buildings_export = buildings_export.to_crs("EPSG:4326")
        except Exception as e:
            print(f"  ⚠ Could not use projected CRS: {str(e)}")
            print("  Falling back to geographic CRS centroids")
            buildings_export['geometry'] = buildings_export.geometry.centroid

        # Save to GeoPackage
        buildings_export.to_file(output_gpkg, driver='GPKG', layer='buildings')
        print(f"     {output_gpkg.name}")
        print(f"     Features: {len(buildings_export):,}")
        print(f"     Layer: buildings")

        # ===== CALCULATE STATISTICS (VECTORIZED) =====
        total_buildings = len(buildings_in_basins)

        # Count by building type (vectorized)
        type_counts = buildings_in_basins['building_type'].value_counts().reset_index()
        type_counts.columns = ['building_type', 'count']
        type_counts = type_counts.sort_values('building_type').reset_index(drop=True)
        type_counts['category'] = 'DETAIL'

        # Create summary row
        summary_row = pd.DataFrame({
            'category': ['TOTAL'],
            'building_type': ['All Buildings'],
            'count': [total_buildings]
        })

        # Combine: summary first, then details
        result = pd.concat(
            [summary_row, type_counts[['category', 'building_type', 'count']]],
            ignore_index=True
        )

    # ===== SAVE CSV =====
    print(f"\n2. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"     {output_csv.name}")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("BUILDING STATISTICS")
    print("=" * 60)
    print(f"Total buildings: {total_buildings:,}")
    if len(type_counts) > 0:
        print(f"Unique building types: {len(type_counts)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. buildings_affected.gpkg (point centroids)")
    print("  2. building_statistics.csv")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    building_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons.parquet"

    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Building_Impact"

    results = calculate_basin_buildings_from_parquet(
        basin_file,
        building_parquet,
        output_folder
    )

    print("\n\nStatistics table:")
    print(results.head(20))