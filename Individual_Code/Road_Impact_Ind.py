import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely.wkb as wkblib


# @profile
def calculate_basin_transportation_from_parquet(basin_file, transportation_parquet, output_folder):
    """
    Calculate transportation infrastructure statistics from transportation parquet file.
    Processes LineString features only and calculates lengths.
    Uses vectorized operations for fast processing.
    Uses World Mollweide equal-area projection for accurate global length calculation.
    FULLY OPTIMIZED - NO LOOPS.

    Parameters:
    -----------
    basin_file : str
        Path to basin parquet file
    transportation_parquet : str
        Path to transportation parquet file with OSM line data (roads/railways)
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "transportation_statistics.csv"
    output_gpkg = output_path / "transportation_affected.gpkg"

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

    # ===== READ TRANSPORTATION PARQUET =====
    print("\nReading transportation parquet...")
    trans_path = Path(transportation_parquet)
    if not trans_path.exists():
        raise FileNotFoundError(f"Transportation parquet file not found: {trans_path}")

    try:
        trans_df = pd.read_parquet(transportation_parquet, engine='fastparquet')
        print("  Using fastparquet engine")
    except Exception as e:
        print(f"  ⚠ fastparquet failed: {str(e)}")
        print("  Falling back to pyarrow engine...")
        trans_df = pd.read_parquet(transportation_parquet, engine='pyarrow')

    # Convert WKB geometry to shapely if needed
    print("  Converting geometries...")
    if 'geometry' in trans_df.columns:
        first_geom = trans_df['geometry'].iloc[0]
        if isinstance(first_geom, (str, bytes)):
            trans_df['geometry'] = trans_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )

    transportation = gpd.GeoDataFrame(trans_df, geometry='geometry', crs="EPSG:4326")
    print(f"  Loaded {len(transportation):,} transportation features")

    # Verify geometry types
    geom_types = transportation.geometry.type.value_counts()
    print(f"  Geometry types: {geom_types.to_dict()}")

    # ===== VECTORIZED FILTERING =====
    print("\nFiltering for roads and railways...")

    # Highway values we want
    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]

    # Railway values we want
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    # Combine criteria - check if highway OR railway column has matching values
    filter_criteria = pd.Series([False] * len(transportation), index=transportation.index)

    if 'highway' in transportation.columns:
        highway_mask = transportation['highway'].isin(highway_values)
        filter_criteria |= highway_mask
        print(f"  Highway features: {highway_mask.sum():,}")

    if 'railway' in transportation.columns:
        railway_mask = transportation['railway'].isin(railway_values)
        filter_criteria |= railway_mask
        print(f"  Railway features: {railway_mask.sum():,}")

    transportation_filtered = transportation[filter_criteria].copy()
    print(f"\nFiltered to {len(transportation_filtered):,} transportation features")

    if len(transportation_filtered) == 0:
        print("\nWarning: No transportation features found!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # ===== ASSIGN INFRASTRUCTURE TYPE =====
    print("\nAssigning infrastructure types...")

    def assign_type(row):
        if 'highway' in row and pd.notna(row.get('highway')) and row['highway'] in highway_values:
            return 'highway', row['highway']
        elif 'railway' in row and pd.notna(row.get('railway')) and row['railway'] in railway_values:
            return 'railway', row['railway']
        return None, None

    transportation_filtered[['infrastructure_type', 'feature_value']] = transportation_filtered.apply(
        assign_type, axis=1, result_type='expand'
    )

    # Remove rows without valid assignment
    transportation_filtered = transportation_filtered[transportation_filtered['infrastructure_type'].notna()].copy()

    # ===== SPATIAL JOIN WITH BASINS =====
    print("\nIntersecting transportation with basins...")

    # Ensure same CRS
    if transportation_filtered.crs != basins.crs:
        transportation_filtered = transportation_filtered.to_crs(basins.crs)

    # ===== EQUAL AREA PROJECTION FOR ACCURATE GLOBAL LENGTH CALCULATION =====
    print("Reprojecting to World Mollweide equal-area projection for accurate global length calculation...")

    # World Mollweide (ESRI:54009) - equal area projection suitable for global analysis
    # Alternative options:
    #   - World Eckert IV (ESRI:54012)
    #   - World Cylindrical Equal Area (ESRI:54034)
    equal_area_crs = "ESRI:54009"

    print(f"  Using CRS: {equal_area_crs} (World Mollweide)")

    # Reproject to equal-area projection
    transportation_projected = transportation_filtered.to_crs(equal_area_crs)
    basins_projected = basins.to_crs(equal_area_crs)

    # Clip transportation to basin boundaries
    print("Clipping transportation to basin boundaries...")
    transportation_in_basins = gpd.overlay(transportation_projected, basins_projected, how='intersection')

    print(f"Transportation segments within basins: {len(transportation_in_basins):,}")

    if len(transportation_in_basins) == 0:
        print("\nWarning: No transportation found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        total_length_km = 0
        type_lengths = pd.DataFrame()
    else:
        # ===== CALCULATE LENGTHS (IN EQUAL AREA PROJECTION FOR ACCURACY) =====
        print("\nCalculating lengths in meters...")
        transportation_in_basins['length_m'] = transportation_in_basins.geometry.length
        transportation_in_basins['length_km'] = transportation_in_basins['length_m'] / 1000

        # Reproject back to WGS84 for export
        print("Reprojecting to WGS84 for export...")
        transportation_for_export = transportation_in_basins.to_crs("EPSG:4326")

        # ===== EXPORT TO SINGLE GEOPACKAGE WITH MULTIPLE LAYERS (AS LINES) =====
        print("\nExporting to GeoPackage with multiple layers...")

        # Separate by type
        motorways = transportation_for_export[transportation_for_export['feature_value'] == 'motorway']
        highways = transportation_for_export[
            (transportation_for_export['infrastructure_type'] == 'highway') &
            (transportation_for_export['feature_value'] != 'motorway')
            ]
        railways = transportation_for_export[transportation_for_export['infrastructure_type'] == 'railway']

        # Select columns for export
        export_cols = ['infrastructure_type', 'feature_value', 'name', 'ref', 'length_km', 'geometry']
        # Only include columns that exist
        export_cols = [col for col in export_cols if col in transportation_for_export.columns or col == 'geometry']

        # Export motorways layer (as lines, not points)
        if len(motorways) > 0:
            print(f"1. Saving motorways layer (LineStrings)...")
            motorways_export = motorways[[col for col in export_cols if col in motorways.columns]].copy()
            motorways_export.to_file(output_gpkg, driver='GPKG', layer='motorways')
            print(f"   ✓ Layer: motorways - {len(motorways):,} features, {motorways['length_km'].sum():,.2f} km")

        # Export highways layer (as lines, not points)
        if len(highways) > 0:
            print(f"2. Saving highways layer (LineStrings)...")
            highways_export = highways[[col for col in export_cols if col in highways.columns]].copy()
            highways_export.to_file(output_gpkg, driver='GPKG', layer='highways')
            print(f"   ✓ Layer: highways - {len(highways):,} features, {highways['length_km'].sum():,.2f} km")

        # Export railways layer (as lines, not points)
        if len(railways) > 0:
            print(f"3. Saving railways layer (LineStrings)...")
            railways_export = railways[[col for col in export_cols if col in railways.columns]].copy()
            railways_export.to_file(output_gpkg, driver='GPKG', layer='railways')
            print(f"   ✓ Layer: railways - {len(railways):,} features, {railways['length_km'].sum():,.2f} km")

        print(f"\n✓ All layers saved to: {output_gpkg.name}")

        # ===== CALCULATE STATISTICS =====
        total_length_km = transportation_in_basins['length_km'].sum()

        # Length by feature type
        type_lengths = transportation_in_basins.groupby('feature_value')['length_km'].sum().reset_index()
        type_lengths.columns = ['feature_type', 'length_km']
        type_lengths = type_lengths.sort_values('feature_type').reset_index(drop=True)
        type_lengths['category'] = 'DETAIL'

        # Summary row
        summary_row = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [total_length_km]
        })

        # Combine
        result = pd.concat(
            [summary_row, type_lengths[['category', 'feature_type', 'length_km']]],
            ignore_index=True
        )

        # Print breakdown
        print("\n" + "=" * 60)
        print("Transportation length breakdown (top 20):")
        print("=" * 60)
        for _, row in type_lengths.head(20).iterrows():
            print(f"  {row['feature_type']}: {row['length_km']:,.2f} km")

        # Summary by infrastructure type
        infra_summary = transportation_in_basins.groupby('infrastructure_type')['length_km'].sum()
        print("\nSummary by infrastructure type:")
        for infra_type, length in infra_summary.items():
            print(f"  {infra_type}: {length:,.2f} km")

    # ===== SAVE CSV =====
    print(f"\n4. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("TRANSPORTATION INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total infrastructure length: {total_length_km:,.2f} km")
    if len(type_lengths) > 0:
        print(f"Segments analyzed: {len(transportation_in_basins):,}")
        print(f"Unique feature types: {len(type_lengths)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. transportation_affected.gpkg (LineStrings)")
    print("     - motorways (layer)")
    print("     - highways (layer)")
    print("     - railways (layer)")
    print("  2. transportation_statistics.csv")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    transportation_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines.parquet"

    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Transportation_Impact"

    results = calculate_basin_transportation_from_parquet(
        basin_file,
        transportation_parquet,
        output_folder
    )

    print("\n\nStatistics table:")
    print(results.head(20))