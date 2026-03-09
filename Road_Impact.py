import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely.wkb as wkblib
from shapely.geometry import LineString, Point
import numpy as np


def calculate_basin_transportation_from_parquet(basin_file, transportation_parquet, output_folder):
    """
    Calculate transportation infrastructure statistics from transportation parquet file.
    Reconstructs LineStrings from points and calculates lengths.
    Uses vectorized operations for fast processing.

    Parameters:
    -----------
    basin_file : str
        Path to basin parquet file
    transportation_parquet : str
        Path to transportation parquet file with OSM road/rail data (as points)
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "transportation_statistics.csv"
    output_gpkg_motorway = output_path / "transportation_motorway.gpkg"
    output_gpkg_highway = output_path / "transportation_highways.gpkg"
    output_gpkg_railway = output_path / "transportation_railway.gpkg"

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

    # Filter
    filter_criteria = pd.Series([False] * len(transportation), index=transportation.index)

    if 'highway' in transportation.columns:
        filter_criteria |= transportation['highway'].isin(highway_values)

    if 'railway' in transportation.columns:
        filter_criteria |= transportation['railway'].isin(railway_values)

    transportation_filtered = transportation[filter_criteria].copy()
    print(f"Filtered to {len(transportation_filtered):,} transportation features")

    if len(transportation_filtered) == 0:
        print("\nWarning: No transportation features found!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # ===== RECONSTRUCT LINESTRINGS FROM POINTS =====
    print("\nReconstructing LineStrings from point geometries...")

    # Group by way ID (osm_id or @id) to reconstruct line geometries
    if '@id' in transportation_filtered.columns:
        way_id_col = '@id'
    elif 'osm_id' in transportation_filtered.columns:
        way_id_col = 'osm_id'
    else:
        raise ValueError("No way ID column found (@id or osm_id)")

    # Group by way ID and reconstruct geometries
    reconstructed_ways = []

    print(f"  Processing {transportation_filtered[way_id_col].nunique():,} unique ways...")

    for way_id, group in transportation_filtered.groupby(way_id_col):
        # Sort by node sequence if available
        if '@sequence' in group.columns:
            group = group.sort_values('@sequence')
        elif 'node_sequence' in group.columns:
            group = group.sort_values('node_sequence')

        # Extract coordinates from Point geometries
        coords = [(geom.x, geom.y) for geom in group.geometry]

        # Need at least 2 points to make a line
        if len(coords) >= 2:
            # Determine infrastructure type and value
            if 'highway' in group.columns and pd.notna(group['highway'].iloc[0]):
                infra_type = 'highway'
                feature_value = group['highway'].iloc[0]
            elif 'railway' in group.columns and pd.notna(group['railway'].iloc[0]):
                infra_type = 'railway'
                feature_value = group['railway'].iloc[0]
            else:
                continue

            # Create LineString
            line = LineString(coords)

            # Get metadata from first point in the way
            reconstructed_ways.append({
                'geometry': line,
                'infrastructure_type': infra_type,
                'feature_value': feature_value,
                'osm_id': way_id,
                'name': group['name'].iloc[0] if 'name' in group.columns else '',
                'ref': group['ref'].iloc[0] if 'ref' in group.columns else ''
            })

    # Convert to GeoDataFrame
    transportation_lines = gpd.GeoDataFrame(reconstructed_ways, crs="EPSG:4326")
    print(f"  Reconstructed {len(transportation_lines):,} LineStrings")

    # ===== SHOW BREAKDOWN =====
    print("\nBreakdown by infrastructure type:")
    print(transportation_lines['infrastructure_type'].value_counts())

    print("\nTop 20 feature types found:")
    print(transportation_lines['feature_value'].value_counts().head(20))

    # ===== SPATIAL JOIN WITH BASINS =====
    print("\nIntersecting transportation with basins...")

    # Ensure same CRS
    if transportation_lines.crs != basins.crs:
        transportation_lines = transportation_lines.to_crs(basins.crs)

    # Reproject to UTM for accurate length calculation
    print("\nReprojecting to UTM for accurate length calculation...")
    basins_wgs84 = basins.to_crs("EPSG:4326") if basins.crs.to_epsg() != 4326 else basins
    utm_crs = basins_wgs84.estimate_utm_crs()

    transportation_utm = transportation_lines.to_crs(utm_crs)
    basins_utm = basins.to_crs(utm_crs)

    # Clip transportation to basin boundaries
    print("Clipping transportation to basin boundaries...")
    transportation_in_basins = gpd.overlay(transportation_utm, basins_utm, how='intersection')

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
        # ===== CALCULATE LENGTHS =====
        print("\nCalculating lengths...")
        transportation_in_basins['length_m'] = transportation_in_basins.geometry.length
        transportation_in_basins['length_km'] = transportation_in_basins['length_m'] / 1000

        # Reproject back to WGS84 for export
        transportation_for_export = transportation_in_basins.to_crs("EPSG:4326")

        # ===== EXPORT GEOPACKAGES =====
        print("\nExporting GeoPackages...")

        # Separate by type
        motorways = transportation_for_export[transportation_for_export['feature_value'] == 'motorway']
        highways = transportation_for_export[
            (transportation_for_export['infrastructure_type'] == 'highway') &
            (transportation_for_export['feature_value'] != 'motorway')
        ]
        railways = transportation_for_export[transportation_for_export['infrastructure_type'] == 'railway']

        # Export motorways
        if len(motorways) > 0:
            print(f"1. Saving motorways...")
            motorways_export = motorways[['infrastructure_type', 'feature_value', 'name', 'ref', 'length_km', 'geometry']].copy()
            motorways_export.to_file(output_gpkg_motorway, driver='GPKG', layer='motorways')
            print(f"   ✓ {output_gpkg_motorway.name} - {len(motorways):,} features, {motorways['length_km'].sum():,.2f} km")

        # Export highways
        if len(highways) > 0:
            print(f"2. Saving highways...")
            highways_export = highways[['infrastructure_type', 'feature_value', 'name', 'ref', 'length_km', 'geometry']].copy()
            highways_export.to_file(output_gpkg_highway, driver='GPKG', layer='highways')
            print(f"   ✓ {output_gpkg_highway.name} - {len(highways):,} features, {highways['length_km'].sum():,.2f} km")

        # Export railways
        if len(railways) > 0:
            print(f"3. Saving railways...")
            railways_export = railways[['infrastructure_type', 'feature_value', 'name', 'ref', 'length_km', 'geometry']].copy()
            railways_export.to_file(output_gpkg_railway, driver='GPKG', layer='railways')
            print(f"   ✓ {output_gpkg_railway.name} - {len(railways):,} features, {railways['length_km'].sum():,.2f} km")

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

        # Summary by type
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
    print("  1. transportation_motorway.gpkg (motorways)")
    print("  2. transportation_highways.gpkg (other highways)")
    print("  3. transportation_railway.gpkg (railways)")
    print("  4. transportation_statistics.csv")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    transportation_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america.parquet"

    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Transportation_Impact"

    results = calculate_basin_transportation_from_parquet(
        basin_file,
        transportation_parquet,
        output_folder
    )

    print("\n\nStatistics table:")
    print(results.head(20))