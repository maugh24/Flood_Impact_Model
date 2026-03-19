import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely.wkb as wkblib

#@profile
def calculate_basin_infrastructure_from_parquet(basin_file, infrastructure_parquet, output_folder):
    """
    Calculate critical infrastructure statistics from infrastructure parquet file.
    Uses vectorized operations for fast processing.

    Parameters:
    -----------
    basin_file : str
        Path to basin parquet file
    infrastructure_parquet : str
        Path to infrastructure parquet file with OSM data
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "infrastructure_statistics.csv"
    output_gpkg = output_path / "infrastructure_affected.gpkg"

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

        # Check what type it is
        if isinstance(first_geom, (str, bytes)):
            # String or bytes - need WKB conversion
            basins_df['geometry'] = basins_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )
        # else it's already a geometry object, no conversion needed

    basins = gpd.GeoDataFrame(basins_df, geometry='geometry', crs="EPSG:4326")
    print(f"Loaded {len(basins)} basins")

    # ===== READ INFRASTRUCTURE PARQUET =====
    print("\nReading infrastructure parquet...")
    infra_path = Path(infrastructure_parquet)
    if not infra_path.exists():
        raise FileNotFoundError(f"Infrastructure parquet file not found: {infra_path}")

    try:
        infra_df = pd.read_parquet(infrastructure_parquet, engine='fastparquet')
        print("  Using fastparquet engine")
    except Exception as e:
        print(f"  ⚠ fastparquet failed: {str(e)}")
        print("  Falling back to pyarrow engine...")
        infra_df = pd.read_parquet(infrastructure_parquet, engine='pyarrow')

    # Convert WKB geometry to shapely if needed
    print("  Converting geometries...")
    if 'geometry' in infra_df.columns:
        first_geom = infra_df['geometry'].iloc[0]

        # Check what type it is
        if isinstance(first_geom, (str, bytes)):
            # String or bytes - need WKB conversion
            infra_df['geometry'] = infra_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )
        # else it's already a geometry object, no conversion needed

    infrastructure = gpd.GeoDataFrame(infra_df, geometry='geometry', crs="EPSG:4326")
    print(f"  Loaded {len(infrastructure):,} infrastructure features")

    # ===== VECTORIZED FILTERING =====
    print("\nFiltering for critical infrastructure types...")

    # Start with a DataFrame of False values
    filter_criteria = pd.Series([False] * len(infrastructure), index=infrastructure.index)

    # Add each criterion if the column exists
    if 'aeroway' in infrastructure.columns:
        filter_criteria |= (infrastructure['aeroway'] == 'runway')

    if 'amenity' in infrastructure.columns:
        filter_criteria |= infrastructure['amenity'].isin([
            'college', 'university', 'doctors', 'hospital',
            'fire_station', 'police', 'townhall'
        ])

    if 'building' in infrastructure.columns:
        filter_criteria |= infrastructure['building'].isin([
            'college', 'fire_station', 'government', 'hospital',
            'school', 'university', 'military'
        ])

    if 'man_made' in infrastructure.columns:
        filter_criteria |= infrastructure['man_made'].isin(['water_works', 'water_well'])

    if 'military' in infrastructure.columns:
        filter_criteria |= infrastructure['military'].isin(['airfield', 'base'])

    if 'power' in infrastructure.columns:
        filter_criteria |= (infrastructure['power'] == 'plant')

    if 'waterway' in infrastructure.columns:
        filter_criteria |= (infrastructure['waterway'] == 'dam')

    # Apply filter
    infrastructure_filtered = infrastructure[filter_criteria].copy()
    print(f"Filtered to {len(infrastructure_filtered):,} critical infrastructure features")

    if len(infrastructure_filtered) == 0:
        print("\nWarning: No critical infrastructure found!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'infrastructure_type': ['All Infrastructure'],
            'count': [0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # ===== VECTORIZED TAG TYPE ASSIGNMENT =====
    print("\nAssigning tag types...")

    def assign_tag_type(row):
        """Vectorized tag type assignment - handles missing columns."""
        # Check aeroway
        if 'aeroway' in row and pd.notna(row['aeroway']) and row['aeroway'] == 'runway':
            return 'aeroway', row['aeroway']
        # Check amenity
        elif 'amenity' in row and pd.notna(row['amenity']) and row['amenity'] in [
            'college', 'university', 'doctors', 'hospital', 'fire_station', 'police', 'townhall'
        ]:
            return 'amenity', row['amenity']
        # Check building
        elif 'building' in row and pd.notna(row['building']) and row['building'] in [
            'college', 'fire_station', 'government', 'hospital', 'school', 'university', 'military'
        ]:
            return 'building', row['building']
        # Check man_made
        elif 'man_made' in row and pd.notna(row['man_made']) and row['man_made'] in ['water_works', 'water_well']:
            return 'man_made', row['man_made']
        # Check military
        elif 'military' in row and pd.notna(row['military']) and row['military'] in ['airfield', 'base']:
            return 'military', row['military']
        # Check power
        elif 'power' in row and pd.notna(row['power']) and row['power'] == 'plant':
            return 'power', row['power']
        # Check waterway
        elif 'waterway' in row and pd.notna(row['waterway']) and row['waterway'] == 'dam':
            return 'waterway', row['waterway']
        else:
            return None, None

    # Apply vectorized assignment
    infrastructure_filtered[['tag_type', 'infrastructure_type']] = infrastructure_filtered.apply(
        assign_tag_type, axis=1, result_type='expand'
    )

    # Remove any rows where assignment failed
    infrastructure_filtered = infrastructure_filtered[infrastructure_filtered['tag_type'].notna()].copy()

    # ===== SHOW BREAKDOWN =====
    print("\nBreakdown by tag type:")
    print(infrastructure_filtered['tag_type'].value_counts())

    print("\nInfrastructure types found:")
    print(infrastructure_filtered['infrastructure_type'].value_counts())

    # ===== SPATIAL JOIN WITH BASINS =====
    print("\nIntersecting infrastructure with basins...")

    # Ensure same CRS
    if infrastructure_filtered.crs != basins.crs:
        infrastructure_filtered = infrastructure_filtered.to_crs(basins.crs)

    # Vectorized spatial join
    infrastructure_in_basins = gpd.sjoin(
        infrastructure_filtered,
        basins,
        how='inner',
        predicate='intersects'
    )

    print(f"Infrastructure within basins: {len(infrastructure_in_basins):,}")

    if len(infrastructure_in_basins) == 0:
        print("\nWarning: No infrastructure found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'infrastructure_type': ['All Infrastructure'],
            'count': [0]
        })
        total_infrastructure = 0
        tag_type_counts = {}
        type_counts = pd.DataFrame()
    else:
        # ===== EXPORT GEOPACKAGE =====
        print(f"\n1. Saving infrastructure to GeoPackage (as point centroids)...")

        # Reproject to WGS84
        infrastructure_for_export = infrastructure_in_basins.to_crs("EPSG:4326")

        # Select columns
        export_columns = ['tag_type', 'infrastructure_type', 'name', 'osm_id', 'geometry']

        # Handle missing columns gracefully
        available_cols = [col for col in export_columns if col in infrastructure_for_export.columns]
        if 'geometry' not in available_cols:
            available_cols.append('geometry')

        infrastructure_export = infrastructure_for_export[available_cols].copy()

        # Convert to centroids - use projected CRS for accurate centroid calculation
        # First reproject to a suitable projected CRS (UTM zone for the region)
        # For Central America, UTM Zone 16N (EPSG:32616) is appropriate
        try:
            # Try to use UTM Zone 16N for Central America
            infrastructure_projected = infrastructure_export.to_crs("EPSG:32616")
            infrastructure_export['geometry'] = infrastructure_projected.geometry.centroid
            # Reproject centroids back to WGS84
            infrastructure_export = infrastructure_export.to_crs("EPSG:4326")
        except Exception as e:
            # Fallback: calculate centroids in geographic CRS (less accurate but works)
            print(f"  ⚠ Could not use projected CRS: {str(e)}")
            print("  Falling back to geographic CRS centroids (less accurate)")
            infrastructure_export['geometry'] = infrastructure_export.geometry.centroid

        # Save to GeoPackage (no field name length limits!)
        infrastructure_export.to_file(output_gpkg, driver='GPKG', layer='infrastructure')
        print(f"   ✓ {output_gpkg.name}")
        print(f"     Features: {len(infrastructure_export):,}")
        print(f"     Layer: infrastructure")

        # ===== CALCULATE STATISTICS (VECTORIZED) =====
        total_infrastructure = len(infrastructure_in_basins)

        # Count by tag type (vectorized)
        tag_type_counts = infrastructure_in_basins['tag_type'].value_counts().to_dict()

        # Count by infrastructure type (vectorized)
        type_counts = infrastructure_in_basins['infrastructure_type'].value_counts().reset_index()
        type_counts.columns = ['infrastructure_type', 'count']
        type_counts = type_counts.sort_values('infrastructure_type').reset_index(drop=True)
        type_counts['category'] = 'DETAIL'

        # Create summary rows
        summary_rows = [
            {'category': 'TOTAL', 'infrastructure_type': 'All Critical Infrastructure', 'count': total_infrastructure}
        ]

        # Add subtotals
        for tag_type in sorted(tag_type_counts.keys()):
            summary_rows.append({
                'category': 'SUBTOTAL',
                'infrastructure_type': f'All {tag_type.capitalize()}',
                'count': tag_type_counts[tag_type]
            })

        summary_df = pd.DataFrame(summary_rows)

        # Combine
        result = pd.concat(
            [summary_df, type_counts[['category', 'infrastructure_type', 'count']]],
            ignore_index=True
        )

        # Print breakdown
        print("\n" + "=" * 60)
        print("Critical infrastructure breakdown by tag type:")
        print("=" * 60)
        for tag_type, count in sorted(tag_type_counts.items()):
            print(f"  {tag_type}: {count:,}")

        print("\nInfrastructure types:")
        print("=" * 60)
        for _, row in type_counts.iterrows():
            print(f"  {row['infrastructure_type']}: {row['count']:,}")

    # ===== SAVE CSV =====
    print(f"\n2. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("CRITICAL INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total critical infrastructure: {total_infrastructure:,}")
    if len(type_counts) > 0:
        print(f"Unique infrastructure types: {len(type_counts)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. infrastructure_affected.gpkg (point centroids)")
    print("  2. infrastructure_statistics.csv")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    infrastructure_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america.parquet"

    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Critical_Infrastructure_Impact"

    results = calculate_basin_infrastructure_from_parquet(
        basin_file,
        infrastructure_parquet,
        output_folder
    )

    print("\n\nStatistics table:")
    print(results)