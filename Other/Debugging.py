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

    # ===== DIAGNOSTIC: INSPECT THE DATA STRUCTURE =====
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Inspecting parquet structure")
    print("=" * 60)
    print(f"\nColumns in parquet file:")
    print(trans_df.columns.tolist())
    print(f"\nFirst 5 rows:")
    print(trans_df.head())
    print(f"\nData types:")
    print(trans_df.dtypes)

    # Check for highway/railway columns
    if 'highway' in trans_df.columns:
        print(f"\nUnique highway values (first 20):")
        print(trans_df['highway'].value_counts().head(20))
    else:
        print(f"\n⚠ No 'highway' column found")

    if 'railway' in trans_df.columns:
        print(f"\nUnique railway values:")
        print(trans_df['railway'].value_counts())
    else:
        print(f"\n⚠ No 'railway' column found")

    # Check for common OSM tag column patterns
    tag_columns = [col for col in trans_df.columns if col.startswith('tag') or ':' in col]
    if tag_columns:
        print(f"\nFound potential tag columns: {tag_columns}")

    print("=" * 60)

    # Convert WKB geometry to shapely if needed
    print("\nConverting geometries...")
    if 'geometry' in trans_df.columns:
        first_geom = trans_df['geometry'].iloc[0]
        if isinstance(first_geom, (str, bytes)):
            trans_df['geometry'] = trans_df['geometry'].apply(
                lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x)
            )

    transportation = gpd.GeoDataFrame(trans_df, geometry='geometry', crs="EPSG:4326")
    print(f"  Loaded {len(transportation):,} transportation features")

    # ===== FLEXIBLE FILTERING =====
    print("\nFiltering for roads and railways...")

    # Highway values we want
    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]

    # Railway values we want
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    # Try different column name patterns
    filter_criteria = pd.Series([False] * len(transportation), index=transportation.index)

    # Pattern 1: Direct columns
    if 'highway' in transportation.columns:
        filter_criteria |= transportation['highway'].isin(highway_values)
        print(f"  Found {filter_criteria.sum():,} features using 'highway' column")

    if 'railway' in transportation.columns:
        filter_criteria |= transportation['railway'].isin(railway_values)
        print(f"  Found {filter_criteria.sum():,} features using 'railway' column")

    # Pattern 2: Try 'tags' column (common in OSM parquet exports)
    if 'tags' in transportation.columns:
        print("  Checking 'tags' column...")

        # Tags might be stored as dict or JSON string
        def check_tags(tags):
            if pd.isna(tags):
                return False
            if isinstance(tags, dict):
                return (tags.get('highway') in highway_values or
                        tags.get('railway') in railway_values)
            return False

        tag_matches = transportation['tags'].apply(check_tags)
        filter_criteria |= tag_matches
        print(f"  Found {tag_matches.sum():,} additional features from 'tags' column")

    # Pattern 3: Try individual tag columns like 'tag.highway', 'tag.railway'
    for prefix in ['tag.', 'tags.', '@']:
        highway_col = f'{prefix}highway'
        railway_col = f'{prefix}railway'

        if highway_col in transportation.columns:
            filter_criteria |= transportation[highway_col].isin(highway_values)
            print(f"  Found features using '{highway_col}' column")

        if railway_col in transportation.columns:
            filter_criteria |= transportation[railway_col].isin(railway_values)
            print(f"  Found features using '{railway_col}' column")

    transportation_filtered = transportation[filter_criteria].copy()
    print(f"\nTotal filtered: {len(transportation_filtered):,} transportation features")

    if len(transportation_filtered) == 0:
        print("\n⚠ WARNING: No transportation features found after filtering!")
        print("\nPlease check the diagnostic output above to see:")
        print("  1. What columns exist in your parquet file")
        print("  2. What values are in the highway/railway columns")
        print("  3. Whether the data structure matches what the code expects")

        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # ===== REST OF THE CODE CONTINUES AS BEFORE =====
    # ... (continue with LineString reconstruction, etc.)

    print("\n✓ Filtering successful! Continuing with LineString reconstruction...")

    # Return for now to see diagnostic output
    return None


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