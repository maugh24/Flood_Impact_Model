import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely.wkb as wkb
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
import warnings

warnings.filterwarnings('ignore')


def convert_geopackage_to_parquet(gpkg_file, output_parquet):
    """
    Convert GeoPackage file with multiple layers into a single Parquet file.
    Processes only MultiPolygon geometries, ignoring all other geometry types.

    Parameters:
    -----------
    gpkg_file : str
        Path to input GeoPackage file
    output_parquet : str
        Path to output Parquet file
    """

    gpkg_path = Path(gpkg_file)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage file not found: {gpkg_path}")

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing GeoPackage: {gpkg_path.name}")
    print("=" * 70)

    # Get all layers in the GeoPackage
    import fiona
    available_layers = fiona.listlayers(str(gpkg_path))
    print(f"\nFound {len(available_layers)} layers in GeoPackage:")
    for i, layer in enumerate(available_layers, 1):
        print(f"  {i}. {layer}")

    print(f"\nExtracting MultiPolygon features from all layers into single Parquet file...")
    print("=" * 70)

    # Collect all layers
    all_dataframes = []
    total_features = 0

    # Process each layer
    for layer_idx, layer_name in enumerate(available_layers, 1):
        try:
            # Read the layer
            gdf = gpd.read_file(gpkg_path, layer=layer_name)

            if len(gdf) == 0:
                continue

            # Filter for MultiPolygon geometries only
            multipolygon_mask = gdf.geometry.type == 'MultiPolygon'
            multipolygons = gdf[multipolygon_mask].copy()

            if len(multipolygons) == 0:
                continue

            # Only print for layers with MultiPolygon features
            print(f"\n[{layer_idx}/{len(available_layers)}] Processing layer: {layer_name}")
            print(f"  ✓ Loaded {len(gdf):,} features, filtered to {len(multipolygons):,} MultiPolygon features")

            # Add layer name as a column
            multipolygons['layer_name'] = layer_name

            # Store original geometry type
            multipolygons['geometry_type'] = multipolygons.geometry.type

            # Store CRS
            if multipolygons.crs is not None:
                multipolygons['crs'] = str(multipolygons.crs)

            all_dataframes.append(multipolygons)
            total_features += len(multipolygons)

        except Exception as e:
            print(f"  ✗ Error reading layer '{layer_name}': {str(e)}")
            continue

    if not all_dataframes:
        raise ValueError("No MultiPolygon features found in any layer!")

    # Combine all layers
    print(f"\n" + "=" * 70)
    print(f"Combining {len(all_dataframes)} layer(s)...")
    combined_gdf = pd.concat(all_dataframes, ignore_index=True)
    print(f"✓ Combined {total_features:,} total features")

    # Show combined statistics
    print(f"\nCombined layer statistics:")
    print(f"  Layers: {combined_gdf['layer_name'].value_counts().to_dict()}")
    print(f"  Geometry types: {combined_gdf['geometry_type'].value_counts().to_dict()}")

    # Convert geometries to WKB format for Parquet
    print(f"\nConverting geometries to WKB format...")
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(
        lambda geom: wkb.dumps(geom, hex=True) if geom is not None else None
    )

    # Convert to regular DataFrame
    df_export = pd.DataFrame(combined_gdf)

    # ===== FIX DATA TYPES FOR PARQUET COMPATIBILITY =====
    print(f"Fixing data types for Parquet compatibility...")

    # Convert problematic columns to compatible types
    for col in df_export.columns:
        # Skip geometry column (already converted to WKB string)
        if col == 'geometry':
            continue

        # Get column dtype
        dtype = df_export[col].dtype

        # Convert object/string columns with issues
        if dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            # Convert to string explicitly (handles Arrow/PyArrow string types)
            df_export[col] = df_export[col].astype(str)
            # Replace 'None' string with actual None
            df_export[col] = df_export[col].replace('None', None)

        # Convert int64 IDs to string (osm_id often too large for some systems)
        elif col in ['osm_id', '@id', 'id'] and pd.api.types.is_integer_dtype(dtype):
            print(f"  Converting {col} from {dtype} to string")
            df_export[col] = df_export[col].astype(str)

        # Handle any remaining Arrow types
        elif hasattr(dtype, 'pyarrow_dtype'):
            print(f"  Converting Arrow type column: {col}")
            df_export[col] = df_export[col].astype(str)

    # Save to Parquet using PyArrow (more robust than fastparquet for this case)
    print(f"Writing to Parquet using PyArrow engine...")
    try:
        df_export.to_parquet(
            output_path,
            engine='pyarrow',  # PyArrow is more robust for mixed types
            compression='snappy',
            index=False
        )
    except Exception as e:
        print(f"  ⚠ PyArrow failed: {e}")
        print(f"  Trying with fastparquet...")

        # Fallback: Try fastparquet with additional type conversions
        # Ensure all string columns are truly string type
        for col in df_export.select_dtypes(include=['object']).columns:
            df_export[col] = df_export[col].fillna('').astype(str)

        df_export.to_parquet(
            output_path,
            engine='fastparquet',
            compression='snappy',
            index=False
        )

    # Verify file
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"✓ Output file: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Total features: {len(df_export):,}")
    print(f"  Layers combined: {len(all_dataframes)}")
    print("=" * 70)

    print("\nTo read this file back:")
    print("  import pandas as pd")
    print("  import geopandas as gpd")
    print("  import shapely.wkb as wkb")
    print(f"  df = pd.read_parquet('{output_path.name}')")
    print("  df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x, hex=True))")
    print("  gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=df['crs'].iloc[0])")


def verify_parquet(parquet_file, sample_size=10):
    """
    Verify a converted parquet file by reading it back and showing sample data.

    Parameters:
    -----------
    parquet_file : str
        Path to parquet file to verify
    sample_size : int
        Number of sample rows to display
    """

    print(f"\nVerifying Parquet file: {Path(parquet_file).name}")
    print("=" * 70)

    # Read parquet
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df):,} features")

    # Show columns
    print(f"\nColumns ({len(df.columns)}): {', '.join(df.columns.tolist())}")

    # Show layer distribution
    if 'layer_name' in df.columns:
        print(f"\nLayer distribution:")
        for layer, count in df['layer_name'].value_counts().items():
            print(f"  - {layer}: {count:,}")

    # Show geometry types
    if 'geometry_type' in df.columns:
        print(f"\nGeometry type distribution:")
        for geom_type, count in df['geometry_type'].value_counts().items():
            print(f"  - {geom_type}: {count:,}")

    # Show CRS
    if 'crs' in df.columns and len(df) > 0:
        unique_crs = df['crs'].unique()
        print(f"\nCRS: {unique_crs[0]}")
        if len(unique_crs) > 1:
            print(f"  ⚠ Warning: Multiple CRS found: {unique_crs}")

    # Convert geometries back
    print(f"\nConverting WKB back to geometries...")
    df['geometry'] = df['geometry'].apply(
        lambda x: wkb.loads(x, hex=True) if pd.notna(x) and x != '' else None
    )

    # Create GeoDataFrame
    if 'crs' in df.columns and len(df) > 0:
        gdf = gpd.GeoDataFrame(df.drop(columns=['crs']), geometry='geometry', crs=df['crs'].iloc[0])
    else:
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

    print(f"✓ Successfully converted to GeoDataFrame")

    # Show sample from each layer
    print(f"\nSample data:")
    if 'layer_name' in gdf.columns:
        for layer in gdf['layer_name'].unique()[:3]:  # Show first 3 layers
            print(f"\n  Layer: {layer}")
            layer_data = gdf[gdf['layer_name'] == layer].head(min(sample_size, 3))
            print(f"  {layer_data[['geometry_type', 'geometry']].to_string(index=False)}")
    else:
        print(gdf.head(sample_size))

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE - File is valid!")
    print("=" * 70)

    return gdf


# Usage
if __name__ == "__main__":
    # Convert GeoPackage to single Parquet file
    gpkg_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_GPKG\central-america.gpkg"
    output_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-polygons.parquet"

    convert_geopackage_to_parquet(gpkg_file, output_parquet)

    # Verify the conversion (optional)
    # verify_parquet(output_parquet)