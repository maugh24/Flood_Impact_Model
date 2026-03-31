import geopandas as gpd
import pandas as pd
from pathlib import Path


# @profile
def calculate_basin_transportation(basin_file,rivers, transportation_parquet, output_folder,index):

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output files
    output_csv = output_path / f"transportation_statistics{index}.csv"
    output_gpkg = output_path / f"transportation_affected{index}.gpkg"

    # ===== READ BASINS =====
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)])

    # ===== READ TRANSPORTATION PARQUET =====
    transportation = gpd.read_parquet(transportation_parquet)

    # ===== VECTORIZED FILTERING =====
    # Highway values we want
    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]

    # Railway values we want+
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    # Combine criteria - check if highway OR railway column has matching values
    filter_criteria = pd.Series([False] * len(transportation), index=transportation.index)

    if 'highway' in transportation.columns:
        highway_mask = transportation['highway'].isin(highway_values)
        filter_criteria |= highway_mask

    if 'railway' in transportation.columns:
        railway_mask = transportation['railway'].isin(railway_values)
        filter_criteria |= railway_mask

    transportation_filtered = transportation[filter_criteria].copy()

    if len(transportation_filtered) == 0:
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # ===== ASSIGN INFRASTRUCTURE TYPE =====
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

    # Ensure same CRS
    if transportation_filtered.crs != basins.crs:
        transportation_filtered = transportation_filtered.to_crs(basins.crs)

    # ===== EQUAL AREA PROJECTION FOR ACCURATE GLOBAL LENGTH CALCULATION =====
    cea_crs = "ESRI:54034"

    # Reproject to equal-area projection
    transportation_projected = transportation_filtered.to_crs(cea_crs)
    basins_projected = basins.to_crs({'proj': 'cea'})


    # Clip transportation to basin boundaries
    transportation_in_basins = gpd.overlay(
        transportation_projected,
        basins_projected,
        how='intersection'
    )

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
        transportation_in_basins['length_m'] = transportation_in_basins.geometry.length
        transportation_in_basins['length_km'] = transportation_in_basins['length_m'] / 1000

        # Reproject back to WGS84 for export
        transportation_for_export = transportation_in_basins.to_crs("EPSG:4326")

        # ===== EXPORT TO SINGLE GEOPACKAGE WITH MULTIPLE LAYERS (AS LINES) =====
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
            motorways_export = motorways[[col for col in export_cols if col in motorways.columns]].copy()
            motorways_export.to_file(output_gpkg, driver='GPKG', layer='motorways')

        # Export highways layer (as lines, not points)
        if len(highways) > 0:
            highways_export = highways[[col for col in export_cols if col in highways.columns]].copy()
            highways_export.to_file(output_gpkg, driver='GPKG', layer='highways')

        # Export railways layer (as lines, not points)
        if len(railways) > 0:
            railways_export = railways[[col for col in export_cols if col in railways.columns]].copy()
            railways_export.to_file(output_gpkg, driver='GPKG', layer='railways')

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

    # ===== SAVE CSV =====
    result.to_csv(output_csv, index=False)
    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    transportation_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines.parquet"

    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Transportation_Impact"

    results = calculate_basin_transportation(
        basin_file,
        transportation_parquet,
        output_folder
    )