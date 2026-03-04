import geopandas as gpd
import pandas as pd
from pathlib import Path
import osmium
from shapely.geometry import LineString


class TransportationExtractor(osmium.SimpleHandler):
    """Extract road and railway features from PBF file using specific OSM tags."""

    def __init__(self, bbox=None):
        osmium.SimpleHandler.__init__(self)
        self.features = []
        self.processed_count = 0
        self.bbox = bbox

        # Define specific highway values we want to extract (UPDATED LIST)
        self.highway_values = {
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
            'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
            'living_street', 'busway', 'footway', 'cycleway'
        }

        # Define specific railway values we want to extract (UPDATED LIST)
        self.railway_values = {
            'light_rail', 'monorail', 'rail', 'subway', 'tram'
        }

    def _in_bbox(self, nodes):
        """Check if any node in the way is within bounding box."""
        if self.bbox is None:
            return True

        for node in nodes:
            try:
                # Check if node has valid location
                if node.location.valid():
                    if (self.bbox[0] <= node.lon <= self.bbox[2] and
                            self.bbox[1] <= node.lat <= self.bbox[3]):
                        return True
            except:
                # Skip nodes with invalid locations
                continue

        return False

    def way(self, w):
        """Extract road and railway ways (linear features)."""
        self.processed_count += 1
        if self.processed_count % 100000 == 0:
            print(f"  Processed {self.processed_count:,} ways, found {len(self.features):,} features...", end='\r')

        # Check if this is a highway or railway we want
        highway = w.tags.get('highway')
        railway = w.tags.get('railway')

        feature_type = None
        feature_value = None

        # Prioritize highway, then railway
        if highway and highway in self.highway_values:
            feature_type = 'highway'
            feature_value = highway
        elif railway and railway in self.railway_values:
            feature_type = 'railway'
            feature_value = railway

        if feature_type and feature_value:
            # Check bounding box
            if not self._in_bbox(w.nodes):
                return

            try:
                # Create LineString from nodes (only include valid locations)
                coords = []
                for node in w.nodes:
                    try:
                        if node.location.valid():
                            coords.append((node.lon, node.lat))
                    except:
                        continue

                if len(coords) >= 2:  # Need at least 2 points for a line
                    geometry = LineString(coords)

                    self.features.append({
                        'geometry': geometry,
                        'infrastructure_type': feature_type,  # 'highway' or 'railway'
                        'feature_value': feature_value,  # specific type (e.g., 'primary', 'rail')
                        'osm_id': w.id,
                        'name': w.tags.get('name', ''),
                        'ref': w.tags.get('ref', '')  # Reference number
                    })
            except Exception as e:
                pass  # Skip invalid geometries


def calculate_basin_transportation_from_pbf(basin_file, pbf_files, output_folder):
    """
    Calculate road and railway length statistics from PBF file(s).
    Saves CSV statistics and separate shapefiles for motorways, other highways, and railways.

    Parameters:
    -----------
    basin_file : str
        Path to basin shapefile or parquet file
    pbf_files : str or list
        Path to PBF file, folder of PBF files, or list of files
    output_folder : str
        Path to output folder (will be created if it doesn't exist)
    """

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    # Define output files
    output_csv = output_path / "transportation_statistics.csv"
    output_geojson = output_path / "transportation_all.geojson"
    output_shp_motorway = output_path / "transportation_motorway.shp"
    output_shp_highway = output_path / "transportation_roads.shp"
    output_shp_railway = output_path / "transportation_railway.shp"

    # Read basins
    basin_path = Path(basin_file)
    print(f"\nReading basin data from {basin_path.suffix} file...")

    if basin_path.suffix == '.parquet':
        basins = gpd.read_parquet(basin_file)
    elif basin_path.suffix in ['.shp', '.geojson', '.gpkg']:
        basins = gpd.read_file(basin_file)
    else:
        raise ValueError(f"Unsupported file format: {basin_path.suffix}")

    print(f"Loaded {len(basins)} basins")

    # Get bounding box of basins to speed up PBF processing
    if basins.crs.to_epsg() != 4326:
        basins_wgs84 = basins.to_crs("EPSG:4326")
    else:
        basins_wgs84 = basins

    bbox = basins_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    print(f"Basin bounding box: {bbox}")

    # Find PBF files
    if isinstance(pbf_files, str):
        pbf_path = Path(pbf_files)
        if pbf_path.is_dir():
            files = list(pbf_path.glob("*.osm.pbf")) + list(pbf_path.glob("*.pbf"))
        else:
            files = [pbf_path]
    else:
        files = [Path(f) for f in pbf_files]

    if not files:
        raise FileNotFoundError("No PBF files found")

    print(f"Found {len(files)} PBF file(s) to process")

    # Extract transportation features from all PBF files
    all_features = []

    for pbf_file in files:
        print(f"\nExtracting transportation infrastructure from: {pbf_file.name}")
        print(f"This may take several minutes for large files...")

        handler = TransportationExtractor(bbox=bbox)
        handler.apply_file(str(pbf_file), locations=True)

        print(f"\n  Found {len(handler.features):,} transportation segments in bounding box")

        if handler.features:
            all_features.extend(handler.features)

    if not all_features:
        print("\nWarning: No transportation features found in PBF files within basin area!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # Convert to GeoDataFrame
    print(f"\nConverting {len(all_features):,} transportation segments to GeoDataFrame...")
    features = gpd.GeoDataFrame(all_features, crs="EPSG:4326")
    print(f"Total transportation segments extracted: {len(features):,}")

    # Show what we found
    print("\nBreakdown by infrastructure type:")
    print(features['infrastructure_type'].value_counts())

    print("\nTop 20 feature types found:")
    print(features['feature_value'].value_counts().head(20))

    # Reproject to match basins
    if basins.crs != features.crs:
        print(f"\nReprojecting features to match basins CRS: {basins.crs}...")
        features = features.to_crs(basins.crs)

    # If basins are in geographic coordinates, reproject to a meter-based CRS for accurate length
    if basins.crs.is_geographic:
        print("\nBasins are in geographic coordinates. Reprojecting to UTM for accurate length calculation...")
        utm_crs = basins_wgs84.estimate_utm_crs()

        features = features.to_crs(utm_crs)
        basins_projected = basins.to_crs(utm_crs)
    else:
        basins_projected = basins

    # Spatial intersection - clip features to basins
    print("\nIntersecting transportation infrastructure with basins (clipping to basin boundaries)...")
    features_in_basins = gpd.overlay(features, basins_projected, how='intersection')

    print(f"Transportation segments within basins: {len(features_in_basins):,}")

    if len(features_in_basins) == 0:
        print("\nWarning: No transportation features found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [0.0]
        })
    else:
        # Calculate length for each segment in meters
        print("\nCalculating transportation infrastructure lengths...")
        features_in_basins['length_m'] = features_in_basins.geometry.length
        features_in_basins['length_km'] = features_in_basins['length_m'] / 1000

        # === EXPORT FILES ===
        print(f"\nPreparing transportation features for export...")

        # Reproject back to WGS84 for GeoJSON/Shapefile
        features_for_export = features_in_basins.to_crs("EPSG:4326")

        # Select columns for export
        export_columns = [
            'infrastructure_type',  # 'highway' or 'railway'
            'feature_value',  # specific type (e.g., 'primary', 'rail')
            'name',  # road/railway name
            'ref',  # reference number
            'length_km',  # calculated length
            'geometry'  # the actual geometry
        ]

        features_export = features_for_export[export_columns].copy()

        # 1. Save ALL features to GeoJSON
        print(f"\n1. Saving all transportation features to GeoJSON...")
        features_export.to_file(output_geojson, driver='GeoJSON')
        print(f"   ✓ {output_geojson.name}")
        print(f"     Features: {len(features_export):,}")

        # 2. Separate and save MOTORWAYS
        print(f"\n2. Saving motorway features to Shapefile...")
        motorways = features_export[features_export['feature_value'] == 'motorway'].copy()
        if len(motorways) > 0:
            motorways_shp = motorways.rename(columns={
                'infrastructure_type': 'infra_type',
                'feature_value': 'feat_val',
                'length_km': 'length_km'
            })
            motorways_shp.to_file(output_shp_motorway)
            print(f"   ✓ {output_shp_motorway.name}")
            print(f"     Features: {len(motorways):,}")
            print(f"     Total length: {motorways['length_km'].sum():,.2f} km")
        else:
            print(f"   ⚠ No motorway features found")

        # 3. Separate and save OTHER HIGHWAYS (excluding motorway)
        print(f"\n3. Saving highway features (non-motorway) to Shapefile...")
        highways = features_export[
            (features_export['infrastructure_type'] == 'highway') &
            (features_export['feature_value'] != 'motorway')
            ].copy()
        if len(highways) > 0:
            highways_shp = highways.rename(columns={
                'infrastructure_type': 'infra_type',
                'feature_value': 'feat_val',
                'length_km': 'length_km'
            })
            highways_shp.to_file(output_shp_highway)
            print(f"   ✓ {output_shp_highway.name}")
            print(f"     Features: {len(highways):,}")
            print(f"     Total length: {highways['length_km'].sum():,.2f} km")
        else:
            print(f"   ⚠ No highway features found")

        # 4. Separate and save RAILWAYS
        print(f"\n4. Saving railway features to Shapefile...")
        railways = features_export[features_export['infrastructure_type'] == 'railway'].copy()
        if len(railways) > 0:
            railways_shp = railways.rename(columns={
                'infrastructure_type': 'infra_type',
                'feature_value': 'feat_val',
                'length_km': 'length_km'
            })
            railways_shp.to_file(output_shp_railway)
            print(f"   ✓ {output_shp_railway.name}")
            print(f"     Features: {len(railways):,}")
            print(f"     Total length: {railways['length_km'].sum():,.2f} km")
        else:
            print(f"   ⚠ No railway features found")

        # === CALCULATE STATISTICS ===
        # Calculate total length
        total_length_km = features_in_basins['length_km'].sum()

        # Calculate length by feature type
        type_lengths = features_in_basins.groupby('feature_value')['length_km'].sum().reset_index()
        type_lengths.columns = ['feature_type', 'length_km']
        type_lengths = type_lengths.sort_values('feature_type').reset_index(drop=True)

        # Add category column for detailed rows
        type_lengths['category'] = 'DETAIL'

        # Create summary row
        summary_row = pd.DataFrame({
            'category': ['TOTAL'],
            'feature_type': ['All Transportation'],
            'length_km': [total_length_km]
        })

        # Combine: summary first, then details
        result = pd.concat([summary_row, type_lengths[['category', 'feature_type', 'length_km']]],
                           ignore_index=True)

        print("\n" + "=" * 60)
        print("Transportation infrastructure length breakdown (top 20):")
        print("=" * 60)
        for _, row in type_lengths.head(20).iterrows():
            print(f"  {row['feature_type']}: {row['length_km']:,.2f} km")

        # Summary by infrastructure type (highway vs railway)
        infra_summary = features_in_basins.groupby('infrastructure_type')['length_km'].sum()
        print("\nSummary by infrastructure type:")
        for infra_type, length in infra_summary.items():
            print(f"  {infra_type}: {length:,.2f} km")

    # Save statistics to CSV
    print(f"\n5. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("TRANSPORTATION INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total infrastructure length: {total_length_km:,.2f} km")
    print(f"Segments analyzed: {len(features_in_basins):,}")
    print(f"Unique feature types: {len(type_lengths)}")
    print("\nFiles created in: {output_path}")
    print("  1. transportation_all.geojson (all features)")
    print("  2. transportation_motorway.shp (motorways only)")
    print("  3. transportation_roads.shp (other highways)")
    print("  4. transportation_railway.shp (railways only)")
    print("  5. transportation_statistics.csv (length statistics)")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    pbf_files = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM\central-america-260204.osm.pbf"

    # Output folder (will be created if it doesn't exist)
    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Transportation_Impact"

    results = calculate_basin_transportation_from_pbf(
        basin_file,
        pbf_files,
        output_folder
    )

    print("\n\nStatistics table (first 20 rows):")
    print(results.head(20))