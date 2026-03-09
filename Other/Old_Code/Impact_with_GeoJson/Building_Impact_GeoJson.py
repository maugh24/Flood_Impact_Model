import geopandas as gpd
import pandas as pd
from pathlib import Path
import osmium
import shapely.wkb as wkblib
from shapely.geometry import Point, LineString


class CriticalInfrastructureExtractor(osmium.SimpleHandler):
    """Extract critical infrastructure from PBF file using specific OSM tags."""

    def __init__(self, bbox=None):
        osmium.SimpleHandler.__init__(self)
        self.structures = []
        self.processed_count = 0
        self.bbox = bbox

        # Define specific values we want to extract for each tag type
        self.aeroway_values = {'runway'}

        self.amenity_values = {
            'college', 'university', 'doctors', 'hospital',
            'fire_station', 'police', 'townhall'
        }

        self.building_values = {
            'college', 'fire_station', 'government', 'hospital',
            'school', 'university', 'military'
        }

        self.man_made_values = {'water_works', 'water_well'}

        self.military_values = {'airfield', 'base'}

        self.power_values = {'plant'}

        self.waterway_values = {'dam'}

    def _in_bbox(self, lon, lat):
        """Check if point is within bounding box."""
        if self.bbox is None:
            return True
        return (self.bbox[0] <= lon <= self.bbox[2] and
                self.bbox[1] <= lat <= self.bbox[3])

    def _check_feature(self, osm_tags):
        """
        Check if feature has any critical infrastructure tags we want.
        Returns (is_valid, tag_type, tag_value)
        """
        # Check each tag type in priority order
        aeroway = osm_tags.get('aeroway')
        if aeroway and aeroway in self.aeroway_values:
            return True, 'aeroway', aeroway

        amenity = osm_tags.get('amenity')
        if amenity and amenity in self.amenity_values:
            return True, 'amenity', amenity

        building = osm_tags.get('building')
        if building and building in self.building_values:
            return True, 'building', building

        man_made = osm_tags.get('man_made')
        if man_made and man_made in self.man_made_values:
            return True, 'man_made', man_made

        military = osm_tags.get('military')
        if military and military in self.military_values:
            return True, 'military', military

        power = osm_tags.get('power')
        if power and power in self.power_values:
            return True, 'power', power

        waterway = osm_tags.get('waterway')
        if waterway and waterway in self.waterway_values:
            return True, 'waterway', waterway

        return False, None, None

    def node(self, n):
        """Extract point features."""
        self.processed_count += 1
        if self.processed_count % 1000000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Check bounding box
        try:
            if not self._in_bbox(n.location.lon, n.location.lat):
                return
        except:
            return

        # Check if this feature matches our criteria
        is_valid, tag_type, tag_value = self._check_feature(dict(n.tags))

        if is_valid:
            self.structures.append({
                'geometry': Point(n.location.lon, n.location.lat),
                'tag_type': tag_type,
                'infrastructure_type': tag_value,
                'osm_id': n.id,
                'name': n.tags.get('name', ''),
                'feature_class': 'node'
            })

    def way(self, w):
        """Extract way features (can be lines or areas)."""
        self.processed_count += 1
        if self.processed_count % 100000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Check if this feature matches our criteria
        is_valid, tag_type, tag_value = self._check_feature(dict(w.tags))

        if is_valid:
            try:
                # Create LineString from nodes (for linear features like runways, dams)
                coords = []
                for node in w.nodes:
                    try:
                        if node.location.valid():
                            coords.append((node.lon, node.lat))
                    except:
                        continue

                if len(coords) >= 2:
                    geometry = LineString(coords)

                    self.structures.append({
                        'geometry': geometry,
                        'tag_type': tag_type,
                        'infrastructure_type': tag_value,
                        'osm_id': w.id,
                        'name': w.tags.get('name', ''),
                        'feature_class': 'way'
                    })
            except Exception as e:
                pass

    def area(self, a):
        """Extract area/polygon features."""
        self.processed_count += 1
        if self.processed_count % 100000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Check if this feature matches our criteria
        is_valid, tag_type, tag_value = self._check_feature(dict(a.tags))

        if is_valid:
            try:
                wkb = wkblib.loads(a.wkb, hex=True)

                self.structures.append({
                    'geometry': wkb,
                    'tag_type': tag_type,
                    'infrastructure_type': tag_value,
                    'osm_id': a.id,
                    'name': a.tags.get('name', ''),
                    'feature_class': 'area'
                })
            except Exception as e:
                pass


def calculate_basin_infrastructure_from_pbf(basin_file, pbf_files, output_folder):
    """
    Calculate critical infrastructure statistics from PBF file(s).
    Creates CSV with statistics and spatial files (GeoJSON and Shapefile) for visualization.

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
    output_csv = output_path / "infrastructure_statistics.csv"
    output_geojson = output_path / "infrastructure_affected.geojson"
    output_shapefile = output_path / "infrastructure_affected.shp"

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

    bbox = basins_wgs84.total_bounds
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

    # Extract infrastructure from all PBF files
    all_infrastructure = []

    for pbf_file in files:
        print(f"\nExtracting critical infrastructure from: {pbf_file.name}")
        print(f"This may take several minutes for large files...")

        handler = CriticalInfrastructureExtractor(bbox=bbox)
        handler.apply_file(str(pbf_file), locations=True)

        print(f"\n  Found {len(handler.structures):,} infrastructure features in bounding box")

        if handler.structures:
            all_infrastructure.extend(handler.structures)

    if not all_infrastructure:
        print("\nWarning: No infrastructure found in PBF files within basin area!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'infrastructure_type': ['All Infrastructure'],
            'count': [0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # Convert to GeoDataFrame
    print(f"\nConverting {len(all_infrastructure):,} infrastructure features to GeoDataFrame...")
    infrastructure = gpd.GeoDataFrame(all_infrastructure, crs="EPSG:4326")
    print(f"Total infrastructure extracted: {len(infrastructure):,}")

    # Show what we found
    print("\nBreakdown by tag type:")
    print(infrastructure['tag_type'].value_counts())

    print("\nInfrastructure types found:")
    print(infrastructure['infrastructure_type'].value_counts())

    # Reproject to match basins
    if infrastructure.crs != basins.crs:
        print(f"\nReprojecting infrastructure to {basins.crs}...")
        infrastructure = infrastructure.to_crs(basins.crs)

    # Spatial join
    print("\nIntersecting infrastructure with basins...")
    infrastructure_in_basins = gpd.sjoin(infrastructure, basins, how='inner', predicate='intersects')

    print(f"Infrastructure within basins: {len(infrastructure_in_basins):,}")

    if len(infrastructure_in_basins) == 0:
        print("\nWarning: No infrastructure found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'infrastructure_type': ['All Infrastructure'],
            'count': [0]
        })
    else:
        # === EXPORT SPATIAL FILES ===
        print(f"\nPreparing infrastructure features for export...")

        # Reproject back to WGS84 for GeoJSON/Shapefile
        infrastructure_for_export = infrastructure_in_basins.to_crs("EPSG:4326")

        # Select columns for export
        export_columns = [
            'tag_type',
            'infrastructure_type',
            'name',
            'osm_id',
            'geometry'
        ]

        infrastructure_export = infrastructure_for_export[export_columns].copy()

        # 1. Save to GeoJSON (keeps original geometries)
        print(f"\n1. Saving infrastructure to GeoJSON...")
        infrastructure_export.to_file(output_geojson, driver='GeoJSON')
        print(f"   ✓ {output_geojson.name}")
        print(f"     Features: {len(infrastructure_export):,}")

        # 2. Save to Shapefile (convert to point centroids)
        print(f"\n2. Saving infrastructure to Shapefile (as point centroids)...")
        infrastructure_shp = infrastructure_export.copy()
        infrastructure_shp = infrastructure_shp.rename(columns={
            'tag_type': 'tag_type',
            'infrastructure_type': 'infra_type',
            'name': 'name',
            'osm_id': 'osm_id'
        })

        # Convert all geometries to centroids (points)
        infrastructure_shp['geometry'] = infrastructure_shp.geometry.centroid

        infrastructure_shp.to_file(output_shapefile)
        print(f"   ✓ {output_shapefile.name}")
        print(f"     Features: {len(infrastructure_shp):,}")
        print(f"     Note: All features converted to point centroids")

        # === CALCULATE STATISTICS ===
        total_infrastructure = len(infrastructure_in_basins)

        # Count by tag type
        tag_type_counts = infrastructure_in_basins.groupby('tag_type').size().to_dict()

        # Count by each specific infrastructure type
        type_counts = infrastructure_in_basins['infrastructure_type'].value_counts().reset_index()
        type_counts.columns = ['infrastructure_type', 'count']
        type_counts = type_counts.sort_values('infrastructure_type').reset_index(drop=True)

        # Add category column for detailed rows
        type_counts['category'] = 'DETAIL'

        # Create summary rows
        summary_rows = []

        # Overall total
        summary_rows.append({
            'category': 'TOTAL',
            'infrastructure_type': 'All Critical Infrastructure',
            'count': total_infrastructure
        })

        # Subtotals by tag type
        for tag_type in sorted(tag_type_counts.keys()):
            summary_rows.append({
                'category': 'SUBTOTAL',
                'infrastructure_type': f'All {tag_type.capitalize()}',
                'count': tag_type_counts[tag_type]
            })

        summary_df = pd.DataFrame(summary_rows)

        # Combine: summaries first, then details
        result = pd.concat([summary_df, type_counts[['category', 'infrastructure_type', 'count']]],
                           ignore_index=True)

        print("\n" + "=" * 60)
        print("Critical infrastructure breakdown by tag type:")
        print("=" * 60)
        for tag_type, count in sorted(tag_type_counts.items()):
            print(f"  {tag_type}: {count:,}")

        print("\nInfrastructure types:")
        print("=" * 60)
        for _, row in type_counts.iterrows():
            print(f"  {row['infrastructure_type']}: {row['count']:,}")

    # Save statistics to CSV
    print(f"\n3. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("CRITICAL INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total critical infrastructure: {total_infrastructure:,}")
    print(f"Unique infrastructure types: {len(type_counts)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. infrastructure_affected.geojson (original geometries)")
    print("  2. infrastructure_affected.shp (point centroids)")
    print("  3. infrastructure_statistics.csv")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    pbf_files = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM\central-america-260204.osm.pbf"

    # Output folder
    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Critical_Infrastructure_Impact"

    results = calculate_basin_infrastructure_from_pbf(
        basin_file,
        pbf_files,
        output_folder
    )

    print("\n\nStatistics table:")
    print(results)