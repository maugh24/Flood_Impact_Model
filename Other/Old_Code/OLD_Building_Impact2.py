import geopandas as gpd
import pandas as pd
from pathlib import Path
import osmium
import shapely.wkb as wkblib
from shapely.geometry import Point, LineString

#@profile
class CriticalInfrastructureExtractor(osmium.SimpleHandler):
    """Extract critical infrastructure from PBF file using specific OSM tags - OPTIMIZED."""

    def __init__(self, bbox=None):
        osmium.SimpleHandler.__init__(self)
        self.structures = []
        self.processed_count = 0
        self.bbox_tuple = bbox  # Cache as tuple for faster access

        # Use frozenset for O(1) membership testing (faster than set)
        self.aeroway_values = frozenset(['runway'])

        self.amenity_values = frozenset([
            'college', 'university', 'doctors', 'hospital',
            'fire_station', 'police', 'townhall'
        ])

        self.building_values = frozenset([
            'college', 'fire_station', 'government', 'hospital',
            'school', 'university', 'military'
        ])

        self.man_made_values = frozenset(['water_works', 'water_well'])

        self.military_values = frozenset(['airfield', 'base'])

        self.power_values = frozenset(['plant'])

        self.waterway_values = frozenset(['dam'])

    def _in_bbox(self, lon, lat):
        """Quick bounding box check - optimized."""
        if self.bbox_tuple is None:
            return True
        return (self.bbox_tuple[0] <= lon <= self.bbox_tuple[2] and
                self.bbox_tuple[1] <= lat <= self.bbox_tuple[3])

    def _check_feature_fast(self, tags_dict):
        """
        Optimized feature checking with early returns and frozenset lookups.
        Returns (is_valid, tag_type, tag_value)
        """
        # Check in order of rarity (rarest first = faster elimination)
        aeroway = tags_dict.get('aeroway')
        if aeroway in self.aeroway_values:  # frozenset membership is O(1)
            return True, 'aeroway', aeroway

        amenity = tags_dict.get('amenity')
        if amenity in self.amenity_values:
            return True, 'amenity', amenity

        building = tags_dict.get('building')
        if building in self.building_values:
            return True, 'building', building

        man_made = tags_dict.get('man_made')
        if man_made in self.man_made_values:
            return True, 'man_made', man_made

        military = tags_dict.get('military')
        if military in self.military_values:
            return True, 'military', military

        power = tags_dict.get('power')
        if power in self.power_values:
            return True, 'power', power

        waterway = tags_dict.get('waterway')
        if waterway in self.waterway_values:
            return True, 'waterway', waterway

        return False, None, None

    def node(self, n):
        """Extract point features - optimized."""
        # Update counter every 5M instead of 1M to reduce overhead
        self.processed_count += 1
        if self.processed_count % 5000000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Quick bbox check
        try:
            if not self._in_bbox(n.location.lon, n.location.lat):
                return
        except:
            return

        # Convert tags to dict ONCE
        tags = dict(n.tags)
        is_valid, tag_type, tag_value = self._check_feature_fast(tags)

        if is_valid:
            self.structures.append({
                'geometry': Point(n.location.lon, n.location.lat),
                'tag_type': tag_type,
                'infrastructure_type': tag_value,
                'osm_id': n.id,
                'name': tags.get('name', ''),  # Use cached tags dict
                'feature_class': 'node'
            })

    def way(self, w):
        """Extract way features - optimized."""
        self.processed_count += 1
        if self.processed_count % 500000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Convert tags to dict ONCE
        tags = dict(w.tags)
        is_valid, tag_type, tag_value = self._check_feature_fast(tags)

        if is_valid:
            try:
                # Build coordinates list efficiently using list comprehension
                coords = [(node.lon, node.lat) for node in w.nodes if node.location.valid()]

                if len(coords) >= 2:
                    geometry = LineString(coords)

                    self.structures.append({
                        'geometry': geometry,
                        'tag_type': tag_type,
                        'infrastructure_type': tag_value,
                        'osm_id': w.id,
                        'name': tags.get('name', ''),  # Use cached tags dict
                        'feature_class': 'way'
                    })
            except:
                pass

    def area(self, a):
        """Extract area/polygon features - optimized."""
        self.processed_count += 1
        if self.processed_count % 500000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Convert tags to dict ONCE
        tags = dict(a.tags)
        is_valid, tag_type, tag_value = self._check_feature_fast(tags)

        if is_valid:
            try:
                wkb = wkblib.loads(a.wkb, hex=True)

                self.structures.append({
                    'geometry': wkb,
                    'tag_type': tag_type,
                    'infrastructure_type': tag_value,
                    'osm_id': a.id,
                    'name': tags.get('name', ''),  # Use cached tags dict
                    'feature_class': 'area'
                })
            except:
                pass

@profile
def calculate_basin_infrastructure_from_pbf(basin_file, pbf_files, output_folder):
    """
    Calculate critical infrastructure statistics from PBF file(s).
    Creates CSV with statistics and a shapefile for visualization.
    OPTIMIZED VERSION.

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
    output_shapefile = output_path / "infrastructure_affected.shp"

    # Read basins
    basin_path = Path(basin_file)
    print(f"\nReading basin data from {basin_path.suffix} file...")

    if basin_path.suffix == '.parquet':
        # Read parquet with fastparquet for better performance
        df = pd.read_parquet(basin_file, engine='fastparquet')
        # Convert WKB geometry column to shapely geometries
        df['geometry'] = df['geometry'].apply(lambda x: wkblib.loads(x, hex=True) if isinstance(x, str) else wkblib.loads(x))
        basins = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    elif basin_path.suffix in ['.shp', '.geojson', '.gpkg']:
        basins = gpd.read_file(basin_file)
    else:
        raise ValueError(f"Unsupported file format: {basin_path.suffix}")

    print(f"Loaded {len(basins)} basins")

    # Get bounding box of basins
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
        # Define defaults so later code can reference these variables safely
        total_infrastructure = 0
        tag_type_counts = {}
        type_counts = pd.DataFrame({'infrastructure_type': [], 'count': []})
    else:
        # === EXPORT SPATIAL FILES ===
        print(f"\nPreparing infrastructure features for export...")

        # Reproject back to WGS84 for Shapefile
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

        # Save to Shapefile (convert to point centroids)
        print(f"\n1. Saving infrastructure to Shapefile (as point centroids)...")
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
    print(f"\n2. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("CRITICAL INFRASTRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total critical infrastructure: {total_infrastructure:,}")
    print(f"Unique infrastructure types: {len(type_counts)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. infrastructure_affected.shp (point centroids)")
    print("  2. infrastructure_statistics.csv")
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