import geopandas as gpd
import pandas as pd
from pathlib import Path
import osmium
import shapely.wkb as wkblib
from shapely.geometry import Point


class StructureExtractor(osmium.SimpleHandler):
    """Extract structures from PBF file using aeroway, amenity, and building tags."""

    def __init__(self, bbox=None):
        osmium.SimpleHandler.__init__(self)
        self.structures = []
        self.processed_count = 0
        self.bbox = bbox

        # Define specific aeroway values we want to extract
        self.aeroway_values = {
            'aerodrome', 'aircraft_crossing', 'apron', 'gate', 'hangar', 'helipad',
            'heliport', 'navigationaid', 'runway', 'spaceport', 'taxiway', 'terminal'
        }

        # Define specific amenity values we want to extract
        self.amenity_values = {
            # Food & Drink
            'bar', 'biergarten', 'cafe', 'fast_food', 'food_court', 'ice_cream', 'pub', 'restaurant',

            # Education
            'college', 'dancing_school', 'driving_school', 'first_aid_school', 'kindergarten',
            'language_school', 'library', 'surf_school', 'toy_library', 'research_institute',
            'training', 'music_school', 'school', 'traffic_park', 'university',

            # Transportation
            'bicycle_rental', 'bicycle_wash', 'boat_rental', 'boat_storage', 'boat_sharing',
            'bus_station', 'car_rental', 'car_sharing', 'car_wash', 'vehicle_inspection',
            'charging_station', 'driver_training', 'ferry_terminal', 'fuel', 'motorcycle_parking',
            'parking', 'taxi', 'weighbridge', 'payment_terminal',

            # Financial
            'bank', 'bureau_de_change', 'money_transfer', 'payment_centre',

            # Healthcare
            'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy',
            'social_facility', 'veterinary',

            # Entertainment
            'arts_centre', 'casino', 'cinema', 'community_centre', 'conference_centre',
            'events_venue', 'exhibition_centre', 'fountain', 'gambling', 'music_venue',
            'nightclub', 'planetarium', 'social_centre', 'stage', 'studio', 'swingerclub', 'theatre',

            # Public Services
            'courthouse', 'fire_station', 'police', 'post_depot', 'post_office', 'prison',
            'ranger_station', 'townhall',

            # Other
            'check_in', 'lounge', 'mailroom', 'waste_transfer_station', 'animal_shelter',
            'animal_training', 'crematorium', 'dive_centre', 'funeral_hall', 'grave_yard',
            'internet_cafe', 'marketplace', 'monastery', 'mortuary', 'place_of_worship', 'refugee_site'
        }

        # Define specific building values we want to extract
        self.building_values = {
            # Residential
            'apartments', 'barracks', 'bungalow', 'cabin', 'detached', 'annex', 'dormitory',
            'farm', 'ger', 'hotel', 'house', 'houseboat', 'residential', 'semidetached_house',
            'static_caravan', 'stilt_house', 'terrace', 'trullo',

            # Commercial
            'commercial', 'industrial', 'office', 'retail', 'supermarket', 'warehouse',

            # Religious
            'religious', 'cathedral', 'chapel', 'church', 'kingdom_hall', 'monastery',
            'mosque', 'presbytery', 'shrine', 'synagogue', 'temple',

            # Civic
            'bakehouse', 'bridge', 'civic', 'clock_tower', 'college', 'fire_station',
            'government', 'gatehouse', 'hospital', 'kindergarten', 'museum', 'public',
            'school', 'toilets', 'train_station', 'transportation', 'university',

            # Agricultural
            'barn', 'conservatory', 'cowshed', 'farm_auxiliary', 'greenhouse', 'stable', 'livestock',

            # Sports
            'grandstand', 'pavilion', 'riding_hall', 'sports_hall', 'sports_centre', 'stadium',

            # Storage & Utility
            'allotment_house', 'boathouse', 'hangar', 'hut', 'shed', 'carport', 'garage',
            'garages', 'parking', 'digester', 'service', 'tech_cab', 'transformer_tower',
            'water_tower', 'storage_tank', 'silo',

            # Other
            'beach_hut', 'castle', 'guardhouse', 'military', 'outbuilding', 'pagoda',
            'quonset_hut', 'ruins', 'ship', 'tower', 'windmill'
        }

    def _in_bbox(self, lon, lat):
        """Check if point is within bounding box."""
        if self.bbox is None:
            return True
        return (self.bbox[0] <= lon <= self.bbox[2] and
                self.bbox[1] <= lat <= self.bbox[3])

    def _check_feature(self, osm_tags):
        """
        Check if feature has aeroway, amenity, or building tags we want.
        Returns (is_valid, tag_type, tag_value)
        """
        # Check aeroway tag first (most specific)
        aeroway = osm_tags.get('aeroway')
        if aeroway and aeroway in self.aeroway_values:
            return True, 'aeroway', aeroway

        # Check amenity tag
        amenity = osm_tags.get('amenity')
        if amenity and amenity in self.amenity_values:
            return True, 'amenity', amenity

        # Check building tag
        building = osm_tags.get('building')
        if building and building in self.building_values:
            return True, 'building', building

        return False, None, None

    def node(self, n):
        """Extract point features."""
        self.processed_count += 1
        if self.processed_count % 1000000 == 0:
            print(f"  Processed {self.processed_count:,} features, found {len(self.structures):,} structures...",
                  end='\r')

        # Check bounding box
        if not self._in_bbox(n.location.lon, n.location.lat):
            return

        # Check if this feature matches our criteria
        is_valid, tag_type, tag_value = self._check_feature(dict(n.tags))

        if is_valid:
            self.structures.append({
                'geometry': Point(n.location.lon, n.location.lat),
                'tag_type': tag_type,  # 'aeroway', 'amenity', or 'building'
                'structure_type': tag_value,  # specific value
                'osm_id': n.id,
                'name': n.tags.get('name', ''),
                'addr_street': n.tags.get('addr:street', ''),
                'addr_housenumber': n.tags.get('addr:housenumber', ''),
                'feature_type': 'node'
            })

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
                    'structure_type': tag_value,
                    'osm_id': a.id,
                    'name': a.tags.get('name', ''),
                    'addr_street': a.tags.get('addr:street', ''),
                    'addr_housenumber': a.tags.get('addr:housenumber', ''),
                    'feature_type': 'area'
                })
            except Exception as e:
                pass  # Skip invalid geometries


def calculate_basin_structures_from_pbf(basin_file, pbf_files, output_folder):
    """
    Calculate structure statistics from PBF file(s) using aeroway, amenity, and building tags.
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
    output_csv = output_path / "structure_statistics.csv"
    output_geojson = output_path / "structures_affected.geojson"
    output_shapefile = output_path / "structures_affected.shp"

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

    # Extract structures from all PBF files
    all_structures = []

    for pbf_file in files:
        print(f"\nExtracting structures from: {pbf_file.name}")
        print(f"This may take several minutes for large files...")

        handler = StructureExtractor(bbox=bbox)
        handler.apply_file(str(pbf_file))

        print(f"\n  Found {len(handler.structures):,} structures in bounding box")

        if handler.structures:
            all_structures.extend(handler.structures)

    if not all_structures:
        print("\nWarning: No structures found in PBF files within basin area!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'structure_type': ['All Structures'],
            'count': [0]
        })
        result.to_csv(output_csv, index=False)
        return result

    # Convert to GeoDataFrame
    print(f"\nConverting {len(all_structures):,} structures to GeoDataFrame...")
    structures = gpd.GeoDataFrame(all_structures, crs="EPSG:4326")
    print(f"Total structures extracted: {len(structures):,}")

    # Show what we found
    print("\nBreakdown by tag type:")
    print(structures['tag_type'].value_counts())

    print("\nTop 20 structure types found:")
    print(structures['structure_type'].value_counts().head(20))

    # Reproject to match basins
    if structures.crs != basins.crs:
        print(f"\nReprojecting structures to {basins.crs}...")
        structures = structures.to_crs(basins.crs)

    # Spatial join
    print("\nIntersecting structures with basins...")
    structures_in_basins = gpd.sjoin(structures, basins, how='inner', predicate='intersects')

    print(f"Structures within basins: {len(structures_in_basins):,}")

    if len(structures_in_basins) == 0:
        print("\nWarning: No structures found within basins!")
        result = pd.DataFrame({
            'category': ['TOTAL'],
            'structure_type': ['All Structures'],
            'count': [0]
        })
    else:
        # === EXPORT SPATIAL FILES ===
        print(f"\nPreparing structure features for export...")

        # Reproject back to WGS84 for GeoJSON/Shapefile
        structures_for_export = structures_in_basins.to_crs("EPSG:4326")

        # Select columns for export (include all useful attributes)
        export_columns = [
            'tag_type',  # 'aeroway', 'amenity', or 'building'
            'structure_type',  # Specific type
            'name',  # Structure name
            'addr_street',  # Street address
            'addr_housenumber',  # House number
            'osm_id',  # OSM ID for reference
            'geometry'  # The actual geometry
        ]

        structures_export = structures_for_export[export_columns].copy()

        # 1. Save to GeoJSON
        print(f"\n1. Saving structures to GeoJSON...")
        structures_export.to_file(output_geojson, driver='GeoJSON')
        print(f"   ✓ {output_geojson.name}")
        print(f"     Features: {len(structures_export):,}")

        # 2. Save to Shapefile
        print(f"\n2. Saving structures to Shapefile...")
        # Shapefile field names limited to 10 characters
        structures_shp = structures_export.copy()
        structures_shp = structures_shp.rename(columns={
            'tag_type': 'tag_type',  # 8 chars
            'structure_type': 'struct_typ',  # 10 chars
            'name': 'name',  # 4 chars
            'addr_street': 'addr_st',  # 7 chars
            'addr_housenumber': 'addr_num',  # 8 chars
            'osm_id': 'osm_id'  # 6 chars
        })
        structures_shp.to_file(output_shapefile)
        print(f"   ✓ {output_shapefile.name}")
        print(f"     Features: {len(structures_shp):,}")
        print(f"     Note: Field names shortened for shapefile compatibility")

        # === CALCULATE STATISTICS ===
        total_structures = len(structures_in_basins)

        # Count by tag type (aeroway, amenity, building)
        tag_type_counts = structures_in_basins.groupby('tag_type').size().to_dict()

        # Count by each specific structure type
        type_counts = structures_in_basins['structure_type'].value_counts().reset_index()
        type_counts.columns = ['structure_type', 'count']
        type_counts = type_counts.sort_values('structure_type').reset_index(drop=True)

        # Add category column for detailed rows
        type_counts['category'] = 'DETAIL'

        # Create summary rows
        summary_rows = []

        # Overall total
        summary_rows.append({
            'category': 'TOTAL',
            'structure_type': 'All Structures',
            'count': total_structures
        })

        # Subtotals by tag type
        for tag_type in ['aeroway', 'amenity', 'building']:
            if tag_type in tag_type_counts:
                summary_rows.append({
                    'category': 'SUBTOTAL',
                    'structure_type': f'All {tag_type.capitalize()}',
                    'count': tag_type_counts[tag_type]
                })

        summary_df = pd.DataFrame(summary_rows)

        # Combine: summaries first, then details
        result = pd.concat([summary_df, type_counts[['category', 'structure_type', 'count']]],
                           ignore_index=True)

        print("\n" + "=" * 60)
        print("Structure breakdown by tag type:")
        print("=" * 60)
        for tag_type, count in tag_type_counts.items():
            print(f"  {tag_type}: {count:,}")

        print("\nTop 20 structure types:")
        print("=" * 60)
        for _, row in type_counts.head(20).iterrows():
            print(f"  {row['structure_type']}: {row['count']:,}")

    # Save statistics to CSV
    print(f"\n3. Saving statistics CSV...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ {output_csv.name}")

    # Print final summary
    print("\n" + "=" * 60)
    print("STRUCTURE STATISTICS")
    print("=" * 60)
    print(f"Total structures: {total_structures:,}")
    print(f"Unique structure types: {len(type_counts)}")
    print(f"\nFiles created in: {output_path}")
    print("  1. structures_affected.geojson (for web mapping/ArcGIS Online)")
    print("  2. structures_affected.shp (for ArcGIS Desktop/Pro)")
    print("  3. structure_statistics.csv (summary statistics)")
    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    pbf_files = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM\central-america-260204.osm.pbf"

    # Output folder (will be created if it doesn't exist)
    output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Structure_Impact"

    results = calculate_basin_structures_from_pbf(
        basin_file,
        pbf_files,
        output_folder
    )

    print("\n\nStatistics table (first 25 rows):")
    print(results.head(25))