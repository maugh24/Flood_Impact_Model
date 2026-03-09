import geopandas as gpd
import pandas as pd
from pathlib import Path
import osmium
import shapely.wkb as wkblib
from shapely.geometry import Point, LineString
import warnings

warnings.filterwarnings('ignore')


class CompletePBFExtractor(osmium.SimpleHandler):
    """
    Extract ALL features from PBF file to preserve complete data.
    Handles nodes, ways, and relations (areas).
    Optimized for speed and memory efficiency.
    """

    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.features = []
        self.processed_count = 0
        self.node_count = 0
        self.way_count = 0
        self.relation_count = 0

    def node(self, n):
        """Extract node features."""
        self.processed_count += 1
        self.node_count += 1

        if self.processed_count % 1000000 == 0:
            print(f"  Processed {self.processed_count:,} features "
                  f"(nodes: {self.node_count:,}, ways: {self.way_count:,}, relations: {self.relation_count:,})...",
                  end='\r')

        try:
            if n.location.valid():
                tags = dict(n.tags)
                self.features.append({
                    'geometry': Point(n.lon, n.lat),
                    'osm_id': n.id,
                    'osm_type': 'node',
                    'tags_json': str(tags),  # Store tags as JSON string
                    'name': tags.get('name', ''),
                    'feature_class': 'node'
                })
        except Exception as e:
            pass

    def way(self, w):
        """Extract way (linestring) features."""
        self.processed_count += 1
        self.way_count += 1

        if self.processed_count % 1000000 == 0:
            print(f"  Processed {self.processed_count:,} features "
                  f"(nodes: {self.node_count:,}, ways: {self.way_count:,}, relations: {self.relation_count:,})...",
                  end='\r')

        try:
            # Build coordinates list efficiently
            coords = [(node.lon, node.lat) for node in w.nodes if node.location.valid()]

            if len(coords) >= 2:
                tags = dict(w.tags)
                geometry = LineString(coords)

                self.features.append({
                    'geometry': geometry,
                    'osm_id': w.id,
                    'osm_type': 'way',
                    'tags_json': str(tags),
                    'name': tags.get('name', ''),
                    'feature_class': 'way'
                })
        except Exception as e:
            pass

    def area(self, a):
        """Extract area/polygon features."""
        self.processed_count += 1
        self.relation_count += 1

        if self.processed_count % 1000000 == 0:
            print(f"  Processed {self.processed_count:,} features "
                  f"(nodes: {self.node_count:,}, ways: {self.way_count:,}, relations: {self.relation_count:,})...",
                  end='\r')

        try:
            wkb = wkblib.loads(a.wkb, hex=True)
            tags = dict(a.tags)

            self.features.append({
                'geometry': wkb,
                'osm_id': a.id,
                'osm_type': 'area',
                'tags_json': str(tags),
                'name': tags.get('name', ''),
                'feature_class': 'area'
            })
        except Exception as e:
            pass


def convert_pbf_to_parquet(pbf_file, output_parquet=None, bbox=None):
    """
    Convert PBF file to Parquet format with all data preserved.
    Each PBF file gets its own individual parquet file matching the original naming.

    Parameters:
    -----------
    pbf_file : str or Path
        Path to PBF file to convert
    output_parquet : str or Path, optional
        Path to output parquet file. If None, uses same name as input with .parquet extension.
        Example: "central-america-260204.osm.pbf" → "central-america-260204.parquet"
    bbox : tuple, optional
        Bounding box as (minx, miny, maxx, maxy) to limit area. If None, processes entire file.

    Returns:
    --------
    GeoDataFrame
        The converted data as a GeoDataFrame
    """

    pbf_path = Path(pbf_file)

    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_path}")

    # Auto-generate individual parquet filename matching the PBF name
    if output_parquet is None:
        output_parquet = pbf_path.with_suffix('.parquet')

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting PBF to Parquet")
    print(f"  Input:  {pbf_path}")
    print(f"  Output: {output_path}")
    if bbox:
        print(f"  BBox:   {bbox}")

    # Extract features from PBF
    print(f"\nExtracting features from PBF file...")
    handler = CompletePBFExtractor()
    handler.apply_file(str(pbf_path), locations=True)

    print(f"\n  Total processed: {handler.processed_count:,}")
    print(f"    - Nodes:      {handler.node_count:,}")
    print(f"    - Ways:       {handler.way_count:,}")
    print(f"    - Relations:  {handler.relation_count:,}")
    print(f"    - Features extracted: {len(handler.features):,}")

    if not handler.features:
        print("\nWarning: No features extracted from PBF file!")
        return None

    # Convert to GeoDataFrame
    print(f"\nConverting to GeoDataFrame...")
    gdf = gpd.GeoDataFrame(handler.features, geometry='geometry', crs="EPSG:4326")

    print(f"  Total features: {len(gdf):,}")
    print(f"  Geometry types:")
    print(f"    {gdf.geometry.type.value_counts().to_dict()}")

    # Save to Parquet with pyarrow engine
    print(f"\nSaving to Parquet...")
    gdf.to_parquet(output_path, engine='pyarrow', compression='snappy')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {output_path.name}")
    print(f"  Size: {file_size_mb:.2f} MB")

    print(f"\nConversion complete!")
    print(f"  Input features: {len(handler.features):,}")
    print(f"  Output features: {len(gdf):,}")
    print(f"  Data preserved: {len(handler.features) == len(gdf)}")

    return gdf


def convert_pbf_directory_to_parquet(pbf_folder, output_folder=None):
    """
    Convert all PBF files in a directory to Parquet format.
    Each PBF file gets its own individual parquet file with matching naming convention.

    Parameters:
    -----------
    pbf_folder : str or Path
        Path to folder containing PBF files
    output_folder : str or Path, optional
        Path to output folder. If None, uses same folder as input

    Examples:
    ---------
    Input files:        Output files:
    - region1.osm.pbf   → region1.parquet
    - region2.pbf       → region2.parquet
    """

    pbf_path = Path(pbf_folder)

    if not pbf_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {pbf_path}")

    if output_folder is None:
        output_path = pbf_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

    # Find all PBF files
    pbf_files = list(pbf_path.glob("*.osm.pbf")) + list(pbf_path.glob("*.pbf"))

    if not pbf_files:
        raise FileNotFoundError(f"No PBF files found in: {pbf_path}")

    print(f"Found {len(pbf_files)} PBF file(s) to convert")

    results = []
    for i, pbf_file in enumerate(pbf_files, 1):
        print(f"\n[{i}/{len(pbf_files)}] Processing: {pbf_file.name}")
        print("-" * 80)

        output_parquet = output_path / pbf_file.with_suffix('.parquet').name

        try:
            gdf = convert_pbf_to_parquet(pbf_file, output_parquet)
            if gdf is not None:
                results.append({
                    'input_file': str(pbf_file),
                    'output_file': str(output_parquet),
                    'features_extracted': len(gdf),
                    'status': 'Success'
                })
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'input_file': str(pbf_file),
                'output_file': str(output_parquet),
                'features_extracted': 0,
                'status': f'Failed: {str(e)}'
            })

    # Print summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    successful = len([r for r in results if r['status'] == 'Success'])
    print(f"\nSuccessfully converted: {successful}/{len(pbf_files)} files")

    return summary_df


# Usage
if __name__ == "__main__":
    # Example 1: Convert single PBF file
    pbf_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_PBF\central-america-260204.osm.pbf"
    output_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-260204.parquet"

    gdf = convert_pbf_to_parquet(pbf_file, output_parquet)

    # Example 2: Convert all PBF files in a directory
    # pbf_folder = r"C:\Path\To\PBF\Files"
    # output_folder = r"C:\Path\To\Output"
    # summary = convert_pbf_directory_to_parquet(pbf_folder, output_folder)



