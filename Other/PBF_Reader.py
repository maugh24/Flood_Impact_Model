import osmium


class OSMInspector(osmium.SimpleHandler):
    """Inspect the structure of a PBF file."""

    def __init__(self, max_samples=100):
        osmium.SimpleHandler.__init__(self)
        self.max_samples = max_samples
        self.node_count = 0
        self.way_count = 0
        self.relation_count = 0
        self.node_samples = []
        self.way_samples = []
        self.relation_samples = []
        self.all_tags = set()

    def node(self, n):
        """Sample nodes (points)."""
        self.node_count += 1
        if len(self.node_samples) < self.max_samples and len(n.tags) > 0:
            self.node_samples.append({
                'id': n.id,
                'tags': dict(n.tags),
                'location': (n.location.lon, n.location.lat)
            })
            for tag in n.tags:
                self.all_tags.add(tag.k)

    def way(self, w):
        """Sample ways (lines/polygons)."""
        self.way_count += 1
        if len(self.way_samples) < self.max_samples and len(w.tags) > 0:
            self.way_samples.append({
                'id': w.id,
                'tags': dict(w.tags)
            })
            for tag in w.tags:
                self.all_tags.add(tag.k)

    def relation(self, r):
        """Sample relations."""
        self.relation_count += 1
        if len(self.relation_samples) < self.max_samples and len(r.tags) > 0:
            self.relation_samples.append({
                'id': r.id,
                'tags': dict(r.tags)
            })
            for tag in r.tags:
                self.all_tags.add(tag.k)


def inspect_pbf(pbf_file, max_samples=100):
    """
    Inspect the structure of a PBF file without loading everything.

    Parameters:
    -----------
    pbf_file : str
        Path to .osm.pbf file
    max_samples : int
        Number of features to sample (default: 100)
    """
    print(f"Inspecting PBF file: {pbf_file}")
    print("=" * 60)

    inspector = OSMInspector(max_samples=max_samples)
    inspector.apply_file(str(pbf_file))

    print(f"\nTotal counts:")
    print(f"  Nodes: {inspector.node_count:,}")
    print(f"  Ways: {inspector.way_count:,}")
    print(f"  Relations: {inspector.relation_count:,}")

    print(f"\nAll tag keys found (first 50):")
    for i, tag in enumerate(sorted(inspector.all_tags)[:50]):
        print(f"  - {tag}")

    print(f"\n{'=' * 60}")
    print("SAMPLE NODES (points with tags):")
    print(f"{'=' * 60}")
    for i, node in enumerate(inspector.node_samples[:5]):
        print(f"\nNode {i + 1} (ID: {node['id']}):")
        for key, value in node['tags'].items():
            print(f"  {key}: {value}")

    print(f"\n{'=' * 60}")
    print("SAMPLE WAYS (buildings/roads with tags):")
    print(f"{'=' * 60}")
    for i, way in enumerate(inspector.way_samples[:5]):
        print(f"\nWay {i + 1} (ID: {way['id']}):")
        for key, value in way['tags'].items():
            print(f"  {key}: {value}")

    print(f"\n{'=' * 60}")
    print("Looking for building/amenity-related features:")
    print(f"{'=' * 60}")

    # Find features with building or amenity tags
    building_nodes = [n for n in inspector.node_samples if 'amenity' in n['tags'] or 'building' in n['tags']]
    building_ways = [w for w in inspector.way_samples if 'amenity' in w['tags'] or 'building' in w['tags']]

    print(f"\nNodes with building/amenity tags: {len(building_nodes)}")
    for node in building_nodes[:3]:
        print(f"  {node['tags']}")

    print(f"\nWays with building/amenity tags: {len(building_ways)}")
    for way in building_ways[:3]:
        print(f"  {way['tags']}")

    return inspector


# Usage
if __name__ == "__main__":
    pbf_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM\central-america-260204.osm.pbf"

    inspector = inspect_pbf(pbf_file, max_samples=100)