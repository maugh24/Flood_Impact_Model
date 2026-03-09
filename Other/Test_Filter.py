import pandas as pd

# Read the transportation parquet
transportation_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america.parquet"

print("Reading parquet file...")
try:
    trans_df = pd.read_parquet(transportation_parquet, engine='fastparquet')
    print("✓ Using fastparquet engine")
except Exception as e:
    print(f"⚠ fastparquet failed: {str(e)}")
    trans_df = pd.read_parquet(transportation_parquet, engine='pyarrow')

print(f"Total records: {len(trans_df):,}")

# Extract highway from other_tags
print("\nExtracting highway values...")
trans_df['highway'] = trans_df['other_tags'].str.extract(
    r'"highway"=>"([^"]+)"', expand=False
)

# Extract railway from other_tags
print("Extracting railway values...")
trans_df['railway'] = trans_df['other_tags'].str.extract(
    r'"railway"=>"([^"]+)"', expand=False
)

print(f"\nRecords with highway: {trans_df['highway'].notna().sum():,}")
print(f"Records with railway: {trans_df['railway'].notna().sum():,}")

print(f"\nUnique highway values (top 30):")
print(trans_df['highway'].value_counts().head(30))

print(f"\nUnique railway values:")
print(trans_df['railway'].value_counts())

# Now test the filter
highway_values = frozenset([
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'busway', 'footway', 'cycleway'
])

railway_values = frozenset(['light_rail', 'monorail', 'rail', 'subway', 'tram'])

filter_criteria = pd.Series([False] * len(trans_df), index=trans_df.index)
filter_criteria |= trans_df['highway'].isin(highway_values)
filter_criteria |= trans_df['railway'].isin(railway_values)

filtered = trans_df[filter_criteria]
print(f"\n\nRecords matching filter: {len(filtered):,}")
print(f"\nBreakdown by highway type:")
print(filtered['highway'].value_counts())
print(f"\nBreakdown by railway type:")
print(filtered['railway'].value_counts())

