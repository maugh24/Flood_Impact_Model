import pandas as pd
import geopandas as gpd
from pathlib import Path

# Read the transportation parquet
transportation_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america.parquet"

print("Reading parquet file...")
try:
    trans_df = pd.read_parquet(transportation_parquet, engine='fastparquet')
    print("✓ Using fastparquet engine")
except Exception as e:
    print(f"⚠ fastparquet failed: {str(e)}")
    print("Falling back to pyarrow engine...")
    trans_df = pd.read_parquet(transportation_parquet, engine='pyarrow')

print(f"\nDataFrame shape: {trans_df.shape}")
print(f"\nColumn names and types:")
print(trans_df.dtypes)

print(f"\nFirst few rows:")
print(trans_df.head())

print(f"\n\n=== CHECKING FOR HIGHWAY/RAILWAY COLUMNS ===")

if 'highway' in trans_df.columns:
    print(f"\n'highway' column exists!")
    print(f"  Non-null values: {trans_df['highway'].notna().sum()}")
    print(f"  Unique values (first 50):")
    print(trans_df['highway'].unique()[:50])
    print(f"\n  Value counts (top 20):")
    print(trans_df['highway'].value_counts().head(20))
else:
    print(f"\n✗ 'highway' column NOT found")

if 'railway' in trans_df.columns:
    print(f"\n'railway' column exists!")
    print(f"  Non-null values: {trans_df['railway'].notna().sum()}")
    print(f"  Unique values:")
    print(trans_df['railway'].unique())
    print(f"\n  Value counts:")
    print(trans_df['railway'].value_counts())
else:
    print(f"\n✗ 'railway' column NOT found")

# Check what transportation-related columns exist
print(f"\n\n=== ALL COLUMNS ===")
for col in trans_df.columns:
    if col != 'geometry':
        non_null = trans_df[col].notna().sum()
        unique = trans_df[col].nunique()
        print(f"\n{col}:")
        print(f"  Non-null: {non_null}")
        print(f"  Unique: {unique}")
        if unique <= 30:
            print(f"  Values: {trans_df[col].unique()}")
        else:
            print(f"  Sample values: {trans_df[col].unique()[:20]}")

