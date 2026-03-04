import pandas as pd
from pathlib import Path


def combine_flood_impacts(population_csv, farmland_csv, building_csv, output_csv):
    """
    Combine population, farmland, and building impact statistics into one detailed report.
    Creates a multi-row format suitable for report tables.

    Parameters:
    -----------
    population_csv : str
        Path to population statistics CSV
    farmland_csv : str
        Path to farmland statistics CSV
    building_csv : str
        Path to building statistics CSV (with dynamic OSM tag columns)
    output_csv : str
        Path for final combined output CSV
    """

    print("=" * 60)
    print("COMBINING FLOOD IMPACT ASSESSMENTS")
    print("=" * 60)

    # Read each impact assessment
    print("\nReading impact assessments...")

    detailed_report = []

    # === POPULATION IMPACT ===
    try:
        population_df = pd.read_csv(population_csv)
        print(f"✓ Population impact loaded")

        total_population = population_df['total_population'].iloc[0]

        detailed_report.append({
            'Impact_Category': 'POPULATION',
            'Metric': 'Total Population Affected',
            'Value': total_population,
            'Unit': 'people'
        })

    except FileNotFoundError:
        print(f"✗ Population impact not found: {population_csv}")
        detailed_report.append({
            'Impact_Category': 'POPULATION',
            'Metric': 'Total Population Affected',
            'Value': 0,
            'Unit': 'people'
        })
    except Exception as e:
        print(f"✗ Error reading population data: {e}")

    # === FARMLAND IMPACT ===
    try:
        farmland_df = pd.read_csv(farmland_csv)
        print(f"✓ Farmland impact loaded")

        farmland_area = farmland_df['total_farmland_area_km2'].iloc[0]
        farmland_cells = farmland_df['total_farmland_cells'].iloc[0]
        total_area = farmland_df['total_area_km2'].iloc[0]
        farmland_pct = farmland_df['farmland_percentage'].iloc[0]

        detailed_report.append({
            'Impact_Category': 'FARMLAND',
            'Metric': 'Farmland Area Affected',
            'Value': farmland_area,
            'Unit': 'km²'
        })

        detailed_report.append({
            'Impact_Category': 'FARMLAND',
            'Metric': 'Total Basin Area Analyzed',
            'Value': total_area,
            'Unit': 'km²'
        })

        detailed_report.append({
            'Impact_Category': 'FARMLAND',
            'Metric': 'Farmland Coverage',
            'Value': farmland_pct,
            'Unit': '%'
        })

        detailed_report.append({
            'Impact_Category': 'FARMLAND',
            'Metric': 'Farmland Cells (10m pixels)',
            'Value': farmland_cells,
            'Unit': 'cells'
        })

    except FileNotFoundError:
        print(f"✗ Farmland impact not found: {farmland_csv}")
        detailed_report.extend([
            {'Impact_Category': 'FARMLAND', 'Metric': 'Farmland Area Affected', 'Value': 0, 'Unit': 'km²'},
            {'Impact_Category': 'FARMLAND', 'Metric': 'Total Basin Area Analyzed', 'Value': 0, 'Unit': 'km²'},
            {'Impact_Category': 'FARMLAND', 'Metric': 'Farmland Coverage', 'Value': 0, 'Unit': '%'},
        ])
    except Exception as e:
        print(f"✗ Error reading farmland data: {e}")

    # === INFRASTRUCTURE IMPACT (BUILDINGS) ===
    try:
        building_df = pd.read_csv(building_csv)
        print(f"✓ Infrastructure impact loaded")

        # The building CSV now has dynamic columns based on OSM tags found
        # Get total features (this should always be the first column)
        total_features = building_df['total_features'].iloc[0] if 'total_features' in building_df.columns else 0

        # Add total first
        detailed_report.append({
            'Impact_Category': 'INFRASTRUCTURE',
            'Metric': 'Total Features Affected',
            'Value': total_features,
            'Unit': 'features'
        })

        # Friendly names for OSM tags
        osm_tag_names = {
            'amenity': 'Amenities',
            'building': 'Buildings',
            'aeroway': 'Airport Infrastructure',
            'highway': 'Road Infrastructure',
            'railway': 'Railway Infrastructure',
            'shop': 'Shops & Retail',
            'tourism': 'Tourism Facilities',
            'leisure': 'Leisure Facilities',
            'office': 'Office Buildings',
            'landuse': 'Land Use Areas',
            'natural': 'Natural Features',
            'waterway': 'Waterways',
            'power': 'Power Infrastructure',
            'man_made': 'Man-made Structures',
            'craft': 'Craft Facilities',
            'emergency': 'Emergency Services',
            'healthcare': 'Healthcare Facilities',
            'historic': 'Historic Sites',
            'military': 'Military Facilities',
            'sport': 'Sports Facilities',
            'telecom': 'Telecom Infrastructure',
            'public_transport': 'Public Transport'
        }

        # Add breakdown by OSM tag (skip 'total_features' column)
        for col in building_df.columns:
            if col != 'total_features':
                display_name = osm_tag_names.get(col, col.replace('_', ' ').title())
                value = building_df[col].iloc[0]

                # Only include tags with non-zero values
                if value > 0:
                    detailed_report.append({
                        'Impact_Category': 'INFRASTRUCTURE',
                        'Metric': display_name,
                        'Value': value,
                        'Unit': 'features'
                    })

    except FileNotFoundError:
        print(f"✗ Infrastructure impact not found: {building_csv}")
        detailed_report.append({
            'Impact_Category': 'INFRASTRUCTURE',
            'Metric': 'Total Features Affected',
            'Value': 0,
            'Unit': 'features'
        })
    except Exception as e:
        print(f"✗ Error reading infrastructure data: {e}")
        print(f"  Error details: {str(e)}")

    # Convert to DataFrame
    result = pd.DataFrame(detailed_report)

    # Save combined results
    result.to_csv(output_csv, index=False)
    print(f"\nFinal combined report saved to: {output_csv}")

    # Print formatted report
    print("\n" + "=" * 60)
    print("FINAL FLOOD IMPACT ASSESSMENT")
    print("=" * 60)

    current_category = None
    for _, row in result.iterrows():
        if row['Impact_Category'] != current_category:
            current_category = row['Impact_Category']
            print(f"\n{current_category}:")

        # Format the value based on type
        value = row['Value']
        if row['Unit'] in ['km²', '%']:
            print(f"  {row['Metric']}: {value:,.2f} {row['Unit']}")
        else:
            print(f"  {row['Metric']}: {value:,.0f} {row['Unit']}")

    print("=" * 60)

    return result


# Usage
if __name__ == "__main__":
    # Paths to individual impact assessments
    population_csv = r"C:\C_Drive_Brians_Stuff\Python_Projects\population_statistics.csv"
    farmland_csv = r"C:\C_Drive_Brians_Stuff\Python_Projects\farmland_statistics.csv"
    building_csv = r"C:\C_Drive_Brians_Stuff\Python_Projects\building_statistics.csv"

    # Output path for combined assessment
    output_csv = r"C:\C_Drive_Brians_Stuff\Python_Projects\Final_Flood_Impact.csv"

    # Combine all impacts
    final_report = combine_flood_impacts(
        population_csv,
        farmland_csv,
        building_csv,
        output_csv
    )

    print("\n✓ Final flood impact assessment complete!")
    print(f"\nThe report can now be imported into your document.")