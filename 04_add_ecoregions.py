import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def normalize_lon_and_join(group, ecoregions_shp_path, base_dir='connectivity_matrix'):
    # 1. Load the data
    group_dir = os.path.join(base_dir, group)
    input_path = os.path.join(group_dir, f"{group}_integrated_climate_connectivity.csv")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # 2. Convert Longitude from 0-360 to -180-180
    # Formula: (lon + 180) % 360 - 180
    print("Normalizing longitude from 0-360 to -180-180...")
    df['Longitude_Original'] = df['Longitude'] # Keep a backup just in case
    df['Longitude'] = (df['Longitude'] + 180) % 360 - 180

    # 3. Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf_reefs = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 4. Load Ecoregions
    gdf_meow = gpd.read_file(ecoregions_shp_path)
    if gdf_meow.crs != gdf_reefs.crs:
        gdf_meow = gdf_meow.to_crs(gdf_reefs.crs)

    # 5. Robust Spatial Join (Nearest Neighbor)
    print("Performing spatial join with normalized coordinates...")
    joined = gpd.sjoin_nearest(
        gdf_reefs, 
        gdf_meow[['ECOREGION', 'PROVINCE', 'REALM', 'geometry']], 
        how='left', 
        max_distance=0.1, # Approx 10km buffer for edge reefs
        distance_col="dist_to_poly"
    )

    # Handle potential duplicates from the nearest join
    joined = joined.drop_duplicates(subset=['Reef_ID'])

    # 6. Save the updated file
    output_path = os.path.join(group_dir, f"{group}_integrated_climate_connectivity_ecoregions.csv")
    
    # Clean up temporary spatial columns before saving
    final_df = joined.drop(columns=['geometry', 'index_right', 'dist_to_poly'])
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Data saved to: {output_path}")
    print(f"Sample Longitude: Original {final_df['Longitude_Original'].iloc[0]} -> New {final_df['Longitude'].iloc[0]}")

if __name__ == "__main__":
    # Update this path to your local MEOW shapefile
    meow_path = "MEOW/meow_ecos.shp"
    normalize_lon_and_join('competitive', meow_path)