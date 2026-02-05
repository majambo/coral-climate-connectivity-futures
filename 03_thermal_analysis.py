import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import NearestNDInterpolator

def analyze_climate_corridors(group, cmip6_dir, base_dir='connectivity_matrix'):
    # 1. Paths & Merging (Keep as before)
    group_dir = os.path.join(base_dir, group)
    metrics_path = os.path.join(group_dir, f"{group}_graph_metrics.csv")
    coords_path = os.path.join(group_dir, f"{group}_reef_coordinates.csv")
    
    df_metrics = pd.read_csv(metrics_path)
    df_coords = pd.read_csv(coords_path)
    master_df = pd.merge(df_metrics, df_coords[['Reef_ID', 'Latitude', 'Longitude']], on='Reef_ID')

    # 2. Sorting Logic (Keep as before)
    all_nc_files = [f for f in os.listdir(cmip6_dir) if f.endswith('.nc')]
    def sorting_key(filename):
        if 'baseline' in filename: return (0, 0)
        parts = filename.replace('.nc', '').split('_')
        ssp_val = 1 if 'ssp245' in parts else 2
        try: year = int(parts[-1])
        except: year = 9999
        return (ssp_val, year)

    lttmax_files = sorted(all_nc_files, key=sorting_key)

    # 3. Process Climate Scenarios with Spatial Interpolation
    for nc_file in lttmax_files:
        scenario_name = nc_file.replace('.nc', '')
        file_path = os.path.join(cmip6_dir, nc_file)
        print(f"  → Processing: {scenario_name}")
        
        with xr.open_dataset(file_path) as ds:
            temp_var = 'thetao_ltmax'
            # Pull data and squeeze time dimension
            data = ds[temp_var].isel(time=0).load()
            
            # Create a mask of where data actually exists (Ocean only)
            mask = ~np.isnan(data)
            
            # Get coordinates of ocean pixels
            lon_grid, lat_grid = np.meshgrid(ds.longitude.values, ds.latitude.values)
            ocean_points = np.column_stack((lat_grid[mask], lon_grid[mask]))
            ocean_values = data.values[mask]

            # Build the interpolator: finds the nearest NON-NAN ocean pixel
            interpolator = NearestNDInterpolator(ocean_points, ocean_values)

            # Extract temperatures for all reef coordinates
            # This handles both points in the water and points slightly on land
            reef_points = master_df[['Latitude', 'Longitude']].values
            master_df[scenario_name] = interpolator(reef_points)

    # 4. Save Final Integrated Dataset
    output_path = os.path.join(group_dir, f"{group}_integrated_climate_connectivity.csv")
    master_df.to_csv(output_path, index=False)
    print(f"\n✅ Success! Missing values interpolated to nearest ocean grid.")

if __name__ == "__main__":
    cmip6_path = "/Volumes/share/Staff/mq41637526/Maina/habitat/parcels/cmip6"
    analyze_climate_corridors('competitive', cmip6_path)