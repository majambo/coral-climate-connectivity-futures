import os
import pandas as pd
import numpy as np

def calculate_all_refugia_trajectories(group, base_dir='connectivity_matrix', weight_bio=0.5, weight_climate=0.5):
    # 1. Paths
    group_dir = os.path.join(base_dir, group)
    input_path = os.path.join(group_dir, f"{group}_integrated_climate_connectivity_ecoregions.csv")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    # 2. Load Data
    df = pd.read_csv(input_path)
    
    # 3. Identify Baseline and Future Columns
    baseline_col = 'baseline_lttmax_2000_2019'
    # Find all columns that look like 'thetao_ltmax_ssp...'
    future_cols = [c for c in df.columns if 'thetao_ltmax_ssp' in c]
    future_cols = sorted(future_cols) # Ensures chronological order

    print(f"Analyzing {len(future_cols)} scenarios for {group}...")

    # 4. Pre-calculate Biological Score (Static across time)
    # Filter for sources first
    is_source = df['ss_ratio'] > 1.0
    # Threshold: coolest 25% at baseline
    temp_threshold = df[baseline_col].quantile(0.25)
    is_cool_baseline = df[baseline_col] <= temp_threshold
    
    # Static Bio Score
    min_log_ss = np.log1p(df['ss_ratio'].min())
    max_log_ss = np.log1p(df['ss_ratio'].max())
    df['score_bio'] = (np.log1p(df['ss_ratio']) - min_log_ss) / (max_log_ss - min_log_ss)

    # 5. Loop through every Scenario and Decade
    for f_col in future_cols:
        scenario_label = f_col.replace('thetao_ltmax_', '') # e.g., ssp585_2040
        
        # A. Warming Delta for this specific decade
        warming_delta = df[f_col] - df[baseline_col]
        
        # B. Climate Score (1 = least warming, 0 = most warming)
        # We normalize based on the range of warming WITHIN this specific decade
        d_min, d_max = warming_delta.min(), warming_delta.max()
        score_climate = 1 - ((warming_delta - d_min) / (d_max - d_min))
        
        # C. Calculate Refugia Quality for this decade
        quality_col = f"Quality_{scenario_label}"
        df[quality_col] = (df['score_bio'] * weight_bio) + (score_climate * weight_climate)
        
        # D. Apply the "Donor Candidate" mask (Must be a source and cool at baseline)
        # Reefs that don't meet the baseline "cool source" criteria get a 0
        df.loc[~(is_source & is_cool_baseline), quality_col] = 0

    # 6. Export the Master Trajectory Table
    output_path = os.path.join(group_dir, f"{group}_refugia_trajectories_master.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Master trajectory table saved.")
    
    # 7. Quick Comparison: 2020 vs 2090 (SSP585)
    print("\n--- Top 5 Ecoregions: 2020 vs 2090 (SSP585) ---")
    rank_2020 = df.groupby('ECOREGION')['Quality_ssp245_2020'].mean().sort_values(ascending=False).head(5)
    rank_2090 = df.groupby('ECOREGION')['Quality_ssp245_2090'].mean().sort_values(ascending=False).head(5)
    
    print("\n[Rank in 2020]")
    print(rank_2020)
    print("\n[Rank in 2090]")
    print(rank_2090)

if __name__ == "__main__":
    calculate_all_refugia_trajectories('competitive')