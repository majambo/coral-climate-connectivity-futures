import os
import pandas as pd

def create_ecoregion_delta(group, base_dir='connectivity_matrix'):
    # 1. Load the trajectory master table created in the previous step
    group_dir = os.path.join(base_dir, group)
    input_path = os.path.join(group_dir, f"{group}_refugia_trajectories_master.csv")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run the trajectory script first.")
        return

    df = pd.read_csv(input_path)

    # 2. Aggregate Quality by Ecoregion for Start and End points
    # We use SSP585 as the stress-test scenario
    start_col = 'Quality_ssp585_2020'
    end_col = 'Quality_ssp585_2090'

    # Filter out rows where quality is 0 (non-donors) to get true regional potential
    eco_summary = df[df[start_col] > 0].groupby('ECOREGION')[[start_col, end_col]].mean()

    # 3. Calculate Ranks
    # rank(ascending=False) gives the #1 spot to the highest quality
    eco_summary['Rank_2020'] = eco_summary[start_col].rank(ascending=False, method='min')
    eco_summary['Rank_2090'] = eco_summary[end_col].rank(ascending=False, method='min')

    # 4. Calculate the Delta (Rank Change)
    # NOTE: Since Rank 1 is better than Rank 10, (Rank_2020 - Rank_2090) 
    # results in a positive number if an ecoregion moves UP the list.
    eco_summary['Rank_Delta'] = eco_summary['Rank_2020'] - eco_summary['Rank_2090']
    
    # 5. Calculate Absolute Quality Change (Degradation)
    eco_summary['Quality_Retention'] = (eco_summary[end_col] / eco_summary[start_col]) * 100

    # 6. Sort by Delta to see the "Winners" (those that gained the most relative importance)
    eco_summary = eco_summary.sort_values('Rank_Delta', ascending=False)

    # 7. Export Results
    output_path = os.path.join(group_dir, f"{group}_ecoregion_rank_delta.csv")
    eco_summary.to_csv(output_path)

    print(f"✅ Delta CSV saved to: {output_path}")
    print("\n--- Top 5 'Climatic Winners' (Gained most rank positions) ---")
    print(eco_summary[['Rank_2020', 'Rank_2090', 'Rank_Delta']].head(5))

    print("\n--- Top 5 'Climatic Losers' (Dropped most rank positions) ---")
    print(eco_summary[['Rank_2020', 'Rank_2090', 'Rank_Delta']].tail(5))

if __name__ == "__main__":
    create_ecoregion_delta('competitive')