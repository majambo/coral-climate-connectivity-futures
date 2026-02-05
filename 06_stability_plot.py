import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_scenario_comparison(group, base_dir='connectivity_matrix'):
    # 1. Load the master trajectory table
    group_dir = os.path.join(base_dir, group)
    input_path = os.path.join(group_dir, f"{group}_refugia_trajectories_master.csv")
    
    if not os.path.exists(input_path):
        print("❌ Error: Master trajectory file not found.")
        return

    df = pd.read_csv(input_path)

    # 2. Aggregate average quality for the year 2090 across both SSPs
    # We only care about donor candidates (Quality > 0)
    comparison = df[df['Quality_ssp585_2020'] > 0].groupby('ECOREGION')[['Quality_ssp245_2090', 'Quality_ssp585_2090']].mean()
    
    # Sort by the higher scenario to keep the plot organized
    comparison = comparison.sort_values('Quality_ssp245_2090', ascending=True)

    # 3. Create the Plot
    plt.figure(figsize=(12, 10))
    
    # Define Y-axis positions
    y_range = range(len(comparison))
    
    # Draw the connecting lines (The 'Dumbbell')
    plt.hlines(y=y_range, xmin=comparison['Quality_ssp585_2090'], xmax=comparison['Quality_ssp245_2090'], 
               color='grey', alpha=0.5, linewidth=1, zorder=1)
    
    # Plot SSP245 points (Moderate)
    plt.scatter(comparison['Quality_ssp245_2090'], y_range, color='skyblue', label='SSP2-4.5 (2090)', s=80, zorder=2, edgecolors='k')
    
    # Plot SSP585 points (High)
    plt.scatter(comparison['Quality_ssp585_2090'], y_range, color='crimson', label='SSP5-8.5 (2090)', s=80, zorder=2, edgecolors='k')

    # 4. Formatting
    plt.yticks(y_range, comparison.index, fontsize=9)
    plt.title(f"Scenario Comparison: {group.capitalize()} Refugia Quality (2090)", fontsize=14)
    plt.xlabel("Refugia Quality Score", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Highlight "Disappearing" Refugia
    # These are regions where the score drops below a critical threshold (e.g., 0.3) in SSP585
    threshold = 0.3
    plt.axvline(x=threshold, color='red', linestyle=':', alpha=0.6)
    plt.text(threshold + 0.01, -1, "Survival Threshold", color='red', verticalalignment='bottom', fontsize=10)

    # Save
    output_img = os.path.join(group_dir, f"{group}_scenario_comparison.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Scenario comparison plot saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_scenario_comparison('competitive')