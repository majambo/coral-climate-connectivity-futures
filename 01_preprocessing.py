import sys
import os
import glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
VALID_GROUPS = ['competitive', 'stress_tolerant', 'weedy']
DEFAULT_DATA_FOLDER = 'output_final'
SETTLEMENT_FLAG = 6  # Deletion reason 6 = Settled

def find_zarr_files(data_folder, group):
    """Broad search to ensure we catch your specific structure"""
    search_patterns = [
        os.path.join(data_folder, group, "*.zarr"),
        os.path.join(data_folder, f"{group}_*", "*.zarr"),
        os.path.join(data_folder, "**", f"{group}_*.zarr")
    ]
    zarr_files = []
    for pattern in search_patterns:
        zarr_files.extend(glob.glob(pattern, recursive=True))
    return sorted(list(set([f for f in zarr_files if os.path.isdir(f)])))

def run_preprocessing(group, data_folder=DEFAULT_DATA_FOLDER):
    print(f"\n{'='*60}\nPROCESSING GROUP: {group.upper()}\n{'='*60}")
    
    zarr_files = find_zarr_files(data_folder, group)
    if not zarr_files:
        print(f"❌ No files found in {data_folder} for group {group}")
        return

    # 1. DISCOVER ALL REEFS (The 'Original' Reliability)
    print("Step 1: Identifying unique Reef IDs across all files...")
    all_reefs_set = set()
    for f in zarr_files:
        with xr.open_zarr(f, consolidated=False) as ds:
            # Source reef is at observation 0
            src = ds['source_reef'].isel(obs=0).values.astype(np.int32)
            # Settled contains the destination IDs
            dst = ds['settled'].values.flatten().astype(np.int32)
            all_reefs_set.update(np.unique(src))
            all_reefs_set.update(np.unique(dst[dst > 0]))

    all_reefs = sorted(list(all_reefs_set))
    reef_to_idx = {reef_id: idx for idx, reef_id in enumerate(all_reefs)}
    n_reefs = len(all_reefs)
    print(f"Found {n_reefs} unique reefs.")

    # 2. BUILD MATRIX (Optimized Sparse Accumulation)
    print("\nStep 2: Processing settlement data into sparse matrix...")
    adj_matrix = sparse.lil_matrix((n_reefs, n_reefs), dtype=np.float64)
    total_settlements = 0

    for f in zarr_files:
        try:
            with xr.open_zarr(f, consolidated=False) as ds:
                deletion = np.nanmax(ds['deletion_reason'].values, axis=1).astype(np.int32)
                settled = np.nanmax(ds['settled'].values, axis=1).astype(np.int32)
                source_reef = ds['source_reef'].isel(obs=0).values.astype(np.int32)
                
                # Filter for successful settlement
                mask = (deletion == SETTLEMENT_FLAG) & (settled > 0)
                if np.any(mask):
                    s_ids = source_reef[mask]
                    d_ids = settled[mask]
                    for s, d in zip(s_ids, d_ids):
                        adj_matrix[reef_to_idx[s], reef_to_idx[d]] += 1
                    total_settlements += np.sum(mask)
            print(f"  ✓ {os.path.basename(f)} ({np.sum(mask)} settled)")
        except Exception as e:
            print(f"  ✗ Error in {f}: {e}")

    # 3. CONVERT AND SAVE
    adj_matrix = adj_matrix.tocsr()
    output_dir = os.path.join('connectivity_matrix', group)
    os.makedirs(output_dir, exist_ok=True)
    
    sparse.save_npz(os.path.join(output_dir, f"{group}_adjacency.npz"), adj_matrix)
    
    # Save a CSV with basic metrics for confirmation
    metrics = pd.DataFrame({'Reef_ID': all_reefs, 'Out_Flux': np.array(adj_matrix.sum(axis=1)).flatten()})
    metrics.to_csv(os.path.join(output_dir, f"{group}_summary_metrics.csv"), index=False)

    print(f"\nSUCCESS: Files saved in {output_dir}")
    print(f"Total Settlements Recorded: {total_settlements:,}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=VALID_GROUPS)
    parser.add_argument('--data-folder', default=DEFAULT_DATA_FOLDER)
    args = parser.parse_args()
    run_preprocessing(args.group, args.data_folder)