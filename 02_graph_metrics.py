import os
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx

def calculate_metrics(group, base_dir='connectivity_matrix'):
    # 1. Load the data
    group_dir = os.path.join(base_dir, group)
    adj_path = os.path.join(group_dir, f"{group}_adjacency.npz")
    # We load the summary metrics created in the last step to get Reef IDs
    metrics_path = os.path.join(group_dir, f"{group}_summary_metrics.csv")
    
    adj_matrix = sparse.load_npz(adj_path)
    df = pd.read_csv(metrics_path)
    
    # 2. Source-Sink Metrics (Vectorized for speed)
    # Out-degree: Number of reefs this reef sends larvae to
    out_degree = np.diff(adj_matrix.tocsr().indptr)
    
    # In-degree: Number of reefs this reef receives larvae from
    in_degree = np.diff(adj_matrix.tocsc().indptr)
    
    # Strength (Flux totals)
    out_strength = np.array(adj_matrix.sum(axis=1)).flatten()
    in_strength = np.array(adj_matrix.sum(axis=0)).flatten()
    
    # 3. Source-Sink Ratio
    # A value > 1 indicates a net Source; < 1 indicates a net Sink
    with np.errstate(divide='ignore', invalid='ignore'):
        ss_ratio = out_strength / in_strength
        ss_ratio = np.nan_to_num(ss_ratio, nan=0, posinf=out_strength.max())

    # 4. Integrate into DataFrame
    df['out_degree'] = out_degree
    df['in_degree'] = in_degree
    df['in_strength'] = in_strength
    df['ss_ratio'] = ss_ratio
    
    # Classify the Reef
    df['role'] = 'Neutral'
    df.loc[(df['ss_ratio'] > 1.2), 'role'] = 'Source'
    df.loc[(df['ss_ratio'] < 0.8) & (df['in_strength'] > 0), 'role'] = 'Sink'
    
    # 5. Graph Theory Metrics (using NetworkX)
    # Note: For very large networks, Betweenness Centrality can be slow
    G = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    
    # PageRank: Importance based on the importance of incoming connections
    pagerank = nx.pagerank(G, weight='weight')
    df['pagerank'] = df.index.map(pagerank)
    
    # 6. Save expanded metrics
    output_path = os.path.join(group_dir, f"{group}_graph_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Graph metrics saved to {output_path}")

if __name__ == "__main__":
    calculate_metrics('competitive')