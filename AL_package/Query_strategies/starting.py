import numpy as np
import torch

def query_balanced_samples(X_pool, cluster_labels, difficulty_label, ini_batch, target_label):
    """
    Query approximately equal samples from each cluster with difficulty_label

    Parameters:
    - X_pool: Dataset (not used for querying, but provides length reference)
    - cluster_labels: Array of cluster assignments for each sample
    - difficulty_label: List of groups (e.g., 0,1,2,3,...) assigned to each sample
    - ini_batch: Total number of samples to query

    Returns:
    - queried_indices: List of selected sample indices
    """

    difficulty_indices = np.where(np.array(difficulty_label) == target_label)[0]

    cluster_dict = {}
    for idx in difficulty_indices:
        cluster = cluster_labels[idx]
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(idx)

    n_clusters = len(cluster_dict)
    samples_per_cluster = ini_batch // n_clusters  
    remainder = ini_batch % n_clusters  

    queried_indices = []

    for cluster, indices in cluster_dict.items():
        n_samples = min(samples_per_cluster, len(indices))
        sampled = np.random.choice(indices, size=n_samples, replace=False).tolist()
        queried_indices.extend(sampled)

    if remainder > 0:
        extra_samples = np.random.choice(difficulty_indices, size=remainder, replace=False).tolist()
        queried_indices.extend(extra_samples)

    return queried_indices