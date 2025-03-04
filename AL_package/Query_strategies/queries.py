import numpy as np
import torch
from models.models import MolecularModel 
import random

"""
The query functions must return the indices from the self.remaining indices list that must be probed.
"""

def random_query(X, y, cluster_labels=None, difficulty_label=None, batch_size=10, target_label=0, model: MolecularModel=None, use_uncertainty=False):
    """
    Randomly select `batch_size` samples from the pool.
    """
    queried_indices = np.random.choice(len(X), size=batch_size, replace=False)
    return queried_indices

def unc_calc(model: MolecularModel):
    """
    Calculate uncertainty by computing variance over ensemble predictions.
    """
    predictions, var = model.predict()
    return var

def most_unc_query(X, y, cluster_labels=None, difficulty_label=None, batch_size=10, target_label=0, model: MolecularModel=None, use_uncertainty=False):
    """
    Query the samples with the highest uncertainty
    """
    var = unc_calc(model)

    ### Get the indices of the var vector with the highest ensemble variances/uncertainties
    sorted_indices_relative = torch.argsort(var, descending=True)[:batch_size]

    return sorted_indices_relative


#def most_unc_query(X, y, cluster_labels=None, difficulty_label=None, batch_size=10, target_label=0, model: MolecularModel=None, use_uncertainty=False):
#    """
#    Query the samples with the highest uncertainty
#    """
#    var = unc_calc(model)
#    queried_indices = torch.argsort(var, descending=True)[:batch_size]
#    print("CHECK INDICES", queried_indices)
#    return queried_indices



def greedy_query(X, y, cluster_labels=None, difficulty_label=None, batch_size=10, target_label=0, model: MolecularModel=None, use_uncertainty=False):
    """
    Query the samples with the best predicted properties
    """
    pred, var = model.predict()
    print("VAR:",var[:5])
    print("PRED SHAPE", pred.shape)
    print("5 first predictions", pred[:5])
    sorted_indices_relative = torch.argsort(pred, descending=False)[:batch_size]
    print("SORTED INDICES", sorted_indices_relative)
    return sorted_indices_relative


def query_balanced_samples(X, y, cluster_labels=None, difficulty_label=None, batch_size=10, target_label=0, model: MolecularModel=None, use_uncertainty=False):
    """
    Query approximately equal samples from each cluster with difficulty_label
    If use_uncertainty=True, selects the most uncertain samples within each cluster
    """
    if cluster_labels is None or difficulty_label is None:
        return random_query(X, y, batch_size=batch_size, model=model) 
    
    difficulty_indices = np.where(np.array(difficulty_label) == target_label)[0]
    cluster_dict = {}
    var = unc_calc(model) if use_uncertainty else None
    
    for idx in difficulty_indices:
        cluster = cluster_labels[idx]
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(idx)
    
    n_clusters = len(cluster_dict)

    samples_per_cluster = batch_size // n_clusters  
    remainder = batch_size % n_clusters  
    
    queried_indices = []
    
    for cluster, indices in cluster_dict.items():
        if use_uncertainty:
            sorted_indices = sorted(indices, key=lambda i: var[i], reverse=True)
        else:
            sorted_indices = random.sample(indices, len(indices)) 
        
        n_samples = min(samples_per_cluster, len(sorted_indices))
        sampled = sorted_indices[:n_samples] 
        queried_indices.extend(sampled)
    
    if remainder > 0:
        extra_samples = np.random.choice(difficulty_indices, size=remainder, replace=False).tolist()
        queried_indices.extend(extra_samples)
    
    return queried_indices