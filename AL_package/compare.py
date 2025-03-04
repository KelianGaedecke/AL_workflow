import torch
import os
import shutil
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.functional import l1_loss
import matplotlib.pyplot as plt
from chemprop import data, featurizers, models, nn
import astartes as ast
from utils.cluster import (
    molecular_fingerprints, determine_optimal_clusters,
    cluster_fingerprints, divide_into_groups_based_on_score
)
from Query_strategies.starting import query_balanced_samples
import sys
from Query_strategies.queries import random_query, unc_calc, most_unc_query, query_balanced_samples, greedy_query
from models.models import MolecularModel 
import pandas as pd

df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_workflow/AL_package/data/qm9_1000_data.csv")

def run_experiment(query_fn, num_iters=2, batch_size=10, top_k_percent=10):
    """Run an experiment with a given query function and return metrics."""

    df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_workflow/AL_package/data/qm9_1000_data.csv")

    target_columns = ["gap"]
    X_pool = df.loc[:,'SMILES'].values
    y_pool = df.loc[:,target_columns].values

    model = MolecularModel(n_models=3, X_pool=X_pool, y_pool=y_pool, iter_per_group=1)
    
    model.start(ini_batch=10)
    model.train(query_fn=query_fn, num_iters=num_iters, batch_size=batch_size, train_type='mix', use_uncertainty=True)
    
    top_k_fractions = model.top_k_history
    test_losses = model.loss_history_plot

    return top_k_fractions, test_losses

def plot_results(results, query_names, top_k_percent):
    """Plot top-k fractions and test losses."""
    plt.figure(figsize=(12, 6))
    
    # Plotting Top K Fractions
    plt.subplot(1, 2, 1)
    for name, (top_k_fractions, _) in results.items():
        # Extract the fractions from the tuples in the list
        fractions = [f for f in top_k_fractions]
        plt.plot(fractions, label=name)
    plt.xlabel("Iteration")
    plt.ylabel(f"Fraction of Top {top_k_percent}%")
    plt.title(f"Top {top_k_percent}% Overlap vs. Iteration")
    plt.legend()

    # Plotting Test Losses
    plt.subplot(1, 2, 2)
    for name, (_, test_losses) in results.items():
        # Extract the losses from the tuples in the list
        losses = [l for l in test_losses]
        plt.plot(losses, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs. Iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query_functions = {
        "Greedy Query": greedy_query,
        "Uncertainty Query": most_unc_query,
        "Random Query": random_query,
        "Balanced Query": query_balanced_samples
    }

    results = {}
    top_k_percent = 10
    num_iters = 2
    batch_size = 10

    for name, query_fn in query_functions.items():
        top_k_fractions, test_losses = run_experiment(query_fn, num_iters, batch_size, top_k_percent)
        results[name] = (top_k_fractions, test_losses)

    plot_results(results, query_functions.keys(), top_k_percent)
