import numpy as np
import torch
import pandas as pd
from models.models_test import MolecularModel
from Query_strategies.queries import query_balanced_samples

df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_REPO/data/qm9_1000_data.csv")


target_columns = ["gap"]
X_pool = df.loc[:,'SMILES'].values
y_pool = df.loc[:,target_columns].values

model = MolecularModel(n_models=2, X_pool=X_pool, y_pool=y_pool, difficulty_metric="silhouette")

# Run a simple training loop
print("\n========== STARTING TEST TRAINING ==========")
model.train(num_iters=3, batch_size=5)

# Check updated difficulty metrics
print("\n========== TESTING DIFFICULTY UPDATES ==========")
print("Updated Uncertainty Values:", model.uncertainty_vals)
print("Updated Loss Values:", model.loss_vals)

# Check if training loss history is recorded
print("\n========== TRAINING LOSS HISTORY ==========")
print(model.loss_history)
