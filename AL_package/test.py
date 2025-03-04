import sys
import os
import numpy as np
from Query_strategies.queries import random_query, unc_calc, most_unc_query, query_balanced_samples, greedy_query
from models.models import MolecularModel 
import pandas as pd


df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_workflow/AL_package/data/qm9_1000_data.csv")

target_columns = ["gap"]
X_pool = df.loc[:,'SMILES'].values
y_pool = df.loc[:,target_columns].values

n_models = 2  
model = MolecularModel(n_models=n_models, X_pool = X_pool, y_pool = y_pool, iter_per_group=2)

print(f"Initial pool size: {len(model.X_pool)}")
print(f"Initial pool ys: {len(model.y_pool)}")

model.start(ini_batch = 10)#, mod = 'representation')
model.train(num_iters=2, query_fn=query_balanced_samples, batch_size=10, train_type='mix',use_uncertainty = True)


print(f"Remaining pool size: {len(model.X_pool)}")
print(f"Remaining pool ys: {len(model.y_pool)}")
print(model.evaluate())

model.get_top_k()


all_indices = np.concatenate([np.array(indices) for indices in model.queried_indices_history])

unique, counts = np.unique(all_indices, return_counts=True)
duplicates = unique[counts > 1]

if len(duplicates) > 0:
    print("Duplicate indices found:", duplicates)
else:
    print("No duplicates found.")

model._scatter_plot()

print("THE END")
