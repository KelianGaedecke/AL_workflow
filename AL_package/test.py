import sys
import os
import numpy as np
from Query_strategies.queries import random_query, unc_calc, most_unc_query, query_balanced_samples, greedy_query
from models.models import MolecularModel  
import pandas as pd


df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_workflow/AL_package/data/qm9_10000_data.csv")
#print(len(df)*0.0045)

target_columns = ["gap"]
X_pool = df.loc[:,'SMILES'].values
y_pool = df.loc[:,target_columns].values

# Initialize the model
n_models = 3  
model = MolecularModel(n_models=n_models, X_pool = X_pool, y_pool = y_pool, iter_per_group=2)

print(f"Initial pool size: {len(model.X_pool)}")
print(f"Initial pool ys: {len(model.y_pool)}")

model.start(ini_batch = 100)#, mod = 'representation')
model.train(num_iters=9, query_fn=most_unc_query, batch_size=30, train_type='ext',use_uncertainty = False)


print(f"Remaining pool size: {len(model.X_pool)}")
print(f"Remaining pool ys: {len(model.y_pool)}")
print(model.evaluate())

print("THE END")
