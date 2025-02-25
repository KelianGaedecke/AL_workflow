import sys
import os
import numpy as np
from Query_strategies.queries import random_query, unc_calc, most_unc_query, query_balanced_samples
from models.models import MolecularModel  
import pandas as pd


# Generate synthetic data
#num_samples = 50  # Pool size
#X_pool = [f"C{n}" for n in range(num_samples)]  # Fake SMILES strings
#y_pool = np.random.rand(num_samples).tolist()  # Random target values

df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_REPO/data/qm9_1000_data.csv")


target_columns = ["gap"]
X_pool = df.loc[:,'SMILES'].values
y_pool = df.loc[:,target_columns].values

# Initialize the model
n_models = 3  
model = MolecularModel(n_models=n_models, X_pool = X_pool, y_pool = y_pool, iter_per_group=2)

print(f"Initial pool size: {len(model.X_pool)}")
print(f"Initial pool ys: {len(model.y_pool)}")

model.start(ini_batch = 10, mod = 'representation')
model.train(num_iters=5, query_fn=query_balanced_samples, batch_size=5, use_uncertainty = True)


#def train(self, query_fn, num_iters=4, ini_batch = 10, batch_size = 10, train_type = 'new', use_uncertainty = False)

# Print results
print(f"Remaining pool size: {len(model.X_pool)}")
print(f"Remaining pool ys: {len(model.y_pool)}")

print("THE END")