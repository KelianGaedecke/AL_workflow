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
import sys
from Query_strategies.queries import random_query, unc_calc, most_unc_query, query_balanced_samples, greedy_query
from models.models import MolecularModel 
import pandas as pd


l = [100, 500, 1000, 5000, 10000]
df = pd.read_csv("/Users/keliangaedecke/Desktop/MA_THESIS/code/AL_workflow/AL_package/data/qm9_20000_data.csv")
target_columns = ["gap"]

X_pool = df.loc[:,'SMILES'].values
y_pool = df.loc[:,target_columns].values

error = []

for i in range(len(l)):
    model = MolecularModel(n_models=3, X_pool = X_pool, y_pool = y_pool, iter_per_group=1)
    model.start(ini_batch = l[i])
    error.append(model.evaluate())

    print(error)


#DID WE JUST FORGET THE SCALER??????


plt.plot(l, error)
plt.xlabel("train size")
plt.ylabel("Loss")
plt.title("Learning curve")
plt.show()
