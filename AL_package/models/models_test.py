import torch
import os
import shutil
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.stats import norm
from chemprop import data, featurizers, models, nn
from utils.cluster import (
    molecular_fingerprints, determine_optimal_clusters, 
    cluster_fingerprints, divide_into_groups_based_on_score
)
from Query_strategies.starting import query_balanced_samples


class MolecularModel:
    def __init__(
        self, 
        n_models, 
        X_pool, 
        y_pool,  
        scaler=None, 
        checkpoint_dir="checkpoints/",
        difficulty_metric="silhouette"
    ):
        """
        Initialize the ensemble of models and store the dataset.
        """
        # Training parameters
        self.iter = 0
        self.checkpoint_dir = checkpoint_dir
        self.difficulty_metric = difficulty_metric  # "silhouette", "uncertainty", "loss"

        # Ensure checkpoint directory is reset
        self._setup_checkpoints()

        # Dataset & labels
        self.scaler = scaler  
        self.X_og_pool = X_pool
        self.y_og_pool = y_pool
        self.X_pool = X_pool 
        self.y_pool = y_pool  

        self._compute_cluster_info()

        self.ensemble = self._initialize_ensemble(n_models)


    def _setup_checkpoints(self):
        """Ensure the checkpoint directory is clean"""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir) 
        os.makedirs(self.checkpoint_dir) 

    def _compute_cluster_info(self):
        """Compute molecular fingerprints and clustering labels"""
        self.fingerprints = molecular_fingerprints(self.X_pool, radius=2, n_bits=2048)
        num_cluster = determine_optimal_clusters(self.fingerprints, max_clusters=10)
        self.cluster_labels, self.silhouette_vals, self.kmeans = cluster_fingerprints(self.fingerprints, num_cluster)

        self.uncertainty_vals = np.ones(len(self.X_pool))  
        self.loss_vals = np.ones(len(self.X_pool))  

        print("DIFFICULTY MEASURES INITIALIZED.")

    def start(self, ini_batch=10, mod="random"):
        """Initial dataset selection and model training"""
        


        self.iter += 1



    def _initialize_ensemble(self, n_models):
        """Initialize ensemble of models"""
        ensemble = []
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler) if self.scaler else None
        ffn = nn.RegressionFFN(output_transform=output_transform)
        batch_norm = True
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

        for _ in range(n_models):
            ensemble.append(models.MPNN(mp, agg, ffn, batch_norm, metric_list))
        return ensemble

    def _train_model(self, model, train_loader, model_idx):
        """Helper function to train a single model"""
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    filename=f"model-iter={self.iter + 1}-model={model_idx}",  
                    save_last=True
                )
            ]
        )
        trainer.fit(model, train_loader)

    def _prepare_training_data(self, queried_indices):
        """Prepare training data from selected samples"""
        X_train = [self.X_pool[i] for i in queried_indices]
        y_train = [self.y_pool[i] for i in queried_indices]

        self.X_pool = np.delete(self.X_pool, queried_indices, axis=0).tolist()
        self.y_pool = np.delete(self.y_pool, queried_indices, axis=0).tolist()
        self.cluster_labels = np.delete(self.cluster_labels, queried_indices, axis=0).tolist()
        self.silhouette_vals = np.delete(self.silhouette_vals, queried_indices, axis=0).tolist()
        self.uncertainty_vals = np.delete(self.uncertainty_vals, queried_indices, axis=0).tolist()
        self.loss_vals = np.delete(self.loss_vals, queried_indices, axis=0).tolist()
        self.fingerprints = np.delete(self.fingerprints, queried_indices, axis=0).tolist()

        train_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(X_train, y_train)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)
        train_loader = data.build_dataloader(train_dset, num_workers=0)

        self.scaler = train_dset.normalize_targets()

        return train_loader

    def _compute_cdf(self, values):
        """Compute CDF for a difficulty metric"""
        sorted_vals = np.sort(values)
        cdf = np.searchsorted(sorted_vals, values, side="right") / len(values)
        return cdf

    def _update_difficulty_metrics(self):
        """Update difficulty metrics after each training step"""
        _, self.uncertainty_vals = self.predict()  
        self.loss_vals = np.array(self.loss_vals) / np.max(self.loss_vals) 

    def _select_training_samples(self, batch_size):
        """Select training samples based on difficulty metric"""
        if self.difficulty_metric == "silhouette":
            difficulty_scores = self.silhouette_vals
        elif self.difficulty_metric == "uncertainty":
            difficulty_scores = self.uncertainty_vals
        elif self.difficulty_metric == "loss":
            difficulty_scores = self.loss_vals
        else:
            raise ValueError("Invalid difficulty metric")

        cdf_vals = self._compute_cdf(difficulty_scores)
        selected_indices = np.argsort(cdf_vals)[:batch_size]
        return selected_indices.tolist()

    def train(self, num_iters=4, batch_size=10):
        """Train the model iteratively using active learning"""
        if self.iter == 0:
            self.start(batch_size)

        for _ in range(num_iters):
            if len(self.X_pool) < batch_size:
                print("Not enough data left in the pool for another iteration.")
                break

            queried_indices = self._select_training_samples(batch_size)

            train_loader = self._prepare_training_data(queried_indices)

            for model_idx, model in enumerate(self.ensemble):
                if self.iter > 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"model-iter={self.iter}-model={model_idx}.ckpt")
                    print(f"Loading checkpoint for model {model_idx} from: {checkpoint_path}")
                    model = models.MPNN.load_from_checkpoint(checkpoint_path)

                self._train_model(model, train_loader, model_idx)

            self.iter += 1  

            self._update_difficulty_metrics()

            print(f"ITER {self.iter} completed.")

 
