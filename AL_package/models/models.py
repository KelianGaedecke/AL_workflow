import torch
import os
import shutil
import random
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
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
        iter_per_group=None, 
        n_groups=3, 
        scaler=None, 
        checkpoint_dir="checkpoints/"
    ):
        """
        Initialize the ensemble of models and store the dataset
        """
        # Training parameters
        self.iter = 0
        self.n_groups = n_groups
        self.target_label = 0
        self.iter_per_group = iter_per_group if iter_per_group else n_groups
        self.checkpoint_dir = checkpoint_dir

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
        """Check the checkpoint directory is clean"""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)

    def _compute_cluster_info(self):
        """Compute molecular fingerprints and clustering labels"""
        self.fingerprints = molecular_fingerprints(self.X_pool, radius=2, n_bits=2048)
        num_cluster = determine_optimal_clusters(self.fingerprints, max_clusters=10)
        self.cluster_labels, self.silhouette_vals, self.kmeans = cluster_fingerprints(self.fingerprints, num_cluster)
        self.difficulty_label = divide_into_groups_based_on_score(self.silhouette_vals, self.n_groups)
        print("DIFFICULTY INDICES:", self.difficulty_label)

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
        self.difficulty_label = np.delete(self.difficulty_label, queried_indices, axis=0).tolist()
        self.fingerprints = np.delete(self.fingerprints, queried_indices, axis=0).tolist()

        train_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(X_train, y_train)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)
        train_loader = data.build_dataloader(train_dset, num_workers=0)

        self.scaler = train_dset.normalize_targets()

        return train_loader

    def start(self, ini_batch=10, mod="random"):
        """Initial dataset selection and model training"""
        queried_indices = (
            np.random.choice(len(self.X_pool), size=ini_batch, replace=False)
            if mod == "random"
            else query_balanced_samples(self.X_pool, self.cluster_labels, self.difficulty_label, ini_batch, target_label=0)
        )

        train_loader = self._prepare_training_data(queried_indices)

        for model_idx, model in enumerate(self.ensemble):
            self._train_model(model, train_loader, model_idx)

        self.iter += 1
    

    def train(self, query_fn, num_iters=4, batch_size=10, train_type="new", use_uncertainty=False):
        """Train the model iteratively using active learning and plot training loss"""
        if self.iter == 0:
            self.start(batch_size)
    
        X_train, y_train = [], []  # Track training data if train_type="ext"
        self.loss_history = []  # Store loss history
    
        for _ in range(num_iters):
            if len(self.X_pool) < batch_size:
                print("Not enough data left in the pool for another iteration.")
                break
    
            queried_indices = query_fn(
                self.X_pool, self.y_pool, self.cluster_labels, 
                self.difficulty_label, batch_size, self.target_label, model=self, use_uncertainty=use_uncertainty
            )
    
            # Retrieve new samples
            X_new = [self.X_pool[i] for i in queried_indices]
            y_new = [self.y_pool[i] for i in queried_indices]
    
            # Handle train_type: either overwrite or extend training set
            if train_type == "new":
                X_train = X_new
                y_train = y_new
            elif train_type == "ext":
                X_train.extend(X_new)
                y_train.extend(y_new)
    
            # Prepare data loader
            train_loader = self._prepare_training_data(queried_indices)
    
            # Store model losses for this iteration
            iteration_losses = []
    
            # Train models
            for model_idx, model in enumerate(self.ensemble):
                if self.iter > 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"model-iter={self.iter}-model={model_idx}.ckpt")
                    print(f"Loading checkpoint for model {model_idx} from: {checkpoint_path}")
                    model = models.MPNN.load_from_checkpoint(checkpoint_path)
    
                # Train model
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
    
                # Extract training loss
                loss = trainer.callback_metrics.get("train_loss_epoch", None)
                if loss is not None:
                    iteration_losses.append(loss.item())
    
            # Compute average and standard deviation of losses
            if iteration_losses:
                avg_loss = np.mean(iteration_losses)
                std_loss = np.std(iteration_losses)
                self.loss_history.append((self.iter, avg_loss, std_loss))
    
            self.iter += 1
            self.target_label = self.iter // self.iter_per_group
            if self.target_label > self.n_groups:
                self.target_label = self.n_groups


    
            print(f"ITER {self.iter} completed.")
            print(f"CURRENT CLUSTER GROUP INVESTIGATED {self.target_label}")
    
        # Plot training loss
        self._plot_training_loss()
    
    def _plot_training_loss(self):
        """Plot the training loss with standard deviation"""
        if not self.loss_history:
            print("No loss history recorded.")
            return
    
        iterations, avg_losses, std_losses = zip(*self.loss_history)
    
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, avg_losses, label="Avg Training Loss", color="blue")
        plt.fill_between(iterations, 
                         np.array(avg_losses) - np.array(std_losses), 
                         np.array(avg_losses) + np.array(std_losses), 
                         color="blue", alpha=0.2, label="Std Dev")
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Trend")
        plt.legend()
        plt.show()



    def predict(self):
        """Make predictions using the trained ensemble models"""
        pool_data = [data.MoleculeDatapoint.from_smi(x, 0) for x in self.X_pool]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)

        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            model_predictions = trainer.predict(model, prediction_dataloader)
            all_predictions.append(torch.concat(model_predictions))

        predictions = torch.mean(torch.stack(all_predictions), dim=0)
        variance = torch.var(torch.stack(all_predictions), dim=0).squeeze()

        return predictions, variance


    def evaluate(self, X_test, y_test):
        """Evaluate the model on the given test data."""

