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


class MolecularModel:
    def __init__(
        self, 
        n_models, 
        X_pool, 
        y_pool, 
        iter_per_group=None, 
        n_groups=3, 
        scaler=None, 
        checkpoint_dir="checkpoints/",
        results_dir="results/"
    ):
        """
        Initialize the ensemble of models and store the dataset
        """

        self.iter = 0
        self.n_groups = n_groups
        self.target_label = 0
        self.iter_per_group = iter_per_group if iter_per_group else n_groups
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        self._setup_checkpoints()

        self.scaler = scaler 

        self.X_og_pool = X_pool
        self.y_og_pool = y_pool
        self.X_pool, self.X_val, self.X_test, self.y_pool, self.y_val, self.y_test = ast.train_val_test_split(
                X_pool,
                y_pool,
                train_size=0.7,
                val_size=0.2,
                test_size=0.1,
                sampler = "random"
        )
        ############# TEST ZONE #############

        train_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(self.X_pool, self.y_pool)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)

        #self.scaler = train_dset.normalize_targets()

        ############# TEST ZONE #############


        self._compute_cluster_info()
        self.ensemble = self._initialize_ensemble(n_models)
        self.queried_indices_history = []
        self.mask = np.ones(len(self.X_pool), dtype=bool)
        self.remaining_indices = np.where(self.mask)[0]

        self.top_k_history = []
        self.test_loss_history = []
        self.loss_history = []
        self.loss_history_plot = []



    def _setup_checkpoints(self):
        """Check the checkpoint directory is clean"""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir)
    
    def _setup_results_folder(self):
        """Ensure the results directory is clean"""
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

    def _compute_cluster_info(self):
        """Compute molecular fingerprints and clustering labels"""
        self.fingerprints = molecular_fingerprints(self.X_pool, radius=2, n_bits=2048)
        num_cluster = determine_optimal_clusters(self.fingerprints, max_clusters=10)
        self.cluster_labels, self.silhouette_vals, self.kmeans = cluster_fingerprints(self.fingerprints, num_cluster)
        self.difficulty_label = divide_into_groups_based_on_score(self.silhouette_vals, self.n_groups)

    def _initialize_ensemble(self, n_models):
        """Initialize ensemble of models"""
        ensemble = []
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        #output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler) if self.scaler else None
        #ffn = nn.RegressionFFN(output_transform=output_transform)
        ffn = nn.RegressionFFN()
        batch_norm = True
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

        for _ in range(n_models):
            ensemble.append(models.MPNN(mp, agg, ffn, batch_norm, metric_list))
        return ensemble

    def _train_model(self, model, train_loader, val_loader, model_idx):
        """Helper function to train a single model"""
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1,
            max_epochs=20,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    filename=f"model-iter={self.iter + 1}-model={model_idx}",  
                    save_last=True
                )
            ]
        )
        trainer.fit(model, train_loader, val_loader)

        checkpoint_path = os.path.join(self.checkpoint_dir, f"model-iter={self.iter + 1}-model={model_idx}.ckpt")
        print(f"Loading checkpoint for model {model_idx} from: {checkpoint_path}")
        self.ensemble[model_idx] = models.MPNN.load_from_checkpoint(checkpoint_path)

    def _prepare_training_data(self, queried_indices):
        """Prepare training data from selected samples"""

        if len(queried_indices) == 0 :
            print("WARNING: Queried indices are empty. No training data available.")
            return None 
        
        previous_mask = self.mask
        print(f"Queried indices: {queried_indices}")
        print(f"Mask before update (sum of True values): {np.sum(previous_mask)}")
        self.mask[queried_indices] = False

        if np.all(self.mask == previous_mask):
            print("WARNING: Masking has not changed. No training data available.")

        print(f"Mask after update (sum of True values): {np.sum(self.mask)}")

        ### get the corresponding X and y values of the queried indices
        X_train = [self.X_pool[i] for i in queried_indices]
        y_train = [self.y_pool[i] for i in queried_indices]
        ### update the remaining indices using the mask vector previously updated
        self.remaining_indices = np.where(self.mask)[0]

        train_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(X_train, y_train)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)
        train_loader = data.build_dataloader(train_dset, num_workers=0)

        ############# TEST ZONE #############

        #self.scaler = train_dset.normalize_targets()

        ############# TEST ZONE #############

        return train_loader 

    def start(self, ini_batch=10, mod="random"):
        """Initial dataset selection and model training"""
        queried_indices = np.array(
            np.random.choice(len(self.remaining_indices), size=ini_batch, replace=False)
            if mod == "random"
            else query_balanced_samples(self.remaining_indices, self.cluster_labels, self.difficulty_label, ini_batch, target_label=0)
        )

        queried_indices = self.remaining_indices[queried_indices]


        self.queried_indices_history.append(queried_indices)

        train_loader = self._prepare_training_data(queried_indices)

        val_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(self.X_val, self.y_val)]
        val_dset = data.MoleculeDataset(val_data, featurizers.SimpleMoleculeMolGraphFeaturizer())
        #val_dset.normalize_targets(self.scaler)
        val_loader = data.build_dataloader(val_dset, num_workers=0)

        for model_idx, model in enumerate(self.ensemble):
            self._train_model(model, train_loader, val_loader, model_idx)

        self.top_k_history.append(self.get_top_k())
        self.test_loss_history.append((self.iter, self.evaluate()))
        self.loss_history_plot.append(self.evaluate())
        self.iter += 1

    

    def train(self, query_fn, num_iters=4, batch_size=10, train_type="new", use_uncertainty=False):
        """Train the model iteratively using active learning and plot training loss"""
        if self.iter == 0:
            self.start(batch_size)

        self.loss_history = []  
        self.val_loss_history = []  
        self.test_loss_history = []
        self.loss_history_plot = []
    
        for _ in range(num_iters):
            if len(self.remaining_indices) < batch_size:
                print("Not enough data left in the pool for another iteration.")
                break
    
            target_in_remaining = np.array(query_fn(
                self.remaining_indices, 
                self.y_pool[self.remaining_indices], 
                self.cluster_labels[self.remaining_indices], 
                self.difficulty_label[self.remaining_indices], 
                batch_size,
                self.target_label, 
                model=self, 
                use_uncertainty=use_uncertainty
            ))

            queried_indices = self.remaining_indices[target_in_remaining]

            if train_type == "mix":
                all_historical_indices = np.concatenate(self.queried_indices_history)
                
                n = min(int(len(queried_indices)//2), len(all_historical_indices))  
                random_indices = np.random.choice(all_historical_indices, size=n, replace=False)

                
                all_indices = np.concatenate([queried_indices, random_indices])
            
                train_loader = self._prepare_training_data(all_indices)

            self.queried_indices_history.append(queried_indices)

            if train_type == "new":
                train_loader = self._prepare_training_data(queried_indices)
            if train_type == "ext":
                all_indices = np.concatenate([np.array(indices) for indices in self.queried_indices_history])
                train_loader = self._prepare_training_data(all_indices)

            if train_loader is None:
                print("Skipping iteration due to empty training dataset.")
                continue
    
            val_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(self.X_val, self.y_val)]
            val_dset = data.MoleculeDataset(val_data, featurizers.SimpleMoleculeMolGraphFeaturizer())
            val_loader = data.build_dataloader(val_dset, num_workers=0)
    
            iteration_losses = []
            iteration_val_losses = []

            for model_idx, model in enumerate(self.ensemble):
                if self.iter > 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"model-iter={self.iter}-model={model_idx}.ckpt")
                    print(f"Loading checkpoint for model {model_idx} from: {checkpoint_path}")
                    self.ensemble[model_idx] = models.MPNN.load_from_checkpoint(checkpoint_path)
                

                trainer = pl.Trainer(
                    logger=False,
                    enable_checkpointing=True,
                    enable_progress_bar=True,
                    accelerator="cpu",
                    devices=1,
                    max_epochs=20,
                    callbacks=[
                        ModelCheckpoint(
                            dirpath=self.checkpoint_dir,
                            filename=f"model-iter={self.iter + 1}-model={model_idx}", 
                            monitor="val_loss", 
                            mode="min",
                            save_last=True
                        )
                    ]
                )
                
                trainer.fit(model, train_loader, val_loader)

                checkpoint_path = os.path.join(self.checkpoint_dir, f"model-iter={self.iter + 1}-model={model_idx}.ckpt")
                print(f"Loading checkpoint for model {model_idx} from: {checkpoint_path}")
                self.ensemble[model_idx] = models.MPNN.load_from_checkpoint(checkpoint_path)

                loss = trainer.callback_metrics.get("train_loss_epoch", None)
                val_loss = trainer.callback_metrics.get("val_loss", None)

                print(f"train_loss: {loss}, val_loss: {val_loss}")

                if loss is not None:
                    iteration_losses.append(loss.item())
                if val_loss is not None:
                    iteration_val_losses.append(val_loss.item())
    
            if iteration_losses:
                print("ITERATION LOSSES", iteration_losses)
                avg_loss = np.mean(iteration_losses)
                std_loss = np.std(iteration_losses)
                self.loss_history.append((self.iter, avg_loss, std_loss))
    
            if iteration_val_losses:
                print("ITERATION VAL LOSSES", iteration_val_losses)
                avg_val_loss = np.mean(iteration_val_losses)
                std_val_loss = np.std(iteration_val_losses)
                self.val_loss_history.append((self.iter, avg_val_loss, std_val_loss))
            

            self.top_k_history.append(self.get_top_k())
            self.loss_history_plot.append(self.evaluate())
            self.test_loss_history.append((self.iter, self.evaluate()))

            self.iter += 1
            self.target_label = self.iter // self.iter_per_group
            if self.target_label > self.n_groups - 1:
                self.target_label = self.n_groups -1
    

            results_txt_path = os.path.join(self.results_dir, "training_results.txt")

            with open(results_txt_path, "a") as f:
                f.write(f"{self.iter}, {loss:.4f}, {val_loss:.4f} \n")

    
        self._plot_training_and_validation_loss()
    
    def _plot_training_and_validation_loss(self):
        """Plot the training and validation loss with standard deviation"""
        if not self.loss_history or not self.val_loss_history:
            print("No loss history recorded.")
            return
    
        iterations, avg_losses, std_losses = zip(*self.loss_history)
        val_iterations, avg_val_losses, std_val_losses = zip(*self.val_loss_history) # laaaaa
        error_iterations, avg_test_losses = zip(*self.test_loss_history)


    
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, avg_losses, label="Avg Training Loss", color="blue")
        plt.fill_between(iterations, 
                         np.array(avg_losses) - np.array(std_losses), 
                         np.array(avg_losses) + np.array(std_losses), 
                         color="blue", alpha=0.2, label="Train Std Dev")
        plt.plot(error_iterations, avg_test_losses, color="orange", label="Test Loss")
        
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Trend")
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "loss_plot.png"))
        plt.show()


    def predict(self):
        """Make predictions using the trained ensemble models"""

        pool_data = [data.MoleculeDatapoint.from_smi(self.X_pool[i], 0) for i in self.remaining_indices]
        
        assert len(pool_data) == len(self.remaining_indices), "Mismatch in selected molecules!"
        
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)
    
        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            
            model_predictions = trainer.predict(model, prediction_dataloader)
            
            all_predictions.append(torch.concat(model_predictions))
    
        predictions = torch.mean(torch.stack(all_predictions), dim=0).squeeze()
        variance = torch.var(torch.stack(all_predictions), dim=0).squeeze()
    
        return predictions, variance 
    

    def evaluate(self):
        """Evaluate the model on the given test data"""
        pool_data = [data.MoleculeDatapoint.from_smi(x, 0) for x in self.X_test]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)
    
        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            model_predictions = trainer.predict(model, prediction_dataloader)
            all_predictions.append(torch.concat(model_predictions))
    
        predictions = torch.mean(torch.stack(all_predictions), dim=0).squeeze()
    

        error = l1_loss(predictions, torch.tensor(self.y_test, dtype=torch.float32).squeeze()).item()

        return error 


    def get_top_k(self, top_k_percent=10):
        """Evaluate the model on the given test data"""
        pool_data = [data.MoleculeDatapoint.from_smi(x, 0) for x in self.X_og_pool]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)

        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            model_predictions = trainer.predict(model, prediction_dataloader)
            all_predictions.append(torch.concat(model_predictions))

        predictions = torch.mean(torch.stack(all_predictions), dim=0).squeeze()
        
        print(f"POOL DATA LEN: {len(predictions)}")
        percent = len(predictions*top_k_percent)//100
        print(f"Percent: {percent}")
        top_pred_indices = np.argsort(predictions)[:percent]
        print("TOP PRED",top_pred_indices)
        
        true_value = self.y_og_pool.squeeze()

        top_true_indices = np.argsort(true_value)[:percent]
        top_pred_indices = top_pred_indices.cpu().numpy()

        print("TOP TRUE",top_true_indices)

        common_indices = np.isin(top_pred_indices, top_true_indices)

        num_common = np.sum(common_indices)
        print("NUM COMMON:",num_common)
        print(f"FOUND {num_common*100/(percent)} out of the top {top_k_percent}%")

        return num_common/(percent)
    
    def _scatter_plot(self):
        """
        Plots the true target values (self.y_og_pool) vs. the average predictions 
        of the model ensemble using self.X_og_pool.
        Includes a red dotted line indicating perfect prediction.
        """

        pool_data = [data.MoleculeDatapoint.from_smi(x, 0) for x in self.X_og_pool]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)

        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            model_predictions = trainer.predict(model, prediction_dataloader)
            all_predictions.append(torch.concat(model_predictions))

        predictions = torch.mean(torch.stack(all_predictions), dim=0).squeeze().detach().numpy()
        true_values = self.y_og_pool.squeeze()

        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.xlabel("True Target Values (self.y_og_pool)")
        plt.ylabel("Average Predictions")
        plt.grid(True)

        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', label="Perfect Correlation")

        plt.legend() 
        plt.savefig(os.path.join(self.results_dir, "scatter_plot.png"))
        plt.show()













    

        