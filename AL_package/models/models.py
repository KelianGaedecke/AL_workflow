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

        #print("SCALER:", self.scaler)

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
        print("TRAIN DATA:", train_data), "TRAIN DATA LENGTH:", len(train_data)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)

        self.scaler = train_dset.normalize_targets()

        #print("SCALER:", self.scaler)

        ############# TEST ZONE #############


        self._compute_cluster_info()
        self.ensemble = self._initialize_ensemble(n_models)
        self.queried_indices_history = []
        self.mask = np.ones(len(self.X_pool), dtype=bool)
        self.remaining_indices = np.where(self.mask)[0]



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
        output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler) if self.scaler else None
        ffn = nn.RegressionFFN(output_transform=output_transform)
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

    def _prepare_training_data(self, queried_indices):
        """Prepare training data from selected samples"""

        if len(queried_indices) == 0 :
            print("WARNING: Queried indices are empty. No training data available.")
            return None 
        
        self.mask[queried_indices] = False

        X_train = [self.X_pool[i] for i in queried_indices]
        y_train = [self.y_pool[i] for i in queried_indices]
        
        # Keep track of the remaining indices
        print("REMAINING INICES B4:",self.remaining_indices)
        self.remaining_indices = np.where(self.mask)[0]
        print("REMAINING INICES AFTER:",self.remaining_indices)

        train_data = [data.MoleculeDatapoint.from_smi(x, y) for x, y in zip(X_train, y_train)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer)
        train_loader = data.build_dataloader(train_dset, num_workers=0)

        ############# TEST ZONE #############

        self.scaler = train_dset.normalize_targets()

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
        val_dset.normalize_targets(self.scaler)
        val_loader = data.build_dataloader(val_dset, num_workers=0)

        for model_idx, model in enumerate(self.ensemble):
            self._train_model(model, train_loader, val_loader, model_idx)

        self.iter += 1
    

    def train(self, query_fn, num_iters=4, batch_size=10, train_type="new", use_uncertainty=False):
        """Train the model iteratively using active learning and plot training loss"""
        if self.iter == 0:
            self.start(batch_size)
            X_train, y_train = [], [] 

        self.loss_history = []  
        self.val_loss_history = []  
        self.test_loss_history = []
    
        for _ in range(num_iters):
            if len(self.remaining_indices) < batch_size:
                print("Not enough data left in the pool for another iteration.")
                break
    
            queried_indices = np.array(query_fn(
                self.remaining_indices, self.y_pool, self.cluster_labels, 
                self.difficulty_label, batch_size,
                self.target_label, model=self, use_uncertainty=use_uncertainty
            ))


            queried_indices = self.remaining_indices[queried_indices]

            print("QUERIED INDICES:", queried_indices)
            self.queried_indices_history.append(queried_indices)
            print("QUERIED INDICES HISTORY:", self.queried_indices_history)

            X_new = [self.X_pool[i] for i in queried_indices]
            y_new = [self.y_pool[i] for i in queried_indices]


            ###### BUILDING / TEST ######

            #if train_type == "new":
            #    X_train = X_new
            #    y_train = y_new
            #elif train_type == "ext":
            #    X_train.extend(X_new)
            #    y_train.extend(y_new)


            if train_type == "new":
                train_loader = self._prepare_training_data(queried_indices)
            if train_type == "ext":
                all_indices = np.concatenate([np.array(indices) for indices in self.queried_indices_history])
                train_loader = self._prepare_training_data(all_indices)

            ###### BUILDING / TEST ######

            train_loader = self._prepare_training_data(queried_indices)

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
    
                # Extract training loss
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
        val_iterations, avg_val_losses, std_val_losses = zip(*self.val_loss_history)
        error_iterations, avg_test_losses = zip(*self.test_loss_history)


    
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, avg_losses, label="Avg Training Loss", color="blue")
        plt.fill_between(iterations, 
                         np.array(avg_losses) - np.array(std_losses), 
                         np.array(avg_losses) + np.array(std_losses), 
                         color="blue", alpha=0.2, label="Train Std Dev")
        
        plt.plot(val_iterations, avg_val_losses, label="Avg Validation Loss", color="green")
        plt.fill_between(val_iterations, 
                         np.array(avg_val_losses) - np.array(std_val_losses), 
                         np.array(avg_val_losses) + np.array(std_val_losses), 
                         color="green", alpha=0.2, label="Val Std Dev")
        
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
        
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        pool_dset = data.MoleculeDataset(pool_data, featurizer)
        
        # Create the data loader
        prediction_dataloader = data.build_dataloader(pool_dset, shuffle=False)
    
        all_predictions = []
        for model in self.ensemble:
            trainer = pl.Trainer(accelerator="cpu", devices=1)
            
            model_predictions = trainer.predict(model, prediction_dataloader)
            
            all_predictions.append(torch.concat(model_predictions))
    
        predictions = torch.mean(torch.stack(all_predictions), dim=0)
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
        predictions = predictions.detach().numpy()

        k = max(1, int(len(self.y_og_pool) * top_k_percent / 100))

        sorted_indices_pred = np.argsort(predictions)[:k]
        top_k_pred = set(tuple(self.X_og_pool[i]) for i in sorted_indices_pred)  

        self.y_og_pool_flat = self.y_og_pool.flatten() 
        sorted_indices_true = np.argsort(self.y_og_pool_flat)[:k]
        top_k_true = set(tuple(self.X_og_pool[i]) for i in sorted_indices_true)  
    
        overlap = top_k_pred & top_k_true 
        overlap_percentage = (len(overlap) / len(top_k_true)) * 100
        
        results_txt_path = os.path.join(self.results_dir, "training_results.txt")
        with open(results_txt_path, "a") as f:
                f.write(f"\n L1 ERROR :{error:.4f}, PERCENTAGE OF TOP {top_k_percent:.4f}  % : {overlap_percentage:.4f}% \n\n")
    
        return error, overlap_percentage