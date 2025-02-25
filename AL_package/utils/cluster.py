from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def molecular_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Compute Morgan fingerprints for a list of SMILES strings.
    
    Parameters:
    - smiles_list: list of SMILES strings
    - radius: Morgan fingerprint radius (default=2)
    - n_bits: Number of bits in the fingerprint (default=2048)
    
    Returns:
    - A PyTorch tensor of Morgan fingerprints
    """
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule
        if mol:  # Ensure the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(torch.tensor(fp, dtype=torch.float32))  # Convert to tensor
        else:
            print(f"Warning: Invalid SMILES '{smiles}'")
            fingerprints.append(torch.zeros(n_bits))  # Fallback for invalid SMILES
    
    return torch.stack(fingerprints)


def determine_optimal_clusters(fingerprints: torch.Tensor, max_clusters: int = 10):
    """
    Determine the optimal number of clusters using the Elbow Method.
    """
    X = fingerprints.cpu().numpy()

    # Store inertia (sum of squared distances of samples to their cluster center)
    inertias = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    #plt.figure(figsize=(8, 5))
    #plt.plot(range(2, max_clusters + 1), inertias, marker='o', label="Inertia (Elbow Method)")
    #plt.xlabel("Number of Clusters")
    #plt.ylabel("Inertia")
    #plt.title("Elbow Method to Determine Optimal Clusters")
    #plt.legend()
    #plt.show()

    # Find the "elbow" point
    optimal_clusters = inertias.index(min(inertias[1:], key=lambda x: abs(x - inertias[0]))) + 2
    print(f"Optimal number of clusters (based on Elbow Method): {optimal_clusters}")

    return optimal_clusters

def cluster_fingerprints(fingerprints: torch.Tensor, n_clusters: int):
    """
    Cluster molecular fingerprints using K-Means clustering.
    """
    X = fingerprints.cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate Silhouette Coefficients for each point
    silhouette_vals = silhouette_samples(X, cluster_labels)

    return cluster_labels, silhouette_vals, kmeans

def plot_pca(fingerprints: torch.Tensor, cluster_labels: list, filtered_indices: list):
    """
    Plot the 2D PCA projection of the molecular fingerprints.
    """
    X = fingerprints.cpu().numpy()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Filtered data for plotting
    X_pca_filtered = X_pca[filtered_indices]
    cluster_labels_filtered = [cluster_labels[i] for i in filtered_indices]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca_filtered[:, 0], X_pca_filtered[:, 1], c=cluster_labels_filtered, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("PCA of Molecular Fingerprints (Filtered by Silhouette Score)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def divide_into_groups_based_on_score(silhouette_vals, n_groups):
    """
    Assign each sample to a group based on its Silhouette score.

    Parameters:
    - silhouette_vals: List or array of silhouette scores.
    - n_groups: Number of groups to divide the data into.

    Returns:
    - groups: A NumPy array where each index corresponds to a sample and contains the group number.
    """
    sorted_indices = np.argsort(silhouette_vals)[::-1]  # Sort indices by silhouette score (descending)
    group_size = len(silhouette_vals) // n_groups
    remainder = len(silhouette_vals) % n_groups

    groups = np.zeros(len(silhouette_vals), dtype=int)  # Initialize group array

    start = 0
    for group in range(n_groups):
        end = start + group_size + (1 if group < remainder else 0)  # Distribute remainder
        groups[sorted_indices[start:end]] = group  # Assign group labels
        start = end

    return groups

