"""
Utility functions for VAE preprocessing and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict
import torch


def preprocess_gene_expression(
    counts_df: pd.DataFrame,
    method: str = 'log_normalize',
    scale: str = 'standard',
    filter_low_variance: bool = True,
    variance_threshold: float = 0.01
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess gene expression count data for VAE.
    
    Parameters:
    -----------
    counts_df : pd.DataFrame
        Raw count data [samples x genes]
    method : str
        Normalization method: 'log_normalize', 'tpm', or 'none'
    scale : str
        Scaling method: 'standard', 'minmax', or 'none'
    filter_low_variance : bool
        Whether to filter low variance genes
    variance_threshold : float
        Variance threshold for filtering
    
    Returns:
    --------
    processed_df : pd.DataFrame
        Processed gene expression data
    metadata : dict
        Preprocessing metadata (scalers, filtered genes, etc.)
    """
    print(f"Original shape: {counts_df.shape}")
    
    # Remove constant columns
    non_constant = counts_df.var() > 0
    counts_df = counts_df.loc[:, non_constant]
    print(f"After removing constant genes: {counts_df.shape}")
    
    # Normalization
    if method == 'log_normalize':
        # Log2(x + 1) transformation
        df_normalized = np.log2(counts_df + 1)
    elif method == 'tpm':
        # TPM-like normalization
        df_normalized = (counts_df / counts_df.sum(axis=1).values.reshape(-1, 1)) * 1e6
        df_normalized = np.log2(df_normalized + 1)
    else:
        df_normalized = counts_df.copy()
    
    # Filter low variance genes
    if filter_low_variance:
        gene_variance = df_normalized.var(axis=0)
        high_var_genes = gene_variance > variance_threshold
        df_filtered = df_normalized.loc[:, high_var_genes]
        print(f"After filtering low variance genes: {df_filtered.shape}")
    else:
        df_filtered = df_normalized
    
    # Scaling
    scaler = None
    if scale == 'standard':
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_filtered),
            index=df_filtered.index,
            columns=df_filtered.columns
        )
    elif scale == 'minmax':
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_filtered),
            index=df_filtered.index,
            columns=df_filtered.columns
        )
    else:
        df_scaled = df_filtered
    
    # Handle NaN values
    if df_scaled.isna().any().any():
        print(f"Warning: {df_scaled.isna().sum().sum()} NaN values found, filling with 0")
        df_scaled = df_scaled.fillna(0)
    
    # Handle infinite values
    if np.isinf(df_scaled.values).any():
        print(f"Warning: Infinite values found, replacing with column max/min")
        df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan)
        df_scaled = df_scaled.fillna(df_scaled.max())
    
    metadata = {
        'method': method,
        'scale': scale,
        'scaler': scaler,
        'filtered_genes': df_scaled.columns.tolist(),
        'original_shape': counts_df.shape,
        'final_shape': df_scaled.shape
    }
    
    print(f"Final shape: {df_scaled.shape}")
    
    return df_scaled, metadata


def plot_latent_space(
    model,
    data_loader,
    labels: Optional[pd.Series] = None,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Visualize latent space using PCA (if dim > 2) or direct plotting.
    
    Parameters:
    -----------
    model : VAE
        Trained VAE model
    data_loader : DataLoader
        Data loader
    labels : pd.Series, optional
        Labels for coloring points
    device : str
        Device to use
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    model.eval()
    model = model.to(device)
    
    # Extract latent representations
    latent_vectors = []
    batch_labels = []
    
    with torch.no_grad():
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            latent_vectors.append(mu.cpu().numpy())
            batch_labels.extend(batch_label.numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Apply PCA if latent dim > 2
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
        explained_var = pca.explained_variance_ratio_
        title_suffix = f'(PCA: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%} var)'
    else:
        latent_2d = latent_vectors
        title_suffix = ''
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Colored by labels (if available)
    if labels is not None and len(set(batch_labels)) > 1:
        scatter = axes[0].scatter(
            latent_2d[:, 0], latent_2d[:, 1],
            c=batch_labels, cmap='tab20', alpha=0.6, s=20
        )
        axes[0].set_title(f'Latent Space (by labels) {title_suffix}')
        plt.colorbar(scatter, ax=axes[0])
    else:
        axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
        axes[0].set_title(f'Latent Space {title_suffix}')
    
    axes[0].set_xlabel('Latent Dim 1')
    axes[0].set_ylabel('Latent Dim 2')
    
    # Plot 2: Density plot
    axes[1].hexbin(latent_2d[:, 0], latent_2d[:, 1], gridsize=30, cmap='Blues')
    axes[1].set_title(f'Latent Space Density {title_suffix}')
    axes[1].set_xlabel('Latent Dim 1')
    axes[1].set_ylabel('Latent Dim 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent space plot to: {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4)
):
    """
    Plot training history.
    
    Parameters:
    -----------
    history : dict
        Training history from train_vae()
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon_loss'], 'b-', label='Train')
    if history['val_recon_loss']:
        axes[1].plot(epochs, history['val_recon_loss'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(epochs, history['train_kl_loss'], 'b-', label='Train')
    if history['val_kl_loss']:
        axes[2].plot(epochs, history['val_kl_loss'], 'r-', label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to: {save_path}")
    
    plt.show()


def plot_reconstruction_quality(
    model,
    data_loader,
    n_samples: int = 5,
    device: str = 'cpu',
    save_path: Optional[str] = None
):
    """
    Plot original vs reconstructed gene expression profiles.
    
    Parameters:
    -----------
    model : VAE
        Trained VAE model
    data_loader : DataLoader
        Data loader
    n_samples : int
        Number of samples to plot
    device : str
        Device to use
    save_path : str, optional
        Path to save figure
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch
    batch_data, _ = next(iter(data_loader))
    batch_data = batch_data.to(device)
    
    with torch.no_grad():
        recon_batch, _, _ = model(batch_data)
    
    # Convert to numpy
    original = batch_data[:n_samples].cpu().numpy()
    reconstructed = recon_batch[:n_samples].cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    
    for i in range(n_samples):
        # Original
        axes[i, 0].plot(original[i], alpha=0.7, linewidth=0.5)
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].set_xlabel('Gene Index')
        axes[i, 0].set_ylabel('Expression')
        
        # Reconstructed
        axes[i, 1].plot(reconstructed[i], alpha=0.7, linewidth=0.5, color='orange')
        axes[i, 1].set_title(f'Sample {i+1}: Reconstructed')
        axes[i, 1].set_xlabel('Gene Index')
        axes[i, 1].set_ylabel('Expression')
        
        # Compute correlation
        corr = np.corrcoef(original[i], reconstructed[i])[0, 1]
        axes[i, 1].text(0.02, 0.98, f'Correlation: {corr:.3f}',
                       transform=axes[i, 1].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction quality plot to: {save_path}")
    
    plt.show()

