"""
Utility functions for Contrastive VAE visualization and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from typing import Optional, Tuple, Dict, List
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')


def plot_latent_space_by_treatment(
    model,
    data_loader,
    treatments: pd.Series,
    device: str = 'cpu',
    method: str = 'pca',
    highlight_treatments: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    max_treatments_legend: int = 20
):
    """
    Visualize latent space colored by treatment (compound).
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Trained model
    data_loader : DataLoader
        Data loader
    treatments : pd.Series
        Treatment labels
    device : str
        Device
    method : str
        Dimensionality reduction: 'pca' or 'tsne'
    highlight_treatments : list, optional
        Specific treatments to highlight
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    max_treatments_legend : int
        Maximum treatments to show in legend
    """
    model.eval()
    model = model.to(device)
    
    # Extract latent representations
    latent_vectors = []
    treatment_names = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            latent_vectors.append(mu.cpu().numpy())
            treatment_names.extend(batch_treatments)
    
    latent_vectors = np.vstack(latent_vectors)
    treatment_names = np.array(treatment_names)
    
    # Get unique treatments
    unique_treatments = pd.Series(treatment_names).unique()
    treatment_to_color = {t: i for i, t in enumerate(unique_treatments)}
    colors = np.array([treatment_to_color[t] for t in treatment_names])
    
    # Apply dimensionality reduction
    if latent_vectors.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)//2))
            latent_2d = reducer.fit_transform(latent_vectors)
            title_suffix = 't-SNE'
        else:  # PCA
            reducer = PCA(n_components=2)
            latent_2d = reducer.fit_transform(latent_vectors)
            explained_var = reducer.explained_variance_ratio_
            title_suffix = f'PCA ({explained_var[0]:.1%} + {explained_var[1]:.1%} var)'
    else:
        latent_2d = latent_vectors
        title_suffix = 'Latent Space'
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: All treatments (small points)
    scatter = axes[0].scatter(
        latent_2d[:, 0], latent_2d[:, 1],
        c=colors, cmap='tab20', alpha=0.5, s=10
    )
    axes[0].set_title(f'All Treatments - {title_suffix}')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    
    # Plot 2: Highlighted treatments or controls
    if highlight_treatments:
        # Show only highlighted treatments
        for i, treatment in enumerate(highlight_treatments[:10]):  # Max 10 for clarity
            mask = treatment_names == treatment
            if mask.sum() > 0:
                axes[1].scatter(
                    latent_2d[mask, 0], latent_2d[mask, 1],
                    label=treatment, alpha=0.7, s=50
                )
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].set_title('Highlighted Treatments')
    else:
        # Show controls (DMSO, Blank, RNA)
        controls = ['DMSO', 'Blank', 'RNA', 'dmso', 'blank', 'rna']
        for control in controls:
            mask = np.char.lower(treatment_names.astype(str)) == control.lower()
            if mask.sum() > 0:
                axes[1].scatter(
                    latent_2d[mask, 0], latent_2d[mask, 1],
                    label=control, alpha=0.7, s=50
                )
        
        # Add a few random compounds for reference
        compound_mask = ~np.isin(treatment_names, controls)
        if compound_mask.sum() > 0:
            axes[1].scatter(
                latent_2d[compound_mask, 0], latent_2d[compound_mask, 1],
                c='gray', alpha=0.2, s=10, label='Other compounds'
            )
        
        axes[1].legend(fontsize=8)
        axes[1].set_title('Controls vs Compounds')
    
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    
    # Plot 3: Density plot
    h = axes[2].hexbin(latent_2d[:, 0], latent_2d[:, 1], gridsize=40, cmap='Blues')
    axes[2].set_title('Density Plot')
    axes[2].set_xlabel('Component 1')
    axes[2].set_ylabel('Component 2')
    plt.colorbar(h, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent space plot to: {save_path}")
    
    plt.show()


def plot_treatment_clusters(
    model,
    data_loader,
    treatments: pd.Series,
    n_treatments: int = 10,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot latent space for specific treatments showing replicates.
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Trained model
    data_loader : DataLoader
        Data loader
    treatments : pd.Series
        Treatment labels
    n_treatments : int
        Number of treatments to show
    device : str
        Device
    save_path : str, optional
        Save path
    figsize : tuple
        Figure size
    """
    model.eval()
    model = model.to(device)
    
    # Extract latent representations
    latent_vectors = []
    treatment_names = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            latent_vectors.append(mu.cpu().numpy())
            treatment_names.extend(batch_treatments)
    
    latent_vectors = np.vstack(latent_vectors)
    treatment_names = np.array(treatment_names)
    
    # Select treatments with most replicates (excluding controls)
    treatment_counts = pd.Series(treatment_names).value_counts()
    treatment_counts = treatment_counts[~treatment_counts.index.isin(['DMSO', 'Blank', 'RNA', 'dmso'])]
    selected_treatments = treatment_counts.head(n_treatments).index.tolist()
    
    # Add DMSO as reference
    if 'DMSO' in treatment_names or 'dmso' in treatment_names:
        selected_treatments = ['DMSO'] + selected_treatments
    
    # PCA for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Create subplot grid
    n_cols = 3
    n_rows = (len(selected_treatments) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot each treatment
    for idx, treatment in enumerate(selected_treatments):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot all samples as background (gray)
        ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c='lightgray', alpha=0.1, s=5)
        
        # Highlight this treatment
        mask = treatment_names == treatment
        if mask.sum() > 0:
            replicates_2d = latent_2d[mask]
            ax.scatter(
                replicates_2d[:, 0], replicates_2d[:, 1],
                c='red', s=100, alpha=0.8, edgecolors='black', linewidth=1
            )
            
            # Compute replicate spread (std)
            spread_x = np.std(replicates_2d[:, 0])
            spread_y = np.std(replicates_2d[:, 1])
            
            ax.set_title(f'{treatment}\n({mask.sum()} replicates, σ={spread_x:.2f},{spread_y:.2f})')
        else:
            ax.set_title(f'{treatment}\n(not found)')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    # Hide unused subplots
    for idx in range(len(selected_treatments), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Replicate Clustering in Latent Space', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved treatment clusters plot to: {save_path}")
    
    plt.show()


def compute_replicate_agreement(
    model,
    data_loader,
    treatments: pd.Series,
    device: str = 'cpu',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute within-treatment replicate agreement in latent space.
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Trained model
    data_loader : DataLoader
        Data loader
    treatments : pd.Series
        Treatment labels
    device : str
        Device
    verbose : bool
        Print summary
    
    Returns:
    --------
    agreement_df : pd.DataFrame
        DataFrame with treatment, n_replicates, mean_distance, std_distance
    """
    model.eval()
    model = model.to(device)
    
    # Extract latent representations
    latent_vectors = []
    treatment_names = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            latent_vectors.append(mu.cpu().numpy())
            treatment_names.extend(batch_treatments)
    
    latent_vectors = np.vstack(latent_vectors)
    treatment_names = np.array(treatment_names)
    
    # Compute agreement for each treatment
    results = []
    
    for treatment in pd.Series(treatment_names).unique():
        mask = treatment_names == treatment
        n_replicates = mask.sum()
        
        if n_replicates > 1:
            # Get latent vectors for this treatment
            treatment_latent = latent_vectors[mask]
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(treatment_latent, metric='euclidean')
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            median_dist = np.median(distances)
        else:
            mean_dist = np.nan
            std_dist = np.nan
            median_dist = np.nan
        
        results.append({
            'treatment': treatment,
            'n_replicates': n_replicates,
            'mean_distance': mean_dist,
            'median_distance': median_dist,
            'std_distance': std_dist
        })
    
    agreement_df = pd.DataFrame(results).sort_values('mean_distance')
    
    if verbose:
        print("="*70)
        print("Replicate Agreement in Latent Space")
        print("="*70)
        print(f"\nTreatments with replicates: {(agreement_df['n_replicates'] > 1).sum()}")
        print(f"Average replicate distance: {agreement_df['mean_distance'].mean():.4f}")
        print(f"\nMost consistent (lowest distance):")
        print(agreement_df[agreement_df['n_replicates'] > 1].head(10).to_string())
        print(f"\nLeast consistent (highest distance):")
        print(agreement_df[agreement_df['n_replicates'] > 1].tail(10).to_string())
    
    return agreement_df


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
):
    """
    Plot training history for Contrastive VAE.
    
    Parameters:
    -----------
    history : dict
        Training history
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
    if history['val_recon_loss']:
        axes[1].plot(epochs, history['val_recon_loss'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(epochs, history['train_kl_loss'], 'b-', label='Train', linewidth=2)
    if history['val_kl_loss']:
        axes[2].plot(epochs, history['val_kl_loss'], 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Contrastive loss (NEW!)
    axes[3].plot(epochs, history['train_contrastive_loss'], 'b-', label='Train', linewidth=2)
    if history['val_contrastive_loss']:
        axes[3].plot(epochs, history['val_contrastive_loss'], 'r-', label='Validation', linewidth=2)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('InfoNCE Loss')
    axes[3].set_title('InfoNCE Contrastive Loss')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to: {save_path}")
    
    plt.show()


def plot_replicate_similarity_heatmap(
    model,
    data_loader,
    treatments: pd.Series,
    n_treatments: int = 20,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Heatmap showing cosine similarity between replicates in latent space.
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Trained model
    data_loader : DataLoader
        Data loader  
    treatments : pd.Series
        Treatment labels
    n_treatments : int
        Number of treatments to show
    device : str
        Device
    save_path : str, optional
        Save path
    figsize : tuple
        Figure size
    """
    model.eval()
    model = model.to(device)
    
    # Extract latent representations
    latent_vectors = []
    treatment_names = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            latent_vectors.append(mu.cpu().numpy())
            treatment_names.extend(batch_treatments)
    
    latent_vectors = np.vstack(latent_vectors)
    treatment_names = np.array(treatment_names)
    
    # Select treatments with most replicates
    treatment_counts = pd.Series(treatment_names).value_counts()
    treatment_counts = treatment_counts[treatment_counts >= 2]  # At least 2 replicates
    selected_treatments = treatment_counts.head(n_treatments).index.tolist()
    
    # Build similarity matrix
    selected_indices = []
    labels = []
    
    for treatment in selected_treatments:
        mask = treatment_names == treatment
        indices = np.where(mask)[0]
        selected_indices.extend(indices)
        labels.extend([f"{treatment[:20]}_r{i+1}" for i in range(len(indices))])
    
    selected_latent = latent_vectors[selected_indices]
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(selected_latent)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='RdYlBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Replicate Similarity in Latent Space\n(InfoNCE encourages replicates to cluster)')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity heatmap to: {save_path}")
    
    plt.show()


def visualize_specific_compounds(
    model,
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    compound_list: List[str],
    device: str = 'cpu',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize specific compounds in latent space with their replicates.
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Trained model
    data : pd.DataFrame
        Processed gene expression data
    metadata : pd.DataFrame
        Metadata with treatment column
    compound_list : list
        List of compound names to visualize
    device : str
        Device
    save_path : str, optional
        Save path
    figsize : tuple
        Figure size
    """
    model.eval()
    model = model.to(device)
    
    # Get latent representations
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data.values).to(device)
        latent, _ = model.encode(data_tensor)
        latent = latent.cpu().numpy()
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background: all other samples
    ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c='lightgray', alpha=0.2, s=10, label='Other')
    
    # Highlight each compound
    colors = plt.cm.tab10(np.linspace(0, 1, len(compound_list)))
    
    for compound, color in zip(compound_list, colors):
        mask = metadata['treatment'] == compound
        if mask.sum() > 0:
            compound_latent = latent_2d[mask]
            ax.scatter(
                compound_latent[:, 0], compound_latent[:, 1],
                c=[color], s=150, alpha=0.8,
                edgecolors='black', linewidth=2,
                label=f'{compound} (n={mask.sum()})'
            )
            
            # Draw convex hull or circle around replicates
            if len(compound_latent) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(compound_latent)
                    for simplex in hull.simplices:
                        ax.plot(
                            compound_latent[simplex, 0],
                            compound_latent[simplex, 1],
                            color=color, alpha=0.3, linewidth=1
                        )
                except:
                    pass
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Specific Compounds in Latent Space\n(Replicates should cluster together)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved compound visualization to: {save_path}")
    
    plt.show()

