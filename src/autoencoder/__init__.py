"""
Autoencoder models for transcriptomics data analysis.
"""

from .vae import (
    VAE, 
    train_vae, 
    GeneExpressionDataset,
    preprocess_gene_expression,
    plot_latent_space,
    plot_training_history,
    plot_reconstruction_quality
)

from .contrastive_vae import (
    ContrastiveVAE,
    train_contrastive_vae,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment,
    plot_treatment_clusters,
    compute_replicate_agreement
)

from .triplet_vae import (
    TripletVAE,
    train_triplet_vae,
    TripletGeneExpressionDataset,
    weighted_triplet_loss
)

__all__ = [
    # Standard VAE
    'VAE', 
    'train_vae', 
    'GeneExpressionDataset',
    'preprocess_gene_expression',
    'plot_latent_space',
    'plot_training_history',
    'plot_reconstruction_quality',
    # Contrastive VAE
    'ContrastiveVAE',
    'train_contrastive_vae',
    'ContrastiveGeneExpressionDataset',
    'plot_latent_space_by_treatment',
    'plot_treatment_clusters',
    'compute_replicate_agreement',
    # Triplet VAE
    'TripletVAE',
    'train_triplet_vae',
    'TripletGeneExpressionDataset',
    'weighted_triplet_loss'
]

