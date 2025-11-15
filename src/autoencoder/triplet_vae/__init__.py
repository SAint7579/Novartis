"""
Triplet VAE with LogFC-weighted loss.

Supports two loss variants:
1. Quadruplet hinge loss (loss.py) - explicit anchor/positive/negative triplets
2. InfoNCE contrastive loss (loss_infonce.py) - batch-wise contrastive learning
"""

from .model import TripletVAE
from .loss import (
    weighted_triplet_loss, 
    weighted_multi_negative_triplet_loss,
    compute_logfc_weights_positive,
    compute_logfc_weights_dmso_negative,
    compute_logfc_weights_compound_negative
)
from .loss_infonce import (
    triplet_vae_infonce_loss,
    logfc_weighted_info_nce_loss,
    compute_logfc_similarity_weights
)
from .dataset import TripletGeneExpressionDataset
from .dataset_fast import FastTripletGeneExpressionDataset
from .train import train_triplet_vae
from .loss_fast import triplet_vae_loss_fast

__all__ = [
    'TripletVAE',
    # Quadruplet loss
    'weighted_triplet_loss',
    'weighted_multi_negative_triplet_loss',
    'compute_logfc_weights_positive',
    'compute_logfc_weights_dmso_negative',
    'compute_logfc_weights_compound_negative',
    # InfoNCE loss
    'triplet_vae_infonce_loss',
    'logfc_weighted_info_nce_loss',
    'compute_logfc_similarity_weights',
    # Datasets
    'TripletGeneExpressionDataset',
    'FastTripletGeneExpressionDataset',
    # Training
    'train_triplet_vae',
    'triplet_vae_loss_fast'
]

