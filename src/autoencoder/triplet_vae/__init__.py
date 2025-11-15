"""
Triplet VAE with LogFC-weighted loss.
Uses biological effect size (logFC) to weight the importance of triplet pairs.
"""

from .model import TripletVAE
from .loss import (
    weighted_triplet_loss, 
    weighted_multi_negative_triplet_loss,
    compute_logfc_weights_positive,
    compute_logfc_weights_dmso_negative,
    compute_logfc_weights_compound_negative
)
from .dataset import TripletGeneExpressionDataset
from .train import train_triplet_vae

__all__ = [
    'TripletVAE',
    'weighted_triplet_loss',
    'weighted_multi_negative_triplet_loss',
    'compute_logfc_weights_positive',
    'compute_logfc_weights_dmso_negative',
    'compute_logfc_weights_compound_negative',
    'TripletGeneExpressionDataset',
    'train_triplet_vae'
]

