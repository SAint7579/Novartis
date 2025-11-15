"""
Contrastive Variational Autoencoder (VAE) with InfoNCE loss.
Leverages replicate structure to learn better representations.

Supports two variants:
1. Standard InfoNCE (loss.py) - uniform weighting
2. LogFC-weighted InfoNCE (loss_logfc.py) - biological weighting
"""

from .model import ContrastiveVAE
from .loss import info_nce_loss, contrastive_vae_loss
from .loss_logfc import contrastive_vae_logfc_loss, logfc_weighted_info_nce_loss
from .dataset import ContrastiveGeneExpressionDataset
from .train import train_contrastive_vae
from .utils import plot_latent_space_by_treatment, plot_treatment_clusters, compute_replicate_agreement

__all__ = [
    'ContrastiveVAE',
    'info_nce_loss',
    'contrastive_vae_loss',
    'contrastive_vae_logfc_loss',
    'logfc_weighted_info_nce_loss',
    'ContrastiveGeneExpressionDataset',
    'train_contrastive_vae',
    'plot_latent_space_by_treatment',
    'plot_treatment_clusters',
    'compute_replicate_agreement'
]

