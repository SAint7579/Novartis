"""
Contrastive Variational Autoencoder (VAE) with InfoNCE loss.
Leverages replicate structure to learn better representations.
"""

from .model import ContrastiveVAE
from .loss import info_nce_loss, contrastive_vae_loss
from .dataset import ContrastiveGeneExpressionDataset
from .train import train_contrastive_vae
from .utils import plot_latent_space_by_treatment, plot_treatment_clusters, compute_replicate_agreement

__all__ = [
    'ContrastiveVAE',
    'info_nce_loss',
    'contrastive_vae_loss',
    'ContrastiveGeneExpressionDataset',
    'train_contrastive_vae',
    'plot_latent_space_by_treatment',
    'plot_treatment_clusters',
    'compute_replicate_agreement'
]

