"""
Variational Autoencoder (VAE) for gene expression data.
"""

from .model import VAE
from .train import train_vae
from .dataset import GeneExpressionDataset
from .utils import preprocess_gene_expression, plot_latent_space, plot_training_history, plot_reconstruction_quality

__all__ = [
    'VAE',
    'train_vae',
    'GeneExpressionDataset',
    'preprocess_gene_expression',
    'plot_latent_space',
    'plot_training_history',
    'plot_reconstruction_quality'
]

