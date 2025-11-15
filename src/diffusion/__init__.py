"""
Diffusion-based perturbation prediction.

Uses conditional diffusion in VAE latent space to predict post-perturbation
gene expression from baseline state and compound SMILES.
"""

from .smiles_encoder import (
    SMILESEncoder, load_smiles_dict, precompute_smiles_embeddings, normalize_compound_id
)
from .diffusion_model import PerturbationDiffusionModel
from .linear_baseline import LinearPerturbationModel

__all__ = [
    'SMILESEncoder',
    'load_smiles_dict',
    'precompute_smiles_embeddings',
    'normalize_compound_id',
    'PerturbationDiffusionModel',
    'LinearPerturbationModel'
]

