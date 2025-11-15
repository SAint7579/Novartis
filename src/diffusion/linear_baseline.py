"""
Linear baseline for perturbation prediction.

Simple MLP that predicts post-perturbation latent from baseline + SMILES.
Used as a simpler baseline to compare against diffusion model.
"""

import torch
import torch.nn as nn


class LinearPerturbationModel(nn.Module):
    """
    Simple MLP baseline for perturbation prediction.
    
    Predicts: (baseline_latent, smiles_embedding) -> post_perturbation_latent
    """
    
    def __init__(self,
                 latent_dim: int = 64,
                 smiles_dim: int = 256,
                 hidden_dims: list = [512, 512, 256]):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.smiles_dim = smiles_dim
        
        # Build MLP
        layers = []
        input_dim = latent_dim + smiles_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, baseline_latent: torch.Tensor, smiles_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict post-perturbation latent.
        
        Parameters:
        -----------
        baseline_latent : torch.Tensor [batch_size, latent_dim]
            Baseline (DMSO) latent embedding
        smiles_emb : torch.Tensor [batch_size, smiles_dim]
            SMILES embedding
        
        Returns:
        --------
        pred_latent : torch.Tensor [batch_size, latent_dim]
            Predicted post-perturbation latent
        """
        # Concatenate inputs
        x = torch.cat([baseline_latent, smiles_emb], dim=-1)
        
        # Predict perturbation
        pred_latent = self.network(x)
        
        return pred_latent

