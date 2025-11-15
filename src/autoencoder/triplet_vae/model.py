"""
Triplet VAE model with LogFC-weighted loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class TripletVAE(nn.Module):
    """
    VAE with Triplet Loss weighted by biological effect size (logFC).
    
    Triplets:
    - Anchor: Sample from treatment
    - Positive: Replicate of same treatment
    - Negative: DMSO control sample
    
    Weight: exp(-β * d_logFC(anchor, positive))
    - Small logFC between replicates -> high weight (should be similar)
    - Large logFC -> lower weight (allows more variation)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize Triplet VAE.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (genes)
        latent_dim : int
            Latent space dimension
        hidden_dims : list, optional
            Hidden layer dimensions
        dropout : float
            Dropout rate
        use_batch_norm : bool
            Use batch normalization
        """
        super(TripletVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        reversed_dims = [latent_dim] + hidden_dims[::-1]
        
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(reversed_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        h = self.decoder(z)
        return self.output_layer(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_latent(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """Get latent representation."""
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)

