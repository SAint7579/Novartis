"""
Contrastive VAE model with InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ContrastiveVAE(nn.Module):
    """
    Contrastive Variational Autoencoder with InfoNCE loss.
    
    Learns representations that:
    1. Reconstruct gene expression (VAE objective)
    2. Group replicates of same perturbation together (InfoNCE)
    3. Separate different perturbations (InfoNCE)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        projection_dim: Optional[int] = None
    ):
        """
        Initialize Contrastive VAE.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (genes)
        latent_dim : int
            Dimension of latent space
        hidden_dims : list, optional
            List of hidden layer dimensions
        dropout : float
            Dropout rate
        use_batch_norm : bool
            Whether to use batch normalization
        projection_dim : int, optional
            Dimension of projection head for contrastive learning
            If None, uses latent_dim
        """
        super(ContrastiveVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim or latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.hidden_dims = hidden_dims
        
        # Build Encoder
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
        
        # Projection head for contrastive learning
        # Maps latent representation to projection space
        if projection_dim != latent_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, projection_dim)
            )
        else:
            self.projection_head = nn.Identity()
        
        # Build Decoder
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
        """
        Encode input to latent space parameters.
        
        Returns:
        --------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latent representation for contrastive learning.
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent representation
        
        Returns:
        --------
        z_proj : torch.Tensor
            Projected representation for contrastive loss
        """
        return self.projection_head(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed input."""
        h = self.decoder(z)
        recon_x = self.output_layer(h)
        return recon_x
    
    def forward(
        self, 
        x: torch.Tensor,
        return_projection: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Contrastive VAE.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data
        return_projection : bool
            Whether to return projection for contrastive loss
        
        Returns:
        --------
        recon_x : torch.Tensor
            Reconstructed input
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        z_proj : torch.Tensor, optional
            Projected latent for contrastive learning
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        if return_projection:
            z_proj = self.project(mu)  # Use mu (deterministic) for contrastive
            return recon_x, mu, logvar, z_proj
        else:
            return recon_x, mu, logvar, None
    
    def get_latent(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """Get latent representation."""
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)
    
    def get_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Get projection for contrastive learning."""
        mu, _ = self.encode(x)
        return self.project(mu)

