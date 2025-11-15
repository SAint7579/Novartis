"""
Variational Autoencoder (VAE) model for gene expression data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VAE(nn.Module):
    """
    Variational Autoencoder for gene expression data.
    
    Architecture:
    - Encoder: input -> hidden layers -> mu and logvar (latent space)
    - Decoder: latent -> hidden layers -> reconstructed input
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize VAE.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (genes)
        latent_dim : int
            Dimension of latent space
        hidden_dims : list, optional
            List of hidden layer dimensions. Default: [512, 256, 128]
        dropout : float
            Dropout rate for regularization
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Default hidden dimensions
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
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
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
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data [batch_size, input_dim]
        
        Returns:
        --------
        mu : torch.Tensor
            Mean of latent distribution [batch_size, latent_dim]
        logvar : torch.Tensor
            Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        
        Returns:
        --------
        z : torch.Tensor
            Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed input.
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent vector [batch_size, latent_dim]
        
        Returns:
        --------
        recon_x : torch.Tensor
            Reconstructed input [batch_size, input_dim]
        """
        h = self.decoder(z)
        recon_x = self.output_layer(h)
        return recon_x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data [batch_size, input_dim]
        
        Returns:
        --------
        recon_x : torch.Tensor
            Reconstructed input [batch_size, input_dim]
        mu : torch.Tensor
            Mean of latent distribution [batch_size, latent_dim]
        logvar : torch.Tensor
            Log variance of latent distribution [batch_size, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_latent(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Get latent representation of input.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data [batch_size, input_dim]
        use_mean : bool
            If True, return mu; if False, sample from distribution
        
        Returns:
        --------
        z : torch.Tensor
            Latent representation [batch_size, latent_dim]
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)
    
    def generate(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from random latent vectors.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        device : str
            Device to use ('cpu' or 'cuda')
        
        Returns:
        --------
        samples : torch.Tensor
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function: MSE reconstruction loss + KL divergence.
    
    Parameters:
    -----------
    recon_x : torch.Tensor
        Reconstructed input
    x : torch.Tensor
        Original input
    mu : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution
    beta : float
        Weight for KL divergence term (beta-VAE)
    
    Returns:
    --------
    total_loss : torch.Tensor
        Total loss
    recon_loss : torch.Tensor
        Reconstruction loss (MSE)
    kl_loss : torch.Tensor
        KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence loss
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

