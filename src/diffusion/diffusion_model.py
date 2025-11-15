"""
Conditional diffusion model for perturbation prediction.

Uses denoising diffusion with cross-attention conditioning on:
- Compound SMILES embedding
- Baseline VAE embedding

Predicts post-perturbation VAE embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CrossAttentionBlock(nn.Module):
    """Multi-headed cross-attention for conditioning."""
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Parameters:
        -----------
        x : torch.Tensor [batch_size, dim]
            Query (noisy embedding)
        context : torch.Tensor [batch_size, context_dim]
            Key/Value (conditioning)
        
        Returns:
        --------
        out : torch.Tensor [batch_size, dim]
            Attended output
        """
        batch_size = x.shape[0]
        
        # Compute Q, K, V
        q = self.to_q(x)  # [B, dim]
        k = self.to_k(context)  # [B, dim]
        v = self.to_v(context)  # [B, dim]
        
        # Reshape for multi-head
        q = q.view(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhd,bhd->bh', q, k) * scale  # [B, H]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bh,bhd->bhd', attn, v)  # [B, H, D]
        out = out.reshape(batch_size, self.dim)
        
        # Output projection
        out = self.to_out(out)
        
        # Residual + norm
        return self.norm(x + out)


class PerturbationDiffusionModel(nn.Module):
    """
    Conditional diffusion model for perturbation prediction.
    
    Predicts noise in VAE latent space, conditioned on:
    - SMILES embedding (compound information)
    - Baseline VAE embedding (initial state)
    - Diffusion timestep
    """
    
    def __init__(self,
                 latent_dim: int = 64,
                 smiles_dim: int = 256,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_timesteps: int = 1000):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.smiles_dim = smiles_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Condition embedding (SMILES + baseline)
        self.condition_proj = nn.Sequential(
            nn.Linear(smiles_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Cross-attention blocks
        self.cross_attn1 = CrossAttentionBlock(hidden_dim, hidden_dim, num_heads)
        self.cross_attn2 = CrossAttentionBlock(hidden_dim, hidden_dim, num_heads)
        
        # Feed-forward blocks
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection (predict noise)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Diffusion schedule (cosine schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule for diffusion noise.
        
        From "Improved Denoising Diffusion Probabilistic Models"
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self,
                x_t: torch.Tensor,
                timesteps: torch.Tensor,
                smiles_emb: torch.Tensor,
                baseline_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in noisy latent embedding.
        
        Parameters:
        -----------
        x_t : torch.Tensor [batch_size, latent_dim]
            Noisy latent embedding at timestep t
        timesteps : torch.Tensor [batch_size]
            Diffusion timesteps
        smiles_emb : torch.Tensor [batch_size, smiles_dim]
            SMILES embedding of compound
        baseline_emb : torch.Tensor [batch_size, latent_dim]
            Baseline VAE embedding
        
        Returns:
        --------
        noise_pred : torch.Tensor [batch_size, latent_dim]
            Predicted noise
        """
        # Time embedding
        t_emb = timesteps.float().unsqueeze(-1) / self.num_timesteps
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]
        
        # Input embedding
        x_emb = self.input_proj(x_t)  # [B, hidden_dim]
        x_emb = x_emb + t_emb  # Add time information
        
        # Condition embedding (SMILES + baseline)
        cond = torch.cat([smiles_emb, baseline_emb], dim=-1)
        cond_emb = self.condition_proj(cond)  # [B, hidden_dim]
        
        # Cross-attention conditioning
        h = self.cross_attn1(x_emb, cond_emb)
        h = h + self.ff1(h)
        
        h = self.cross_attn2(h, cond_emb)
        h = h + self.ff2(h)
        
        # Predict noise
        noise_pred = self.output_proj(h)
        
        return noise_pred
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean sample.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self,
                 x_t: torch.Tensor,
                 t: int,
                 smiles_emb: torch.Tensor,
                 baseline_emb: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        
        p(x_{t-1} | x_t) sampling
        """
        batch_size = x_t.shape[0]
        timesteps = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self(x_t, timesteps, smiles_emb, baseline_emb)
        
        # Compute x_{t-1}
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        
        beta_t = self.betas[t]
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(1.0 - beta_t)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alpha_t * (x_t - beta_t / torch.sqrt(1.0 - alpha_t) * noise_pred)
        
        if t == 0:
            return model_mean
        else:
            # Add noise
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return model_mean + sigma_t * noise
    
    @torch.no_grad()
    def sample(self,
               smiles_emb: torch.Tensor,
               baseline_emb: torch.Tensor,
               num_steps: int = None) -> torch.Tensor:
        """
        Generate post-perturbation embedding via reverse diffusion.
        
        Parameters:
        -----------
        smiles_emb : torch.Tensor [batch_size, smiles_dim]
            SMILES embedding
        baseline_emb : torch.Tensor [batch_size, latent_dim]
            Baseline embedding
        num_steps : int, optional
            Number of diffusion steps (default: self.num_timesteps)
        
        Returns:
        --------
        x_0 : torch.Tensor [batch_size, latent_dim]
            Predicted post-perturbation embedding
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        batch_size = smiles_emb.shape[0]
        device = smiles_emb.device
        
        # Start from pure noise
        x_t = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Reverse diffusion
        for t in reversed(range(num_steps)):
            x_t = self.p_sample(x_t, t, smiles_emb, baseline_emb)
        
        return x_t

