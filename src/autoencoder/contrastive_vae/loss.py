"""
Loss functions for Contrastive VAE with InfoNCE.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def info_nce_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    normalize: bool = True
) -> torch.Tensor:
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    Pulls together samples with same label (replicates of same perturbation).
    Pushes apart samples with different labels (different perturbations).
    
    Parameters:
    -----------
    z : torch.Tensor
        Latent representations [batch_size, latent_dim]
    labels : torch.Tensor
        Treatment labels [batch_size]
    temperature : float
        Temperature parameter for contrastive loss (lower = stricter)
    normalize : bool
        Whether to L2-normalize embeddings
    
    Returns:
    --------
    loss : torch.Tensor
        InfoNCE loss
    """
    batch_size = z.shape[0]
    
    if batch_size < 2:
        return torch.tensor(0.0, device=z.device)
    
    # L2 normalize embeddings (common in contrastive learning)
    if normalize:
        z = F.normalize(z, p=2, dim=1)
    
    # Compute similarity matrix [batch_size, batch_size]
    similarity_matrix = torch.mm(z, z.t()) / temperature
    
    # Create mask for positive pairs (same treatment)
    # labels: [batch_size] -> expand to [batch_size, batch_size]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(z.device)
    
    # Remove diagonal (sample with itself)
    mask = mask - torch.eye(batch_size, device=z.device)
    
    # For each sample, find number of positive pairs
    positives_per_sample = mask.sum(dim=1)
    
    # If a sample has no positives (unique treatment in batch), skip it
    valid_samples = positives_per_sample > 0
    
    if valid_samples.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    
    # Compute InfoNCE loss
    # exp_sim: [batch_size, batch_size]
    exp_sim = torch.exp(similarity_matrix)
    
    # Sum of all similarities (denominator)
    # Subtract diagonal to exclude self-similarity
    sum_exp_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
    
    # Sum of positive pair similarities (numerator)
    positive_sim = (exp_sim * mask).sum(dim=1)
    
    # InfoNCE loss: -log(sum of positive similarities / sum of all similarities)
    # Only for samples that have positives
    loss = -torch.log(positive_sim[valid_samples] / (sum_exp_sim[valid_samples] + 1e-8))
    
    # Average over valid samples
    loss = loss.mean()
    
    return loss


def contrastive_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z_proj: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for Contrastive VAE.
    
    Total Loss = Reconstruction Loss + β * KL Divergence + γ * InfoNCE Loss
    
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
    z_proj : torch.Tensor
        Projected latent for contrastive learning
    labels : torch.Tensor
        Treatment labels for grouping replicates
    beta : float
        Weight for KL divergence (β-VAE)
    gamma : float
        Weight for InfoNCE contrastive loss
    temperature : float
        Temperature for InfoNCE
    
    Returns:
    --------
    total_loss : torch.Tensor
        Total combined loss
    recon_loss : torch.Tensor
        Reconstruction loss (MSE)
    kl_loss : torch.Tensor
        KL divergence loss
    contrastive_loss : torch.Tensor
        InfoNCE contrastive loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # InfoNCE contrastive loss
    contrastive_loss = info_nce_loss(z_proj, labels, temperature=temperature)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + gamma * contrastive_loss
    
    return total_loss, recon_loss, kl_loss, contrastive_loss


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (alternative to InfoNCE).
    
    Similar to InfoNCE but averages over all positive pairs.
    From: https://arxiv.org/abs/2004.11362
    
    Parameters:
    -----------
    z : torch.Tensor
        Latent representations
    labels : torch.Tensor
        Treatment labels
    temperature : float
        Temperature parameter
    
    Returns:
    --------
    loss : torch.Tensor
        Supervised contrastive loss
    """
    batch_size = z.shape[0]
    
    if batch_size < 2:
        return torch.tensor(0.0, device=z.device)
    
    # Normalize
    z = F.normalize(z, p=2, dim=1)
    
    # Compute similarity
    similarity = torch.mm(z, z.t()) / temperature
    
    # Create mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(z.device)
    
    # Remove diagonal
    mask_no_diag = mask - torch.eye(batch_size, device=z.device)
    
    # Count positives per sample
    num_positives = mask_no_diag.sum(dim=1)
    
    # Samples with at least one positive
    valid_samples = num_positives > 0
    
    if valid_samples.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    
    # Compute loss
    # log-sum-exp trick for numerical stability
    max_sim = similarity.max(dim=1, keepdim=True)[0]
    exp_sim = torch.exp(similarity - max_sim)
    
    # Log of sum of exponentials (denominator)
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).view(-1, 1) + 1e-8)
    
    # Average log probability of positive pairs
    # For each sample, average over all its positive pairs
    log_prob_pos = (similarity - max_sim - log_sum_exp) * mask_no_diag
    
    # Sum over positives and divide by number of positives
    loss_per_sample = -log_prob_pos.sum(dim=1) / (num_positives + 1e-8)
    
    # Average over valid samples
    loss = loss_per_sample[valid_samples].mean()
    
    return loss

