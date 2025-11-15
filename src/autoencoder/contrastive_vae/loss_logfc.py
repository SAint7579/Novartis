"""
LogFC-weighted InfoNCE loss for Contrastive VAE.

Combines the projection head architecture (no task interference) with
logFC-based weighting (biological similarity).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_logfc_similarity_weights(
    anchor_expr: torch.Tensor,
    other_expr: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute logFC-based similarity weights between anchor and other samples.
    
    Higher weight = more similar logFC patterns (should be closer in latent space)
    
    Parameters:
    -----------
    anchor_expr : torch.Tensor [batch_size, n_genes]
        Anchor expression values
    other_expr : torch.Tensor [batch_size, batch_size, n_genes]
        Other expression values (pairwise)
    dmso_mean : torch.Tensor [n_genes]
        Mean DMSO expression (baseline)
    beta : float
        Temperature for exponential weighting
    
    Returns:
    --------
    weights : torch.Tensor [batch_size, batch_size]
        Similarity weights (higher = more similar logFC)
    """
    # Compute logFC for anchor [batch_size, n_genes]
    anchor_logfc = anchor_expr - dmso_mean.unsqueeze(0)
    
    # Compute logFC for all others [batch_size, batch_size, n_genes]
    other_logfc = other_expr - dmso_mean.unsqueeze(0).unsqueeze(0)
    
    # Expand anchor for pairwise comparison [batch_size, 1, n_genes]
    anchor_logfc_expanded = anchor_logfc.unsqueeze(1)
    
    # Pairwise distance [batch_size, batch_size]
    logfc_distance = torch.norm(anchor_logfc_expanded - other_logfc, p=2, dim=2)
    
    # Normalize by number of genes
    logfc_distance = logfc_distance / np.sqrt(anchor_expr.shape[-1])
    
    # Convert distance to similarity weight
    # Small distance -> high weight, Large distance -> low weight
    weights = torch.exp(-beta * torch.clamp(logfc_distance, max=20.0))
    
    return weights


def logfc_weighted_info_nce_loss(
    z: torch.Tensor,
    expr: torch.Tensor,
    labels: torch.Tensor,
    dmso_mean: torch.Tensor,
    temperature: float = 0.1,
    logfc_beta: float = 0.1,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    InfoNCE loss with logFC-based weighting.
    
    Uses projection space (z) for contrastive learning while incorporating
    biological similarity (logFC) for loss weighting.
    
    Parameters:
    -----------
    z : torch.Tensor [batch_size, projection_dim]
        Projected latent representations (from projection head)
    expr : torch.Tensor [batch_size, n_genes]
        Expression values (for logFC weighting)
    labels : torch.Tensor [batch_size]
        Treatment labels (for identifying replicates)
    dmso_mean : torch.Tensor [n_genes]
        Mean DMSO baseline expression
    temperature : float
        Temperature for similarity scaling
    logfc_beta : float
        Temperature for logFC weighting
    normalize : bool
        Whether to L2-normalize latent vectors
    
    Returns:
    --------
    loss : torch.Tensor
        LogFC-weighted InfoNCE loss
    avg_weight : torch.Tensor
        Average logFC weight (for monitoring)
    """
    batch_size = z.shape[0]
    
    if batch_size < 2:
        return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    
    # Normalize embeddings
    if normalize:
        z = F.normalize(z, p=2, dim=1)
    
    # Compute cosine similarity matrix [batch_size, batch_size]
    similarity_matrix = torch.mm(z, z.t()) / temperature
    
    # Create positive pair mask (same treatment)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels, labels.t()).float().to(z.device)
    pos_mask = pos_mask - torch.eye(batch_size, device=z.device)  # Remove diagonal
    
    # Compute standard InfoNCE loss
    exp_sim = torch.exp(similarity_matrix)
    
    # Positive similarities (numerator)
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    
    # All similarities except self (denominator)
    all_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
    
    # Samples with at least one positive
    has_positives = pos_mask.sum(dim=1) > 0
    
    if has_positives.sum() == 0:
        return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    
    # Standard InfoNCE loss per sample
    sample_losses = -torch.log(pos_sim[has_positives] / (all_sim[has_positives] + 1e-8))
    
    # Compute logFC-based weights for positives only
    expr_expanded = expr.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, batch_size, n_genes]
    logfc_weights = compute_logfc_similarity_weights(
        expr, 
        expr_expanded,
        dmso_mean,
        beta=logfc_beta
    )
    
    # Average logFC weight for each sample's positives
    pos_logfc_weights = logfc_weights * pos_mask
    num_positives = pos_mask.sum(dim=1).clamp(min=1e-8)
    avg_pos_weight = pos_logfc_weights.sum(dim=1) / num_positives
    
    # Weight the InfoNCE loss by logFC consistency
    # Samples with similar logFC patterns (high weight) contribute more
    weighted_losses = sample_losses * avg_pos_weight[has_positives]
    loss = weighted_losses.mean()
    
    # Average weight for monitoring
    avg_weight = avg_pos_weight[has_positives].mean() if has_positives.any() else torch.tensor(0.0, device=z.device)
    
    return loss, avg_weight


def contrastive_vae_logfc_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z_proj: torch.Tensor,
    labels: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 0.1,
    logfc_beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for Contrastive VAE with LogFC-weighted InfoNCE.
    
    Total Loss = Reconstruction + β * KL + γ * LogFC-InfoNCE
    
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
    dmso_mean : torch.Tensor
        Mean DMSO baseline expression
    beta : float
        Weight for KL divergence (β-VAE)
    gamma : float
        Weight for InfoNCE contrastive loss
    temperature : float
        Temperature for InfoNCE
    logfc_beta : float
        Temperature for logFC weighting
    
    Returns:
    --------
    total_loss, recon_loss, kl_loss, contrastive_loss, avg_weight
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # LogFC-weighted InfoNCE contrastive loss
    contrastive_loss, avg_weight = logfc_weighted_info_nce_loss(
        z_proj, x, labels, dmso_mean,
        temperature=temperature,
        logfc_beta=logfc_beta
    )
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + gamma * contrastive_loss
    
    return total_loss, recon_loss, kl_loss, contrastive_loss, avg_weight

