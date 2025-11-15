"""
Triplet loss weighted by logFC for biological relevance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_logfc_weights_positive(
    anchor_expr: torch.Tensor,
    positive_expr: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute weights for anchor-positive pairs.
    
    Weight = exp(-β * d_logFC(anchor, positive))
    Small logFC difference → HIGH weight (replicates should be similar)
    
    Parameters:
    -----------
    anchor_expr : torch.Tensor
        Anchor expression [batch, genes]
    positive_expr : torch.Tensor
        Positive expression [batch, genes]
    dmso_mean : torch.Tensor
        DMSO mean expression [genes]
    beta : float
        Temperature parameter
    
    Returns:
    --------
    weights : torch.Tensor
        Weights [batch]
    """
    anchor_logfc = anchor_expr - dmso_mean
    positive_logfc = positive_expr - dmso_mean
    
    logfc_distance = torch.norm(anchor_logfc - positive_logfc, p=2, dim=1)
    logfc_distance = logfc_distance / np.sqrt(anchor_expr.shape[1])
    
    # Small distance → high weight
    weights = torch.exp(-beta * torch.clamp(logfc_distance, max=20.0))
    
    return weights


def compute_logfc_weights_dmso_negative(
    anchor_expr: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute weights for anchor-DMSO pairs.
    
    Weight = 1 - exp(-β * ||logFC(anchor)||)
    Large perturbation effect → HIGH weight (should be far from DMSO)
    
    Parameters:
    -----------
    anchor_expr : torch.Tensor
        Anchor expression [batch, genes]
    dmso_mean : torch.Tensor
        DMSO mean expression [genes]
    beta : float
        Temperature parameter
    
    Returns:
    --------
    weights : torch.Tensor
        Weights [batch]
    """
    anchor_logfc = anchor_expr - dmso_mean
    
    # Magnitude of perturbation
    logfc_magnitude = torch.norm(anchor_logfc, p=2, dim=1)
    logfc_magnitude = logfc_magnitude / np.sqrt(anchor_expr.shape[1])
    
    # Large perturbation → high weight (should be far from baseline)
    weights = 1 - torch.exp(-beta * torch.clamp(logfc_magnitude, max=20.0))
    
    return weights


def compute_logfc_weights_compound_negative(
    anchor_expr: torch.Tensor,
    negative_expr: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute weights for anchor-other_compound pairs.
    
    Weight = 1 - exp(-β * d_logFC(anchor, negative))
    Large logFC difference → HIGH weight (different biology, should be far)
    
    Parameters:
    -----------
    anchor_expr : torch.Tensor
        Anchor expression [batch, genes]
    negative_expr : torch.Tensor
        Negative compound expression [batch, genes]
    dmso_mean : torch.Tensor
        DMSO mean expression [genes]
    beta : float
        Temperature parameter
    
    Returns:
    --------
    weights : torch.Tensor
        Weights [batch]
    """
    anchor_logfc = anchor_expr - dmso_mean
    negative_logfc = negative_expr - dmso_mean
    
    logfc_distance = torch.norm(anchor_logfc - negative_logfc, p=2, dim=1)
    logfc_distance = logfc_distance / np.sqrt(anchor_expr.shape[1])
    
    # Large difference → high weight (different biology, enforce separation)
    weights = 1 - torch.exp(-beta * torch.clamp(logfc_distance, max=20.0))
    
    return weights


def weighted_multi_negative_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative_dmso: torch.Tensor,
    negative_compound: torch.Tensor,
    pos_weight: torch.Tensor,
    dmso_weight: torch.Tensor,
    compound_weight: torch.Tensor,
    margin: float = 1.0,
    use_cosine: bool = True
) -> torch.Tensor:
    """
    Multi-negative triplet loss with separate logFC weights for each component.
    
    Loss = pos_weight × d(a,p) + 
           dmso_weight × max(0, margin - d(a,n_dmso)) +
           compound_weight × max(0, margin - d(a,n_compound))
    
    Where:
    - pos_weight: Small logFC(a,p) → HIGH (enforce similarity)
    - dmso_weight: Large ||logFC(a)|| → HIGH (strong effect, far from baseline)
    - compound_weight: Large logFC(a,n) → HIGH (different biology, separate)
    
    Parameters:
    -----------
    anchor : torch.Tensor
        Anchor latent [batch, latent_dim]
    positive : torch.Tensor
        Positive latent [batch, latent_dim]
    negative_dmso : torch.Tensor
        DMSO negative latent [batch, latent_dim]
    negative_compound : torch.Tensor
        Other compound negative latent [batch, latent_dim]
    pos_weight : torch.Tensor
        Weights for anchor-positive [batch]
    dmso_weight : torch.Tensor
        Weights for anchor-DMSO [batch]
    compound_weight : torch.Tensor
        Weights for anchor-compound [batch]
    margin : float
        Triplet margin
    use_cosine : bool
        Use cosine distance
    
    Returns:
    --------
    loss : torch.Tensor
        Weighted loss
    """
    if use_cosine:
        # Normalize to unit sphere
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_dmso_norm = F.normalize(negative_dmso, p=2, dim=1)
        negative_compound_norm = F.normalize(negative_compound, p=2, dim=1)
        
        # Cosine similarities
        cosine_sim_ap = (anchor_norm * positive_norm).sum(dim=1)
        cosine_sim_an_dmso = (anchor_norm * negative_dmso_norm).sum(dim=1)
        cosine_sim_an_compound = (anchor_norm * negative_compound_norm).sum(dim=1)
        
        # Convert to distances
        dist_ap = 1 - cosine_sim_ap
        dist_an_dmso = 1 - cosine_sim_an_dmso
        dist_an_compound = 1 - cosine_sim_an_compound
    else:
        # Euclidean distances
        dist_ap = F.pairwise_distance(anchor, positive, p=2)
        dist_an_dmso = F.pairwise_distance(anchor, negative_dmso, p=2)
        dist_an_compound = F.pairwise_distance(anchor, negative_compound, p=2)
    
    # Separate losses with separate weights
    # Pull together: anchor and positive (weighted by similarity)
    loss_positive = pos_weight * dist_ap
    
    # Push apart: anchor from DMSO (weighted by perturbation strength)
    loss_dmso = dmso_weight * F.relu(margin - dist_an_dmso)
    
    # Push apart: anchor from other compounds (weighted by biological difference)
    loss_compound = compound_weight * F.relu(margin - dist_an_compound)
    
    # Combined loss
    total_loss = loss_positive + loss_dmso + loss_compound
    
    return total_loss.mean()


# Keep old function for backward compatibility
def weighted_triplet_loss(anchor, positive, negative, weights, margin=1.0, use_cosine=True):
    """Single-negative version for backward compatibility."""
    return weighted_multi_negative_triplet_loss(
        anchor, positive, negative, negative, 
        weights, weights, weights,  # Use same weight for all
        margin, use_cosine
    )


def triplet_vae_loss(
    recon_anchor: torch.Tensor,
    recon_positive: torch.Tensor,
    recon_dmso: torch.Tensor,
    recon_compound: torch.Tensor,
    anchor_expr: torch.Tensor,
    positive_expr: torch.Tensor,
    dmso_expr: torch.Tensor,
    compound_expr: torch.Tensor,
    mu_anchor: torch.Tensor,
    logvar_anchor: torch.Tensor,
    mu_positive: torch.Tensor,
    logvar_positive: torch.Tensor,
    mu_dmso: torch.Tensor,
    logvar_dmso: torch.Tensor,
    mu_compound: torch.Tensor,
    logvar_compound: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    margin: float = 1.0,
    logfc_beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete loss for Triplet VAE with dual negatives.
    Reconstructs ALL samples (anchor, positive, DMSO, other compound).
    """
    # Reconstruction loss for ALL samples
    batch_size = anchor_expr.size(0)
    recon_loss = (
        F.mse_loss(recon_anchor, anchor_expr, reduction='sum') +
        F.mse_loss(recon_positive, positive_expr, reduction='sum') +
        F.mse_loss(recon_dmso, dmso_expr, reduction='sum') +
        F.mse_loss(recon_compound, compound_expr, reduction='sum')
    ) / (4 * batch_size)  # Average over all 4 samples
    
    # KL divergence for ALL samples
    kl_loss = (
        -0.5 * torch.sum(1 + logvar_anchor - mu_anchor.pow(2) - logvar_anchor.exp()) +
        -0.5 * torch.sum(1 + logvar_positive - mu_positive.pow(2) - logvar_positive.exp()) +
        -0.5 * torch.sum(1 + logvar_dmso - mu_dmso.pow(2) - logvar_dmso.exp()) +
        -0.5 * torch.sum(1 + logvar_compound - mu_compound.pow(2) - logvar_compound.exp())
    ) / (4 * batch_size)
    
    # Compute logFC-based weights for each component
    # 1. Positive: small logFC diff → high weight (enforce similarity)
    pos_weight = compute_logfc_weights_positive(anchor_expr, positive_expr, dmso_mean, beta=logfc_beta)
    
    # 2. DMSO: large perturbation → high weight (enforce separation from baseline)
    dmso_weight = compute_logfc_weights_dmso_negative(anchor_expr, dmso_mean, beta=logfc_beta)
    
    # 3. Compound: large logFC diff → high weight (different biology, enforce separation)
    compound_weight = compute_logfc_weights_compound_negative(anchor_expr, compound_expr, dmso_mean, beta=logfc_beta)
    
    # Multi-negative triplet loss with separate weights
    triplet_loss = weighted_multi_negative_triplet_loss(
        mu_anchor, mu_positive, 
        mu_dmso, mu_compound,
        pos_weight, dmso_weight, compound_weight,
        margin=margin, use_cosine=True
    )
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + gamma * triplet_loss
    
    avg_weight = pos_weight.mean()  # Monitor positive weights
    
    return total_loss, recon_loss, kl_loss, triplet_loss, avg_weight

