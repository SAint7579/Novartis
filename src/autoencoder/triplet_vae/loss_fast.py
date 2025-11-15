"""
Fast loss functions using pre-computed logFC.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_logfc_weights_from_precomputed(
    anchor_logfc: torch.Tensor,
    positive_logfc: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Compute positive weights from pre-computed logFC."""
    logfc_distance = torch.norm(anchor_logfc - positive_logfc, p=2, dim=1)
    logfc_distance = logfc_distance / np.sqrt(anchor_logfc.shape[1])
    weights = torch.exp(-beta * torch.clamp(logfc_distance, max=20.0))
    return weights


def compute_dmso_weights_from_precomputed(
    anchor_logfc: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Compute DMSO weights from pre-computed logFC."""
    logfc_magnitude = torch.norm(anchor_logfc, p=2, dim=1)
    logfc_magnitude = logfc_magnitude / np.sqrt(anchor_logfc.shape[1])
    weights = 1 - torch.exp(-beta * torch.clamp(logfc_magnitude, max=20.0))
    return weights


def compute_compound_weights_from_precomputed(
    anchor_logfc: torch.Tensor,
    compound_logfc: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Compute compound negative weights from pre-computed logFC."""
    logfc_distance = torch.norm(anchor_logfc - compound_logfc, p=2, dim=1)
    logfc_distance = logfc_distance / np.sqrt(anchor_logfc.shape[1])
    weights = 1 - torch.exp(-beta * torch.clamp(logfc_distance, max=20.0))
    return weights


def triplet_vae_loss_fast(
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
    anchor_logfc: torch.Tensor,
    positive_logfc: torch.Tensor,
    dmso_logfc: torch.Tensor,
    compound_logfc: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    margin: float = 1.0,
    logfc_beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast loss using pre-computed logFC.
    """
    batch_size = anchor_expr.size(0)
    
    # Reconstruction loss
    recon_loss = (
        F.mse_loss(recon_anchor, anchor_expr, reduction='sum') +
        F.mse_loss(recon_positive, positive_expr, reduction='sum') +
        F.mse_loss(recon_dmso, dmso_expr, reduction='sum') +
        F.mse_loss(recon_compound, compound_expr, reduction='sum')
    ) / (4 * batch_size)
    
    # KL divergence
    kl_loss = (
        -0.5 * torch.sum(1 + logvar_anchor - mu_anchor.pow(2) - logvar_anchor.exp()) +
        -0.5 * torch.sum(1 + logvar_positive - mu_positive.pow(2) - logvar_positive.exp()) +
        -0.5 * torch.sum(1 + logvar_dmso - mu_dmso.pow(2) - logvar_dmso.exp()) +
        -0.5 * torch.sum(1 + logvar_compound - mu_compound.pow(2) - logvar_compound.exp())
    ) / (4 * batch_size)
    
    # Compute weights from PRE-COMPUTED logFC (fast!)
    pos_weight = compute_logfc_weights_from_precomputed(anchor_logfc, positive_logfc, beta=logfc_beta)
    dmso_weight = compute_dmso_weights_from_precomputed(anchor_logfc, beta=logfc_beta)
    compound_weight = compute_compound_weights_from_precomputed(anchor_logfc, compound_logfc, beta=logfc_beta)
    
    # Cosine distances
    mu_anchor_norm = F.normalize(mu_anchor, p=2, dim=1)
    mu_positive_norm = F.normalize(mu_positive, p=2, dim=1)
    mu_dmso_norm = F.normalize(mu_dmso, p=2, dim=1)
    mu_compound_norm = F.normalize(mu_compound, p=2, dim=1)
    
    dist_ap = 1 - (mu_anchor_norm * mu_positive_norm).sum(dim=1)
    dist_an_dmso = 1 - (mu_anchor_norm * mu_dmso_norm).sum(dim=1)
    dist_an_compound = 1 - (mu_anchor_norm * mu_compound_norm).sum(dim=1)
    
    # Weighted losses
    loss_positive = pos_weight * dist_ap
    loss_dmso = dmso_weight * F.relu(margin - dist_an_dmso)
    loss_compound = compound_weight * F.relu(margin - dist_an_compound)
    
    triplet_loss = (loss_positive + loss_dmso + loss_compound).mean()
    
    # Total
    total_loss = recon_loss + beta * kl_loss + gamma * triplet_loss
    
    return total_loss, recon_loss, kl_loss, triplet_loss, pos_weight.mean()

