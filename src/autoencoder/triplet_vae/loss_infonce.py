"""
LogFC-weighted InfoNCE loss for Triplet VAE.

Combines InfoNCE contrastive learning with logFC-based weighting.
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
    Lower weight = more different logFC patterns (should be farther in latent space)
    
    Parameters:
    -----------
    anchor_expr : torch.Tensor [batch_size, n_genes]
        Anchor expression values
    other_expr : torch.Tensor [batch_size, n_genes] or [batch_size, batch_size, n_genes]
        Other expression values (can be batch or pairwise)
    dmso_mean : torch.Tensor [n_genes]
        Mean DMSO expression (baseline)
    beta : float
        Temperature for exponential weighting
    
    Returns:
    --------
    weights : torch.Tensor
        Similarity weights (higher = more similar logFC)
    """
    # Compute logFC for anchor
    anchor_logfc = anchor_expr - dmso_mean.unsqueeze(0)  # [batch_size, n_genes]
    
    # Handle different shapes
    if other_expr.dim() == 2:
        # [batch_size, n_genes]
        other_logfc = other_expr - dmso_mean.unsqueeze(0)
        # Pairwise distance
        logfc_distance = torch.norm(anchor_logfc - other_logfc, p=2, dim=1)
    elif other_expr.dim() == 3:
        # [batch_size, batch_size, n_genes]
        other_logfc = other_expr - dmso_mean.unsqueeze(0).unsqueeze(0)
        anchor_logfc_expanded = anchor_logfc.unsqueeze(1)  # [batch_size, 1, n_genes]
        # Pairwise distance
        logfc_distance = torch.norm(anchor_logfc_expanded - other_logfc, p=2, dim=2)
    else:
        raise ValueError(f"Unexpected shape for other_expr: {other_expr.shape}")
    
    # Normalize by number of genes
    logfc_distance = logfc_distance / np.sqrt(anchor_expr.shape[-1])
    
    # Convert distance to similarity weight
    # Small distance -> high weight
    # Large distance -> low weight
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
    
    For each anchor:
    - Positives: replicates (same label), weighted by logFC similarity
    - Negatives: all others, weighted by logFC dissimilarity
    
    Parameters:
    -----------
    z : torch.Tensor [batch_size, latent_dim]
        Latent representations
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
    
    # Compute logFC-based weights for all pairs
    # Shape: [batch_size, batch_size]
    expr_expanded = expr.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, batch_size, n_genes]
    logfc_weights = compute_logfc_similarity_weights(
        expr, 
        expr_expanded,
        dmso_mean,
        beta=logfc_beta
    )
    
    # For positives: use logFC weight as-is (high weight = similar logFC)
    # For negatives: invert weight (high weight = different logFC, should be pushed apart more)
    neg_mask = 1.0 - pos_mask - torch.eye(batch_size, device=z.device)
    
    # Weight positive similarities
    pos_weights = logfc_weights * pos_mask
    
    # Weight negative similarities (inverted)
    neg_weights = (1.0 - logfc_weights) * neg_mask
    
    # Compute weighted InfoNCE
    exp_sim = torch.exp(similarity_matrix)
    
    # Weighted positive similarities (numerator)
    weighted_pos_sim = (exp_sim * pos_weights).sum(dim=1)
    
    # Weighted negative similarities + unweighted positives (denominator)
    # We include positives in denominator but with their weights
    weighted_all_sim = (exp_sim * (pos_weights + neg_weights)).sum(dim=1)
    
    # Samples with at least one positive
    has_positives = pos_mask.sum(dim=1) > 0
    
    if has_positives.sum() == 0:
        return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    
    # InfoNCE loss: -log(weighted positive / weighted total)
    loss = -torch.log(weighted_pos_sim[has_positives] / (weighted_all_sim[has_positives] + 1e-8))
    loss = loss.mean()
    
    # Average weight for monitoring
    avg_weight = logfc_weights[pos_mask > 0].mean() if (pos_mask > 0).any() else torch.tensor(0.0, device=z.device)
    
    return loss, avg_weight


def triplet_vae_infonce_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    labels: torch.Tensor,
    dmso_mean: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 0.1,
    logfc_beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for Triplet VAE with LogFC-weighted InfoNCE.
    
    Total Loss = Reconstruction + β * KL + γ * LogFC-InfoNCE
    
    Parameters:
    -----------
    recon_x : torch.Tensor
        Reconstructed expression
    x : torch.Tensor
        Original expression
    mu : torch.Tensor
        Latent mean (used as latent representation)
    logvar : torch.Tensor
        Latent log variance
    labels : torch.Tensor
        Treatment labels
    dmso_mean : torch.Tensor
        DMSO baseline
    beta : float
        KL weight
    gamma : float
        InfoNCE weight
    temperature : float
        InfoNCE temperature
    logfc_beta : float
        LogFC weighting temperature
    
    Returns:
    --------
    total_loss, recon_loss, kl_loss, infonce_loss, avg_weight
    """
    batch_size = x.size(0)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    # LogFC-weighted InfoNCE
    infonce_loss, avg_weight = logfc_weighted_info_nce_loss(
        mu, x, labels, dmso_mean,
        temperature=temperature,
        logfc_beta=logfc_beta,
        normalize=True
    )
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + gamma * infonce_loss
    
    return total_loss, recon_loss, kl_loss, infonce_loss, avg_weight

