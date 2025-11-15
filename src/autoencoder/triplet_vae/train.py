"""
Training utilities for Triplet VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm

from .model import TripletVAE
from .loss import triplet_vae_loss


def train_triplet_vae(
    model: TripletVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    gamma: float = 1.0,
    margin: float = 1.0,
    logfc_beta: float = 1.0,
    device: str = 'cpu',
    patience: int = 15,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train Triplet VAE with logFC-weighted triplet loss.
    
    Parameters:
    -----------
    model : TripletVAE
        Model to train
    train_loader : DataLoader
        Training data (returns triplets)
    val_loader : DataLoader, optional
        Validation data
    epochs : int
        Training epochs
    learning_rate : float
        Learning rate
    beta : float
        KL divergence weight
    gamma : float
        Triplet loss weight
    margin : float
        Triplet margin
    logfc_beta : float
        LogFC weighting temperature
    device : str
        Device
    patience : int
        Early stopping patience
    save_path : str, optional
        Save path for best model
    verbose : bool
        Print progress
    
    Returns:
    --------
    history : dict
        Training history
    """
    model = model.to(device)
    dmso_mean = train_loader.dataset.dmso_mean.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_triplet_loss': [],
        'train_avg_weight': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_triplet_loss': [],
        'val_avg_weight': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_recon = []
        train_kl = []
        train_triplet = []
        train_weights = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        
        for anchor, positive, negative_dmso, negative_compound in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative_dmso = negative_dmso.to(device)
            negative_compound = negative_compound.to(device)
            
            # Forward pass for all four - RECONSTRUCT ALL
            recon_anchor, mu_anchor, logvar_anchor = model(anchor)
            recon_positive, mu_positive, logvar_positive = model(positive)
            recon_dmso, mu_dmso, logvar_dmso = model(negative_dmso)
            recon_compound, mu_compound, logvar_compound = model(negative_compound)
            
            # Compute loss - now reconstructs ALL samples including DMSO
            loss, recon_loss, kl_loss, triplet_loss, avg_weight = triplet_vae_loss(
                recon_anchor, recon_positive, recon_dmso, recon_compound,
                anchor, positive, negative_dmso, negative_compound,
                mu_anchor, logvar_anchor, mu_positive, logvar_positive,
                mu_dmso, logvar_dmso, mu_compound, logvar_compound,
                dmso_mean,
                beta=beta, gamma=gamma, margin=margin, logfc_beta=logfc_beta
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Record
            train_losses.append(loss.item())
            train_recon.append(recon_loss.item())
            train_kl.append(kl_loss.item())
            train_triplet.append(triplet_loss.item())
            train_weights.append(avg_weight.item())
            
            if verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.2f}',
                    'triplet': f'{triplet_loss.item():.3f}',
                    'weight': f'{avg_weight.item():.3f}'
                })
        
        history['train_loss'].append(np.mean(train_losses))
        history['train_recon_loss'].append(np.mean(train_recon))
        history['train_kl_loss'].append(np.mean(train_kl))
        history['train_triplet_loss'].append(np.mean(train_triplet))
        history['train_avg_weight'].append(np.mean(train_weights))
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            val_recon = []
            val_kl = []
            val_triplet = []
            val_weights = []
            
            val_dmso_mean = val_loader.dataset.dmso_mean.to(device)
            
            with torch.no_grad():
                for anchor, positive, negative_dmso, negative_compound in val_loader:
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative_dmso = negative_dmso.to(device)
                    negative_compound = negative_compound.to(device)
                    
                    recon_anchor, mu_anchor, logvar_anchor = model(anchor)
                    recon_positive, mu_positive, logvar_positive = model(positive)
                    recon_dmso, mu_dmso, logvar_dmso = model(negative_dmso)
                    recon_compound, mu_compound, logvar_compound = model(negative_compound)
                    
                    loss, recon_loss, kl_loss, triplet_loss, avg_weight = triplet_vae_loss(
                        recon_anchor, recon_positive, recon_dmso, recon_compound,
                        anchor, positive, negative_dmso, negative_compound,
                        mu_anchor, logvar_anchor, mu_positive, logvar_positive,
                        mu_dmso, logvar_dmso, mu_compound, logvar_compound,
                        val_dmso_mean,
                        beta=beta, gamma=gamma, margin=margin, logfc_beta=logfc_beta
                    )
                    
                    val_losses.append(loss.item())
                    val_recon.append(recon_loss.item())
                    val_kl.append(kl_loss.item())
                    val_triplet.append(triplet_loss.item())
                    val_weights.append(avg_weight.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            history['val_recon_loss'].append(np.mean(val_recon))
            history['val_kl_loss'].append(np.mean(val_kl))
            history['val_triplet_loss'].append(np.mean(val_triplet))
            history['val_avg_weight'].append(np.mean(val_weights))
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'dmso_mean': train_loader.dataset.dmso_mean
                    }, save_path)
                    if verbose:
                        print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')
            else:
                patience_counter += 1
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'train: {history["train_loss"][-1]:.2f} - '
                      f'val: {avg_val_loss:.2f} - '
                      f'triplet: {history["val_triplet_loss"][-1]:.3f} - '
                      f'weight: {history["val_avg_weight"][-1]:.3f} - '
                      f'patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - train: {history["train_loss"][-1]:.2f}')
    
    return history

