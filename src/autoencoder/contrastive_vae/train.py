"""
Training utilities for Contrastive VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm

from .model import ContrastiveVAE
from .loss import contrastive_vae_loss


def train_contrastive_vae(
    model: ContrastiveVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 0.1,
    device: str = 'cpu',
    patience: int = 15,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train Contrastive VAE model.
    
    Parameters:
    -----------
    model : ContrastiveVAE
        Contrastive VAE model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    beta : float
        Weight for KL divergence term
    gamma : float
        Weight for InfoNCE contrastive term
    temperature : float
        Temperature for InfoNCE loss
    device : str
        Device ('cpu' or 'cuda')
    patience : int
        Early stopping patience
    save_path : str, optional
        Path to save best model
    verbose : bool
        Print training progress
    
    Returns:
    --------
    history : dict
        Training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_contrastive_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_contrastive_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []
        train_contrastive_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        
        for batch_data, batch_labels, _ in pbar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            recon_batch, mu, logvar, z_proj = model(batch_data, return_projection=True)
            
            # Compute loss
            loss, recon_loss, kl_loss, contrastive_loss = contrastive_vae_loss(
                recon_batch, batch_data, mu, logvar, z_proj, batch_labels,
                beta=beta, gamma=gamma, temperature=temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kl_losses.append(kl_loss.item())
            train_contrastive_losses.append(contrastive_loss.item())
            
            if verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.2f}',
                    'recon': f'{recon_loss.item():.2f}',
                    'kl': f'{kl_loss.item():.2f}',
                    'contrast': f'{contrastive_loss.item():.3f}'
                })
        
        # Record average training losses
        history['train_loss'].append(np.mean(train_losses))
        history['train_recon_loss'].append(np.mean(train_recon_losses))
        history['train_kl_loss'].append(np.mean(train_kl_losses))
        history['train_contrastive_loss'].append(np.mean(train_contrastive_losses))
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_recon_losses = []
            val_kl_losses = []
            val_contrastive_losses = []
            
            with torch.no_grad():
                for batch_data, batch_labels, _ in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    recon_batch, mu, logvar, z_proj = model(batch_data, return_projection=True)
                    loss, recon_loss, kl_loss, contrastive_loss = contrastive_vae_loss(
                        recon_batch, batch_data, mu, logvar, z_proj, batch_labels,
                        beta=beta, gamma=gamma, temperature=temperature
                    )
                    
                    val_losses.append(loss.item())
                    val_recon_losses.append(recon_loss.item())
                    val_kl_losses.append(kl_loss.item())
                    val_contrastive_losses.append(contrastive_loss.item())
            
            # Record average validation losses
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            history['val_recon_loss'].append(np.mean(val_recon_losses))
            history['val_kl_loss'].append(np.mean(val_kl_losses))
            history['val_contrastive_loss'].append(np.mean(val_contrastive_losses))
            
            # Learning rate scheduling
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
                        'treatment_to_idx': model.treatment_to_idx if hasattr(model, 'treatment_to_idx') else None,
                    }, save_path)
                    if verbose:
                        print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')
            else:
                patience_counter += 1
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'train_loss: {history["train_loss"][-1]:.4f} - '
                      f'val_loss: {avg_val_loss:.4f} - '
                      f'contrast: {history["val_contrastive_loss"][-1]:.4f} - '
                      f'patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                break
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'train_loss: {history["train_loss"][-1]:.4f} - '
                      f'contrast: {history["train_contrastive_loss"][-1]:.4f}')
    
    return history

