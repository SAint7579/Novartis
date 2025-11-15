"""
Training utilities for VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm

from .model import VAE, vae_loss


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    device: str = 'cpu',
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train VAE model.
    
    Parameters:
    -----------
    model : VAE
        VAE model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    beta : float
        Beta parameter for beta-VAE (weight of KL term)
    device : str
        Device to use ('cpu' or 'cuda')
    patience : int
        Early stopping patience
    save_path : str, optional
        Path to save best model
    verbose : bool
        Print training progress
    
    Returns:
    --------
    history : dict
        Training history with losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
        
        for batch_data, _ in pbar:
            batch_data = batch_data.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(batch_data)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kl_losses.append(kl_loss.item())
            
            if verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
        
        # Record average training losses
        history['train_loss'].append(np.mean(train_losses))
        history['train_recon_loss'].append(np.mean(train_recon_losses))
        history['train_kl_loss'].append(np.mean(train_kl_losses))
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_recon_losses = []
            val_kl_losses = []
            
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    batch_data = batch_data.to(device)
                    
                    recon_batch, mu, logvar = model(batch_data)
                    loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar, beta)
                    
                    val_losses.append(loss.item())
                    val_recon_losses.append(recon_loss.item())
                    val_kl_losses.append(kl_loss.item())
            
            # Record average validation losses
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            history['val_recon_loss'].append(np.mean(val_recon_losses))
            history['val_kl_loss'].append(np.mean(val_kl_losses))
            
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
                    }, save_path)
                    if verbose:
                        print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')
            else:
                patience_counter += 1
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'train_loss: {history["train_loss"][-1]:.4f} - '
                      f'val_loss: {avg_val_loss:.4f} - '
                      f'patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                break
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'train_loss: {history["train_loss"][-1]:.4f}')
    
    return history


def evaluate_vae(
    model: VAE,
    data_loader: DataLoader,
    device: str = 'cpu',
    beta: float = 1.0
) -> Dict[str, float]:
    """
    Evaluate VAE on data.
    
    Parameters:
    -----------
    model : VAE
        Trained VAE model
    data_loader : DataLoader
        Data loader for evaluation
    device : str
        Device to use
    beta : float
        Beta parameter for loss calculation
    
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            
            recon_batch, mu, logvar = model(batch_data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches
    }

