"""
Train Triplet VAE with LogFC-weighted InfoNCE on HEK293T data.

Uses InfoNCE contrastive learning with logFC-based weighting instead of 
hinge-based triplet loss.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths
project_root = Path(__file__).resolve().parent
os.chdir(str(project_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autoencoder.vae import VAE, preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveGeneExpressionDataset
from src.autoencoder.triplet_vae.loss_infonce import triplet_vae_infonce_loss

# =============================================================================
# Configuration
# =============================================================================

# Data paths
COUNTS_CSV = 'Dataset/csv/HEK293T_Counts.csv'
METADATA_XLSX = 'Dataset/HEK293T_MetaData.xlsx'
OUTPUT_MODEL = 'models/triplet_vae_infonce_hek293t_best.pt'

# Model hyperparameters
INPUT_DIM = None  # Will be set from data
LATENT_DIM = 64
HIDDEN_DIMS = [512, 256, 128]

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 15

# Loss weights
BETA = 1.0           # KL weight
GAMMA = 2.0          # InfoNCE weight
TEMPERATURE = 0.1    # InfoNCE temperature
LOGFC_BETA = 0.1     # LogFC weighting temperature

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# =============================================================================
# Load and preprocess data
# =============================================================================

print("\nLoading data...")
counts_df = pd.read_csv(COUNTS_CSV, header=1, index_col=0)
metadata = pd.read_excel(METADATA_XLSX, header=1)

print("Preprocessing...")
processed_df, scaler = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

# processed_df is [genes x samples], need to transpose to [samples x genes]
processed_df_T = processed_df.T
# Reset index to match metadata row indices
processed_df_T = processed_df_T.reset_index(drop=True)

INPUT_DIM = processed_df_T.shape[1]
print(f"Input dimension: {INPUT_DIM}")
print(f"Number of samples: {processed_df_T.shape[0]}")

# Compute DMSO mean for logFC weighting
dmso_mask = metadata['treatment'] == 'DMSO'
dmso_data = processed_df_T[dmso_mask]  # Now indices align
dmso_mean = torch.tensor(dmso_data.mean(axis=0).values, dtype=torch.float32).to(DEVICE)
print(f"DMSO samples: {dmso_mask.sum()}")

# Create dataset (reuse ContrastiveGeneExpressionDataset)
dataset = ContrastiveGeneExpressionDataset(
    processed_df_T,
    metadata['treatment']
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == 'cuda')
)

print(f"Dataset size: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {len(dataloader)}")

# =============================================================================
# Initialize model
# =============================================================================

print("\nInitializing model...")
model = VAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# Training loop
# =============================================================================

print("\n" + "="*70)
print("Training Triplet VAE with LogFC-weighted InfoNCE")
print("="*70)

best_loss = float('inf')
patience_counter = 0

history = {
    'train_loss': [],
    'recon_loss': [],
    'kl_loss': [],
    'infonce_loss': [],
    'avg_weight': []
}

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'infonce': 0.0,
        'weight': 0.0
    }
    
    for batch_idx, (expr, labels, _) in enumerate(dataloader):
        expr = expr.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = model(expr)
        
        # Compute loss
        loss, recon_loss, kl_loss, infonce_loss, avg_weight = triplet_vae_infonce_loss(
            recon, expr, mu, logvar, labels, dmso_mean,
            beta=BETA,
            gamma=GAMMA,
            temperature=TEMPERATURE,
            logfc_beta=LOGFC_BETA
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        epoch_losses['total'] += loss.item()
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl'] += kl_loss.item()
        epoch_losses['infonce'] += infonce_loss.item()
        epoch_losses['weight'] += avg_weight.item()
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = epoch_losses['total'] / num_batches
    avg_recon = epoch_losses['recon'] / num_batches
    avg_kl = epoch_losses['kl'] / num_batches
    avg_infonce = epoch_losses['infonce'] / num_batches
    avg_weight = epoch_losses['weight'] / num_batches
    
    # Update scheduler
    scheduler.step(avg_loss)
    
    # Record history
    history['train_loss'].append(avg_loss)
    history['recon_loss'].append(avg_recon)
    history['kl_loss'].append(avg_kl)
    history['infonce_loss'].append(avg_infonce)
    history['avg_weight'].append(avg_weight)
    
    # Print progress
    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Recon: {avg_recon:.4f} | "
          f"KL: {avg_kl:.4f} | "
          f"InfoNCE: {avg_infonce:.4f} | "
          f"Weight: {avg_weight:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        
        # Create models directory if needed
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'history': history,
            'config': {
                'input_dim': INPUT_DIM,
                'latent_dim': LATENT_DIM,
                'hidden_dims': HIDDEN_DIMS,
                'beta': BETA,
                'gamma': GAMMA,
                'temperature': TEMPERATURE,
                'logfc_beta': LOGFC_BETA
            }
        }, OUTPUT_MODEL)
        
        print(f"  → Saved best model (loss: {best_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# =============================================================================
# Training complete
# =============================================================================

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"Best loss: {best_loss:.4f}")
print(f"Model saved to: {OUTPUT_MODEL}")
print(f"Total epochs: {len(history['train_loss'])}")

