"""
Train Contrastive VAE with LogFC-weighted InfoNCE on HEK293T data.

Combines:
- Projection head architecture (stable training, no task interference)
- LogFC-based weighting (biological similarity)
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

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset
from src.autoencoder.contrastive_vae.loss_logfc import contrastive_vae_logfc_loss

# =============================================================================
# Configuration
# =============================================================================

# Data paths
COUNTS_CSV = 'Dataset/csv/HEK293T_Counts.csv'
METADATA_XLSX = 'Dataset/HEK293T_MetaData.xlsx'
OUTPUT_MODEL = 'models/contrastive_vae_logfc_hek293t_best.pt'

# Model hyperparameters
LATENT_DIM = 64
PROJECTION_DIM = 128
HIDDEN_DIMS = [512, 256, 128]

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 15

# Loss weights
BETA = 1.0           # KL weight
GAMMA = 0.5          # InfoNCE weight
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
processed_df, _ = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

# Transpose to [samples x genes]
processed_df_T = processed_df.T.reset_index(drop=True)

INPUT_DIM = processed_df_T.shape[1]
print(f"Input dimension: {INPUT_DIM} genes")
print(f"Number of samples: {processed_df_T.shape[0]}")

# Compute DMSO mean for logFC weighting
dmso_mask = metadata['treatment'] == 'DMSO'
dmso_data = processed_df_T[dmso_mask]
dmso_mean = torch.tensor(dmso_data.mean(axis=0).values, dtype=torch.float32).to(DEVICE)
print(f"DMSO samples: {dmso_mask.sum()}")

# Create datasets
print("\nCreating datasets...")
train_dataset, val_dataset = ContrastiveGeneExpressionDataset.create_train_val_split(
    processed_df_T,
    treatments=metadata['treatment'],
    val_split=0.2,
    stratify=True,
    random_state=42
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# =============================================================================
# Initialize model
# =============================================================================

print("\nInitializing Contrastive VAE with LogFC weighting...")
model = ContrastiveVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=0.2,
    use_batch_norm=True,
    projection_dim=PROJECTION_DIM
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Projection dim: {PROJECTION_DIM}")

# =============================================================================
# Training loop
# =============================================================================

print("\n" + "="*70)
print("Training Contrastive VAE with LogFC-weighted InfoNCE")
print("="*70)
print(f"\nHyperparameters:")
print(f"  Beta (KL): {BETA}")
print(f"  Gamma (InfoNCE): {GAMMA}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  LogFC Beta: {LOGFC_BETA}")
print()

best_val_loss = float('inf')
patience_counter = 0

history = {
    'train_loss': [],
    'val_loss': [],
    'train_recon': [],
    'train_kl': [],
    'train_contrastive': [],
    'train_weight': [],
    'val_recon': [],
    'val_kl': [],
    'val_contrastive': [],
    'val_weight': []
}

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'contrastive': 0.0, 'weight': 0.0}
    
    for batch_data, batch_labels, _ in train_loader:
        batch_data = batch_data.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar, z_proj = model(batch_data, return_projection=True)
        
        # Compute loss with logFC weighting
        loss, recon_loss, kl_loss, contrastive_loss, avg_weight = contrastive_vae_logfc_loss(
            recon, batch_data, mu, logvar, z_proj, batch_labels, dmso_mean,
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
        train_losses['total'] += loss.item()
        train_losses['recon'] += recon_loss.item()
        train_losses['kl'] += kl_loss.item()
        train_losses['contrastive'] += contrastive_loss.item()
        train_losses['weight'] += avg_weight.item()
    
    # Average training losses
    num_train_batches = len(train_loader)
    avg_train_loss = train_losses['total'] / num_train_batches
    avg_train_recon = train_losses['recon'] / num_train_batches
    avg_train_kl = train_losses['kl'] / num_train_batches
    avg_train_contrastive = train_losses['contrastive'] / num_train_batches
    avg_train_weight = train_losses['weight'] / num_train_batches
    
    # Validation
    model.eval()
    val_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'contrastive': 0.0, 'weight': 0.0}
    
    with torch.no_grad():
        for batch_data, batch_labels, _ in val_loader:
            batch_data = batch_data.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            recon, mu, logvar, z_proj = model(batch_data, return_projection=True)
            
            loss, recon_loss, kl_loss, contrastive_loss, avg_weight = contrastive_vae_logfc_loss(
                recon, batch_data, mu, logvar, z_proj, batch_labels, dmso_mean,
                beta=BETA,
                gamma=GAMMA,
                temperature=TEMPERATURE,
                logfc_beta=LOGFC_BETA
            )
            
            val_losses['total'] += loss.item()
            val_losses['recon'] += recon_loss.item()
            val_losses['kl'] += kl_loss.item()
            val_losses['contrastive'] += contrastive_loss.item()
            val_losses['weight'] += avg_weight.item()
    
    # Average validation losses
    num_val_batches = len(val_loader)
    avg_val_loss = val_losses['total'] / num_val_batches
    avg_val_recon = val_losses['recon'] / num_val_batches
    avg_val_kl = val_losses['kl'] / num_val_batches
    avg_val_contrastive = val_losses['contrastive'] / num_val_batches
    avg_val_weight = val_losses['weight'] / num_val_batches
    
    # Update scheduler
    scheduler.step(avg_val_loss)
    
    # Record history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_recon'].append(avg_train_recon)
    history['train_kl'].append(avg_train_kl)
    history['train_contrastive'].append(avg_train_contrastive)
    history['train_weight'].append(avg_train_weight)
    history['val_recon'].append(avg_val_recon)
    history['val_kl'].append(avg_val_kl)
    history['val_contrastive'].append(avg_val_contrastive)
    history['val_weight'].append(avg_val_weight)
    
    # Print progress
    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Recon: {avg_train_recon:.4f} | "
          f"KL: {avg_train_kl:.4f} | "
          f"InfoNCE: {avg_train_contrastive:.4f} | "
          f"Weight: {avg_train_weight:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        # Create models directory if needed
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'history': history,
            'config': {
                'input_dim': INPUT_DIM,
                'latent_dim': LATENT_DIM,
                'projection_dim': PROJECTION_DIM,
                'hidden_dims': HIDDEN_DIMS,
                'beta': BETA,
                'gamma': GAMMA,
                'temperature': TEMPERATURE,
                'logfc_beta': LOGFC_BETA
            }
        }, OUTPUT_MODEL)
        
        print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
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
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {OUTPUT_MODEL}")
print(f"Total epochs: {len(history['train_loss'])}")

