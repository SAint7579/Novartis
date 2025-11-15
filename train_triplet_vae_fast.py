"""
FAST Triplet VAE training with optimizations:
1. Pre-computed logFC (10-20× faster)
2. Mixed precision training (2× faster)
3. Larger batch size (1.5× faster)
4. Multi-worker data loading (1.3× faster)

Expected: 6-10 seconds/epoch (vs 2 minutes)
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parent
os.chdir(str(project_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.triplet_vae.model import TripletVAE
from src.autoencoder.triplet_vae.dataset_fast import FastTripletGeneExpressionDataset
from src.autoencoder.triplet_vae.loss_fast import triplet_vae_loss_fast
from src.autoencoder.contrastive_vae import plot_latent_space_by_treatment, ContrastiveGeneExpressionDataset
from src.autoencoder.contrastive_vae.utils import plot_training_history
import numpy as np
from tqdm import tqdm

print("="*70)
print("FAST Triplet VAE Training")
print("Optimizations: Pre-computed logFC + Mixed Precision + Larger Batches")
print("="*70)
print()

# Load data
print("Loading data...")
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard',
    filter_low_variance=True, variance_threshold=0.01
)

# Create FAST dataset with pre-computed logFC
print("\nCreating optimized datasets...")
train_dataset, val_dataset = FastTripletGeneExpressionDataset.create_train_val_split(
    processed_df,
    treatments=metadata['treatment'],
    val_split=0.2,
    dmso_label='DMSO',
    include_compound_negatives=True,
    random_state=42
)

# OPTIMIZED DataLoader settings
train_loader = DataLoader(
    train_dataset, 
    batch_size=256,      # Larger batch (was 64)
    shuffle=True,
    num_workers=4,       # Parallel loading
    pin_memory=True      # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"  Batch size: 256 (optimized)")
print(f"  Workers: 4 (parallel loading)")
print()

# Initialize model
model = TripletVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    hidden_dims=[512, 256, 128],
    dropout=0.2
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimized training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scaler = GradScaler()  # Mixed precision

# Hyperparameters
BETA = 1.0
GAMMA = 0.5
MARGIN = 1.0
LOGFC_BETA = 0.1
EPOCHS = 100
PATIENCE = 15

print(f"\nHyperparameters:")
print(f"  Beta (KL): {BETA}")
print(f"  Gamma (Triplet): {GAMMA}")
print(f"  LogFC Beta: {LOGFC_BETA}")
print(f"  Mixed Precision: {'✓' if device == 'cuda' else '✗'}")
print()

# Training loop
history = {'train_loss': [], 'val_loss': [], 'val_triplet': []}
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for batch in pbar:
        anchor, positive, neg_dmso, neg_comp, anc_lfc, pos_lfc, dmso_lfc, comp_lfc = batch
        
        anchor = anchor.to(device)
        positive = positive.to(device)
        neg_dmso = neg_dmso.to(device)
        neg_comp = neg_comp.to(device)
        anc_lfc = anc_lfc.to(device)
        pos_lfc = pos_lfc.to(device)
        dmso_lfc = dmso_lfc.to(device)
        comp_lfc = comp_lfc.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=(device == 'cuda')):
            recon_anc, mu_anc, logvar_anc = model(anchor)
            recon_pos, mu_pos, logvar_pos = model(positive)
            recon_dmso, mu_dmso, logvar_dmso = model(neg_dmso)
            recon_comp, mu_comp, logvar_comp = model(neg_comp)
            
            loss, recon_loss, kl_loss, triplet_loss, avg_weight = triplet_vae_loss_fast(
                recon_anc, recon_pos, recon_dmso, recon_comp,
                anchor, positive, neg_dmso, neg_comp,
                mu_anc, logvar_anc, mu_pos, logvar_pos,
                mu_dmso, logvar_dmso, mu_comp, logvar_comp,
                anc_lfc, pos_lfc, dmso_lfc, comp_lfc,
                beta=BETA, gamma=GAMMA, margin=MARGIN, logfc_beta=LOGFC_BETA
            )
        
        # Mixed precision backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.2f}', 'triplet': f'{triplet_loss.item():.3f}'})
    
    history['train_loss'].append(np.mean(train_losses))
    
    # Validation
    model.eval()
    val_losses = []
    val_triplets = []
    
    with torch.no_grad():
        for batch in val_loader:
            anchor, positive, neg_dmso, neg_comp, anc_lfc, pos_lfc, dmso_lfc, comp_lfc = batch
            
            anchor = anchor.to(device)
            positive = positive.to(device)
            neg_dmso = neg_dmso.to(device)
            neg_comp = neg_comp.to(device)
            anc_lfc = anc_lfc.to(device)
            pos_lfc = pos_lfc.to(device)
            dmso_lfc = dmso_lfc.to(device)
            comp_lfc = comp_lfc.to(device)
            
            recon_anc, mu_anc, logvar_anc = model(anchor)
            recon_pos, mu_pos, logvar_pos = model(positive)
            recon_dmso, mu_dmso, logvar_dmso = model(neg_dmso)
            recon_comp, mu_comp, logvar_comp = model(neg_comp)
            
            loss, _, _, triplet_loss, _ = triplet_vae_loss_fast(
                recon_anc, recon_pos, recon_dmso, recon_comp,
                anchor, positive, neg_dmso, neg_comp,
                mu_anc, logvar_anc, mu_pos, logvar_pos,
                mu_dmso, logvar_dmso, mu_comp, logvar_comp,
                anc_lfc, pos_lfc, dmso_lfc, comp_lfc,
                beta=BETA, gamma=GAMMA, margin=MARGIN, logfc_beta=LOGFC_BETA
            )
            
            val_losses.append(loss.item())
            val_triplets.append(triplet_loss.item())
    
    avg_val_loss = np.mean(val_losses)
    history['val_loss'].append(avg_val_loss)
    history['val_triplet'].append(np.mean(val_triplets))
    
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'models/triplet_vae2_hek293t_best.pt')
        print(f'  -> Saved best model (val_loss: {best_val_loss:.4f})')
    else:
        patience_counter += 1
    
    print(f'Epoch {epoch+1}/{EPOCHS} - train: {history["train_loss"][-1]:.2f} - '
          f'val: {avg_val_loss:.2f} - triplet: {history["val_triplet"][-1]:.3f} - '
          f'patience: {patience_counter}/{PATIENCE}')
    
    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"Model saved: models/triplet_vae2_hek293t_best.pt")
print(f"\nRun evaluations: python evaluation/run_all_evals.py")

