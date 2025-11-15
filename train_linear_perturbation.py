"""
Train linear baseline for perturbation prediction.

Simple MLP: (baseline_latent, SMILES_embedding) -> post_perturbation_latent

Used as baseline to compare against diffusion model.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup paths
project_root = Path(__file__).resolve().parent
os.chdir(str(project_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE
from src.diffusion.smiles_encoder import SMILESEncoder, load_smiles_dict
from src.diffusion.linear_baseline import LinearPerturbationModel

# =============================================================================
# Configuration
# =============================================================================

COUNTS_CSV = 'Dataset/csv/HEK293T_Counts.csv'
METADATA_XLSX = 'Dataset/HEK293T_MetaData.xlsx'
SMILES_FILE = 'Dataset/SMILES.txt'
VAE_MODEL = 'models/contrastive_vae_hek293t_best.pt'
OUTPUT_DIR = 'models'

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# =============================================================================
# Dataset
# =============================================================================

class PerturbationDataset(Dataset):
    def __init__(self, baseline_latents, perturbed_latents, smiles_strings, treatments):
        self.baseline_latents = baseline_latents
        self.perturbed_latents = perturbed_latents
        self.smiles_strings = smiles_strings
        self.treatments = treatments
    
    def __len__(self):
        return len(self.perturbed_latents)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.baseline_latents[idx]),
            torch.FloatTensor(self.perturbed_latents[idx]),
            self.smiles_strings[idx],
            self.treatments[idx]
        )

# =============================================================================
# Main
# =============================================================================

print("\n" + "="*70)
print("Linear Perturbation Prediction Training")
print("="*70)

# Load data
print("\nLoading data...")
counts_df = pd.read_csv(COUNTS_CSV, header=1, index_col=0)
metadata = pd.read_excel(METADATA_XLSX, header=1)

processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard',
    filter_low_variance=True, variance_threshold=0.01
)

metadata = metadata.set_index('unique_ID').loc[processed_df.index].reset_index()

print(f"Data shape: {processed_df.shape}")

# Load SMILES
print("\nLoading SMILES...")
smiles_dict = load_smiles_dict(SMILES_FILE)

# Load VAE
print(f"\nLoading VAE from {VAE_MODEL}...")
vae = ContrastiveVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    hidden_dims=[512, 256, 128],
    dropout=0.2,
    projection_dim=128
)

checkpoint = torch.load(VAE_MODEL, map_location='cpu', weights_only=False)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()
vae = vae.to(DEVICE)

# Encode all samples
print("\nEncoding samples to VAE latent space...")
all_latents = []
with torch.no_grad():
    data_tensor = torch.FloatTensor(processed_df.values).to(DEVICE)
    for i in range(0, len(data_tensor), 256):
        batch = data_tensor[i:i+256]
        mu, _ = vae.encode(batch)
        all_latents.append(mu.cpu().numpy())

all_latents = np.vstack(all_latents)
print(f"Latent embeddings shape: {all_latents.shape}")

# Compute DMSO baseline
dmso_mask = metadata['treatment'] == 'DMSO'
dmso_latents = all_latents[dmso_mask]
baseline_latent = dmso_latents.mean(axis=0)
print(f"DMSO baseline from {dmso_mask.sum()} samples")

# Prepare dataset
print("\nPreparing training dataset...")
baseline_list = []
perturbed_list = []
smiles_list = []
treatment_list = []

skipped = 0
for idx, treatment in enumerate(metadata['treatment']):
    if treatment in ['DMSO', 'Blank', 'RNA']:
        continue
    
    if treatment not in smiles_dict:
        skipped += 1
        continue
    
    smiles = smiles_dict[treatment]
    
    baseline_list.append(baseline_latent)
    perturbed_list.append(all_latents[idx])
    smiles_list.append(smiles)
    treatment_list.append(treatment)

print(f"Total samples: {len(perturbed_list)}")
print(f"Skipped {skipped} samples without SMILES")

# Split train/val
n_train = int(0.9 * len(perturbed_list))
indices = np.random.permutation(len(perturbed_list))

train_dataset = PerturbationDataset(
    [baseline_list[i] for i in indices[:n_train]],
    [perturbed_list[i] for i in indices[:n_train]],
    [smiles_list[i] for i in indices[:n_train]],
    [treatment_list[i] for i in indices[:n_train]]
)

val_dataset = PerturbationDataset(
    [baseline_list[i] for i in indices[n_train:]],
    [perturbed_list[i] for i in indices[n_train:]],
    [smiles_list[i] for i in indices[n_train:]],
    [treatment_list[i] for i in indices[n_train:]]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Initialize models
print("\nInitializing models...")
smiles_encoder = SMILESEncoder(
    model_name='DeepChem/ChemBERTa-77M-MLM',
    embedding_dim=256,
    freeze_encoder=True
).to(DEVICE)

linear_model = LinearPerturbationModel(
    latent_dim=64,
    smiles_dim=256,
    hidden_dims=[512, 512, 256]
).to(DEVICE)

print(f"SMILES Encoder params: {sum(p.numel() for p in smiles_encoder.parameters()):,}")
print(f"  Trainable: {sum(p.numel() for p in smiles_encoder.parameters() if p.requires_grad):,}")
print(f"Linear Model params: {sum(p.numel() for p in linear_model.parameters()):,}")

# Optimizer
optimizer = optim.Adam(
    list(smiles_encoder.parameters()) + list(linear_model.parameters()),
    lr=LEARNING_RATE
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
print("\n" + "="*70)
print("Training Linear Baseline")
print("="*70)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Train
    smiles_encoder.train()
    linear_model.train()
    train_loss = 0.0
    
    for baseline, target, smiles_strings, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        baseline = baseline.to(DEVICE)
        target = target.to(DEVICE)
        
        # Encode SMILES
        smiles_emb = smiles_encoder(list(smiles_strings))
        
        # Predict
        pred = linear_model(baseline, smiles_emb)
        
        # Loss
        loss = nn.MSELoss()(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(smiles_encoder.parameters()) + list(linear_model.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    smiles_encoder.eval()
    linear_model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for baseline, target, smiles_strings, _ in val_loader:
            baseline = baseline.to(DEVICE)
            target = target.to(DEVICE)
            
            smiles_emb = smiles_encoder(list(smiles_strings))
            pred = linear_model(baseline, smiles_emb)
            loss = nn.MSELoss()(pred, target)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'smiles_encoder': smiles_encoder.state_dict(),
            'linear_model': linear_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }, f"{OUTPUT_DIR}/linear_perturbation_best.pt")
        
        print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {OUTPUT_DIR}/linear_perturbation_best.pt")

