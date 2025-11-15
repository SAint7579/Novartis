"""
Train diffusion model for perturbation prediction.

Usage:
    python train_diffusion_perturbation.py

This will:
1. Load Contrastive VAE
2. Encode all samples to VAE latent space
3. Load SMILES embeddings
4. Train diffusion model to predict: (baseline_latent, SMILES) -> post_perturbation_latent
5. Evaluate on held-out test set
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
from src.diffusion.smiles_encoder import (
    SMILESEncoder, load_smiles_dict, precompute_smiles_embeddings
)
from src.diffusion.diffusion_model import PerturbationDiffusionModel

# =============================================================================
# Configuration
# =============================================================================

# Paths
COUNTS_CSV = 'Dataset/csv/HEK293T_Counts.csv'
METADATA_XLSX = 'Dataset/HEK293T_MetaData.xlsx'
SMILES_FILE = 'Dataset/SMILES.txt'
VAE_MODEL = 'models/contrastive_vae_hek293t_best.pt'
OUTPUT_DIR = 'models'

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# =============================================================================
# Dataset
# =============================================================================

class PerturbationDataset(Dataset):
    """
    Dataset for perturbation prediction.
    
    Returns: (baseline_latent, post_perturbation_latent, smiles_string, 
              cell_line_onehot, concentration, treatment)
    """
    
    def __init__(self, baseline_latents, perturbed_latents, 
                 smiles_strings, cell_lines, concentrations, treatments,
                 num_cell_lines=10):
        self.baseline_latents = baseline_latents
        self.perturbed_latents = perturbed_latents
        self.smiles_strings = smiles_strings
        self.cell_lines = cell_lines  # Integer indices
        self.concentrations = concentrations
        self.treatments = treatments
        self.num_cell_lines = num_cell_lines
    
    def __len__(self):
        return len(self.perturbed_latents)
    
    def __getitem__(self, idx):
        # One-hot encode cell line
        cell_line_onehot = torch.zeros(self.num_cell_lines)
        cell_line_onehot[self.cell_lines[idx]] = 1.0
        
        return (
            torch.FloatTensor(self.baseline_latents[idx]),
            torch.FloatTensor(self.perturbed_latents[idx]),
            self.smiles_strings[idx],
            cell_line_onehot,
            torch.FloatTensor([self.concentrations[idx]]),
            self.treatments[idx]
        )

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Diffusion-based Perturbation Prediction Training")
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
    
    # Encode all samples to VAE latent space
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
    
    # Compute DMSO baseline mean
    dmso_mask = metadata['treatment'] == 'DMSO'
    dmso_latents = all_latents[dmso_mask]
    baseline_latent = dmso_latents.mean(axis=0)
    print(f"DMSO baseline computed from {dmso_mask.sum()} samples")
    
    # Prepare dataset: for each perturbed sample, pair with baseline
    print("\nPreparing training dataset...")
    baseline_list = []
    perturbed_list = []
    smiles_list = []
    cell_line_list = []
    concentration_list = []
    treatment_list = []
    
    skipped = 0
    for idx, treatment in enumerate(metadata['treatment']):
        if treatment == 'DMSO' or treatment == 'Blank' or treatment == 'RNA':
            continue
        
        if treatment not in smiles_dict:
            skipped += 1
            continue
        
        smiles = smiles_dict[treatment]
        
        baseline_list.append(baseline_latent)
        perturbed_list.append(all_latents[idx])
        smiles_list.append(smiles)
        cell_line_list.append(0)  # All HEK293T (cell line 0)
        concentration_list.append(10.0)  # Fixed concentration
        treatment_list.append(treatment)
    
    print(f"Total training samples: {len(perturbed_list)}")
    print(f"Skipped {skipped} samples without SMILES")
    print(f"Cell line: HEK293T (index 0), Concentration: 10.0")
    
    # Split train/val
    n_train = int(0.9 * len(perturbed_list))
    indices = np.random.permutation(len(perturbed_list))
    
    train_dataset = PerturbationDataset(
        [baseline_list[i] for i in indices[:n_train]],
        [perturbed_list[i] for i in indices[:n_train]],
        [smiles_list[i] for i in indices[:n_train]],
        [cell_line_list[i] for i in indices[:n_train]],
        [concentration_list[i] for i in indices[:n_train]],
        [treatment_list[i] for i in indices[:n_train]],
        num_cell_lines=10
    )
    
    val_dataset = PerturbationDataset(
        [baseline_list[i] for i in indices[n_train:]],
        [perturbed_list[i] for i in indices[n_train:]],
        [smiles_list[i] for i in indices[n_train:]],
        [cell_line_list[i] for i in indices[n_train:]],
        [concentration_list[i] for i in indices[n_train:]],
        [treatment_list[i] for i in indices[n_train:]],
        num_cell_lines=10
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize models
    print("\nInitializing models...")
    smiles_encoder = SMILESEncoder(
        model_name='DeepChem/ChemBERTa-77M-MLM',
        embedding_dim=256,
        freeze_encoder=True  # Only train projection head
    ).to(DEVICE)
    
    diffusion_model = PerturbationDiffusionModel(
        latent_dim=64,
        smiles_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_timesteps=1000,
        num_cell_lines=10,
        concentration_dim=1
    ).to(DEVICE)
    
    print(f"SMILES Encoder params: {sum(p.numel() for p in smiles_encoder.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in smiles_encoder.parameters() if p.requires_grad):,}")
    print(f"Diffusion Model params: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        list(smiles_encoder.parameters()) + list(diffusion_model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Train
        smiles_encoder.train()
        diffusion_model.train()
        train_loss = 0.0
        
        for baseline, target, smiles_strings, cell_line, concentration, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            baseline = baseline.to(DEVICE)
            target = target.to(DEVICE)
            cell_line = cell_line.to(DEVICE)
            concentration = concentration.to(DEVICE)
            
            # Encode SMILES strings (batch)
            smiles_emb = smiles_encoder(list(smiles_strings))
            
            # Sample random timesteps
            t = torch.randint(0, diffusion_model.num_timesteps, (baseline.shape[0],), device=DEVICE)
            
            # Sample noise
            noise = torch.randn_like(target)
            
            # Add noise to target
            x_t = diffusion_model.q_sample(target, t, noise)
            
            # Predict noise
            noise_pred = diffusion_model(x_t, t, smiles_emb, baseline, cell_line, concentration)
            
            # Loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(smiles_encoder.parameters()) + list(diffusion_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        smiles_encoder.eval()
        diffusion_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for baseline, target, smiles_strings, cell_line, concentration, _ in val_loader:
                baseline = baseline.to(DEVICE)
                target = target.to(DEVICE)
                cell_line = cell_line.to(DEVICE)
                concentration = concentration.to(DEVICE)
                
                # Encode SMILES strings
                smiles_emb = smiles_encoder(list(smiles_strings))
                t = torch.randint(0, diffusion_model.num_timesteps, (baseline.shape[0],), device=DEVICE)
                noise = torch.randn_like(target)
                x_t = diffusion_model.q_sample(target, t, noise)
                noise_pred = diffusion_model(x_t, t, smiles_emb, baseline, cell_line, concentration)
                loss = nn.MSELoss()(noise_pred, noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'smiles_encoder': smiles_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, f"{OUTPUT_DIR}/diffusion_perturbation_best.pt")
            
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()

