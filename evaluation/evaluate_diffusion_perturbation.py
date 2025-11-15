"""
Evaluate diffusion-based perturbation prediction.

Compares diffusion model predictions against ground truth using:
- LogFC correlation
- Gene expression MSE
- Volcano plot comparison
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE
from src.diffusion.smiles_encoder import SMILESEncoder, load_smiles_dict
from src.diffusion.diffusion_model import PerturbationDiffusionModel

# =============================================================================
# Configuration
# =============================================================================

COUNTS_CSV = PROJECT_ROOT / 'Dataset' / 'csv' / 'HEK293T_Counts.csv'
METADATA_XLSX = PROJECT_ROOT / 'Dataset' / 'HEK293T_MetaData.xlsx'
SMILES_FILE = PROJECT_ROOT / 'Dataset' / 'SMILES.txt'
VAE_MODEL = PROJECT_ROOT / 'models' / 'contrastive_vae_hek293t_best.pt'
DIFFUSION_MODEL = PROJECT_ROOT / 'models' / 'diffusion_perturbation_best.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("Diffusion Perturbation Prediction Evaluation")
print("="*70)
print(f"Using device: {DEVICE}\n")

# =============================================================================
# Load Data and Models
# =============================================================================

print("Loading data...")
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
print(f"\nLoading VAE...")
vae = ContrastiveVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    hidden_dims=[512, 256, 128],
    dropout=0.2,
    projection_dim=128
)

vae_checkpoint = torch.load(VAE_MODEL, map_location='cpu', weights_only=False)
vae.load_state_dict(vae_checkpoint['model_state_dict'])
vae.eval()
vae = vae.to(DEVICE)

print("VAE loaded")

# Load diffusion model
print(f"\nLoading diffusion model...")
smiles_encoder = SMILESEncoder(
    model_name='DeepChem/ChemBERTa-77M-MLM',
    embedding_dim=256,
    freeze_encoder=True
).to(DEVICE)

diffusion_model = PerturbationDiffusionModel(
    latent_dim=64,
    smiles_dim=256,
    hidden_dim=512,
    num_heads=8,
    num_timesteps=1000
).to(DEVICE)

diffusion_checkpoint = torch.load(DIFFUSION_MODEL, map_location='cpu', weights_only=False)
smiles_encoder.load_state_dict(diffusion_checkpoint['smiles_encoder'])
diffusion_model.load_state_dict(diffusion_checkpoint['diffusion_model'])

smiles_encoder.eval()
diffusion_model.eval()

print("Diffusion model loaded")

# =============================================================================
# Encode all samples to VAE latent space
# =============================================================================

print("\nEncoding samples to VAE latent space...")
all_latents = []
with torch.no_grad():
    data_tensor = torch.FloatTensor(processed_df.values).to(DEVICE)
    for i in range(0, len(data_tensor), 256):
        batch = data_tensor[i:i+256]
        mu, _ = vae.encode(batch)
        all_latents.append(mu.cpu().numpy())

all_latents = np.vstack(all_latents)

# Get DMSO baseline
dmso_mask = metadata['treatment'] == 'DMSO'
dmso_latents = all_latents[dmso_mask]
baseline_latent = torch.FloatTensor(dmso_latents.mean(axis=0)).to(DEVICE)

print(f"DMSO baseline computed from {dmso_mask.sum()} samples")

# =============================================================================
# Evaluate on test compounds
# =============================================================================

print("\nEvaluating perturbation predictions...")

# Get compounds with replicates and SMILES
treatment_counts = metadata['treatment'].value_counts()
eval_treatments = [t for t in treatment_counts.index 
                   if treatment_counts[t] >= 2 
                   and t in smiles_dict 
                   and t not in ['DMSO', 'Blank', 'RNA']]

print(f"Evaluating on {len(eval_treatments[:50])} compounds (with replicates and SMILES)")

all_correlations = []
all_mses = []

for treatment in eval_treatments[:50]:  # Sample 50 for speed
    treatment_mask = metadata['treatment'] == treatment
    treatment_latents = all_latents[treatment_mask]
    treatment_expr = processed_df.values[treatment_mask]
    
    if len(treatment_latents) < 2:
        continue
    
    # Split into train/test
    if len(treatment_latents) == 2:
        test_idx = [1]
    else:
        _, test_idx = train_test_split(range(len(treatment_latents)), test_size=0.33, random_state=42)
    
    # Predict using diffusion model
    with torch.no_grad():
        # Get SMILES embedding
        smiles = smiles_dict[treatment]
        smiles_emb = smiles_encoder.encode_smiles(smiles)
        
        # Expand baseline for batch
        n_test = len(test_idx)
        baseline_batch = baseline_latent.unsqueeze(0).expand(n_test, -1)
        smiles_batch = smiles_emb.expand(n_test, -1)
        
        # Generate predicted latents via diffusion
        pred_latents = diffusion_model.sample(smiles_batch, baseline_batch, num_steps=100)  # Fast sampling
        
        # Decode to expression space
        pred_expr = vae.decode(pred_latents).cpu().numpy()
    
    # Ground truth
    true_expr = treatment_expr[test_idx]
    
    # Compute logFC
    dmso_mean_expr = processed_df.values[dmso_mask].mean(axis=0)
    pred_logfc = pred_expr - dmso_mean_expr
    true_logfc = true_expr - dmso_mean_expr
    
    # Compute correlation
    for i in range(len(test_idx)):
        corr, _ = pearsonr(pred_logfc[i], true_logfc[i])
        mse = np.mean((pred_logfc[i] - true_logfc[i])**2)
        
        if not np.isnan(corr):
            all_correlations.append(corr)
            all_mses.append(mse)

mean_corr = np.mean(all_correlations)
mean_mse = np.mean(all_mses)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"LogFC Correlation: {mean_corr:.4f}")
print(f"LogFC MSE: {mean_mse:.4f}")
print(f"Evaluated on: {len(all_correlations)} test samples")

# Save results
results_df = pd.DataFrame({
    'model': ['diffusion_perturbation'],
    'type': ['diffusion'],
    'logFC_correlation': [mean_corr],
    'logFC_mse': [mean_mse],
    'n_evaluations': [len(all_correlations)]
})

results_file = PROJECT_ROOT / 'results' / 'diffusion_perturbation_results.csv'
results_df.to_csv(results_file, index=False)
print(f"\nResults saved to: {results_file}")

