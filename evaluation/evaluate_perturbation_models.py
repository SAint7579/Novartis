"""
Comprehensive evaluation of perturbation prediction models.

Compares:
1. Diffusion model
2. Linear baseline model

Metrics:
- Per-treatment logFC correlation
- Per-treatment MSE
- Overall statistics
- Saves detailed results to Excel
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE
from src.diffusion.smiles_encoder import SMILESEncoder, load_smiles_dict
from src.diffusion.diffusion_model import PerturbationDiffusionModel
from src.diffusion.linear_baseline import LinearPerturbationModel

# =============================================================================
# Configuration
# =============================================================================

COUNTS_CSV = PROJECT_ROOT / 'Dataset' / 'csv' / 'HEK293T_Counts.csv'
METADATA_XLSX = PROJECT_ROOT / 'Dataset' / 'HEK293T_MetaData.xlsx'
SMILES_FILE = PROJECT_ROOT / 'Dataset' / 'SMILES.txt'
VAE_MODEL = PROJECT_ROOT / 'models' / 'contrastive_vae_hek293t_best.pt'
DIFFUSION_MODEL = PROJECT_ROOT / 'models' / 'diffusion_perturbation_best.pt'
LINEAR_MODEL = PROJECT_ROOT / 'models' / 'linear_perturbation_best.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("Perturbation Prediction Model Comparison")
print("="*70)
print(f"Using device: {DEVICE}\n")

# =============================================================================
# Load Data
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

# Encode all samples
print("Encoding samples to latent space...")
all_latents = []
with torch.no_grad():
    data_tensor = torch.FloatTensor(processed_df.values).to(DEVICE)
    for i in range(0, len(data_tensor), 256):
        batch = data_tensor[i:i+256]
        mu, _ = vae.encode(batch)
        all_latents.append(mu.cpu().numpy())

all_latents = np.vstack(all_latents)

# DMSO baseline
dmso_mask = metadata['treatment'] == 'DMSO'
dmso_latents = all_latents[dmso_mask]
baseline_latent = torch.FloatTensor(dmso_latents.mean(axis=0)).to(DEVICE)
dmso_mean_expr = processed_df.values[dmso_mask].mean(axis=0)

print(f"DMSO baseline from {dmso_mask.sum()} samples")

# =============================================================================
# Load Models
# =============================================================================

# SMILES encoder (shared)
print("\nLoading SMILES encoder...")
smiles_encoder = SMILESEncoder(
    model_name='DeepChem/ChemBERTa-77M-MLM',
    embedding_dim=256,
    freeze_encoder=True
).to(DEVICE)

# Check which models exist
models_to_eval = []

if DIFFUSION_MODEL.exists():
    print(f"Loading diffusion model...")
    diffusion_model = PerturbationDiffusionModel(
        latent_dim=64, smiles_dim=256, hidden_dim=512,
        num_heads=8, num_timesteps=1000,
        num_cell_lines=10, concentration_dim=1
    ).to(DEVICE)
    
    diff_checkpoint = torch.load(DIFFUSION_MODEL, map_location='cpu', weights_only=False)
    smiles_encoder.load_state_dict(diff_checkpoint['smiles_encoder'])
    diffusion_model.load_state_dict(diff_checkpoint['diffusion_model'])
    diffusion_model.eval()
    models_to_eval.append(('Diffusion', diffusion_model, smiles_encoder))
    print("  Diffusion model loaded")
else:
    print(f"Diffusion model not found: {DIFFUSION_MODEL}")

if LINEAR_MODEL.exists():
    print(f"Loading linear model...")
    linear_model = LinearPerturbationModel(
        latent_dim=64, smiles_dim=256, hidden_dims=[512, 512, 256]
    ).to(DEVICE)
    
    # Create new SMILES encoder for linear model
    smiles_encoder_linear = SMILESEncoder(
        model_name='DeepChem/ChemBERTa-77M-MLM',
        embedding_dim=256,
        freeze_encoder=True
    ).to(DEVICE)
    
    lin_checkpoint = torch.load(LINEAR_MODEL, map_location='cpu', weights_only=False)
    smiles_encoder_linear.load_state_dict(lin_checkpoint['smiles_encoder'])
    linear_model.load_state_dict(lin_checkpoint['linear_model'])
    linear_model.eval()
    models_to_eval.append(('Linear', linear_model, smiles_encoder_linear))
    print("  Linear model loaded")
else:
    print(f"Linear model not found: {LINEAR_MODEL}")

if len(models_to_eval) == 0:
    print("\nNo trained models found! Train models first:")
    print("  python train_diffusion_perturbation.py")
    print("  python train_linear_perturbation.py")
    exit()

# =============================================================================
# Evaluate on Test Set
# =============================================================================

print("\n" + "="*70)
print("Evaluating Perturbation Predictions")
print("="*70)

# Get compounds with replicates and SMILES
treatment_counts = metadata['treatment'].value_counts()
eval_treatments = [t for t in treatment_counts.index 
                   if treatment_counts[t] >= 2 
                   and t in smiles_dict 
                   and t not in ['DMSO', 'Blank', 'RNA']]

print(f"\nFound {len(eval_treatments)} compounds with replicates and SMILES")
print(f"Sampling 200 compounds for faster evaluation...")
eval_treatments_sample = np.random.choice(eval_treatments, size=min(200, len(eval_treatments)), replace=False)
print(f"Evaluating on {len(eval_treatments_sample)} compounds\n")

# Store results for each model
all_results = {name: [] for name, _, _ in models_to_eval}

for treatment in tqdm(eval_treatments_sample, desc="Evaluating compounds"):
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
    
    # Ground truth
    true_latent = treatment_latents[test_idx]
    true_expr = treatment_expr[test_idx]
    true_logfc = true_expr - dmso_mean_expr
    
    # Get SMILES
    smiles = smiles_dict[treatment]
    
    # Evaluate each model
    for model_name, model, encoder in models_to_eval:
        with torch.no_grad():
            # Encode SMILES
            smiles_emb = encoder.encode_smiles(smiles)
            
            # Predict post-perturbation latents
            n_test = len(test_idx)
            baseline_batch = baseline_latent.unsqueeze(0).expand(n_test, -1)
            smiles_batch = smiles_emb.expand(n_test, -1)
            
            # Cell line and concentration (fixed for HEK293T)
            cell_line_batch = torch.zeros(n_test, 10, device=DEVICE)
            cell_line_batch[:, 0] = 1.0  # HEK293T = cell line 0
            concentration_batch = torch.full((n_test, 1), 10.0, device=DEVICE)
            
            if model_name == 'Diffusion':
                # Generate via diffusion (single sample for speed, 50 steps)
                pred_latents = model.sample(smiles_batch, baseline_batch, cell_line_batch, concentration_batch, num_steps=50).cpu().numpy()
            else:
                # Linear model prediction (doesn't use cell_line/concentration)
                pred_latents = model(baseline_batch, smiles_batch).cpu().numpy()
            
            # Decode to expression
            pred_latents_torch = torch.FloatTensor(pred_latents).to(DEVICE)
            pred_expr = vae.decode(pred_latents_torch).cpu().numpy()
            pred_logfc = pred_expr - dmso_mean_expr
        
        # Compute metrics for each test sample
        for i in range(len(test_idx)):
            # LogFC correlation
            corr, _ = pearsonr(pred_logfc[i], true_logfc[i])
            
            # MSE (latent space)
            latent_mse = np.mean((pred_latents[i] - true_latent[i])**2)
            
            # MSE (expression space)
            expr_mse = np.mean((pred_expr[i] - true_expr[i])**2)
            
            # MSE (logFC space)
            logfc_mse = np.mean((pred_logfc[i] - true_logfc[i])**2)
            
            if not np.isnan(corr):
                all_results[model_name].append({
                    'treatment': treatment,
                    'logFC_correlation': corr,
                    'latent_mse': latent_mse,
                    'expression_mse': expr_mse,
                    'logFC_mse': logfc_mse
                })

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

summary_data = []
detailed_results = {}

for model_name in all_results.keys():
    if len(all_results[model_name]) == 0:
        continue
    
    results_df = pd.DataFrame(all_results[model_name])
    detailed_results[model_name] = results_df
    
    # Compute statistics
    mean_corr = results_df['logFC_correlation'].mean()
    std_corr = results_df['logFC_correlation'].std()
    mean_latent_mse = results_df['latent_mse'].mean()
    mean_expr_mse = results_df['expression_mse'].mean()
    mean_logfc_mse = results_df['logFC_mse'].mean()
    n_samples = len(results_df)
    
    summary_data.append({
        'Model': model_name,
        'LogFC_Corr_Mean': mean_corr,
        'LogFC_Corr_Std': std_corr,
        'Latent_MSE': mean_latent_mse,
        'Expression_MSE': mean_expr_mse,
        'LogFC_MSE': mean_logfc_mse,
        'N_Samples': n_samples
    })
    
    print(f"\n{model_name} Model:")
    print(f"  LogFC Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  Latent MSE: {mean_latent_mse:.4f}")
    print(f"  Expression MSE: {mean_expr_mse:.4f}")
    print(f"  LogFC MSE: {mean_logfc_mse:.4f}")
    print(f"  Samples: {n_samples}")

# =============================================================================
# Save Results
# =============================================================================

# Summary table
summary_df = pd.DataFrame(summary_data)
summary_file = PROJECT_ROOT / 'results' / 'perturbation_model_comparison.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n\nSummary saved to: {summary_file}")

# Detailed per-treatment results to Excel
excel_file = PROJECT_ROOT / 'results' / 'perturbation_per_treatment.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    for model_name, results_df in detailed_results.items():
        # Sort by correlation
        results_df = results_df.sort_values('logFC_correlation', ascending=False)
        results_df.to_excel(writer, sheet_name=model_name, index=False)
    
    # Add summary sheet
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"Per-treatment results saved to: {excel_file}")

# Display comparison table
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(summary_df.to_string(index=False))

print(f"\n✓ Evaluation complete!")
print(f"\nResults:")
print(f"  Summary:       {summary_file}")
print(f"  Per-treatment: {excel_file}")

