"""
Evaluate perturbation prediction task:
1. Pre-perturbed (DMSO) embedding → Post-perturbed (treatment) embedding
2. Decode predicted embedding → gene expression
3. Compare predicted logFC vs ground truth logFC
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset
from src.autoencoder.vae import VAE

# Simple predictor network
class PerturbationPredictor(nn.Module):
    """Small NN to predict post-perturbation embedding from baseline."""
    
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def evaluate_model_perturbation_prediction(model_path, model_type, processed_df, metadata, device='cpu'):
    """
    Evaluate perturbation prediction for one model.
    """
    
    # Load checkpoint first to get config
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get input_dim from checkpoint config if available
    if 'config' in checkpoint and 'input_dim' in checkpoint['config']:
        input_dim = checkpoint['config']['input_dim']
    else:
        input_dim = processed_df.shape[1]
    
    print(f"  Using input_dim: {input_dim}")
    
    # Load model
    if model_type == 'contrastive':
        model = ContrastiveVAE(
            input_dim=input_dim, latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2, projection_dim=128
        )
    elif model_type == 'triplet':
        from src.autoencoder.triplet_vae import TripletVAE
        model = TripletVAE(
            input_dim=input_dim, latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2
        )
    elif model_type == 'infonce':
        # InfoNCE uses standard VAE architecture
        model = VAE(
            input_dim=input_dim, latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2
        )
    else:
        model = VAE(
            input_dim=input_dim, latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Extract latent representations for all samples
    print("  Extracting latent representations...")
    dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_latent = []
    with torch.no_grad():
        for batch_data, _, _ in loader:
            mu, _ = model.encode(batch_data.to(device))
            all_latent.append(mu.cpu().numpy())
    
    latent_embeddings = np.vstack(all_latent)
    
    # Get DMSO baseline
    dmso_mask = metadata['treatment'].str.upper() == 'DMSO'
    dmso_latent = latent_embeddings[dmso_mask]
    dmso_expr = processed_df.values[dmso_mask]
    dmso_mean_latent = dmso_latent.mean(axis=0)
    dmso_mean_expr = dmso_expr.mean(axis=0)
    
    print(f"  DMSO samples: {dmso_mask.sum()}")
    print(f"  DMSO mean embedding shape: {dmso_mean_latent.shape}")
    
    # Select treatments with replicates for evaluation
    treatment_counts = metadata['treatment'].value_counts()
    eval_treatments = treatment_counts[(treatment_counts >= 2) & (treatment_counts.index != 'DMSO') & 
                                       (treatment_counts.index != 'Blank') & (treatment_counts.index != 'RNA')].index[:100]  # Sample 100 treatments
    
    print(f"  Evaluating on {len(eval_treatments)} treatments")
    
    all_correlations = []
    all_mses = []
    
    for treatment in eval_treatments:
        treatment_mask = metadata['treatment'] == treatment
        treatment_latent = latent_embeddings[treatment_mask]
        treatment_expr = processed_df.values[treatment_mask]
        
        if len(treatment_latent) < 2:
            continue
        
        # Split replicates into train/test
        if len(treatment_latent) == 2:
            train_idx = [0]
            test_idx = [1]
        else:
            train_idx, test_idx = train_test_split(range(len(treatment_latent)), test_size=0.33, random_state=42)
        
        # Train small predictor: DMSO_latent → treatment_latent
        predictor = PerturbationPredictor(latent_dim=64).to(device)
        optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
        
        # Prepare training data
        X_train = torch.FloatTensor(np.tile(dmso_mean_latent, (len(train_idx), 1))).to(device)
        y_train = torch.FloatTensor(treatment_latent[train_idx]).to(device)
        
        # Quick training (10 epochs)
        predictor.train()
        for _ in range(10):
            pred = predictor(X_train)
            loss = nn.MSELoss()(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Predict on test replicates
        predictor.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(np.tile(dmso_mean_latent, (len(test_idx), 1))).to(device)
            pred_latent = predictor(X_test).cpu().numpy()
            
            # Decode predicted latent to gene expression
            pred_latent_torch = torch.FloatTensor(pred_latent).to(device)
            pred_expr = model.decode(pred_latent_torch).cpu().numpy()
        
        # Ground truth expression
        true_expr = treatment_expr[test_idx]
        
        # Compute logFC
        pred_logfc = pred_expr - dmso_mean_expr
        true_logfc = true_expr - dmso_mean_expr
        
        # Compute correlation (averaged across test replicates)
        for i in range(len(test_idx)):
            corr, _ = pearsonr(pred_logfc[i], true_logfc[i])
            mse = np.mean((pred_logfc[i] - true_logfc[i])**2)
            
            if not np.isnan(corr):
                all_correlations.append(corr)
                all_mses.append(mse)
    
    mean_corr = np.mean(all_correlations)
    mean_mse = np.mean(all_mses)
    
    return mean_corr, mean_mse, len(all_correlations)


# ============================================================================
# Main Evaluation
# ============================================================================

print("="*70)
print("Perturbation Prediction Evaluation")
print("="*70)
print("Task: Predict post-perturbation from baseline, measure logFC recovery")
print()

# Get paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Load data
print("Loading data...")
counts_df = pd.read_csv(PROJECT_ROOT / 'Dataset' / 'csv' / 'HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel(PROJECT_ROOT / 'Dataset' / 'HEK293T_MetaData.xlsx', header=1)

processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard',
    filter_low_variance=True, variance_threshold=0.01
)

# Find models
models_dir = PROJECT_ROOT / 'models'
model_files = list(models_dir.glob('*.pt'))

# Load existing results if they exist
results_file = PROJECT_ROOT / 'results' / 'perturbation_prediction_accuracy.csv'
if results_file.exists():
    existing_results = pd.read_csv(results_file)
    evaluated_models = set(existing_results['model'].values)
    print(f"Found existing results for {len(evaluated_models)} model(s)")
    print(f"  Already evaluated: {list(evaluated_models)}")
else:
    existing_results = pd.DataFrame()
    evaluated_models = set()

# Filter to only new models
model_files = [f for f in model_files if f.name not in evaluated_models]

print(f"Models to evaluate: {len(model_files)}")
for mf in model_files:
    print(f"  - {mf.name}")
print()

if len(model_files) == 0:
    print("All models already evaluated!")
    print(f"Results in: {results_file}")
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

results = []

for model_file in model_files:
    print(f"{'='*70}")
    print(f"Evaluating: {model_file.name}")
    print(f"{'='*70}")
    
    if 'infonce' in model_file.name.lower():
        model_type = 'infonce'
    elif 'contrastive' in model_file.name.lower():
        model_type = 'contrastive'
    elif 'triplet' in model_file.name.lower():
        model_type = 'triplet'
    else:
        model_type = 'standard'
    
    corr, mse, n_samples = evaluate_model_perturbation_prediction(
        model_file, model_type, processed_df, metadata, device
    )
    
    results.append({
        'model': model_file.name,
        'type': model_type,
        'logFC_correlation': corr,
        'logFC_mse': mse,
        'n_evaluations': n_samples
    })
    
    print(f"\n  LogFC Recovery Correlation: {corr:.4f}")
    print(f"  LogFC MSE: {mse:.4f}")
    print(f"  Evaluated on: {n_samples} test samples")
    print()

# Combine with existing results
if len(results) > 0:
    new_results_df = pd.DataFrame(results)
    if not existing_results.empty:
        results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
    else:
        results_df = new_results_df
    
    # Save updated results
    results_df.to_csv(PROJECT_ROOT / 'results' / 'perturbation_prediction_accuracy.csv', index=False)
    print(f"Updated results saved")
else:
    results_df = existing_results

# Summary
print("\n" + "="*70)
print("SUMMARY: Perturbation Prediction Accuracy (All Models)")
print("="*70)

print(results_df.to_string(index=False))

print(f"\nInterpretation:")
print(f"  Correlation >0.7: Good - latent space captures perturbation effects")
print(f"  Correlation 0.5-0.7: Moderate")  
print(f"  Correlation <0.5: Poor - latent space doesn't capture biology well")

print(f"\nResults in: {PROJECT_ROOT / 'results' / 'perturbation_prediction_accuracy.csv'}")

