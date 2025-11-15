"""
Fast top-k evaluation using a sample of the data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from pathlib import Path
import sys
sys.path.append('.')

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset
from src.autoencoder.vae import VAE

# Sample size for evaluation (faster)
SAMPLE_SIZE = 5000  # Evaluate on 5000 random samples

print("="*70)
print("Top-k Retrieval Evaluation (Fast Version)")
print("="*70)
print(f"Evaluating on {SAMPLE_SIZE} random samples\n")

# Load data
print("Loading data...")
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard',
    filter_low_variance=True, variance_threshold=0.01
)

# Sample data
np.random.seed(42)
sample_indices = np.random.choice(len(processed_df), SAMPLE_SIZE, replace=False)
processed_sample = processed_df.iloc[sample_indices].reset_index(drop=True)
metadata_sample = metadata.iloc[sample_indices].reset_index(drop=True)

print(f"Sampled {SAMPLE_SIZE} samples for evaluation\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Find models
model_files = list(Path('models').glob('*.pt'))
all_results = []

for model_file in model_files:
    print(f"{'='*70}")
    print(f"{model_file.name}")
    print(f"{'='*70}")
    
    model_type = 'contrastive' if 'contrastive' in model_file.name.lower() else 'standard'
    
    # Load model
    if model_type == 'contrastive':
        model = ContrastiveVAE(
            input_dim=processed_sample.shape[1], latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2, projection_dim=128
        )
    else:
        model = VAE(
            input_dim=processed_sample.shape[1], latent_dim=64,
            hidden_dims=[512, 256, 128], dropout=0.2
        )
    
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Extract latent
    dataset = ContrastiveGeneExpressionDataset(processed_sample, metadata_sample['treatment'])
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    latent_list = []
    treatments_list = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in loader:
            mu, _ = model.encode(batch_data.to(device))
            latent_list.append(mu.cpu().numpy())
            treatments_list.extend(batch_treatments)
    
    latent = np.vstack(latent_list)
    treatments = np.array(treatments_list)
    
    # Normalize for cosine similarity
    latent_norm = normalize(latent, norm='l2')
    
    # Compute similarity matrix (now manageable size)
    print("  Computing similarity matrix...")
    similarity = np.dot(latent_norm, latent_norm.T)
    
    # Find top-k neighbors
    print("  Finding top-k neighbors...")
    # For each row, argsort descending (highest similarity first)
    # Then exclude self (index 0)
    sorted_indices = np.argsort(-similarity, axis=1)[:, 1:10]  # Top 9 neighbors (excluding self)
    
    # Compute metrics
    results = {'model': model_file.name, 'type': model_type}
    
    for k in [1, 3, 5, 9]:
        top_k_indices = sorted_indices[:, :k]
        
        accuracies = []
        precisions = []
        
        for i in range(len(latent)):
            sample_treatment = treatments[i]
            neighbor_treatments = treatments[top_k_indices[i]]
            
            is_replicate = neighbor_treatments == sample_treatment
            
            accuracies.append(1.0 if is_replicate.any() else 0.0)
            precisions.append(is_replicate.sum() / k)
        
        results[f'top{k}_acc'] = np.mean(accuracies)
        results[f'top{k}_prec'] = np.mean(precisions)
    
    all_results.append(results)
    
    print(f"\n  Top-1: Acc={results['top1_acc']:.4f}, Prec={results['top1_prec']:.4f}")
    print(f"  Top-3: Acc={results['top3_acc']:.4f}, Prec={results['top3_prec']:.4f}")
    print(f"  Top-5: Acc={results['top5_acc']:.4f}, Prec={results['top5_prec']:.4f}")
    print(f"  Top-9: Acc={results['top9_acc']:.4f}, Prec={results['top9_prec']:.4f}\n")

# Comparison table
print("="*70)
print("COMPARISON TABLE")
print("="*70)
comparison_df = pd.DataFrame(all_results)
print(comparison_df.to_string(index=False))

comparison_df.to_csv('results/topk_metrics.csv', index=False)
print(f"\nSaved to: results/topk_metrics.csv")

