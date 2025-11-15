"""
Evaluate latent space quality using top-k retrieval metrics.
Measures how well replicates cluster together in latent space.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
import sys
sys.path.append('.')

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset
from src.autoencoder.vae import VAE

def extract_latent_features(model, data_loader, device='cpu'):
    """Extract latent representations from model."""
    model.eval()
    model = model.to(device)
    
    all_latent = []
    all_treatments = []
    
    with torch.no_grad():
        for batch_data, _, batch_treatments in data_loader:
            batch_data = batch_data.to(device)
            mu, _ = model.encode(batch_data)
            all_latent.append(mu.cpu().numpy())
            all_treatments.extend(batch_treatments)
    
    latent = np.vstack(all_latent)
    treatments = np.array(all_treatments)
    
    return latent, treatments


def compute_topk_metrics(latent, treatments, k_values=[1, 3, 5, 9], metric='cosine', batch_size=1000):
    """
    Compute top-k accuracy and precision for replicate retrieval (memory-efficient).
    
    Parameters:
    -----------
    latent : np.ndarray
        Latent representations [n_samples, latent_dim]
    treatments : np.ndarray
        Treatment labels [n_samples]
    k_values : list
        List of k values to evaluate
    metric : str
        'cosine' or 'euclidean'
    batch_size : int
        Process in batches to save memory
    
    Returns:
    --------
    results : dict
        Dictionary with top-k accuracy and precision for each k
    """
    n_samples = len(latent)
    max_k = max(k_values)
    
    # Normalize for cosine similarity
    if metric == 'cosine':
        from sklearn.preprocessing import normalize
        latent_norm = normalize(latent, norm='l2')
    else:
        latent_norm = latent
    
    # Store results for each k
    all_accuracies = {k: [] for k in k_values}
    all_precisions = {k: [] for k in k_values}
    
    # Process in batches to avoid memory issues
    print(f"  Computing top-k metrics in batches (batch_size={batch_size})...")
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_latent = latent_norm[start_idx:end_idx]
        
        # Compute distances for this batch against all samples
        if metric == 'cosine':
            # Cosine similarity: higher is better
            similarity = np.dot(batch_latent, latent_norm.T)
            # Convert to distance (lower is better)
            distance = -similarity
        else:
            # Euclidean distance
            distance = euclidean_distances(batch_latent, latent_norm)
        
        # Find top-k nearest neighbors for each sample in batch
        # argsort along axis=1, take top max_k+1 (including self)
        nearest_indices = np.argsort(distance, axis=1)[:, :max_k+1]
        
        # Remove self (first column is always the sample itself with distance 0)
        nearest_indices = nearest_indices[:, 1:]
        
        # Evaluate for each sample in batch
        for i, global_idx in enumerate(range(start_idx, end_idx)):
            sample_treatment = treatments[global_idx]
            
            for k in k_values:
                top_k_neighbors = nearest_indices[i, :k]
                neighbor_treatments = treatments[top_k_neighbors]
                
                # Check replicates
                is_replicate = neighbor_treatments == sample_treatment
                
                # Accuracy: at least one replicate in top-k
                accuracy = 1.0 if is_replicate.any() else 0.0
                all_accuracies[k].append(accuracy)
                
                # Precision: proportion of replicates in top-k
                precision = is_replicate.sum() / k
                all_precisions[k].append(precision)
    
    # Aggregate results
    results = {}
    for k in k_values:
        results[f'top{k}_accuracy'] = np.mean(all_accuracies[k])
        results[f'top{k}_precision'] = np.mean(all_precisions[k])
    
    return results


def evaluate_model(model_path, model_type, processed_df, metadata, device='cpu'):
    """Evaluate a single model."""
    
    # Load model
    if model_type == 'contrastive':
        model = ContrastiveVAE(
            input_dim=processed_df.shape[1],
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            dropout=0.2,
            projection_dim=128
        )
    else:
        model = VAE(
            input_dim=processed_df.shape[1],
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            dropout=0.2
        )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset and loader
    dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Extract latent features
    latent, treatments = extract_latent_features(model, data_loader, device)
    
    # Compute metrics for both cosine and euclidean
    cosine_metrics = compute_topk_metrics(latent, treatments, k_values=[1, 3, 5, 9], metric='cosine')
    euclidean_metrics = compute_topk_metrics(latent, treatments, k_values=[1, 3, 5, 9], metric='euclidean')
    
    return {
        'model': model_path.name,
        'cosine': cosine_metrics,
        'euclidean': euclidean_metrics,
        'epoch': checkpoint.get('epoch', 'unknown')
    }


# ============================================================================
# Main Evaluation
# ============================================================================

print("="*70)
print("Top-k Retrieval Evaluation for Latent Space")
print("="*70)
print()

# Load data
print("Loading data...")
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

print("Preprocessing...")
processed_df, _ = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

# Find all models
models_dir = Path('models')
model_files = list(models_dir.glob('*.pt'))

print(f"\nFound {len(model_files)} model(s)")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# Evaluate each model
all_results = []

for model_file in model_files:
    print(f"{'='*70}")
    print(f"Evaluating: {model_file.name}")
    print(f"{'='*70}")
    
    # Determine model type
    if 'contrastive' in model_file.name.lower():
        model_type = 'contrastive'
    else:
        model_type = 'standard'
    
    print(f"Model type: {model_type} VAE")
    
    try:
        results = evaluate_model(model_file, model_type, processed_df, metadata, device)
        all_results.append(results)
        
        print(f"\nCosine Similarity Metrics:")
        print(f"  Top-1 Accuracy:  {results['cosine']['top1_accuracy']:.4f}")
        print(f"  Top-1 Precision: {results['cosine']['top1_precision']:.4f}")
        print(f"  Top-3 Accuracy:  {results['cosine']['top3_accuracy']:.4f}")
        print(f"  Top-3 Precision: {results['cosine']['top3_precision']:.4f}")
        print(f"  Top-5 Accuracy:  {results['cosine']['top5_accuracy']:.4f}")
        print(f"  Top-5 Precision: {results['cosine']['top5_precision']:.4f}")
        print(f"  Top-9 Accuracy:  {results['cosine']['top9_accuracy']:.4f}")
        print(f"  Top-9 Precision: {results['cosine']['top9_precision']:.4f}")
        
        print(f"\nEuclidean Distance Metrics:")
        print(f"  Top-1 Accuracy:  {results['euclidean']['top1_accuracy']:.4f}")
        print(f"  Top-1 Precision: {results['euclidean']['top1_precision']:.4f}")
        print(f"  Top-3 Accuracy:  {results['euclidean']['top3_accuracy']:.4f}")
        print(f"  Top-3 Precision: {results['euclidean']['top3_precision']:.4f}")
        print(f"  Top-5 Accuracy:  {results['euclidean']['top5_accuracy']:.4f}")
        print(f"  Top-5 Precision: {results['euclidean']['top5_precision']:.4f}")
        print(f"  Top-9 Accuracy:  {results['euclidean']['top9_accuracy']:.4f}")
        print(f"  Top-9 Precision: {results['euclidean']['top9_precision']:.4f}")
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Summary Comparison
# ============================================================================

if len(all_results) > 1:
    print("="*70)
    print("COMPARISON SUMMARY (Cosine Similarity)")
    print("="*70)
    print()
    
    # Create comparison table
    comparison_data = []
    for res in all_results:
        model_name = res['model'].replace('_hek293t_best.pt', '').replace('_', ' ').title()
        comparison_data.append({
            'Model': model_name,
            'Top-1 Acc': f"{res['cosine']['top1_accuracy']:.4f}",
            'Top-1 Prec': f"{res['cosine']['top1_precision']:.4f}",
            'Top-3 Acc': f"{res['cosine']['top3_accuracy']:.4f}",
            'Top-3 Prec': f"{res['cosine']['top3_precision']:.4f}",
            'Top-5 Acc': f"{res['cosine']['top5_accuracy']:.4f}",
            'Top-5 Prec': f"{res['cosine']['top5_precision']:.4f}",
            'Top-9 Acc': f"{res['cosine']['top9_accuracy']:.4f}",
            'Top-9 Prec': f"{res['cosine']['top9_precision']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('results/topk_metrics_comparison.csv', index=False)
    print(f"\nSaved comparison to: results/topk_metrics_comparison.csv")

# ============================================================================
# Interpretation Guide
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
Top-k Accuracy:  Proportion of samples where at least 1 of the top-k 
                 nearest neighbors is a replicate (same treatment)
                 
Top-k Precision: Average proportion of top-k neighbors that are replicates

Higher is better for both!

Expected for your data (3 replicates per treatment):
  Top-1 Accuracy:  0.60-0.80 (at least 1 replicate in nearest neighbor)
  Top-3 Accuracy:  0.85-0.95 (at least 1 replicate in top-3)
  Top-3 Precision: 0.60-0.80 (2 out of 3 neighbors are replicates)
  
Contrastive VAE should have higher metrics than Standard VAE!
""")

