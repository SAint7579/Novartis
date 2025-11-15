"""
Train Triplet VAE with LogFC-weighted loss.

Triplets:
- Anchor: Treatment sample
- Positive: Replicate of same treatment  
- Negative: DMSO control

Weight: exp(-β * logFC_distance(anchor, positive))
- Small logFC distance between replicates -> high weight (biologically similar)
- Large logFC distance -> lower weight (allows variation)
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Project root

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.triplet_vae import (
    TripletVAE,
    train_triplet_vae,
    TripletGeneExpressionDataset
)
from src.autoencoder.contrastive_vae import plot_latent_space_by_treatment, ContrastiveGeneExpressionDataset
from src.autoencoder.contrastive_vae.utils import plot_training_history

print("="*70)
print("Triplet VAE with LogFC-Weighted Loss")
print("="*70)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

print(f"Counts: {counts_df.shape}")
print(f"Metadata: {metadata.shape}")

# ============================================================================
# Preprocess
# ============================================================================

print("\nPreprocessing...")
processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard',
    filter_low_variance=True, variance_threshold=0.01
)

# ============================================================================
# Create Triplet Datasets
# ============================================================================

print("\nCreating triplet datasets...")
print("  Anchor: Treatment samples")
print("  Positive: Replicates of same treatment")
print("  Negative: DMSO controls + other compounds (50/50 mix)")
print("  Distance: Cosine similarity (like InfoNCE)")
print()

train_dataset, val_dataset = TripletGeneExpressionDataset.create_train_val_split(
    processed_df,
    treatments=metadata['treatment'],
    val_split=0.2,
    dmso_label='DMSO',
    include_compound_negatives=True,  # Use both DMSO and other compounds as negatives
    random_state=42
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ============================================================================
# Initialize Model
# ============================================================================

print("\nInitializing Triplet VAE...")
model = TripletVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    hidden_dims=[512, 256, 128],
    dropout=0.2,
    use_batch_norm=True
)

print(f"  Input dim: {processed_df.shape[1]}")
print(f"  Latent dim: 64")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# Train
# ============================================================================

print("\nTraining Triplet VAE...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Hyperparameters
BETA = 1.0         # KL weight
GAMMA = 0.5        # Triplet loss weight
MARGIN = 1.0       # Triplet margin
LOGFC_BETA = 0.1   # LogFC weighting temperature (lower = more uniform weights)

print(f"\nHyperparameters:")
print(f"  Beta (KL): {BETA}")
print(f"  Gamma (Triplet): {GAMMA}")
print(f"  Margin: {MARGIN}")
print(f"  LogFC Beta: {LOGFC_BETA}")
print()

history = train_triplet_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-3,
    beta=BETA,
    gamma=GAMMA,
    margin=MARGIN,
    logfc_beta=LOGFC_BETA,
    device=device,
    patience=15,
    save_path='models/triplet_vae2_hek293t_best.pt',
    verbose=True
)

# ============================================================================
# Visualize
# ============================================================================

print("\nGenerating visualizations...")

# Training history
plot_training_history(history, save_path='results/triplet_vae_training.png')

# Latent space (need to use ContrastiveDataset for visualization)
print("Generating latent space plot...")
vis_dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
vis_loader = DataLoader(vis_dataset, batch_size=128, shuffle=False)

plot_latent_space_by_treatment(
    model, vis_loader, metadata['treatment'],
    device=device, method='pca',
    save_path='results/triplet_vae_latent.png'
)

# ============================================================================
# Extract Features
# ============================================================================

print("\nExtracting latent features...")
model.eval()
all_latent = []

with torch.no_grad():
    for batch_data, _, _ in vis_loader:
        latent = model.get_latent(batch_data.to(device), use_mean=True)
        all_latent.append(latent.cpu().numpy())

import numpy as np
latent_features = np.vstack(all_latent)

latent_df = pd.DataFrame(
    latent_features,
    columns=[f'latent_{i}' for i in range(64)]
)

result_df = pd.concat([metadata, latent_df], axis=1)
result_df.to_csv('results/triplet_vae_latent_features.csv', index=False)

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"{'='*70}")
print(f"\nFiles saved:")
print(f"  - models/triplet_vae2_hek293t_best.pt")
print(f"  - results/triplet_vae_training.png")
print(f"  - results/triplet_vae_latent.png")
print(f"  - results/triplet_vae_latent_features.csv")
print(f"\nFinal triplet loss: {history['val_triplet_loss'][-1]:.4f}")
print(f"Average logFC weight: {history['val_avg_weight'][-1]:.4f}")

