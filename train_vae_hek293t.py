"""
Train VAE on HEK293T gene expression data.
Simple script to get started quickly.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Project root

from src.autoencoder.vae import (
    VAE, 
    train_vae, 
    GeneExpressionDataset,
    preprocess_gene_expression,
    plot_latent_space,
    plot_training_history,
    plot_reconstruction_quality
)

print("="*70)
print("VAE Training for HEK293T Gene Expression Data")
print("="*70)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("Loading data...")
counts_df = pd.read_csv(
    'Dataset/csv/HEK293T_Counts.csv', 
    header=1, 
    index_col=0
)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

print(f"Loaded counts: {counts_df.shape}")
print(f"Loaded metadata: {metadata.shape}")

# ============================================================================
# 2. Preprocess Data
# ============================================================================
print("\nPreprocessing gene expression data...")
processed_df, preprocess_metadata = preprocess_gene_expression(
    counts_df,
    method='log_normalize',  # Log2(x+1) transformation
    scale='standard',         # Standard scaling (z-score)
    filter_low_variance=True, # Remove low variance genes
    variance_threshold=0.01   # Variance threshold
)

# ============================================================================
# 3. Create Datasets
# ============================================================================
print("\nCreating train/validation datasets...")
train_dataset, val_dataset = GeneExpressionDataset.create_train_val_split(
    processed_df,
    labels=metadata['treatment'],
    val_split=0.2,
    random_state=42
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ============================================================================
# 4. Initialize Model
# ============================================================================
print("\nInitializing VAE model...")
input_dim = processed_df.shape[1]  # Number of genes
latent_dim = 64                     # Latent space dimension

model = VAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=[512, 256, 128],   # Encoder/decoder hidden layers
    dropout=0.2,                    # Dropout rate
    use_batch_norm=True             # Use batch normalization
)

print(f"Model architecture:")
print(f"  Input dim: {input_dim}")
print(f"  Latent dim: {latent_dim}")
print(f"  Hidden dims: {[512, 256, 128]}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 5. Train Model
# ============================================================================
print("\nTraining VAE...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

history = train_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-3,
    beta=1.0,              # Beta-VAE parameter (1.0 = standard VAE)
    device=device,
    patience=15,           # Early stopping patience
    save_path='models/vae_hek293t_best.pt',
    verbose=True
)

# ============================================================================
# 6. Visualize Results
# ============================================================================
print("\nGenerating visualizations...")

# Plot training history
plot_training_history(history, save_path='results/vae_training_history.png')

# Plot latent space
plot_latent_space(
    model, 
    val_loader, 
    labels=metadata['treatment'],
    device=device,
    save_path='results/vae_latent_space.png'
)

# Plot reconstruction quality
plot_reconstruction_quality(
    model,
    val_loader,
    n_samples=5,
    device=device,
    save_path='results/vae_reconstruction.png'
)

# ============================================================================
# 7. Extract and Save Latent Representations
# ============================================================================
print("\nExtracting latent representations...")
model.eval()
all_latent = []

with torch.no_grad():
    for batch_data, _ in DataLoader(
        GeneExpressionDataset(processed_df, metadata['treatment']),
        batch_size=64,
        shuffle=False
    ):
        batch_data = batch_data.to(device)
        latent = model.get_latent(batch_data, use_mean=True)
        all_latent.append(latent.cpu().numpy())

import numpy as np
latent_features = np.vstack(all_latent)

# Create DataFrame with latent features
latent_df = pd.DataFrame(
    latent_features,
    index=processed_df.index,
    columns=[f'latent_{i}' for i in range(latent_dim)]
)

# Combine with metadata
latent_with_metadata = pd.concat([metadata, latent_df], axis=1)
latent_with_metadata.to_csv('results/hek293t_latent_features.csv', index=False)

print(f"\nSaved latent features: {latent_df.shape}")
print(f"Output file: results/hek293t_latent_features.csv")

print("\n" + "="*70)
print("Training complete!")
print("="*70)
print("\nGenerated files:")
print("  - models/vae_hek293t_best.pt (model checkpoint)")
print("  - results/vae_training_history.png")
print("  - results/vae_latent_space.png")
print("  - results/vae_reconstruction.png")
print("  - results/hek293t_latent_features.csv")

