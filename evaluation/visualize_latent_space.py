"""
Standalone script to visualize latent space from trained VAE weights.
Generates latent space plot colored by treatment (compound).
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import (
    ContrastiveVAE,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment
)

print("="*70)
print("Latent Space Visualization")
print("="*70)
print()

# ============================================================================
# Configuration
# ============================================================================

# Model weights path
MODEL_PATH = 'models/contrastive_vae_hek293t_best.pt'

# Data paths
COUNTS_CSV = 'Dataset/csv/HEK293T_Counts.csv'
METADATA_XLSX = 'Dataset/HEK293T_MetaData.xlsx'

# Output
OUTPUT_PATH = 'latent_space_by_treatment.png'

# Visualization settings
METHOD = 'pca'  # 'pca' or 'tsne'
HIGHLIGHT_COMPOUNDS = None  # or ['HY_50946', 'HY_18686', 'DMSO']

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
counts_df = pd.read_csv(COUNTS_CSV, header=1, index_col=0)
metadata = pd.read_excel(METADATA_XLSX, header=1)

print(f"Loaded: {counts_df.shape[0]} samples x {counts_df.shape[1]} genes")
print(f"Unique treatments: {metadata['treatment'].nunique()}")

# ============================================================================
# Preprocess (same as training)
# ============================================================================

print("\nPreprocessing...")
processed_df, _ = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

# ============================================================================
# Create Dataset
# ============================================================================

print("\nCreating dataset...")
dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# ============================================================================
# Load Model
# ============================================================================

print(f"\nLoading model from: {MODEL_PATH}")

# Initialize model (same architecture as training)
model = ContrastiveVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    hidden_dims=[512, 256, 128],
    dropout=0.2,
    use_batch_norm=True,
    projection_dim=128
)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully!")
print(f"  Input dim: {processed_df.shape[1]}")
print(f"  Latent dim: 64")
print(f"  Epoch trained: {checkpoint.get('epoch', 'unknown')}")

# ============================================================================
# Generate Visualization
# ============================================================================

print(f"\nGenerating latent space visualization...")
print(f"  Method: {METHOD}")
print(f"  Highlight: {HIGHLIGHT_COMPOUNDS or 'Controls vs Compounds'}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

plot_latent_space_by_treatment(
    model=model,
    data_loader=data_loader,
    treatments=metadata['treatment'],
    device=device,
    method=METHOD,
    highlight_treatments=HIGHLIGHT_COMPOUNDS,
    save_path=OUTPUT_PATH,
    figsize=(16, 6)
)

print(f"\n{'='*70}")
print(f"Visualization saved to: {OUTPUT_PATH}")
print(f"{'='*70}")

