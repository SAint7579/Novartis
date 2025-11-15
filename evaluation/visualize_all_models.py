"""
Visualize latent space for ALL trained models in models/ folder.
Generates plots for each model weight file.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import (
    ContrastiveVAE,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment
)
from src.autoencoder.vae import VAE, GeneExpressionDataset

print("="*70)
print("Batch Latent Space Visualization")
print("="*70)
print()

# ============================================================================
# Configuration
# ============================================================================

# Get paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'latent_plots'
COUNTS_CSV = PROJECT_ROOT / 'Dataset' / 'csv' / 'HEK293T_Counts.csv'
METADATA_XLSX = PROJECT_ROOT / 'Dataset' / 'HEK293T_MetaData.xlsx'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# Find All Model Files
# ============================================================================

print(f"\nSearching for models in: {MODELS_DIR}")
model_files = list(MODELS_DIR.glob('*.pt'))

if not model_files:
    print(f"No model files found in {MODELS_DIR}")
    exit()

print(f"Found {len(model_files)} model(s):")
for mf in model_files:
    print(f"  - {mf.name}")

# ============================================================================
# Load Data (once for all models)
# ============================================================================

print(f"\nLoading data...")
counts_df = pd.read_csv(COUNTS_CSV, header=1, index_col=0)
metadata = pd.read_excel(METADATA_XLSX, header=1)

print(f"Preprocessing...")
processed_df, _ = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

print(f"Data ready: {processed_df.shape[0]} samples x {processed_df.shape[1]} genes")

# ============================================================================
# Process Each Model
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

for model_file in model_files:
    # Check if plots already exist
    output_name = model_file.stem + '_latent.png'
    output_tsne = model_file.stem + '_latent_tsne.png'
    output_path = OUTPUT_DIR / output_name
    output_tsne_path = OUTPUT_DIR / output_tsne
    
    if output_path.exists() and output_tsne_path.exists():
        print(f"Skipping {model_file.name} (plots already exist)")
        continue
    
    print(f"{'='*70}")
    print(f"Processing: {model_file.name}")
    print(f"{'='*70}")
    
    # Determine model type from filename
    if 'infonce' in model_file.name.lower():
        model_type = 'infonce'
    elif 'contrastive' in model_file.name.lower():
        model_type = 'contrastive'
    elif 'triplet' in model_file.name.lower():
        model_type = 'triplet'
    else:
        model_type = 'standard'
    
    print(f"Model type: {model_type} VAE")
    
    try:
        # Initialize model
        if model_type == 'contrastive':
            model = ContrastiveVAE(
                input_dim=processed_df.shape[1],
                latent_dim=64,
                hidden_dims=[512, 256, 128],
                dropout=0.2,
                projection_dim=128
            )
            dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
        elif model_type == 'triplet':
            from src.autoencoder.triplet_vae import TripletVAE
            model = TripletVAE(
                input_dim=processed_df.shape[1],
                latent_dim=64,
                hidden_dims=[512, 256, 128],
                dropout=0.2
            )
            dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
        elif model_type == 'infonce':
            # InfoNCE uses standard VAE architecture
            model = VAE(
                input_dim=processed_df.shape[1],
                latent_dim=64,
                hidden_dims=[512, 256, 128],
                dropout=0.2
            )
            dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
        else:
            model = VAE(
                input_dim=processed_df.shape[1],
                latent_dim=64,
                hidden_dims=[512, 256, 128],
                dropout=0.2
            )
            dataset = GeneExpressionDataset(processed_df, metadata['treatment'])
        
        # Load weights (weights_only=False for compatibility with older PyTorch)
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  Loaded weights from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        print(f"  Generating visualizations...")
        
        # Plot (use contrastive dataset for both - it works with both model types)
        if model_type == 'standard':
            # Re-create with contrastive dataset for visualization compatibility
            dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
            data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        # PCA plot
        if not output_path.exists():
            print(f"  Generating PCA plot...")
            plot_latent_space_by_treatment(
                model=model,
                data_loader=data_loader,
                treatments=metadata['treatment'],
                device=device,
                method='pca',
                highlight_treatments=None,
                save_path=str(output_path),
                figsize=(16, 6)
            )
            print(f"  -> Saved to: {output_path}")
        else:
            print(f"  PCA plot exists, skipping")
        
        # t-SNE plot
        if not output_tsne_path.exists():
            print(f"  Generating t-SNE plot...")
            plot_latent_space_by_treatment(
                model=model,
                data_loader=data_loader,
                treatments=metadata['treatment'],
                device=device,
                method='tsne',
                highlight_treatments=None,
                save_path=str(output_tsne_path),
                figsize=(16, 6)
            )
            print(f"  -> Saved to: {output_tsne_path}")
        else:
            print(f"  t-SNE plot exists, skipping")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\n{'='*70}")
print(f"All visualizations saved to: {OUTPUT_DIR}")
print(f"{'='*70}")
print(f"\nGenerated files:")
for plot_file in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {plot_file.name}")

