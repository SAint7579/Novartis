"""
Simple script to plot latent space from trained model.
Works with both Standard VAE and Contrastive VAE.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
import argparse
sys.path.append('.')

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset, plot_latent_space_by_treatment
from src.autoencoder.vae import VAE, GeneExpressionDataset

# ============================================================================
# Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Plot latent space from trained model')
parser.add_argument('--model', type=str, default='models/contrastive_vae_hek293t_best.pt',
                    help='Path to model weights')
parser.add_argument('--output', type=str, default='latent_space.png',
                    help='Output image path')
parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'],
                    help='Dimensionality reduction method')
parser.add_argument('--highlight', type=str, nargs='*', default=None,
                    help='Compounds to highlight (e.g., HY_50946 HY_18686 DMSO)')
parser.add_argument('--model-type', type=str, default='contrastive', choices=['contrastive', 'standard'],
                    help='Type of VAE model')

args = parser.parse_args()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard', 
    filter_low_variance=True, variance_threshold=0.01
)

# ============================================================================
# Load Model
# ============================================================================

print(f"Loading {args.model_type} VAE from: {args.model}")

if args.model_type == 'contrastive':
    model = ContrastiveVAE(
        input_dim=processed_df.shape[1],
        latent_dim=64,
        hidden_dims=[512, 256, 128],
        dropout=0.2,
        projection_dim=128
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

checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# ============================================================================
# Plot
# ============================================================================

print(f"Generating plot...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

plot_latent_space_by_treatment(
    model=model,
    data_loader=data_loader,
    treatments=metadata['treatment'],
    device=device,
    method=args.method,
    highlight_treatments=args.highlight,
    save_path=args.output
)

print(f"\nSaved to: {args.output}")

