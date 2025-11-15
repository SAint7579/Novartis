"""
Copy-paste this into your Jupyter notebook to quickly train a VAE
"""

# ============================================================================
# SIMPLE VAE TRAINING - COPY PASTE INTO YOUR NOTEBOOK
# ============================================================================

import sys
sys.path.append('..')  # If running from Notebooks folder

import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.autoencoder.vae import (
    VAE, train_vae, GeneExpressionDataset,
    preprocess_gene_expression, plot_latent_space, plot_training_history
)

# Load your data (adjust paths as needed)
dataframe = pd.read_csv('../Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel("../Dataset/HEK293T_MetaData.xlsx", header=1)

# Preprocess
processed_df, _ = preprocess_gene_expression(
    dataframe, method='log_normalize', scale='standard'
)

# Create datasets
train_dataset, val_dataset = GeneExpressionDataset.create_train_val_split(
    processed_df, labels=metadata['treatment'], val_split=0.2
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize model
model = VAE(input_dim=processed_df.shape[1], latent_dim=64)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history = train_vae(
    model, train_loader, val_loader,
    epochs=100, device=device, save_path='../models/vae_model.pt'
)

# Visualize
plot_training_history(history)
plot_latent_space(model, val_loader, labels=metadata['treatment'], device=device)

# Extract features
model.eval()
all_latent = []
full_dataset = GeneExpressionDataset(processed_df, metadata['treatment'])
full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for batch_data, _ in full_loader:
        latent = model.get_latent(batch_data.to(device), use_mean=True)
        all_latent.append(latent.cpu().numpy())

import numpy as np
latent_features = np.vstack(all_latent)

# Create DataFrame with latent features
latent_df = pd.DataFrame(
    latent_features,
    index=processed_df.index,
    columns=[f'latent_{i}' for i in range(64)]
)

# Combine with metadata
result_df = pd.concat([metadata, latent_df], axis=1)
print(f"✓ Done! Latent features shape: {latent_df.shape}")
print(f"✓ Use 'result_df' for downstream analysis")

