"""
Copy-paste this into your Jupyter notebook to use Contrastive VAE
"""

# ============================================================================
# CONTRASTIVE VAE - QUICK START FOR NOTEBOOKS
# ============================================================================

import sys
sys.path.append('..')

import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import (
    ContrastiveVAE,
    train_contrastive_vae,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment,
    plot_treatment_clusters,
    compute_replicate_agreement
)

# Load data
dataframe = pd.read_csv('../Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel("../Dataset/HEK293T_MetaData.xlsx", header=1)

# Preprocess
processed_df, _ = preprocess_gene_expression(
    dataframe, method='log_normalize', scale='standard'
)

# Create contrastive datasets (stratified to keep replicates in both sets)
train_dataset, val_dataset = ContrastiveGeneExpressionDataset.create_train_val_split(
    processed_df, 
    treatments=metadata['treatment'],
    val_split=0.2,
    stratify=True  # IMPORTANT!
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize Contrastive VAE
model = ContrastiveVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    projection_dim=128  # For contrastive learning
)

# Train with InfoNCE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history = train_contrastive_vae(
    model, train_loader, val_loader,
    epochs=100,
    beta=1.0,        # KL weight
    gamma=0.5,       # InfoNCE weight (KEY!)
    temperature=0.1, # Contrastive temperature
    device=device,
    save_path='../models/contrastive_vae.pt'
)

# Visualize latent space COLORED BY TREATMENT
plot_latent_space_by_treatment(
    model, 
    val_loader,
    treatments=metadata['treatment'],
    method='pca',  # or 'tsne'
)

# Show how specific compounds cluster
plot_treatment_clusters(
    model,
    val_loader,
    treatments=metadata['treatment'],
    n_treatments=12
)

# Compute replicate agreement
agreement_df = compute_replicate_agreement(
    model, val_loader, metadata['treatment']
)

# Extract latent features
model.eval()
all_latent = []
full_dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

with torch.no_grad():
    for batch_data, _, _ in full_loader:
        latent = model.get_latent(batch_data.to(device), use_mean=True)
        all_latent.append(latent.cpu().numpy())

import numpy as np
latent_features = np.vstack(all_latent)

# Create DataFrame
latent_df = pd.DataFrame(
    latent_features,
    columns=[f'latent_{i}' for i in range(64)]
)

result_df = pd.concat([metadata, latent_df], axis=1)

print(f"\n✓ Done! Latent features shape: {latent_df.shape}")
print(f"✓ Replicates are now grouped together in latent space!")
print(f"✓ Use 'result_df' for downstream analysis")

# ============================================================================
# VISUALIZE SPECIFIC COMPOUNDS OF INTEREST
# ============================================================================

from src.autoencoder.contrastive_vae.utils import visualize_specific_compounds

# Choose compounds you care about
my_compounds = ['HY_50946', 'HY_18686', 'HY_17592A', 'DMSO']

visualize_specific_compounds(
    model,
    processed_df,
    metadata,
    compound_list=my_compounds
)

print("✓ Replicates should cluster together for each compound!")

