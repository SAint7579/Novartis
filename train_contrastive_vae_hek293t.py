"""
Train Contrastive VAE on HEK293T gene expression data.
Uses InfoNCE loss to group replicates of same perturbation together.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from src.autoencoder.vae import preprocess_gene_expression
from src.autoencoder.contrastive_vae import (
    ContrastiveVAE,
    train_contrastive_vae,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment,
    plot_treatment_clusters,
    compute_replicate_agreement
)
from src.autoencoder.contrastive_vae.utils import (
    plot_training_history,
    plot_replicate_similarity_heatmap,
    visualize_specific_compounds
)

print("="*70)
print("Contrastive VAE Training for HEK293T Gene Expression Data")
print("With InfoNCE Loss for Replicate Grouping")
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
print(f"Unique treatments: {metadata['treatment'].nunique()}")

# Check replicate structure
treatment_counts = metadata['treatment'].value_counts()
print(f"\nReplicate structure:")
print(f"  Treatments with 3 replicates: {(treatment_counts == 3).sum()}")
print(f"  Treatments with >3 replicates: {(treatment_counts > 3).sum()}")
print(f"  Average replicates per treatment: {treatment_counts.mean():.1f}")

# ============================================================================
# 2. Preprocess Data
# ============================================================================
print("\nPreprocessing gene expression data...")
processed_df, preprocess_metadata = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard',
    filter_low_variance=True,
    variance_threshold=0.01
)

# ============================================================================
# 3. Create Contrastive Datasets
# ============================================================================
print("\nCreating contrastive datasets (with replicate grouping)...")
train_dataset, val_dataset = ContrastiveGeneExpressionDataset.create_train_val_split(
    processed_df,
    treatments=metadata['treatment'],
    val_split=0.2,
    stratify=True,  # IMPORTANT: Keep replicates in both train and val
    random_state=42
)

# Use standard DataLoader (shuffle ensures variety)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Treatments in train: {train_dataset.num_treatments}")
print(f"Treatments in val: {val_dataset.num_treatments}")

# ============================================================================
# 4. Initialize Contrastive VAE Model
# ============================================================================
print("\nInitializing Contrastive VAE model...")
input_dim = processed_df.shape[1]
latent_dim = 64
projection_dim = 128  # Larger projection space for contrastive learning

model = ContrastiveVAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=[512, 256, 128],
    dropout=0.2,
    use_batch_norm=True,
    projection_dim=projection_dim
)

print(f"Model architecture:")
print(f"  Input dim: {input_dim}")
print(f"  Latent dim: {latent_dim}")
print(f"  Projection dim: {projection_dim}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 5. Train Contrastive VAE
# ============================================================================
print("\nTraining Contrastive VAE with InfoNCE loss...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Hyperparameters
BETA = 1.0          # KL divergence weight
GAMMA = 0.5         # InfoNCE contrastive weight (KEY PARAMETER!)
TEMPERATURE = 0.1   # Contrastive temperature

print(f"\nHyperparameters:")
print(f"  Beta (KL weight): {BETA}")
print(f"  Gamma (InfoNCE weight): {GAMMA}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Learning rate: 1e-3")
print()

history = train_contrastive_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-3,
    beta=BETA,
    gamma=GAMMA,
    temperature=TEMPERATURE,
    device=device,
    patience=15,
    save_path='models/contrastive_vae_hek293t_best.pt',
    verbose=True
)

# ============================================================================
# 6. Visualize Results
# ============================================================================
print("\nGenerating visualizations...")

# 6.1 Training history
print("  Plotting training history...")
plot_training_history(history, save_path='results/contrastive_vae_training_history.png')

# 6.2 Latent space colored by treatment
print("  Plotting latent space by treatment...")
plot_latent_space_by_treatment(
    model,
    val_loader,
    treatments=metadata['treatment'],
    method='pca',
    save_path='results/contrastive_vae_latent_by_treatment.png'
)

# 6.3 Individual treatment clusters
print("  Plotting treatment clusters...")
plot_treatment_clusters(
    model,
    val_loader,
    treatments=metadata['treatment'],
    n_treatments=12,
    save_path='results/contrastive_vae_treatment_clusters.png'
)

# 6.4 Replicate similarity heatmap
print("  Plotting replicate similarity heatmap...")
plot_replicate_similarity_heatmap(
    model,
    val_loader,
    treatments=metadata['treatment'],
    n_treatments=15,
    save_path='results/contrastive_vae_replicate_heatmap.png'
)

# 6.5 Specific compounds of interest
print("  Visualizing specific compounds...")
# Select a few interesting compounds (adjust as needed)
sample_compounds = metadata['treatment'].value_counts().head(8).index.tolist()
sample_compounds = [c for c in sample_compounds if c not in ['DMSO', 'Blank', 'RNA']][:6]
sample_compounds = ['DMSO'] + sample_compounds[:5] if 'DMSO' in metadata['treatment'].values else sample_compounds

visualize_specific_compounds(
    model,
    processed_df,
    metadata,
    compound_list=sample_compounds,
    save_path='results/contrastive_vae_specific_compounds.png'
)

# ============================================================================
# 7. Compute Replicate Agreement
# ============================================================================
print("\nComputing replicate agreement metrics...")
agreement_df = compute_replicate_agreement(
    model,
    val_loader,
    treatments=metadata['treatment'],
    verbose=True
)

# Save agreement metrics
agreement_df.to_csv('results/contrastive_vae_replicate_agreement.csv', index=False)
print(f"\nSaved replicate agreement to: results/contrastive_vae_replicate_agreement.csv")

# ============================================================================
# 8. Extract and Save Latent Features
# ============================================================================
print("\nExtracting latent representations...")
model.eval()
all_latent = []

full_dataset = ContrastiveGeneExpressionDataset(processed_df, metadata['treatment'])
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

with torch.no_grad():
    for batch_data, _, _ in full_loader:
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
latent_with_metadata.to_csv('results/contrastive_vae_latent_features.csv', index=False)

print(f"Saved latent features: {latent_df.shape}")
print(f"Output file: results/contrastive_vae_latent_features.csv")

# ============================================================================
# 9. Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("Training Complete!")
print("="*70)

print("\nGenerated files:")
print("  MODELS:")
print("    - models/contrastive_vae_hek293t_best.pt")
print("\n  VISUALIZATIONS:")
print("    - results/contrastive_vae_training_history.png")
print("    - results/contrastive_vae_latent_by_treatment.png")
print("    - results/contrastive_vae_treatment_clusters.png")
print("    - results/contrastive_vae_replicate_heatmap.png")
print("    - results/contrastive_vae_specific_compounds.png")
print("\n  DATA:")
print("    - results/contrastive_vae_latent_features.csv")
print("    - results/contrastive_vae_replicate_agreement.csv")

print("\n" + "="*70)
print("Key Metrics:")
print("="*70)
print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
print(f"Final contrastive loss: {history['val_contrastive_loss'][-1]:.4f}")
print(f"Average replicate distance: {agreement_df[agreement_df['n_replicates']>1]['mean_distance'].mean():.4f}")

print("\n💡 Next Steps:")
print("  1. Check visualizations to see if replicates cluster well")
print("  2. Compare with standard VAE (train_vae_hek293t.py)")
print("  3. Use latent features for downstream analysis")
print("  4. Tune gamma if needed (increase for tighter clusters)")

