# Variational Autoencoder (VAE) for Gene Expression Data

A PyTorch implementation of VAE for dimensionality reduction and feature learning from high-dimensional gene expression data.

## Features

- ✅ **Flexible architecture** - Customizable encoder/decoder layers
- ✅ **Beta-VAE support** - Control KL divergence weight
- ✅ **Preprocessing utilities** - Log normalization, scaling, variance filtering
- ✅ **Training utilities** - Early stopping, learning rate scheduling
- ✅ **Visualization tools** - Latent space, training history, reconstruction quality
- ✅ **Easy-to-use API** - Simple interface for quick experimentation

## Quick Start

### Installation

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm
```

### Basic Usage

```python
import pandas as pd
from src.autoencoder.vae import VAE, train_vae, GeneExpressionDataset, preprocess_gene_expression
from torch.utils.data import DataLoader

# Load your data
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)

# Preprocess
processed_df, _ = preprocess_gene_expression(
    counts_df,
    method='log_normalize',
    scale='standard'
)

# Create datasets
train_dataset, val_dataset = GeneExpressionDataset.create_train_val_split(
    processed_df, val_split=0.2
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize and train model
model = VAE(input_dim=processed_df.shape[1], latent_dim=64)

history = train_vae(
    model,
    train_loader,
    val_loader,
    epochs=100,
    device='cuda'
)
```

### Command Line Training

```bash
# Train VAE on your HEK293T data
python train_vae_hek293t.py
```

This will:
1. Load and preprocess your gene expression data
2. Train a VAE model
3. Generate visualizations
4. Save latent features to CSV
5. Save trained model checkpoint

## Model Architecture

```
Encoder:
  Input (genes) → Dense(512) → BatchNorm → ReLU → Dropout
                → Dense(256) → BatchNorm → ReLU → Dropout
                → Dense(128) → BatchNorm → ReLU → Dropout
                → Dense(latent_dim) [mu and logvar]

Latent Space:
  Reparameterization: z = mu + std * epsilon

Decoder:
  Latent → Dense(128) → BatchNorm → ReLU → Dropout
         → Dense(256) → BatchNorm → ReLU → Dropout
         → Dense(512) → BatchNorm → ReLU → Dropout
         → Dense(genes) [reconstructed expression]
```

## Loss Function

```
Total Loss = Reconstruction Loss + β * KL Divergence

Reconstruction Loss = MSE(reconstructed, original)
KL Divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

- **β = 1.0**: Standard VAE
- **β > 1.0**: β-VAE (encourages disentanglement, may reduce reconstruction quality)
- **β < 1.0**: More emphasis on reconstruction

## Configuration Options

### Preprocessing

```python
processed_df, metadata = preprocess_gene_expression(
    counts_df,
    method='log_normalize',      # 'log_normalize', 'tpm', or 'none'
    scale='standard',             # 'standard', 'minmax', or 'none'
    filter_low_variance=True,     # Remove low variance genes
    variance_threshold=0.01       # Variance threshold
)
```

### Model Parameters

```python
model = VAE(
    input_dim=10000,              # Number of genes
    latent_dim=64,                # Latent space dimension
    hidden_dims=[512, 256, 128],  # Encoder/decoder layers
    dropout=0.2,                  # Dropout rate
    use_batch_norm=True           # Batch normalization
)
```

### Training Parameters

```python
history = train_vae(
    model,
    train_loader,
    val_loader,
    epochs=100,                   # Maximum epochs
    learning_rate=1e-3,           # Initial learning rate
    beta=1.0,                     # Beta-VAE parameter
    device='cuda',                # 'cuda' or 'cpu'
    patience=15,                  # Early stopping patience
    save_path='model.pt'          # Save best model
)
```

## Using Latent Features

After training, extract latent representations for downstream tasks:

```python
# Extract latent features
model.eval()
with torch.no_grad():
    latent_features = model.get_latent(data_tensor, use_mean=True)

# Use for:
# - Clustering
# - Classification
# - Visualization (t-SNE, UMAP)
# - Drug response prediction
# - Biomarker discovery
```

## Visualization

```python
from src.autoencoder.vae import plot_latent_space, plot_training_history

# Plot training curves
plot_training_history(history, save_path='training.png')

# Visualize latent space
plot_latent_space(model, data_loader, labels=treatments, save_path='latent.png')

# Check reconstruction quality
plot_reconstruction_quality(model, data_loader, save_path='recon.png')
```

## Advanced: Custom Training Loop

```python
import torch
from src.autoencoder.vae.model import vae_loss

model = VAE(input_dim=10000, latent_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch_data, _ in train_loader:
        # Forward pass
        recon_batch, mu, logvar = model(batch_data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Tips for Gene Expression Data

1. **Preprocessing is crucial**: Always log-transform and scale your data
2. **Filter low variance genes**: Reduces noise and speeds up training
3. **Start with latent_dim=64**: Good balance between compression and reconstruction
4. **Use beta=1.0 initially**: Adjust only if needed for specific applications
5. **Monitor reconstruction**: Gene expression should correlate >0.9 with reconstructions
6. **Early stopping**: Prevents overfitting on high-dimensional data

## Hyperparameter Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `latent_dim` | Compression level | 32-128 for gene data |
| `hidden_dims` | Model capacity | [512, 256, 128] is good default |
| `beta` | Disentanglement | 1.0 (standard), 2-4 (more disentangled) |
| `dropout` | Regularization | 0.1-0.3 |
| `learning_rate` | Convergence speed | 1e-3 to 1e-4 |
| `batch_size` | Training stability | 32-128 |

## Output Files

After running `train_vae_hek293t.py`:

```
models/
  └── vae_hek293t_best.pt          # Trained model checkpoint

results/
  ├── vae_training_history.png     # Loss curves
  ├── vae_latent_space.png         # 2D latent visualization
  ├── vae_reconstruction.png       # Quality assessment
  └── hek293t_latent_features.csv  # Extracted features + metadata
```

## Loading a Trained Model

```python
import torch
from src.autoencoder.vae import VAE

# Initialize model with same architecture
model = VAE(input_dim=10000, latent_dim=64)

# Load checkpoint
checkpoint = torch.load('models/vae_hek293t_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    latent = model.get_latent(new_data)
```

## Troubleshooting

**Q: Loss is NaN or Inf**
- Check for NaN/Inf values in input data
- Reduce learning rate
- Add gradient clipping

**Q: Reconstruction quality is poor**
- Increase latent_dim
- Add more hidden layers
- Reduce beta (focus on reconstruction)
- Train for more epochs

**Q: Latent space looks random**
- Increase beta (encourage structure)
- Train longer
- Check if labels/treatments actually have signal

**Q: Out of memory**
- Reduce batch_size
- Reduce hidden_dims
- Use gradient accumulation

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
- Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics (scVI)

