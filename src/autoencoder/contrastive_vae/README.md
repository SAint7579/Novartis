## Contrastive VAE with InfoNCE Loss

A Variational Autoencoder that leverages your experimental design (3 replicates per perturbation) to learn better representations through contrastive learning.

## 🎯 Key Idea

**Standard VAE**: Learns to reconstruct gene expression

**Contrastive VAE**: Learns to reconstruct gene expression + groups replicates together

### How It Works

For each sample, the model learns to:
1. ✅ **Reconstruct** gene expression (VAE objective)
2. ✅ **Pull together** replicates of the same perturbation (InfoNCE)
3. ✅ **Push apart** different perturbations (InfoNCE)

This creates a latent space where:
- Replicates of HY_50946 cluster tightly together
- HY_50946 is far from HY_18686
- DMSO controls group together
- Biological replicates are close, technical noise is reduced

## 🔬 Loss Function

```
Total Loss = Reconstruction Loss + β * KL Divergence + γ * InfoNCE Loss

InfoNCE pulls replicates together:
- Positive pairs: same compound (3 replicates)
- Negative pairs: different compounds
```

## 📊 Advantages Over Standard VAE

| Metric | Standard VAE | Contrastive VAE |
|--------|--------------|-----------------|
| Uses replicate structure | ❌ No | ✅ Yes |
| Groups same compounds | 🟡 Sometimes | ✅ Always |
| Reduces technical noise | 🟡 Moderate | ✅ Strong |
| Separates compounds | 🟡 Variable | ✅ Enforced |
| Interpretability | Good | **Better** |

## 🚀 Quick Start

```python
from src.autoencoder.contrastive_vae import (
    ContrastiveVAE,
    train_contrastive_vae,
    ContrastiveGeneExpressionDataset,
    plot_latent_space_by_treatment
)

# Create dataset (knows about replicates!)
train_dataset, val_dataset = ContrastiveGeneExpressionDataset.create_train_val_split(
    processed_df,
    treatments=metadata['treatment'],
    stratify=True  # Ensures each treatment in train AND val
)

# Initialize model
model = ContrastiveVAE(
    input_dim=processed_df.shape[1],
    latent_dim=64,
    projection_dim=128  # Projection space for contrastive learning
)

# Train with InfoNCE
history = train_contrastive_vae(
    model, train_loader, val_loader,
    beta=1.0,      # KL weight
    gamma=0.5,     # InfoNCE weight (tune this!)
    temperature=0.1  # Contrastive temperature
)

# Visualize by treatment
plot_latent_space_by_treatment(model, val_loader, metadata['treatment'])
```

## 🎛️ Hyperparameters

### Key Parameters to Tune

**`gamma` (InfoNCE weight)**
- `0.0`: Standard VAE (no contrastive learning)
- `0.1-0.5`: Mild contrastive (recommended start)
- `0.5-2.0`: Strong contrastive (may hurt reconstruction)
- Higher = Replicates cluster tighter, but reconstruction may suffer

**`temperature` (Contrastive temperature)**
- `0.05`: Very strict similarity (tight clusters)
- `0.1`: Balanced (recommended)
- `0.5`: Looser clusters

**`beta` (KL weight)**
- `1.0`: Standard VAE
- `>1.0`: More regularization

### Recommended Settings

**For your HEK293T data (3 replicates):**
```python
gamma = 0.5        # InfoNCE weight
temperature = 0.1  # Contrastive temperature  
beta = 1.0         # Standard VAE regularization
```

## 📈 Expected Results

After training with `gamma=0.5`:

1. **Replicate distance**: Should be 50-70% lower than standard VAE
2. **Reconstruction**: Slightly worse than standard VAE (~5-10%)
3. **Clustering**: Much better - replicates visibly grouped
4. **Downstream performance**: Better for classification/clustering

## 🎨 Visualization Functions

### 1. Latent Space Colored by Treatment

```python
from src.autoencoder.contrastive_vae import plot_latent_space_by_treatment

plot_latent_space_by_treatment(
    model, 
    data_loader,
    treatments=metadata['treatment'],
    method='pca',  # or 'tsne'
    highlight_treatments=['HY_50946', 'HY_18686', 'DMSO'],
    save_path='latent_by_treatment.png'
)
```

Shows:
- All treatments (colored)
- Specific treatments highlighted
- Controls (DMSO) vs compounds
- Density plot

### 2. Individual Treatment Clusters

```python
from src.autoencoder.contrastive_vae import plot_treatment_clusters

plot_treatment_clusters(
    model,
    data_loader,
    treatments=metadata['treatment'],
    n_treatments=12,  # Show top 12 treatments
    save_path='treatment_clusters.png'
)
```

Shows:
- Each treatment's replicates
- How tightly replicates cluster
- Quantifies replicate spread (σ)

### 3. Replicate Agreement Metrics

```python
from src.autoencoder.contrastive_vae import compute_replicate_agreement

agreement_df = compute_replicate_agreement(
    model, data_loader, metadata['treatment']
)

# Treatments with most consistent replicates
print(agreement_df.head(20))
```

### 4. Specific Compounds Visualization

```python
from src.autoencoder.contrastive_vae.utils import visualize_specific_compounds

compounds_of_interest = ['HY_50946', 'HY_18686', 'HY_17592A', 'DMSO']

visualize_specific_compounds(
    model, processed_df, metadata,
    compound_list=compounds_of_interest,
    save_path='my_compounds.png'
)
```

Shows:
- Your compounds of interest with replicates
- Convex hull around replicates
- How well they cluster

## 🔍 Architecture Details

```
Input (genes)
    ↓
Encoder: 512 → 256 → 128
    ↓
Latent Space (μ, σ)  ← VAE reparameterization
    ↓                  ↓
Decoder              Projection Head
    ↓                  ↓
Output (genes)    Contrastive Space
    ↓                  ↓
Reconstruction    InfoNCE Loss
   Loss
```

The **projection head** maps latent space to a specialized space for contrastive learning.

## 📊 Interpreting Results

### Good Training Shows:

- ✅ **Contrastive loss decreasing**: Replicates getting closer
- ✅ **Reconstruction staying good**: Correlation >0.85
- ✅ **KL loss stable**: ~5-20
- ✅ **Val loss tracking train**: Not overfitting

### Warning Signs:

- ❌ **Reconstruction correlation <0.7**: Gamma too high, reduce it
- ❌ **Contrastive loss not decreasing**: Gamma too low, increase it
- ❌ **Val loss >> train loss**: Overfitting, add more dropout

### Tuning Guide:

**If replicates not clustering well:**
- Increase `gamma`: 0.5 → 1.0 → 2.0
- Decrease `temperature`: 0.1 → 0.05
- Train longer

**If reconstruction is poor:**
- Decrease `gamma`: 0.5 → 0.2 → 0.1
- Increase `temperature`: 0.1 → 0.2
- Increase model capacity

## 🆚 Comparison: Standard VAE vs Contrastive VAE

Run both and compare:

```python
# Standard VAE
from src.autoencoder.vae import VAE, train_vae

vae = VAE(input_dim=3311, latent_dim=64)
train_vae(vae, train_loader, val_loader)

# Contrastive VAE
from src.autoencoder.contrastive_vae import ContrastiveVAE, train_contrastive_vae

cvae = ContrastiveVAE(input_dim=3311, latent_dim=64)
train_contrastive_vae(cvae, train_loader, val_loader, gamma=0.5)

# Compare replicate distances
from src.autoencoder.contrastive_vae import compute_replicate_agreement

vae_agreement = compute_replicate_agreement(vae, val_loader, metadata['treatment'])
cvae_agreement = compute_replicate_agreement(cvae, val_loader, metadata['treatment'])

print(f"Standard VAE replicate distance: {vae_agreement['mean_distance'].mean():.4f}")
print(f"Contrastive VAE replicate distance: {cvae_agreement['mean_distance'].mean():.4f}")
```

Expected: **Contrastive VAE distance is 40-60% lower!**

## 💡 Use Cases

This is especially powerful for:

1. **Drug screening**: Find compounds with similar mechanisms
2. **Biomarker discovery**: Identify consistent perturbation signatures
3. **Quality control**: Detect failed replicates (outliers)
4. **Clustering**: Group compounds by transcriptional response
5. **Few-shot learning**: Leverage replicates to learn from less data

## 🧪 Advanced: Custom Temperature Scheduling

```python
# Start with high temperature (easy), decrease over time (harder)
class TemperatureScheduler:
    def __init__(self, start_temp=0.5, end_temp=0.05, epochs=100):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.epochs = epochs
    
    def get_temperature(self, epoch):
        # Linear decay
        return self.start_temp - (self.start_temp - self.end_temp) * (epoch / self.epochs)

# Use in training loop
scheduler = TemperatureScheduler()
for epoch in range(epochs):
    temp = scheduler.get_temperature(epoch)
    train_contrastive_vae(..., temperature=temp)
```

## 📚 References

- InfoNCE: Oord et al. (2018). Representation Learning with Contrastive Predictive Coding
- Supervised Contrastive: Khosla et al. (2020). Supervised Contrastive Learning
- β-VAE: Higgins et al. (2017). beta-VAE
- Application to biology: Lopez et al. (2018). scVI

