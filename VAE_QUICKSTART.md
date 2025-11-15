# VAE for HEK293T Gene Expression - Quick Start Guide

A complete Variational Autoencoder (VAE) implementation for dimensionality reduction and feature learning from your gene expression data.

## 📁 Project Structure

```
Novartis/
├── src/
│   └── autoencoder/
│       └── vae/
│           ├── model.py           # VAE architecture
│           ├── train.py           # Training utilities
│           ├── dataset.py         # Data handling
│           ├── utils.py           # Preprocessing & visualization
│           ├── README.md          # Full documentation
│           └── requirements.txt   # Dependencies
├── Dataset/
│   ├── csv/
│   │   └── HEK293T_Counts.csv    # Your gene expression data
│   └── HEK293T_MetaData.xlsx     # Sample metadata
├── models/                        # Saved model checkpoints
├── results/                       # Output visualizations & features
├── train_vae_hek293t.py          # Ready-to-run training script
└── Notebooks/
    └── train_vae_example.ipynb   # Interactive tutorial
```

## 🚀 Quick Start

### Option 1: Command Line (Easiest)

```bash
# Install dependencies
pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm

# Train VAE on your data
python train_vae_hek293t.py
```

This will:
- ✅ Load and preprocess your HEK293T data
- ✅ Train a VAE model with optimal hyperparameters
- ✅ Generate visualizations (training curves, latent space, reconstructions)
- ✅ Extract and save latent features to CSV
- ✅ Save the trained model

**Output Files:**
- `models/vae_hek293t_best.pt` - Trained model checkpoint
- `results/vae_training_history.png` - Training curves
- `results/vae_latent_space.png` - Latent space visualization
- `results/vae_reconstruction.png` - Reconstruction quality
- `results/hek293t_latent_features.csv` - Latent features + metadata

### Option 2: Jupyter Notebook (Interactive)

Open `Notebooks/train_vae_example.ipynb` and run cells step-by-step to:
- Understand each component
- Experiment with hyperparameters
- Customize visualizations
- Run downstream analyses

### Option 3: Python Script (Custom)

```python
import pandas as pd
from src.autoencoder.vae import VAE, train_vae, GeneExpressionDataset, preprocess_gene_expression
from torch.utils.data import DataLoader

# 1. Load data
counts_df = pd.read_csv('Dataset/csv/HEK293T_Counts.csv', header=1, index_col=0)
metadata = pd.read_excel('Dataset/HEK293T_MetaData.xlsx', header=1)

# 2. Preprocess
processed_df, _ = preprocess_gene_expression(
    counts_df, method='log_normalize', scale='standard'
)

# 3. Create datasets
train_dataset, val_dataset = GeneExpressionDataset.create_train_val_split(
    processed_df, labels=metadata['treatment'], val_split=0.2
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 4. Initialize and train
model = VAE(input_dim=processed_df.shape[1], latent_dim=64)

history = train_vae(
    model, train_loader, val_loader,
    epochs=100, device='cuda', save_path='models/my_vae.pt'
)

# 5. Extract features
model.eval()
latent = model.get_latent(your_data_tensor, use_mean=True)
```

## 📊 What You Get

### 1. Compressed Representation
- **From**: ~15,000-20,000 genes (high-dimensional, noisy)
- **To**: 64-dimensional latent space (compact, denoised)
- **Use for**: Faster downstream analysis, better generalization

### 2. Latent Features CSV
`results/hek293t_latent_features.csv` contains:
- Original metadata (treatment, dose, etc.)
- 64 latent features (`latent_0` to `latent_63`)
- Ready for: clustering, classification, regression, visualization

### 3. Visualizations
- **Training curves**: Monitor convergence and detect overfitting
- **Latent space**: See how samples cluster by treatment
- **Reconstructions**: Verify model quality (correlation >0.9)

## 🎯 Use Cases

### Clustering
```python
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(latent_features)
```

### t-SNE/UMAP Visualization
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2).fit_transform(latent_features)
plt.scatter(tsne[:, 0], tsne[:, 1], c=treatments)
```

### Drug Response Prediction
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(latent_features_train, drug_response_train)
```

### Similarity Search
```python
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(new_sample_latent, all_latent_features)
most_similar = np.argsort(similarities[0])[-10:]  # Top 10
```

## ⚙️ Hyperparameter Tuning

### Key Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `latent_dim` | 64 | Latent space size | 32 (more compression) to 128 (more info) |
| `hidden_dims` | [512,256,128] | Network capacity | Larger for complex patterns |
| `beta` | 1.0 | KL weight | >1.0 for more disentanglement |
| `dropout` | 0.2 | Regularization | 0.1-0.3 based on overfitting |
| `learning_rate` | 1e-3 | Training speed | Lower if training unstable |
| `batch_size` | 64 | Samples per batch | Adjust based on GPU memory |

### Example: More Compression

```python
model = VAE(
    input_dim=10000,
    latent_dim=32,      # Smaller latent space
    hidden_dims=[256, 128, 64]  # Smaller network
)
```

### Example: β-VAE (More Disentangled)

```python
history = train_vae(
    model, train_loader, val_loader,
    beta=4.0  # Higher beta = more disentangled latent factors
)
```

## 🔍 Monitoring Training

Good training shows:
- ✅ **Total loss** decreasing steadily
- ✅ **Reconstruction loss** < 50 after convergence
- ✅ **KL divergence** stabilizing around 5-20
- ✅ **Validation loss** tracking training loss (not diverging)
- ✅ **Reconstruction correlation** > 0.9

Warning signs:
- ❌ Loss = NaN → Reduce learning rate, check for inf/nan in data
- ❌ Reconstruction correlation < 0.8 → Increase latent_dim or train longer
- ❌ Validation loss >> training loss → Overfitting, increase dropout

## 📚 Documentation

- **Full API documentation**: `src/autoencoder/vae/README.md`
- **Model architecture**: `src/autoencoder/vae/model.py`
- **Training utilities**: `src/autoencoder/vae/train.py`
- **Preprocessing**: `src/autoencoder/vae/utils.py`

## 🆘 Troubleshooting

### Out of Memory
```python
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Or reduce model size
model = VAE(input_dim=10000, latent_dim=32, hidden_dims=[256, 128])
```

### Training Too Slow
```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Filter more genes
processed_df, _ = preprocess_gene_expression(
    counts_df, filter_low_variance=True, variance_threshold=0.1  # Higher threshold
)
```

### Poor Reconstruction
```python
# Increase model capacity
model = VAE(latent_dim=128, hidden_dims=[1024, 512, 256])

# Reduce KL weight (focus on reconstruction)
history = train_vae(model, train_loader, val_loader, beta=0.5)

# Train longer
history = train_vae(model, train_loader, val_loader, epochs=200, patience=30)
```

## 🎓 Next Steps

1. **Train the model**: Run `python train_vae_hek293t.py`
2. **Explore latent space**: Open the generated visualizations
3. **Use latent features**: Load `results/hek293t_latent_features.csv`
4. **Downstream analysis**: Clustering, classification, drug response prediction
5. **Experiment**: Try different hyperparameters in the notebook
6. **Compare with PCA**: See if VAE captures more biological signal

## 💡 Tips

- Start with default hyperparameters - they work well for gene expression
- Always visualize latent space to verify biological signal
- Check reconstruction quality - it should be high (>0.9 correlation)
- Use latent features as input to any machine learning model
- Compare VAE features vs raw data performance on downstream tasks

## 📖 Citation

If using VAE in your analysis, consider citing:
- Kingma & Welling (2014). Auto-Encoding Variational Bayes. ICLR.
- Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics. Nature Methods.

---

**Ready to start? Run:** `python train_vae_hek293t.py`

