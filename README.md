# Gene Expression Perturbation Prediction

Deep learning models for predicting gene expression changes after compound perturbations.

## Project Structure

```
Novartis/
├── Dataset/
│   ├── csv/
│   │   └── HEK293T_Counts.csv          # Gene expression counts
│   ├── HEK293T_MetaData.xlsx           # Sample metadata
│   └── SMILES.txt                      # Compound SMILES strings
│
├── src/
│   ├── autoencoder/
│   │   ├── vae/                        # Standard VAE
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── utils.py
│   │   ├── contrastive_vae/            # VAE with InfoNCE loss
│   │   │   ├── model.py                # ContrastiveVAE architecture
│   │   │   ├── loss.py                 # Standard InfoNCE
│   │   │   ├── loss_logfc.py           # LogFC-weighted InfoNCE
│   │   │   ├── dataset.py
│   │   │   ├── train.py
│   │   │   └── utils.py
│   │   └── triplet_vae/                # VAE with triplet/quadruplet loss
│   │       ├── model.py                # TripletVAE architecture
│   │       ├── loss.py                 # Quadruplet hinge loss
│   │       ├── loss_infonce.py         # InfoNCE variant
│   │       ├── loss_fast.py            # Optimized version
│   │       ├── dataset.py
│   │       ├── dataset_fast.py
│   │       └── train.py
│   │
│   └── diffusion/
│       ├── smiles_encoder.py           # Pre-trained ChemBERTa
│       ├── diffusion_model.py          # Conditional diffusion (DDPM)
│       └── linear_baseline.py          # Simple MLP baseline
│
├── models/                              # Trained model checkpoints
│   ├── vae_hek293t_best.pt
│   ├── contrastive_vae_hek293t_best.pt
│   ├── contrastive_vae_logfc_hek293t_best.pt
│   ├── triplet_vae_hek293t_best.pt
│   ├── triplet_vae2_hek293t_best.pt
│   ├── diffusion_perturbation_best.pt
│   └── linear_perturbation_best.pt
│
├── evaluation/                          # Evaluation scripts
│   ├── run_all_evals.py                # Master script (runs all)
│   ├── visualize_all_models.py         # Latent space plots
│   ├── evaluate_topk_fast.py           # Top-k retrieval metrics
│   ├── evaluate_perturbation_prediction.py  # Perturbation accuracy
│   ├── evaluate_perturbation_models.py # Diffusion/linear comparison
│   ├── evaluate_latent_retrieval.py    # Full top-k evaluation
│   └── plot_volcano.py                 # Volcano plots
│
├── results/                             # Evaluation results
│   ├── perturbation_prediction_accuracy.csv
│   ├── perturbation_model_comparison.csv
│   ├── perturbation_per_treatment.xlsx
│   ├── topk_metrics.csv
│   └── volcano_*.png
│
├── latent_plots/                        # Latent space visualizations
│   ├── *_latent.png                    # PCA plots
│   └── *_latent_tsne.png               # t-SNE plots
│
└── Training scripts (project root)
    ├── train_vae_hek293t.py
    ├── train_contrastive_vae_hek293t.py
    ├── train_contrastive_vae_logfc_hek293t.py
    ├── train_triplet_vae_hek293t.py
    ├── train_diffusion_perturbation.py
    └── train_linear_perturbation.py
```

## Models

### 1. **Standard VAE**
- Basic variational autoencoder
- Loss: Reconstruction + KL divergence
- No replicate information

### 2. **Contrastive VAE** ⭐
- Adds InfoNCE contrastive learning
- Groups replicates together in latent space
- Projection head prevents task interference
- **Best performer** for replicate retrieval

### 3. **Contrastive VAE + LogFC**
- Same as Contrastive VAE
- InfoNCE weighted by logFC similarity
- Incorporates biological effect size

### 4. **Triplet VAE**
- Quadruplet loss (anchor, positive, DMSO neg, compound neg)
- LogFC-weighted with cosine distance
- Explicit triplet mining

### 5. **Diffusion Model** 🚀
- Conditional DDPM in VAE latent space
- Inputs: SMILES (ChemBERTa) + baseline latent + cell line + concentration
- Cross-attention conditioning
- Predicts perturbation from compound structure

### 6. **Linear Baseline**
- Simple MLP: (baseline, SMILES) → perturbation
- Baseline for diffusion comparison

## Quick Start

### Install Dependencies
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn openpyxl
pip install transformers tokenizers sentencepiece  # For diffusion
```

### Train Models
```bash
# VAE models
python train_contrastive_vae_hek293t.py
python train_contrastive_vae_logfc_hek293t.py

# Diffusion models (requires VAE trained first)
python train_diffusion_perturbation.py
python train_linear_perturbation.py
```

### Evaluate All Models
```bash
python evaluation/run_all_evals.py
```

Or individual evaluations:
```bash
python evaluation/visualize_all_models.py        # Latent space plots
python evaluation/evaluate_topk_fast.py          # Top-k retrieval
python evaluation/evaluate_perturbation_prediction.py  # Perturbation accuracy
python evaluation/evaluate_perturbation_models.py     # Diffusion comparison
```

### Volcano Plots
```bash
# VAE prediction
python evaluation/plot_volcano.py \
    --model models/contrastive_vae_hek293t_best.pt \
    --treatment HY_50946

# Diffusion prediction
python evaluation/plot_volcano.py \
    --model models/contrastive_vae_hek293t_best.pt \
    --diffusion models/diffusion_perturbation_best.pt \
    --smiles Dataset/SMILES.txt \
    --treatment HY_50946
```

## Key Results

### Top-k Replicate Retrieval
| Model | Top-1 Acc | Top-3 Acc | Top-5 Acc |
|-------|-----------|-----------|-----------|
| Ours | 0.25 | 0.30 | 0.32 |
| Contrastive VAE | 0.12 | 0.13 | 0.13 |
| VAE | 0.08 | 0.10 | 0.11 |

### Perturbation Prediction (LogFC Correlation)
| Model | Correlation | Uses SMILES? | Method |
|-------|-------------|--------------|--------|
| Ours | 0.212 | ❌ | Trains MLP per treatment |
| Contrastive VAE | 0.155 | ❌ | Trains MLP per treatment |
| VAE | 0.112 | ❌ | Trains MLP per treatment |
| Diffusion | TBD | ✅ | Zero-shot from SMILES |
| Linear | TBD | ✅ | Zero-shot from SMILES |

## Architecture Highlights

### Contrastive VAE
```
Input (40778 genes) → Encoder → mu (64D) → Decoder → Reconstruction
                                  ↓
                            Projection (128D) → InfoNCE Loss
```

### Diffusion Model
```
Baseline Latent (64D) ─┐
                       ├─→ [Concat] → Condition (448D)
SMILES (ChemBERTa) ────┤
Cell Line (10D) ───────┤
Concentration (1D) ────┘
                       ↓
        [Cross-Attention Diffusion] (512D hidden, 8 heads)
                       ↓
              Post-Perturbation Latent (64D)
```

## Evaluation Metrics

1. **Top-k Accuracy**: Can the model retrieve replicates?
2. **Top-k Precision**: How many retrieved neighbors are true replicates?
3. **LogFC Correlation**: How well does predicted logFC match ground truth?
4. **Latent MSE**: Prediction error in latent space
5. **Expression MSE**: Prediction error in gene expression space

## Files Overview

### Training
- `train_*_hek293t.py` - Train individual models
- All save to `models/*.pt`

### Evaluation
- `evaluation/run_all_evals.py` - **Run this for complete evaluation**
- Results save to `results/*.csv` and `results/*.xlsx`

### Key Dependencies
- PyTorch
- Transformers (HuggingFace)
- pandas, numpy, scipy
- matplotlib, seaborn
- openpyxl (for Excel output)

