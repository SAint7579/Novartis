"""
Volcano plot comparison: Ground truth vs VAE-predicted logFC.

Shows -log10(p-value) vs logFC for both ground truth and model predictions,
with top upregulated and downregulated genes annotated.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder.vae import VAE, preprocess_gene_expression
from src.autoencoder.contrastive_vae import ContrastiveVAE, ContrastiveGeneExpressionDataset
from src.diffusion.smiles_encoder import SMILESEncoder, load_smiles_dict
from src.diffusion.diffusion_model import PerturbationDiffusionModel


def compute_logfc_and_pvalues(data_df, metadata, treatment, baseline='DMSO'):
    """
    Compute logFC and p-values for a treatment vs baseline.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Expression data [samples x genes]
    metadata : pd.DataFrame
        Metadata with treatment labels
    treatment : str
        Treatment to compare
    baseline : str
        Baseline condition (default: DMSO)
    
    Returns:
    --------
    logfc : np.array
        Log fold changes
    pvalues : np.array
        P-values from t-test
    gene_names : list
        Gene names
    """
    # Get treatment and baseline samples
    treatment_mask = metadata['treatment'] == treatment
    baseline_mask = metadata['treatment'] == baseline
    
    treatment_data = data_df[treatment_mask.values]
    baseline_data = data_df[baseline_mask.values]
    
    # Compute logFC
    treatment_mean = treatment_data.mean(axis=0).values
    baseline_mean = baseline_data.mean(axis=0).values
    logfc = treatment_mean - baseline_mean
    
    # Compute p-values (t-test)
    pvalues = []
    for i in range(data_df.shape[1]):
        _, pval = ttest_ind(treatment_data.iloc[:, i], baseline_data.iloc[:, i])
        pvalues.append(pval)
    pvalues = np.array(pvalues)
    
    # Replace 0 p-values with smallest non-zero value
    pvalues = np.maximum(pvalues, 1e-300)
    
    gene_names = data_df.columns.tolist()
    
    return logfc, pvalues, gene_names


def predict_logfc_with_vae(model, data_df, metadata, treatment, baseline='DMSO', device='cpu'):
    """
    Predict logFC using VAE's latent space.
    
    Strategy:
    1. Encode baseline samples to latent space
    2. Encode treatment samples to latent space
    3. Compute mean latent for each
    4. Decode both means to expression space
    5. Compute logFC from decoded expressions
    
    Parameters:
    -----------
    model : VAE or ContrastiveVAE
        Trained model
    data_df : pd.DataFrame
        Expression data [samples x genes]
    metadata : pd.DataFrame
        Metadata with treatment labels
    treatment : str
        Treatment to predict
    baseline : str
        Baseline condition
    device : str
        Device to use
    
    Returns:
    --------
    pred_logfc : np.array
        Predicted log fold changes
    pred_pvalues : np.array
        P-values from predicted replicates
    """
    model.eval()
    model = model.to(device)
    
    # Get treatment and baseline samples
    treatment_mask = metadata['treatment'] == treatment
    baseline_mask = metadata['treatment'] == baseline
    
    treatment_data = torch.FloatTensor(data_df[treatment_mask.values].values).to(device)
    baseline_data = torch.FloatTensor(data_df[baseline_mask.values].values).to(device)
    
    with torch.no_grad():
        # Encode to latent space
        treatment_mu, _ = model.encode(treatment_data)
        baseline_mu, _ = model.encode(baseline_data)
        
        # Decode back to expression space
        treatment_recon = model.decode(treatment_mu).cpu().numpy()
        baseline_recon = model.decode(baseline_mu).cpu().numpy()
    
    # Compute predicted logFC
    treatment_mean = treatment_recon.mean(axis=0)
    baseline_mean = baseline_recon.mean(axis=0)
    pred_logfc = treatment_mean - baseline_mean
    
    # Compute p-values from reconstructed replicates
    pred_pvalues = []
    for i in range(treatment_recon.shape[1]):
        _, pval = ttest_ind(treatment_recon[:, i], baseline_recon[:, i])
        pred_pvalues.append(pval)
    pred_pvalues = np.array(pred_pvalues)
    pred_pvalues = np.maximum(pred_pvalues, 1e-300)
    
    return pred_logfc, pred_pvalues


def plot_dual_volcano(logfc_true, pval_true, logfc_pred, pval_pred, 
                      gene_names, treatment, save_path=None):
    """Legacy function - calls plot_dual_volcano_with_title."""
    plot_dual_volcano_with_title(logfc_true, pval_true, logfc_pred, pval_pred,
                                  gene_names, treatment, "VAE", save_path)


def plot_dual_volcano_with_title(logfc_true, pval_true, logfc_pred, pval_pred, 
                      gene_names, treatment, model_type="VAE", save_path=None):
    """
    Create side-by-side volcano plots with top genes annotated.
    
    Parameters:
    -----------
    logfc_true : np.array
        True log fold changes
    pval_true : np.array
        True p-values
    logfc_pred : np.array
        Predicted log fold changes
    pval_pred : np.array
        Predicted p-values
    gene_names : list
        Gene names
    treatment : str
        Treatment name
    save_path : str, optional
        Path to save figure
    """
    # Convert to -log10(p)
    neglog10p_true = -np.log10(pval_true)
    neglog10p_pred = -np.log10(pval_pred)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color by significance and effect size
    sig_threshold = -np.log10(0.05)
    fc_threshold = 0.5
    
    for ax, logfc, neglog10p, title in zip(
        axes,
        [logfc_true, logfc_pred],
        [neglog10p_true, neglog10p_pred],
        ['Ground Truth', f'{model_type} Predicted']
    ):
        # Determine colors
        colors = []
        for fc, p in zip(logfc, neglog10p):
            if p < sig_threshold:
                colors.append('gray')  # Not significant
            elif fc > fc_threshold:
                colors.append('red')  # Upregulated
            elif fc < -fc_threshold:
                colors.append('blue')  # Downregulated
            else:
                colors.append('gray')  # Not significant change
        
        # Scatter plot
        ax.scatter(logfc, neglog10p, c=colors, alpha=0.6, s=10)
        
        # Add threshold lines
        ax.axhline(y=sig_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Find top 5 upregulated and downregulated
        sorted_indices = np.argsort(logfc)
        top_down = sorted_indices[:5]  # Most downregulated
        top_up = sorted_indices[-5:]  # Most upregulated
        
        # Annotate top genes
        for idx in list(top_down) + list(top_up):
            ax.annotate(
                gene_names[idx],
                xy=(logfc[idx], neglog10p[idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5)
            )
        
        # Labels and title
        ax.set_xlabel('log2 Fold Change', fontsize=12)
        ax.set_ylabel('-log10(p-value)', fontsize=12)
        ax.set_title(f'{title}\n{treatment} vs DMSO', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Upregulated'),
            Patch(facecolor='blue', label='Downregulated'),
            Patch(facecolor='gray', label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved volcano plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def predict_logfc_with_diffusion(vae_model, diffusion_model, smiles_encoder,
                                   data_df, metadata, treatment, smiles_dict,
                                   baseline='DMSO', device='cpu', num_samples=10):
    """
    Predict logFC using diffusion model.
    
    Parameters:
    -----------
    vae_model : VAE or ContrastiveVAE
        Trained VAE
    diffusion_model : PerturbationDiffusionModel
        Trained diffusion model
    smiles_encoder : SMILESEncoder
        SMILES encoder
    data_df : pd.DataFrame
        Expression data
    metadata : pd.DataFrame
        Metadata
    treatment : str
        Treatment compound
    smiles_dict : dict
        Compound ID -> SMILES mapping
    baseline : str
        Baseline condition
    device : str
        Device to use
    num_samples : int
        Number of diffusion samples to generate
    
    Returns:
    --------
    pred_logfc : np.array
        Predicted log fold changes
    pred_pvalues : np.array
        P-values from generated samples
    """
    vae_model.eval()
    diffusion_model.eval()
    smiles_encoder.eval()
    
    # Get baseline samples and encode
    baseline_mask = metadata['treatment'] == baseline
    baseline_data = torch.FloatTensor(data_df[baseline_mask.values].values).to(device)
    
    with torch.no_grad():
        baseline_mu, _ = vae_model.encode(baseline_data)
        baseline_mean_latent = baseline_mu.mean(dim=0, keepdim=True)
    
    # Get SMILES
    if treatment not in smiles_dict:
        raise ValueError(f"No SMILES for treatment: {treatment}")
    
    smiles = smiles_dict[treatment]
    
    # Encode SMILES
    with torch.no_grad():
        smiles_emb = smiles_encoder.encode_smiles(smiles)
    
    # Generate multiple samples via diffusion
    print(f"  Generating {num_samples} diffusion samples...")
    generated_latents = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Expand for single sample
            baseline_batch = baseline_mean_latent
            smiles_batch = smiles_emb
            
            # Sample via diffusion
            pred_latent = diffusion_model.sample(smiles_batch, baseline_batch, num_steps=100)
            
            # Decode to expression
            pred_expr = vae_model.decode(pred_latent).cpu().numpy()
            generated_latents.append(pred_expr[0])
    
    generated_latents = np.array(generated_latents)  # [num_samples, n_genes]
    
    # Decode baseline
    with torch.no_grad():
        baseline_recon = vae_model.decode(baseline_mu).cpu().numpy()
    
    # Compute logFC
    treatment_mean = generated_latents.mean(axis=0)
    baseline_mean = baseline_recon.mean(axis=0)
    pred_logfc = treatment_mean - baseline_mean
    
    # Compute p-values from generated samples vs baseline
    pred_pvalues = []
    for i in range(generated_latents.shape[1]):
        _, pval = ttest_ind(generated_latents[:, i], baseline_recon[:, i])
        pred_pvalues.append(pval)
    pred_pvalues = np.array(pred_pvalues)
    pred_pvalues = np.maximum(pred_pvalues, 1e-300)
    
    return pred_logfc, pred_pvalues


def print_top_genes(logfc, pvalues, gene_names, title):
    """Print top upregulated and downregulated genes."""
    # Sort by logFC
    sorted_indices = np.argsort(logfc)
    
    print(f"\n{title}")
    print("="*70)
    
    print("\nTop 5 Downregulated Genes:")
    print(f"{'Gene':<15} {'logFC':>10} {'-log10(p)':>12}")
    print("-"*40)
    for idx in sorted_indices[:5]:
        print(f"{gene_names[idx]:<15} {logfc[idx]:>10.3f} {-np.log10(pvalues[idx]):>12.2f}")
    
    print("\nTop 5 Upregulated Genes:")
    print(f"{'Gene':<15} {'logFC':>10} {'-log10(p)':>12}")
    print("-"*40)
    for idx in sorted_indices[-5:]:
        print(f"{gene_names[idx]:<15} {logfc[idx]:>10.3f} {-np.log10(pvalues[idx]):>12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Create volcano plots comparing ground truth vs model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to VAE model weights')
    parser.add_argument('--diffusion', type=str, default=None, help='Path to diffusion model (optional)')
    parser.add_argument('--smiles', type=str, default=None, help='Path to SMILES file (required for diffusion)')
    parser.add_argument('--treatment', type=str, required=True, help='Treatment to analyze (e.g., HY_50946)')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    use_diffusion = args.diffusion is not None
    if use_diffusion and args.smiles is None:
        print("Error: --smiles required when using --diffusion")
        return
    
    # Load data
    print("\nLoading data...")
    counts_df = pd.read_csv(PROJECT_ROOT / 'Dataset' / 'csv' / 'HEK293T_Counts.csv', header=1, index_col=0)
    metadata = pd.read_excel(PROJECT_ROOT / 'Dataset' / 'HEK293T_MetaData.xlsx', header=1)
    
    print("Preprocessing...")
    processed_df, _ = preprocess_gene_expression(
        counts_df,
        method='log_normalize',
        scale='standard',
        filter_low_variance=True,
        variance_threshold=0.01
    )
    
    # Align metadata
    metadata = metadata.set_index('unique_ID').loc[processed_df.index].reset_index()
    
    print(f"Data shape: {processed_df.shape}")
    
    # Check if treatment exists
    if args.treatment not in metadata['treatment'].values:
        print(f"Error: Treatment '{args.treatment}' not found in dataset")
        print(f"Available treatments: {sorted(metadata['treatment'].unique())[:20]}...")
        return
    
    # Load VAE model
    print(f"\nLoading VAE from: {args.model}")
    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Determine model type
    if 'contrastive' in model_path.name.lower():
        vae_model = ContrastiveVAE(
            input_dim=processed_df.shape[1],
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            dropout=0.2,
            projection_dim=128
        )
    else:
        vae_model = VAE(
            input_dim=processed_df.shape[1],
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            dropout=0.2
        )
    
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.eval()
    vae_model = vae_model.to(device)
    
    print(f"VAE loaded successfully")
    
    # Compute ground truth logFC
    print(f"\nComputing ground truth logFC for {args.treatment}...")
    logfc_true, pval_true, gene_names = compute_logfc_and_pvalues(
        processed_df, metadata, args.treatment
    )
    
    # Predict logFC
    if use_diffusion:
        # Load diffusion model
        print(f"Loading diffusion model from: {args.diffusion}")
        smiles_dict = load_smiles_dict(args.smiles)
        
        smiles_encoder = SMILESEncoder(
            model_name='DeepChem/ChemBERTa-77M-MLM',
            embedding_dim=256,
            freeze_encoder=True
        ).to(device)
        
        diffusion_model = PerturbationDiffusionModel(
            latent_dim=64,
            smiles_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_timesteps=1000
        ).to(device)
        
        diff_checkpoint = torch.load(args.diffusion, map_location='cpu', weights_only=False)
        smiles_encoder.load_state_dict(diff_checkpoint['smiles_encoder'])
        diffusion_model.load_state_dict(diff_checkpoint['diffusion_model'])
        
        print(f"Predicting logFC with diffusion model...")
        logfc_pred, pval_pred = predict_logfc_with_diffusion(
            vae_model, diffusion_model, smiles_encoder,
            processed_df, metadata, args.treatment, smiles_dict,
            device=device, num_samples=10
        )
        model_type_str = "Diffusion"
    else:
        print(f"Predicting logFC with VAE...")
        logfc_pred, pval_pred = predict_logfc_with_vae(
            vae_model, processed_df, metadata, args.treatment, device=device
        )
        model_type_str = "VAE"
    
    # Print top genes
    print_top_genes(logfc_true, pval_true, gene_names, "Ground Truth")
    print_top_genes(logfc_pred, pval_pred, gene_names, f"{model_type_str} Predicted")
    
    # Create volcano plot (need to update title)
    print(f"\nCreating volcano plot...")
    if args.output is None:
        suffix = 'diffusion' if use_diffusion else model_path.stem
        output_path = PROJECT_ROOT / 'results' / f'volcano_{args.treatment}_{suffix}.png'
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update plot function call to use correct title
    plot_dual_volcano_with_title(
        logfc_true, pval_true,
        logfc_pred, pval_pred,
        gene_names,
        args.treatment,
        model_type_str,
        save_path=output_path
    )
    
    # Compute correlation
    corr = np.corrcoef(logfc_true, logfc_pred)[0, 1]
    print(f"\nCorrelation between true and predicted logFC: {corr:.4f}")


if __name__ == '__main__':
    main()

