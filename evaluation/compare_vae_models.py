"""
Compare Standard VAE vs Contrastive VAE performance.
Run this after training both models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("VAE Models Comparison")
print("="*70)
print()

# ============================================================================
# Load Results
# ============================================================================

try:
    # Standard VAE results
    vae_latent = pd.read_csv('results/hek293t_latent_features.csv')
    print("✓ Loaded Standard VAE results")
    has_vae = True
except:
    print("✗ Standard VAE results not found. Run: python train_vae_hek293t.py")
    has_vae = False

try:
    # Contrastive VAE results
    cvae_latent = pd.read_csv('results/contrastive_vae_latent_features.csv')
    cvae_agreement = pd.read_csv('results/contrastive_vae_replicate_agreement.csv')
    print("✓ Loaded Contrastive VAE results")
    has_cvae = True
except:
    print("✗ Contrastive VAE results not found. Run: python train_contrastive_vae_hek293t.py")
    has_cvae = False

if not (has_vae and has_cvae):
    print("\nPlease train both models first!")
    exit()

print()

# ============================================================================
# Compare Replicate Distances
# ============================================================================

print("="*70)
print("1. Replicate Consistency Comparison")
print("="*70)

# For Standard VAE, compute replicate distances
from sklearn.metrics.pairwise import euclidean_distances

vae_latent_cols = [c for c in vae_latent.columns if c.startswith('latent_')]
cvae_latent_cols = [c for c in cvae_latent.columns if c.startswith('latent_')]

vae_replicate_distances = []
cvae_replicate_distances = []

for treatment in vae_latent['treatment'].unique():
    if treatment in ['DMSO', 'Blank', 'RNA', 'dmso', 'blank']:
        continue
    
    # Standard VAE
    vae_treatment_mask = vae_latent['treatment'] == treatment
    if vae_treatment_mask.sum() >= 2:
        vae_reps = vae_latent[vae_treatment_mask][vae_latent_cols].values
        vae_dist_matrix = euclidean_distances(vae_reps)
        # Upper triangle (excluding diagonal)
        vae_dists = vae_dist_matrix[np.triu_indices_from(vae_dist_matrix, k=1)]
        vae_replicate_distances.append(np.mean(vae_dists))
    
    # Contrastive VAE
    cvae_treatment_mask = cvae_latent['treatment'] == treatment
    if cvae_treatment_mask.sum() >= 2:
        cvae_reps = cvae_latent[cvae_treatment_mask][cvae_latent_cols].values
        cvae_dist_matrix = euclidean_distances(cvae_reps)
        cvae_dists = cvae_dist_matrix[np.triu_indices_from(cvae_dist_matrix, k=1)]
        cvae_replicate_distances.append(np.mean(cvae_dists))

vae_avg_dist = np.mean(vae_replicate_distances)
cvae_avg_dist = np.mean(cvae_replicate_distances)
improvement = (vae_avg_dist - cvae_avg_dist) / vae_avg_dist * 100

print(f"\nAverage replicate distance in latent space:")
print(f"  Standard VAE:    {vae_avg_dist:.4f}")
print(f"  Contrastive VAE: {cvae_avg_dist:.4f}")
print(f"  Improvement:     {improvement:.1f}% reduction ✓")

# ============================================================================
# Visualize Comparison
# ============================================================================

print("\n" + "="*70)
print("2. Visualization")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Replicate distance distribution
axes[0].hist(vae_replicate_distances, bins=50, alpha=0.6, label='Standard VAE', color='blue')
axes[0].hist(cvae_replicate_distances, bins=50, alpha=0.6, label='Contrastive VAE', color='red')
axes[0].axvline(vae_avg_dist, color='blue', linestyle='--', linewidth=2, label=f'VAE mean: {vae_avg_dist:.2f}')
axes[0].axvline(cvae_avg_dist, color='red', linestyle='--', linewidth=2, label=f'CVAE mean: {cvae_avg_dist:.2f}')
axes[0].set_xlabel('Mean Pairwise Distance Between Replicates')
axes[0].set_ylabel('Number of Treatments')
axes[0].set_title('Replicate Consistency\n(Lower = Better)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Box plot comparison
data_for_plot = [vae_replicate_distances, cvae_replicate_distances]
bp = axes[1].boxplot(data_for_plot, labels=['Standard VAE', 'Contrastive VAE'],
                      patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1].set_ylabel('Replicate Distance')
axes[1].set_title(f'Replicate Consistency Comparison\n({improvement:.1f}% improvement)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/vae_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved comparison plot to: results/vae_comparison.png")
plt.show()

# ============================================================================
# Detailed Statistics
# ============================================================================

print("\n" + "="*70)
print("3. Detailed Statistics")
print("="*70)

print(f"\nStandard VAE:")
print(f"  Mean replicate distance: {vae_avg_dist:.4f}")
print(f"  Median replicate distance: {np.median(vae_replicate_distances):.4f}")
print(f"  Std replicate distance: {np.std(vae_replicate_distances):.4f}")

print(f"\nContrastive VAE:")
print(f"  Mean replicate distance: {cvae_avg_dist:.4f}")
print(f"  Median replicate distance: {np.median(cvae_replicate_distances):.4f}")
print(f"  Std replicate distance: {np.std(cvae_replicate_distances):.4f}")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

if improvement > 30:
    print(f"🎉 Contrastive VAE is {improvement:.1f}% better at grouping replicates!")
    print(f"   Recommended for downstream tasks.")
elif improvement > 10:
    print(f"✓ Contrastive VAE is {improvement:.1f}% better at grouping replicates.")
    print(f"  Modest improvement - consider increasing gamma.")
else:
    print(f"⚠️  Contrastive VAE is only {improvement:.1f}% better.")
    print(f"   Consider increasing gamma parameter or check if InfoNCE is working.")

print(f"\nNext steps:")
print(f"  1. Check visualizations in results/")
print(f"  2. Use latent features for clustering/classification")
print(f"  3. Try adjusting gamma in train_contrastive_vae_hek293t.py")

