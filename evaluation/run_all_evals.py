"""
Run all evaluations for trained VAE models.
Skips already computed results.
"""

import subprocess
import sys
from pathlib import Path

# Change to parent directory to run scripts
import os
os.chdir(Path(__file__).parent.parent)

print("="*70)
print("Running All VAE Evaluations")
print("="*70)
print()

evaluations = [
    ("1. Latent Space Visualization", "evaluation/visualize_all_models.py"),
    ("2. Perturbation Prediction", "evaluation/evaluate_perturbation_prediction.py"),
]

for name, script in evaluations:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"\nWARNING: {script} had errors")

print(f"\n{'='*70}")
print(f"All Evaluations Complete!")
print(f"{'='*70}")
print(f"\nResults:")
print(f"  Visualizations:  latent_plots/")
print(f"  Metrics:         results/perturbation_prediction_accuracy.csv")

