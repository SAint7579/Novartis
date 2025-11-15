"""
Run all evaluations (skips already computed results).
"""

import subprocess
import sys

print("="*70)
print("Running All Evaluations")
print("="*70)
print()

scripts = [
    ('Latent Space Visualization', 'visualize_all_models.py'),
    ('Perturbation Prediction', 'evaluate_perturbation_prediction.py'),
]

for name, script in scripts:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR running {script}")

print(f"\n{'='*70}")
print(f"All Evaluations Complete!")
print(f"{'='*70}")
print(f"\nCheck:")
print(f"  - latent_plots/ for visualizations")
print(f"  - results/perturbation_prediction_accuracy.csv for metrics")

