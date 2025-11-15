"""
Train all VAE models sequentially.
"""

import subprocess
import sys
from pathlib import Path

print("="*70)
print("Training All VAE Models")
print("="*70)
print()

models = [
    ("Standard VAE", "train_vae_hek293t.py"),
    ("Contrastive VAE (InfoNCE)", "train_contrastive_vae_hek293t.py"),
    ("Triplet VAE (LogFC-weighted)", "train_triplet_vae_hek293t.py"),
]

print("Models to train:")
for i, (name, _) in enumerate(models, 1):
    print(f"  {i}. {name}")
print()

for name, script in models:
    model_name = script.replace('train_', '').replace('_hek293t.py', '')
    checkpoint = Path(f'models/{model_name}_best.pt')
    
    if checkpoint.exists():
        print(f"\nSkipping {name} (checkpoint exists: {checkpoint.name})")
        continue
    
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"\nERROR: Training {name} failed!")
        response = input("Continue with next model? (y/n): ")
        if response.lower() != 'y':
            break

print(f"\n{'='*70}")
print(f"Training Complete!")
print(f"{'='*70}")
print(f"\nTrained models saved in: models/")
print(f"\nRun evaluations: python evaluation/run_all_evals.py")

