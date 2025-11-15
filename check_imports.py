"""
Diagnostic script to check if imports work.
Run this on the Linux machine to diagnose the issue.
"""

import sys
import os
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))

print("="*70)
print("Import Diagnostic")
print("="*70)
print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")
print(f"Current dir: {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")
print()

# Check if files exist
print("Checking files:")
files_to_check = [
    'src/__init__.py',
    'src/autoencoder/__init__.py',
    'src/autoencoder/vae/__init__.py',
    'src/autoencoder/vae/utils.py',
    'src/autoencoder/triplet_vae/__init__.py',
    'src/autoencoder/triplet_vae/model.py',
]

for f in files_to_check:
    exists = Path(f).exists()
    print(f"  {'✓' if exists else '✗'} {f}")

print()

# Try imports step by step
print("Testing imports:")

try:
    import src
    print("  ✓ import src")
except Exception as e:
    print(f"  ✗ import src: {e}")

try:
    import src.autoencoder
    print("  ✓ import src.autoencoder")
except Exception as e:
    print(f"  ✗ import src.autoencoder: {e}")

try:
    import src.autoencoder.vae
    print("  ✓ import src.autoencoder.vae")
except Exception as e:
    print(f"  ✗ import src.autoencoder.vae: {e}")

try:
    from src.autoencoder.vae import preprocess_gene_expression
    print(f"  ✓ from src.autoencoder.vae import preprocess_gene_expression")
    print(f"    Location: {preprocess_gene_expression.__module__}")
except Exception as e:
    print(f"  ✗ from src.autoencoder.vae import preprocess_gene_expression")
    print(f"    Error: {e}")

try:
    import src.autoencoder.triplet_vae
    print("  ✓ import src.autoencoder.triplet_vae")
except Exception as e:
    print(f"  ✗ import src.autoencoder.triplet_vae: {e}")

try:
    from src.autoencoder.triplet_vae import TripletVAE
    print(f"  ✓ from src.autoencoder.triplet_vae import TripletVAE")
    print(f"    Location: {TripletVAE.__module__}")
except Exception as e:
    print(f"  ✗ from src.autoencoder.triplet_vae import TripletVAE")
    print(f"    Error: {e}")

print("\n" + "="*70)
print("If all imports show ✓, the problem is elsewhere.")
print("If any show ✗, check the error message above.")
print("="*70)

