"""
Setup script to ensure all modules can be imported.
Run this once on a new machine.
"""

from pathlib import Path

# Create __init__.py files in all directories that need them
directories = [
    'src',
    'src/autoencoder',
    'src/autoencoder/vae',
    'src/autoencoder/contrastive_vae',
    'src/autoencoder/triplet_vae',
    'src/utils'
]

for dir_path in directories:
    init_file = Path(dir_path) / '__init__.py'
    if not init_file.exists():
        init_file.touch()
        print(f"Created: {init_file}")
    else:
        print(f"Exists: {init_file}")

print("\nSetup complete! Python packages are now properly configured.")
print("You can now run: python train_triplet_vae_hek293t.py")

