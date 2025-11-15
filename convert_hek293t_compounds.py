"""
Quick script to convert HEK293T compound codes to SMILES.
Run this to get SMILES for all your HY_ compound codes.
"""

from src.utils import load_compound_metadata_and_convert
import pandas as pd

print("="*70)
print("HEK293T Compound Code to SMILES Converter")
print("="*70)
print()

# Convert compounds
results = load_compound_metadata_and_convert(
    metadata_file='Dataset/HEK293T_MetaData.xlsx',
    compound_column='treatment',
    output_file='Dataset/compound_smiles_mapping.csv',
    try_online=True,
    max_online_queries=1000  # Adjust this number as needed
)

print("\n" + "="*70)
print("Results Preview:")
print("="*70)
print()
print(results.head(20))

print("\n" + "="*70)
print("Summary Statistics:")
print("="*70)
print(f"Total unique compounds: {len(results)}")
print(f"Compounds with SMILES: {results['smiles'].notna().sum()}")
print(f"Success rate: {results['smiles'].notna().sum() / len(results) * 100:.1f}%")
print()
print("Results saved to: Dataset/compound_smiles_mapping.csv")
print("Cache saved to: compound_smiles_cache.json")
print()
print("You can now load and use the SMILES:")
print("  df = pd.read_csv('Dataset/compound_smiles_mapping.csv')")

