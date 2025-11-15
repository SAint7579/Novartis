"""
Smart scraper: MCE → CAS → PubChem → SMILES
Better success rate than direct SMILES scraping!
"""

import sys
sys.path.append('.')

from src.utils.smart_compound_scraper import SmartCompoundScraper
import pandas as pd

print("="*70)
print("Smart Compound Scraper (Two-Step Pipeline)")
print("="*70)
print()
print("Pipeline:")
print("  Step 1: HY code -> MCE website -> Get CAS number + name")
print("  Step 2: CAS/name -> PubChem API -> Get SMILES")
print()
print("This is more robust than direct SMILES scraping!")
print()

# ============================================================================
# Configuration
# ============================================================================

TEST_MODE = True  # Set to False for full run
TEST_LIMIT = 100   # Test with 100 compounds

METADATA_FILE = 'Dataset/HEK293T_MetaData.xlsx'
OUTPUT_FILE = 'Dataset/compound_smiles_smart.csv'
CACHE_FILE = 'smart_compound_cache.json'

# ============================================================================
# Load Data
# ============================================================================

print("Loading metadata...")
metadata = pd.read_excel(METADATA_FILE, header=1)
compound_codes = metadata['treatment'].dropna().unique().tolist()

# Remove controls
compound_codes = [c for c in compound_codes if str(c).upper() not in ['DMSO', 'BLANK', 'RNA']]

print(f"Found {len(compound_codes)} unique compound codes")

if TEST_MODE:
    compound_codes = compound_codes[:TEST_LIMIT]
    print(f"\nTEST MODE: Processing {TEST_LIMIT} compounds")
    print(f"   Set TEST_MODE = False to process all")

# ============================================================================
# Run Smart Scraper
# ============================================================================

print(f"\nInitializing smart scraper...")
scraper = SmartCompoundScraper(cache_file=CACHE_FILE, delay=1.0)

print(f"\nStarting two-step pipeline...")
print(f"Estimated time: {len(compound_codes) * 1.5 / 60:.1f} minutes")
print(f"Press Ctrl+C to stop (progress will be saved)\n")

results_df = scraper.scrape_batch(
    compound_codes,
    max_compounds=None,
    verbose=True,
    save_interval=10
)

# ============================================================================
# Save and Summarize
# ============================================================================

results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*70}")
print(f"FINAL RESULTS")
print(f"{'='*70}")
print(f"Total compounds: {len(results_df)}")
print(f"SMILES found: {results_df['smiles'].notna().sum()} ({results_df['smiles'].notna().sum()/len(results_df)*100:.1f}%)")
print(f"  - Via CAS number: {(results_df['source'] == 'pubchem_cas').sum()}")
print(f"  - Via compound name: {(results_df['source'] == 'pubchem_name').sum()}")
print(f"Not found: {(results_df['source'] == 'not_found').sum()}")

if results_df['smiles'].notna().sum() > 0:
    print(f"\n{'='*70}")
    print(f"Example successful results:")
    print(f"{'='*70}")
    success_df = results_df[results_df['smiles'].notna()].head(10)
    for _, row in success_df.iterrows():
        print(f"\n{row['hy_code']}:")
        print(f"  Name: {row['compound_name']}")
        print(f"  CAS: {row['cas_number']}")
        print(f"  SMILES: {row['smiles'][:70]}...")
        print(f"  Source: {row['source']}")
else:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: No SMILES found!")
    print(f"{'='*70}")
    print(f"\nPossible reasons:")
    print(f"  1. MCE requires login to view product pages")
    print(f"  2. Website structure has changed")
    print(f"  3. Compounds are not publicly available")
    print(f"\nRecommended action:")
    print(f"  Contact MCE: service@medchemexpress.com")
    print(f"  Request: Compound library structure file for your purchase")

print(f"\n{'='*70}")
print(f"Files saved:")
print(f"{'='*70}")
print(f"  Results: {OUTPUT_FILE}")
print(f"  Cache: {CACHE_FILE}")

if TEST_MODE:
    print(f"\n{'='*70}")
    print(f"TEST MODE COMPLETE")
    print(f"{'='*70}")
    if results_df['smiles'].notna().sum() > 5:
        print(f"✓ Pipeline working! Found SMILES for {results_df['smiles'].notna().sum()} compounds")
        print(f"\nTo run full scrape:")
        print(f"  1. Edit scrape_smart.py")
        print(f"  2. Set: TEST_MODE = False")
        print(f"  3. Run: python scrape_smart.py")
        print(f"\nFull run will take ~{len(metadata['treatment'].unique()) * 1.5 / 60:.0f} minutes")
    else:
        print(f"✗ Pipeline not working (found only {results_df['smiles'].notna().sum()} SMILES)")
        print(f"\nLikely issue: MCE pages require login or are blocked")
        print(f"Recommended: Contact MCE for structure file instead")

