# Compound Code to SMILES Converter

Convert compound codes (like MCE HY_ codes) to SMILES structures using multiple data sources.

## Features

- ✅ **Local database lookup** (MoABox compounds)
- ✅ **Online databases** (PubChem, ChEMBL)
- ✅ **Caching** to avoid repeated queries
- ✅ **Batch processing** with rate limiting
- ✅ **Progress tracking**

## Installation

```bash
pip install pandas requests openpyxl
```

## Usage

### Method 1: Quick Script (Recommended for your HEK293T data)

```python
from src.utils import load_compound_metadata_and_convert

# Convert all compounds in your metadata file
results = load_compound_metadata_and_convert(
    metadata_file='Dataset/HEK293T_MetaData.xlsx',
    compound_column='treatment',
    output_file='Dataset/compound_smiles_mapping.csv',
    try_online=True,
    max_online_queries=1000  # Increase if you want more compounds
)

# View results
print(results.head())
print(f"\nFound SMILES for {results['smiles'].notna().sum()} compounds")
```

### Method 2: Using the Converter Class Directly

```python
from src.utils import CompoundSMILESConverter
import pandas as pd

# Initialize converter
converter = CompoundSMILESConverter(cache_file='my_smiles_cache.json')

# Convert a single compound
smiles = converter.convert_single('HY_50946', try_online=True, verbose=True)
print(f"HY_50946 -> {smiles}")

# Convert multiple compounds
compound_codes = ['HY_50946', 'HY_18686', 'HY_17592A', 'HY_19411']
results_df = converter.convert_batch(compound_codes, try_online=True)
print(results_df)
```

### Method 3: Command Line

```bash
# Convert compounds from metadata file
python src/utils/compound_to_smiles.py Dataset/HEK293T_MetaData.xlsx -o results.csv

# With custom column name and query limit
python src/utils/compound_to_smiles.py Dataset/HEK293T_MetaData.xlsx \
    --column treatment \
    --output compound_mapping.csv \
    --max-queries 500
```

### Method 4: In Jupyter Notebook

```python
import sys
sys.path.append('..')

from src.utils import CompoundSMILESConverter
import pandas as pd

# Load your metadata
metadata = pd.read_excel('../Dataset/HEK293T_MetaData.xlsx', header=1)

# Get unique compound codes
compound_codes = metadata['treatment'].unique()

# Convert to SMILES
converter = CompoundSMILESConverter()
results = converter.convert_batch(
    compound_codes.tolist(),
    try_online=True,
    max_online_queries=500  # Limit online queries to respect API limits
)

# Merge back with your metadata
metadata_with_smiles = metadata.merge(
    results[['compound_code', 'smiles']], 
    left_on='treatment', 
    right_on='compound_code',
    how='left'
)

# Save
results.to_csv('../Dataset/compound_smiles_mapping.csv', index=False)
```

## How It Works

The converter tries multiple sources in order:

1. **Cache** - Previously fetched SMILES (fastest)
2. **Local MoABox database** - If your compound is in the Novartis dataset
3. **PubChem API** - Large public chemical database
4. **ChEMBL API** - Bioactive molecules database

## Important Notes

### MCE (MedChemExpress) Codes

Your `HY_` codes are MCE catalog numbers. The converter will:
- Try searching PubChem with both "HY_50946" and "HY-50946" formats
- Check ChEMBL database
- Cache successful results for future use

### Rate Limiting

- Default delay between API calls: 0.2 seconds
- Default max online queries: 100 (to avoid overwhelming APIs)
- Increase `max_online_queries` if you need more, but be respectful of API limits

### Cache File

Results are cached in `compound_smiles_cache.json` by default. This file:
- Stores all successfully found SMILES
- Persists across runs
- Can be shared with collaborators
- Can be manually edited if needed

## Example Output

```
Converting 11361 unique compound codes to SMILES...
Cache size: 0 compounds
Progress: 100/11361
Progress: 200/11361
...

============================================================
Conversion Summary:
  Total compounds: 11361
  Found SMILES: 8542 (75.2%)
  Not found: 2819

Sources:
local_moabox     45
online          8497
not_found       2819

Cache updated. Current cache size: 8542
```

## Troubleshooting

**Q: Why are some compounds not found?**
- MCE codes might not be in public databases yet
- Some compounds are proprietary
- Try searching manually on [PubChem](https://pubchem.ncbi.nlm.nih.gov/) with the code

**Q: Can I add custom SMILES mappings?**
Yes! Edit the cache JSON file directly:
```json
{
  "HY_50946": "CC(C)NCC(O)COc1ccc(CCOCC(C)C)cc1",
  "your_code": "your_smiles_here"
}
```

**Q: How do I get SMILES for ALL compounds?**
- Increase `max_online_queries` to a larger number
- Contact MCE directly for their compound database
- Use the cache file to build up results over multiple runs

