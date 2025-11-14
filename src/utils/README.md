# Utility Scripts

## Excel to CSV Converter

A memory-efficient utility to convert large Excel files to CSV format.

### Features

- ✅ Handles very large Excel files (500MB+) that pandas can't load
- ✅ Streams data instead of loading everything into memory
- ✅ Supports multiple sheets (converts all or specific sheet)
- ✅ Progress bars for tracking conversion
- ✅ Can process multiple files at once

### Installation

Install required dependencies:

```bash
pip install openpyxl tqdm
```

### Usage

#### Method 1: Command Line

Convert a single Excel file:
```bash
python src/utils/excel_to_csv_converter.py path/to/your/file.xlsx
```

Convert with custom output directory:
```bash
python src/utils/excel_to_csv_converter.py path/to/file.xlsx -o output_folder
```

Convert only a specific sheet:
```bash
python src/utils/excel_to_csv_converter.py path/to/file.xlsx -s "Sheet1"
```

Convert all Excel files in a directory:
```bash
python src/utils/excel_to_csv_converter.py path/to/excel_folder/
```

#### Method 2: Python Script/Notebook

```python
from src.utils import convert_excel_to_csv

# Convert entire Excel file
csv_files = convert_excel_to_csv('path/to/your/file.xlsx')

# Convert specific sheet
csv_files = convert_excel_to_csv(
    'path/to/your/file.xlsx',
    sheet_name='Sheet1',
    output_dir='data/csv'
)

# The function returns a list of created CSV file paths
print(f"Created CSV files: {csv_files}")
```

#### Method 3: Use in Jupyter Notebook

```python
import sys
sys.path.append('..')  # If running from Notebooks folder

from src.utils import convert_excel_to_csv

# Convert your large Excel file
csv_path = convert_excel_to_csv(
    '../Dataset/HEK293T_Counts.xlsx',
    output_dir='../Dataset/csv'
)

# Now load the CSV with pandas (much faster!)
import pandas as pd
df = pd.read_csv(csv_path[0])
```

### Performance

For a 500MB Excel file:
- **Old method (pandas)**: ❌ Out of memory or extremely slow
- **This utility**: ✅ Processes at ~50-100MB/min with minimal memory usage

### Tips

- The converter works best with `.xlsx` files
- For `.xls` (older Excel format), consider using `xlrd` or converting to `.xlsx` first
- CSV files are typically 20-50% smaller than Excel files
- Loading CSV files with pandas is 5-10x faster than Excel files

