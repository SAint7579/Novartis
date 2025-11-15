"""
Utility to convert compound codes to SMILES structures.
Supports multiple compound databases and local caching.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import json


class CompoundSMILESConverter:
    """Convert various compound codes to SMILES structures."""
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the converter.
        
        Parameters:
        -----------
        cache_file : str, optional
            Path to cache file for storing fetched SMILES
        """
        self.cache_file = cache_file or "compound_smiles_cache.json"
        self.cache = self._load_cache()
        
        # Load local MoABox data if available
        self.moabox_data = self._load_moabox_data()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load cached SMILES from file."""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _load_moabox_data(self) -> Optional[pd.DataFrame]:
        """Load local MoABox compound data if available."""
        moabox_path = Path("DRUG-seq/data/Novartis_drugseq_U2OS_MoABox/MoABox_compounds_metadata.txt")
        if moabox_path.exists():
            try:
                df = pd.read_csv(moabox_path, sep='\t')
                return df[df['smiles'].notna()].set_index('cmpd_sample_id')['smiles'].to_dict()
            except:
                return None
        return None
    
    def get_smiles_from_pubchem(self, compound_name: str, delay: float = 0.2) -> Optional[str]:
        """
        Fetch SMILES from PubChem by compound name/identifier.
        
        Parameters:
        -----------
        compound_name : str
            Compound name or identifier
        delay : float
            Delay between requests (seconds) to respect API limits
        
        Returns:
        --------
        str or None
            SMILES string if found, None otherwise
        """
        # Check cache first
        if compound_name in self.cache:
            return self.cache[compound_name]
        
        try:
            # Try to search by name
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                self.cache[compound_name] = smiles
                self._save_cache()
                time.sleep(delay)
                return smiles
        except:
            pass
        
        # If name search fails, try with the code itself
        try:
            # Remove HY_ prefix and try as CID or other identifier
            clean_name = compound_name.replace('HY_', 'HY-').replace('_', '-')
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean_name}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                self.cache[compound_name] = smiles
                self._save_cache()
                time.sleep(delay)
                return smiles
        except:
            pass
        
        return None
    
    def get_smiles_from_chembl(self, compound_name: str, delay: float = 0.2) -> Optional[str]:
        """
        Fetch SMILES from ChEMBL database.
        
        Parameters:
        -----------
        compound_name : str
            Compound name or identifier
        delay : float
            Delay between requests (seconds)
        
        Returns:
        --------
        str or None
            SMILES string if found, None otherwise
        """
        if compound_name in self.cache:
            return self.cache[compound_name]
        
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={compound_name}&format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('molecules') and len(data['molecules']) > 0:
                    smiles = data['molecules'][0].get('molecule_structures', {}).get('canonical_smiles')
                    if smiles:
                        self.cache[compound_name] = smiles
                        self._save_cache()
                        time.sleep(delay)
                        return smiles
        except:
            pass
        
        return None
    
    def get_smiles_from_local(self, compound_code: str) -> Optional[str]:
        """
        Get SMILES from local MoABox database.
        
        Parameters:
        -----------
        compound_code : str
            Compound code (e.g., CB-43-EP73)
        
        Returns:
        --------
        str or None
            SMILES string if found, None otherwise
        """
        if self.moabox_data and compound_code in self.moabox_data:
            return self.moabox_data[compound_code]
        return None
    
    def convert_single(
        self, 
        compound_code: str, 
        try_online: bool = True,
        verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a single compound code to SMILES.
        
        Parameters:
        -----------
        compound_code : str
            Compound identifier/code
        try_online : bool
            Whether to try online databases if local lookup fails
        verbose : bool
            Print status messages
        
        Returns:
        --------
        str or None
            SMILES string if found, None otherwise
        """
        # Skip controls
        if compound_code.upper() in ['DMSO', 'BLANK', 'RNA', 'H2O', 'WATER']:
            return None
        
        # Check cache
        if compound_code in self.cache:
            if verbose:
                print(f"✓ {compound_code}: Found in cache")
            return self.cache[compound_code]
        
        # Try local database
        smiles = self.get_smiles_from_local(compound_code)
        if smiles:
            if verbose:
                print(f"✓ {compound_code}: Found in local database")
            return smiles
        
        # Try online databases
        if try_online:
            if verbose:
                print(f"⌛ {compound_code}: Searching online databases...")
            
            # Try PubChem first
            smiles = self.get_smiles_from_pubchem(compound_code)
            if smiles:
                if verbose:
                    print(f"✓ {compound_code}: Found in PubChem")
                return smiles
            
            # Try ChEMBL
            smiles = self.get_smiles_from_chembl(compound_code)
            if smiles:
                if verbose:
                    print(f"✓ {compound_code}: Found in ChEMBL")
                return smiles
            
            if verbose:
                print(f"✗ {compound_code}: Not found")
        
        return None
    
    def convert_batch(
        self, 
        compound_codes: List[str],
        try_online: bool = True,
        max_online_queries: int = 100,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Convert multiple compound codes to SMILES.
        
        Parameters:
        -----------
        compound_codes : list
            List of compound identifiers/codes
        try_online : bool
            Whether to try online databases if local lookup fails
        max_online_queries : int
            Maximum number of online queries to make (to avoid rate limits)
        verbose : bool
            Print progress information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: compound_code, smiles, source
        """
        results = []
        online_query_count = 0
        
        unique_codes = list(set(compound_codes))
        
        if verbose:
            print(f"Converting {len(unique_codes)} unique compound codes to SMILES...")
            print(f"Cache size: {len(self.cache)} compounds")
        
        for i, code in enumerate(unique_codes, 1):
            if verbose and i % 100 == 0:
                print(f"Progress: {i}/{len(unique_codes)}")
            
            # Check if we've hit the online query limit
            should_try_online = try_online and online_query_count < max_online_queries
            
            smiles = self.convert_single(code, try_online=should_try_online, verbose=False)
            
            if smiles:
                # Determine source
                if code in self.cache and code not in (self.moabox_data or {}):
                    source = "cache"
                elif self.moabox_data and code in self.moabox_data:
                    source = "local_moabox"
                else:
                    source = "online"
                    online_query_count += 1
            else:
                source = "not_found"
            
            results.append({
                'compound_code': code,
                'smiles': smiles,
                'source': source
            })
        
        df = pd.DataFrame(results)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Conversion Summary:")
            print(f"  Total compounds: {len(unique_codes)}")
            print(f"  Found SMILES: {df['smiles'].notna().sum()} ({df['smiles'].notna().sum()/len(unique_codes)*100:.1f}%)")
            print(f"  Not found: {df['smiles'].isna().sum()}")
            print(f"\nSources:")
            print(df['source'].value_counts().to_string())
            print(f"\nCache updated. Current cache size: {len(self.cache)}")
        
        return df
    
    def save_to_file(self, df: pd.DataFrame, output_file: str):
        """Save results to file."""
        df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")


# Convenience functions
def load_compound_metadata_and_convert(
    metadata_file: str,
    compound_column: str = 'treatment',
    output_file: Optional[str] = None,
    try_online: bool = True,
    max_online_queries: int = 100
) -> pd.DataFrame:
    """
    Load compound metadata and convert codes to SMILES.
    
    Parameters:
    -----------
    metadata_file : str
        Path to metadata file (Excel or CSV)
    compound_column : str
        Column name containing compound codes
    output_file : str, optional
        Path to save results
    try_online : bool
        Whether to query online databases
    max_online_queries : int
        Maximum online queries
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with compound codes and SMILES
    """
    # Load metadata
    if metadata_file.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_file, header=1)
    else:
        metadata = pd.read_csv(metadata_file, sep='\t')
    
    # Get unique compound codes
    compound_codes = metadata[compound_column].dropna().unique().tolist()
    
    # Convert to SMILES
    converter = CompoundSMILESConverter()
    results_df = converter.convert_batch(
        compound_codes,
        try_online=try_online,
        max_online_queries=max_online_queries
    )
    
    # Save if requested
    if output_file:
        converter.save_to_file(results_df, output_file)
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert compound codes to SMILES structures"
    )
    parser.add_argument(
        "metadata_file",
        help="Path to metadata file containing compound codes"
    )
    parser.add_argument(
        "-c", "--column",
        help="Column name with compound codes (default: 'treatment')",
        default="treatment"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path",
        default="compound_smiles_mapping.csv"
    )
    parser.add_argument(
        "--no-online",
        help="Don't query online databases",
        action="store_true"
    )
    parser.add_argument(
        "--max-queries",
        help="Maximum online queries (default: 100)",
        type=int,
        default=100
    )
    
    args = parser.parse_args()
    
    load_compound_metadata_and_convert(
        metadata_file=args.metadata_file,
        compound_column=args.column,
        output_file=args.output,
        try_online=not args.no_online,
        max_online_queries=args.max_queries
    )

