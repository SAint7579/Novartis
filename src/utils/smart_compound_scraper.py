"""
Smart two-step compound scraper:
1. HY code -> MCE -> CAS number / compound name
2. CAS/name -> PubChem -> SMILES

More robust than trying to scrape SMILES directly from MCE.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import re


class SmartCompoundScraper:
    """
    Two-step scraper: MCE -> CAS/name -> PubChem -> SMILES
    """
    
    def __init__(self, cache_file: str = "smart_compound_cache.json", delay: float = 1.0):
        """
        Initialize smart scraper.
        
        Parameters:
        -----------
        cache_file : str
            Cache file path
        delay : float
            Delay between requests (seconds)
        """
        self.cache_file = cache_file
        self.delay = delay
        self.cache = self._load_cache()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _load_cache(self) -> Dict:
        """Load cache."""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def get_cas_from_mce(self, hy_code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Step 1: Get CAS number and compound name from MCE website.
        
        Parameters:
        -----------
        hy_code : str
            HY code (e.g., HY_50946)
        
        Returns:
        --------
        cas_number : str or None
            CAS registry number
        compound_name : str or None
            Compound name
        """
        # Convert HY_50946 -> HY-50946
        mce_code = hy_code.replace('HY_', 'HY-').replace('_', '-')
        
        try:
            # Try direct product URL
            url = f"https://www.medchemexpress.com/{mce_code}.html"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return None, None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text()
            
            # Extract CAS number
            cas_patterns = [
                r'CAS\s*[:#]?\s*(\d+-\d+-\d+)',
                r'CAS\s+No\.\s*[:#]?\s*(\d+-\d+-\d+)',
                r'(\d+-\d+-\d+)',  # Fallback: any CAS-like pattern
            ]
            
            cas_number = None
            for pattern in cas_patterns:
                matches = re.findall(pattern, page_text)
                if matches:
                    # Validate CAS format
                    for match in matches:
                        if re.match(r'^\d{1,7}-\d{2}-\d$', match):
                            cas_number = match
                            break
                if cas_number:
                    break
            
            # Extract compound name from title or h1
            compound_name = None
            
            # Try title tag
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                # Remove common suffixes
                name = re.sub(r'\s*\|\s*MedChemExpress.*$', '', title_text)
                name = re.sub(r'\s*-\s*MCE.*$', '', name)
                name = re.sub(r'\s*\(Cat\.\s*No\.\s*HY-[^\)]+\)', '', name)
                if name and len(name) > 3:
                    compound_name = name.strip()
            
            # Try h1 tag
            if not compound_name:
                h1 = soup.find('h1')
                if h1:
                    name = h1.get_text().strip()
                    if name and len(name) > 3:
                        compound_name = name
            
            # Try product name div/span
            if not compound_name:
                for tag in soup.find_all(['div', 'span', 'h2'], class_=re.compile(r'product.*name|title', re.I)):
                    name = tag.get_text().strip()
                    if name and len(name) > 3 and mce_code not in name:
                        compound_name = name
                        break
            
            return cas_number, compound_name
        
        except Exception as e:
            print(f"Error fetching from MCE for {hy_code}: {e}")
            return None, None
    
    def get_smiles_from_pubchem_cas(self, cas_number: str) -> Optional[str]:
        """
        Step 2a: Get SMILES from PubChem using CAS number.
        
        Parameters:
        -----------
        cas_number : str
            CAS registry number
        
        Returns:
        --------
        smiles : str or None
            SMILES string
        """
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/CanonicalSMILES/JSON"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except:
            pass
        
        return None
    
    def get_smiles_from_pubchem_name(self, compound_name: str) -> Optional[str]:
        """
        Step 2b: Get SMILES from PubChem using compound name.
        
        Parameters:
        -----------
        compound_name : str
            Compound name
        
        Returns:
        --------
        smiles : str or None
            SMILES string
        """
        try:
            # Clean compound name
            clean_name = re.sub(r'\([^)]*\)', '', compound_name).strip()
            clean_name = clean_name.split(',')[0].strip()  # Take first part if comma-separated
            
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean_name}/property/CanonicalSMILES/JSON"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except:
            pass
        
        return None
    
    def get_compound_info(self, hy_code: str, verbose: bool = False) -> Optional[Dict]:
        """
        Complete pipeline: HY code -> MCE -> CAS/name -> PubChem -> SMILES.
        
        Parameters:
        -----------
        hy_code : str
            HY catalog code
        verbose : bool
            Print progress
        
        Returns:
        --------
        dict or None
            Compound information with SMILES
        """
        # Check cache
        if hy_code in self.cache:
            if verbose:
                smiles_status = "✓" if self.cache[hy_code] and self.cache[hy_code].get('smiles') else "✗"
                print(f"{smiles_status} {hy_code}: Cached")
            return self.cache[hy_code]
        
        # Skip controls
        if hy_code.upper() in ['DMSO', 'BLANK', 'RNA', 'WATER', '']:
            return None
        
        if verbose:
            print(f"⌛ {hy_code}: Processing...")
        
        # Step 1: Get CAS and name from MCE
        cas_number, compound_name = self.get_cas_from_mce(hy_code)
        
        time.sleep(self.delay)  # Respectful delay
        
        if not cas_number and not compound_name:
            if verbose:
                print(f"✗ {hy_code}: Not found on MCE")
            self.cache[hy_code] = None
            self._save_cache()
            return None
        
        if verbose:
            print(f"  -> Found on MCE: CAS={cas_number}, Name={compound_name}")
        
        # Step 2: Get SMILES from PubChem
        smiles = None
        source = None
        
        # Try CAS first (most reliable)
        if cas_number:
            smiles = self.get_smiles_from_pubchem_cas(cas_number)
            if smiles:
                source = 'pubchem_cas'
                if verbose:
                    print(f"  -> SMILES from PubChem (CAS): {smiles[:50]}...")
        
        # Try compound name if CAS failed
        if not smiles and compound_name:
            time.sleep(0.5)  # Small delay between PubChem requests
            smiles = self.get_smiles_from_pubchem_name(compound_name)
            if smiles:
                source = 'pubchem_name'
                if verbose:
                    print(f"  -> SMILES from PubChem (name): {smiles[:50]}...")
        
        # Store result
        result = {
            'hy_code': hy_code,
            'compound_name': compound_name,
            'cas_number': cas_number,
            'smiles': smiles,
            'source': source or 'not_found'
        }
        
        self.cache[hy_code] = result
        self._save_cache()
        
        if verbose:
            if smiles:
                print(f"✓ {hy_code}: Success!")
            else:
                print(f"✗ {hy_code}: SMILES not found in PubChem")
        
        return result
    
    def scrape_batch(
        self,
        hy_codes: List[str],
        max_compounds: Optional[int] = None,
        verbose: bool = True,
        save_interval: int = 10
    ) -> pd.DataFrame:
        """
        Scrape multiple compounds using smart two-step pipeline.
        
        Parameters:
        -----------
        hy_codes : list
            List of HY codes
        max_compounds : int, optional
            Maximum to process
        verbose : bool
            Show progress
        save_interval : int
            Save every N compounds
        
        Returns:
        --------
        pd.DataFrame
            Results with HY code, name, CAS, SMILES
        """
        unique_codes = list(set(hy_codes))
        unique_codes = [c for c in unique_codes if str(c).upper() not in ['DMSO', 'BLANK', 'RNA', '']]
        
        if max_compounds:
            unique_codes = unique_codes[:max_compounds]
        
        if verbose:
            print(f"="*70)
            print(f"Smart Compound Scraper (Two-Step Pipeline)")
            print(f"="*70)
            print(f"Step 1: MCE -> CAS number + compound name")
            print(f"Step 2: CAS/name -> PubChem -> SMILES")
            print(f"")
            print(f"Total compounds: {len(unique_codes)}")
            print(f"Already cached: {sum(1 for c in unique_codes if c in self.cache)}")
            print(f"Estimated time: {len(unique_codes) * 1.5 / 60:.1f} minutes")
            print(f"="*70)
            print()
        
        results = []
        success_count = 0
        
        try:
            pbar = tqdm(unique_codes, desc="Processing") if verbose else unique_codes
            
            for i, code in enumerate(pbar, 1):
                info = self.get_compound_info(code, verbose=False)
                
                if info and info.get('smiles'):
                    results.append(info)
                    success_count += 1
                elif info:
                    results.append(info)
                else:
                    results.append({
                        'hy_code': code,
                        'compound_name': None,
                        'cas_number': None,
                        'smiles': None,
                        'source': 'not_found'
                    })
                
                if verbose:
                    pbar.set_postfix({
                        'found': success_count,
                        'rate': f'{success_count/i*100:.1f}%'
                    })
                
                # Save periodically
                if i % save_interval == 0:
                    self._save_cache()
                    if verbose:
                        print(f"\n  -> Progress saved ({i}/{len(unique_codes)})")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
        
        finally:
            self._save_cache()
        
        df = pd.DataFrame(results)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Results Summary:")
            print(f"{'='*70}")
            print(f"Total processed: {len(df)}")
            print(f"SMILES found: {success_count} ({success_count/len(df)*100:.1f}%)")
            print(f"  - Via CAS: {(df['source'] == 'pubchem_cas').sum()}")
            print(f"  - Via name: {(df['source'] == 'pubchem_name').sum()}")
            print(f"Not found: {(df['source'] == 'not_found').sum()}")
            
            if success_count > 0:
                print(f"\nExample successful results:")
                print(df[df['smiles'].notna()].head(5)[['hy_code', 'compound_name', 'cas_number']].to_string())
        
        return df


def scrape_compounds_smart(
    metadata_file: str,
    compound_column: str = 'treatment',
    output_file: str = 'compound_smiles_smart.csv',
    max_compounds: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to scrape compounds using smart pipeline.
    
    Parameters:
    -----------
    metadata_file : str
        Metadata file with HY codes
    compound_column : str
        Column with compound codes
    output_file : str
        Output CSV path
    max_compounds : int, optional
        Limit for testing
    
    Returns:
    --------
    pd.DataFrame
        Results
    """
    # Load metadata
    if metadata_file.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_file, header=1)
    else:
        metadata = pd.read_csv(metadata_file)
    
    # Get unique compounds
    compound_codes = metadata[compound_column].dropna().unique().tolist()
    
    # Scrape
    scraper = SmartCompoundScraper()
    results_df = scraper.scrape_batch(compound_codes, max_compounds=max_compounds)
    
    # Save
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart two-step compound scraper (MCE -> CAS -> PubChem -> SMILES)"
    )
    parser.add_argument(
        "metadata_file",
        help="Metadata file with HY codes"
    )
    parser.add_argument(
        "-c", "--column",
        default="treatment",
        help="Column with compound codes"
    )
    parser.add_argument(
        "-o", "--output",
        default="compound_smiles_smart.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum compounds to process (for testing)"
    )
    
    args = parser.parse_args()
    
    scrape_compounds_smart(
        metadata_file=args.metadata_file,
        compound_column=args.column,
        output_file=args.output,
        max_compounds=args.max
    )

