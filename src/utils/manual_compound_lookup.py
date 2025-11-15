"""
Manual compound lookup utilities.
Use this if automated scraping fails.
"""

import pandas as pd
import requests
import time
from typing import Optional, Dict, List


class ManualCompoundLookup:
    """
    Helper for manual compound lookups.
    Provides easy methods to add compounds one-by-one or from a file.
    """
    
    def __init__(self, mapping_file: str = "manual_compound_mapping.csv"):
        """
        Initialize manual lookup.
        
        Parameters:
        -----------
        mapping_file : str
            CSV file to store manual mappings
        """
        self.mapping_file = mapping_file
        self.load_mappings()
    
    def load_mappings(self):
        """Load existing mappings."""
        try:
            self.df = pd.read_csv(self.mapping_file)
        except:
            self.df = pd.DataFrame(columns=['hy_code', 'compound_name', 'cas_number', 'smiles', 'source'])
    
    def save_mappings(self):
        """Save mappings."""
        self.df.to_csv(self.mapping_file, index=False)
        print(f"Saved {len(self.df)} compounds to {self.mapping_file}")
    
    def add_compound(
        self,
        hy_code: str,
        smiles: str,
        compound_name: Optional[str] = None,
        cas_number: Optional[str] = None
    ):
        """
        Manually add a compound.
        
        Parameters:
        -----------
        hy_code : str
            HY code
        smiles : str
            SMILES string
        compound_name : str, optional
            Compound name
        cas_number : str, optional
            CAS number
        """
        # Check if already exists
        if hy_code in self.df['hy_code'].values:
            print(f"Updating {hy_code}")
            self.df.loc[self.df['hy_code'] == hy_code, 'smiles'] = smiles
            if compound_name:
                self.df.loc[self.df['hy_code'] == hy_code, 'compound_name'] = compound_name
            if cas_number:
                self.df.loc[self.df['hy_code'] == hy_code, 'cas_number'] = cas_number
        else:
            print(f"Adding {hy_code}")
            new_row = {
                'hy_code': hy_code,
                'compound_name': compound_name,
                'cas_number': cas_number,
                'smiles': smiles,
                'source': 'manual'
            }
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        self.save_mappings()
    
    def lookup_cas_in_pubchem(self, cas_number: str) -> Optional[str]:
        """
        Look up SMILES by CAS number in PubChem.
        
        Parameters:
        -----------
        cas_number : str
            CAS number
        
        Returns:
        --------
        smiles : str or None
            SMILES string
        """
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                return smiles
        except Exception as e:
            print(f"Error looking up CAS {cas_number}: {e}")
        
        return None
    
    def add_from_cas_list(self, cas_mapping: Dict[str, str]):
        """
        Add multiple compounds from a CAS mapping.
        
        Parameters:
        -----------
        cas_mapping : dict
            Dictionary: {hy_code: cas_number}
        
        Example:
        --------
        cas_mapping = {
            'HY_B0152': '73-24-5',  # Adenine
            'HY_50946': '123456-78-9',
        }
        lookup.add_from_cas_list(cas_mapping)
        """
        print(f"Processing {len(cas_mapping)} compounds...")
        
        for hy_code, cas_number in cas_mapping.items():
            print(f"\nLooking up {hy_code} (CAS: {cas_number})...")
            
            smiles = self.lookup_cas_in_pubchem(cas_number)
            
            if smiles:
                self.add_compound(hy_code, smiles, cas_number=cas_number)
                print(f"  -> Success! SMILES: {smiles[:50]}...")
            else:
                print(f"  -> Not found in PubChem")
            
            time.sleep(0.5)  # Respectful delay
    
    def generate_lookup_urls(self, hy_codes: List[str], output_file: str = "compound_lookup_urls.txt"):
        """
        Generate URLs for manual lookup.
        
        Parameters:
        -----------
        hy_codes : list
            List of HY codes
        output_file : str
            Output text file with URLs
        """
        urls = []
        
        for code in hy_codes:
            mce_code = code.replace('HY_', 'HY-').replace('_', '-')
            mce_url = f"https://www.medchemexpress.com/{mce_code}.html"
            urls.append(f"{code}: {mce_url}")
        
        with open(output_file, 'w') as f:
            f.write("# Manual Compound Lookup URLs\n")
            f.write("# Visit each URL, copy CAS number and/or compound name\n")
            f.write("# Then use PubChem to get SMILES\n\n")
            for url in urls:
                f.write(url + "\n")
        
        print(f"Generated {len(urls)} lookup URLs in: {output_file}")
        print(f"\nVisit each URL, then use:")
        print(f"  lookup.add_compound('HY_50946', 'YOUR_SMILES_HERE', cas_number='CAS')")
    
    def get_mapping_dataframe(self) -> pd.DataFrame:
        """Get current mappings as DataFrame."""
        return self.df.copy()


# Convenience functions

def create_example_mapping():
    """Create example mapping file for testing."""
    example_data = {
        'hy_code': ['HY_B0152', 'HY_13030', 'HY_112269'],
        'compound_name': ['Adenine', 'Unknown', 'Unknown'],
        'cas_number': ['73-24-5', None, None],
        'smiles': [
            'Nc1ncnc2[nH]cnc12',  # Adenine
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'COc1ccc(-c2cccc(F)c2NCCCn2cnc(C)c2)cc1OC'
        ],
        'source': ['manual', 'manual', 'manual']
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv('manual_compound_mapping.csv', index=False)
    print("Created example mapping: manual_compound_mapping.csv")
    print("\nYou can add more compounds using:")
    print("  lookup = ManualCompoundLookup()")
    print("  lookup.add_compound('HY_xxxxx', 'SMILES_STRING', cas_number='CAS')")


if __name__ == "__main__":
    print("Manual Compound Lookup Utility")
    print("="*70)
    print()
    print("Usage examples:")
    print()
    print("# 1. Create lookup manager")
    print("from src.utils.manual_compound_lookup import ManualCompoundLookup")
    print("lookup = ManualCompoundLookup()")
    print()
    print("# 2. Add compound manually")
    print("lookup.add_compound(")
    print("    hy_code='HY_B0152',")
    print("    smiles='Nc1ncnc2[nH]cnc12',")
    print("    compound_name='Adenine',")
    print("    cas_number='73-24-5'")
    print(")")
    print()
    print("# 3. Add from CAS list (if you have CAS numbers)")
    print("cas_mapping = {")
    print("    'HY_B0152': '73-24-5',")
    print("    'HY_50946': 'YOUR_CAS_HERE',")
    print("}")
    print("lookup.add_from_cas_list(cas_mapping)")
    print()
    print("# 4. Generate URLs for manual lookup")
    print("lookup.generate_lookup_urls(['HY_50946', 'HY_18686'])")
    print()
    
    # Create example
    print("\nCreating example mapping file...")
    create_example_mapping()

