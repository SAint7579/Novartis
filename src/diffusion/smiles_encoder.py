"""
SMILES encoder using pre-trained HuggingFace models.

Uses ChemBERTa or other pre-trained molecular language models
for high-quality SMILES embeddings.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


def normalize_compound_id(compound_id: str) -> str:
    """
    Normalize compound ID format.
    Converts HY-1234 to HY_1234 for consistency with metadata.
    """
    return compound_id.replace('-', '_')


def load_smiles_dict(smiles_file: str) -> Dict[str, str]:
    """
    Load SMILES dictionary from file.
    
    Format: compound_id<TAB>SMILES_string
    
    Normalizes compound IDs (HY-1234 -> HY_1234) to match metadata format.
    
    Parameters:
    -----------
    smiles_file : str
        Path to SMILES file
    
    Returns:
    --------
    smiles_dict : dict
        Dictionary mapping compound_id -> SMILES string
    """
    smiles_dict = {}
    with open(smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                compound_id = normalize_compound_id(parts[0])  # Normalize HY-X to HY_X
                smiles = parts[1]
                smiles_dict[compound_id] = smiles
    
    print(f"Loaded {len(smiles_dict)} SMILES strings (normalized to HY_XXX format)")
    return smiles_dict


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    if not RDKIT_AVAILABLE:
        return True  # Assume valid if RDKit not available
    
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


class SMILESEncoder(nn.Module):
    """
    Pre-trained SMILES encoder from HuggingFace.
    
    Uses ChemBERTa-77M-MLM or other pre-trained molecular language models.
    
    Available models:
    - 'DeepChem/ChemBERTa-77M-MLM' (default, fast)
    - 'seyonec/ChemBERTa-zinc-base-v1' (trained on ZINC)
    - 'seyonec/PubChem10M_SMILES_BPE_60k' (trained on PubChem)
    """
    
    def __init__(self, 
                 model_name: str = 'DeepChem/ChemBERTa-77M-MLM',
                 embedding_dim: int = 256,
                 freeze_encoder: bool = True):
        """
        Initialize pre-trained SMILES encoder.
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model name
        embedding_dim : int
            Output embedding dimension
        freeze_encoder : bool
            Whether to freeze pre-trained weights
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        print(f"Loading pre-trained SMILES encoder: {model_name}")
        
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"  Encoder frozen (hidden_size: {self.hidden_size})")
        else:
            print(f"  Encoder trainable (hidden_size: {self.hidden_size})")
        
        # Projection head to target embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode SMILES strings to embeddings.
        
        Parameters:
        -----------
        smiles_list : list of str
            SMILES strings
        
        Returns:
        --------
        embeddings : torch.Tensor [batch_size, embedding_dim]
            Dense embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode with pre-trained model
        with torch.set_grad_enabled(not self.training):
            outputs = self.encoder(**inputs)
        
        # Use [CLS] token embedding (or mean pooling)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            hidden_states = outputs.pooler_output
        else:
            # Mean pooling over tokens
            hidden_states = outputs.last_hidden_state.mean(dim=1)
        
        # Project to target dimension
        embeddings = self.projection(hidden_states)
        
        return embeddings
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """
        Encode a single SMILES string.
        
        Parameters:
        -----------
        smiles : str
            SMILES string
        
        Returns:
        --------
        embedding : torch.Tensor [1, embedding_dim]
            Embedding vector
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward([smiles])
        return embedding
    
    def encode_batch(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode a batch of SMILES strings.
        
        Parameters:
        -----------
        smiles_list : list of str
            List of SMILES strings
        
        Returns:
        --------
        embeddings : torch.Tensor [batch_size, embedding_dim]
            Embedding vectors
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(smiles_list)
        return embeddings


def precompute_smiles_embeddings(smiles_dict: Dict[str, str], 
                                  encoder: SMILESEncoder,
                                  batch_size: int = 32,
                                  device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Precompute SMILES embeddings for all compounds.
    
    Parameters:
    -----------
    smiles_dict : dict
        Compound ID -> SMILES mapping
    encoder : SMILESEncoder
        Pre-trained encoder
    batch_size : int
        Batch size for encoding
    device : str
        Device to use
    
    Returns:
    --------
    embeddings : dict
        Compound ID -> embedding tensor
    """
    encoder.eval()
    encoder = encoder.to(device)
    
    compound_ids = list(smiles_dict.keys())
    smiles_list = [smiles_dict[cid] for cid in compound_ids]
    
    all_embeddings = {}
    
    print(f"Encoding {len(compound_ids)} compounds...")
    for i in range(0, len(compound_ids), batch_size):
        batch_ids = compound_ids[i:i+batch_size]
        batch_smiles = smiles_list[i:i+batch_size]
        
        # Encode batch
        batch_embeddings = encoder.encode_batch(batch_smiles)
        
        # Store individual embeddings
        for cid, emb in zip(batch_ids, batch_embeddings):
            all_embeddings[cid] = emb.cpu()
        
        if (i // batch_size + 1) % 100 == 0:
            print(f"  Encoded {i+len(batch_ids)}/{len(compound_ids)}")
    
    print(f"✓ Precomputed {len(all_embeddings)} SMILES embeddings")
    return all_embeddings

