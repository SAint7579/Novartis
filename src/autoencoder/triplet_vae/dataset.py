"""
Dataset for Triplet VAE that generates anchor-positive-negative triplets.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, List
from collections import defaultdict


class TripletGeneExpressionDataset(Dataset):
    """
    Dataset that generates triplets for training:
    - Anchor: Treatment sample
    - Positive: Replicate of same treatment
    - Negative: DMSO control sample
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        treatments: pd.Series,
        dmso_label: str = 'DMSO',
        include_compound_negatives: bool = False
    ):
        """
        Initialize triplet dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Gene expression data [samples x genes]
        treatments : pd.Series
            Treatment labels
        dmso_label : str
            Label for DMSO controls
        include_compound_negatives : bool
            If True, use both DMSO and other compounds as negatives
        """
        self.include_compound_negatives = include_compound_negatives
        self.data = torch.FloatTensor(data.values)
        self.treatments = treatments.values
        self.dmso_label = dmso_label.upper()
        
        # Find DMSO indices
        self.dmso_indices = np.where(
            np.char.upper(self.treatments.astype(str)) == self.dmso_label
        )[0]
        
        if len(self.dmso_indices) == 0:
            raise ValueError(f"No DMSO controls found! Check if label '{dmso_label}' exists in treatments")
        
        # Compute mean DMSO expression (for logFC calculation)
        dmso_data = self.data[self.dmso_indices]
        self.dmso_mean = dmso_data.mean(dim=0)
        
        # Group non-DMSO samples by treatment
        self.treatment_to_indices = defaultdict(list)
        for idx, treatment in enumerate(self.treatments):
            if treatment.upper() != self.dmso_label:
                self.treatment_to_indices[treatment].append(idx)
        
        # Create list of valid anchor indices (non-DMSO with at least 2 replicates)
        self.valid_anchors = []
        for treatment, indices in self.treatment_to_indices.items():
            if len(indices) >= 2:  # Need at least 2 for anchor-positive pair
                self.valid_anchors.extend(indices)
        
        self.valid_anchors = np.array(self.valid_anchors)
        
        print(f"Triplet Dataset Info:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  DMSO controls: {len(self.dmso_indices)}")
        print(f"  Treatments with >=2 replicates: {sum(1 for t, idx in self.treatment_to_indices.items() if len(idx) >= 2)}")
        print(f"  Valid anchor samples: {len(self.valid_anchors)}")
    
    def __len__(self) -> int:
        """Dataset size is the number of valid anchors."""
        return len(self.valid_anchors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a quadruplet: (anchor, positive, negative_dmso, negative_compound).
        
        Returns:
        --------
        anchor : torch.Tensor
            Treatment sample
        positive : torch.Tensor
            Replicate of same treatment
        negative_dmso : torch.Tensor
            DMSO control
        negative_compound : torch.Tensor
            Different compound sample
        """
        # Anchor: sample from valid anchors
        anchor_idx = self.valid_anchors[idx]
        anchor = self.data[anchor_idx]
        anchor_treatment = self.treatments[anchor_idx]
        
        # Positive: random replicate of same treatment (not the anchor itself)
        treatment_indices = self.treatment_to_indices[anchor_treatment]
        positive_candidates = [i for i in treatment_indices if i != anchor_idx]
        
        if len(positive_candidates) == 0:
            positive_idx = anchor_idx
        else:
            positive_idx = np.random.choice(positive_candidates)
        
        positive = self.data[positive_idx]
        
        # Negative 1: DMSO control (baseline)
        negative_dmso_idx = np.random.choice(self.dmso_indices)
        negative_dmso = self.data[negative_dmso_idx]
        
        # Negative 2: Different compound
        if self.include_compound_negatives:
            all_treatments = list(self.treatment_to_indices.keys())
            negative_treatments = [t for t in all_treatments if t != anchor_treatment]
            if negative_treatments:
                neg_treatment = np.random.choice(negative_treatments)
                neg_indices = self.treatment_to_indices[neg_treatment]
                negative_compound_idx = np.random.choice(neg_indices)
                negative_compound = self.data[negative_compound_idx]
            else:
                # Fallback: use another DMSO
                negative_compound = negative_dmso
        else:
            # If not using compound negatives, just duplicate DMSO
            negative_compound = negative_dmso
        
        return anchor, positive, negative_dmso, negative_compound
    
    @staticmethod
    def create_train_val_split(
        data: pd.DataFrame,
        treatments: pd.Series,
        val_split: float = 0.2,
        dmso_label: str = 'DMSO',
        include_compound_negatives: bool = False,
        random_state: int = 42
    ) -> Tuple['TripletGeneExpressionDataset', 'TripletGeneExpressionDataset']:
        """
        Create stratified train/val split.
        
        Ensures each treatment has replicates in both train and val.
        """
        np.random.seed(random_state)
        
        train_indices = []
        val_indices = []
        
        # Always keep all DMSO in both sets (they're negatives)
        dmso_mask = treatments.str.upper() == dmso_label.upper()
        dmso_indices = np.where(dmso_mask)[0]
        
        # For non-DMSO treatments, stratify
        for treatment in treatments[~dmso_mask].unique():
            treatment_mask = treatments == treatment
            treatment_indices = np.where(treatment_mask)[0]
            
            if len(treatment_indices) >= 2:
                np.random.shuffle(treatment_indices)
                n_val = max(1, int(len(treatment_indices) * val_split))
                val_indices.extend(treatment_indices[:n_val])
                train_indices.extend(treatment_indices[n_val:])
            else:
                # Single sample treatments go to training only
                train_indices.extend(treatment_indices)
        
        # Add DMSO to both sets
        train_indices.extend(dmso_indices)
        val_indices.extend(dmso_indices)
        
        # Create datasets
        train_data = data.iloc[train_indices].reset_index(drop=True)
        val_data = data.iloc[val_indices].reset_index(drop=True)
        
        train_treatments = treatments.iloc[train_indices].reset_index(drop=True)
        val_treatments = treatments.iloc[val_indices].reset_index(drop=True)
        
        train_dataset = TripletGeneExpressionDataset(train_data, train_treatments, dmso_label, include_compound_negatives)
        val_dataset = TripletGeneExpressionDataset(val_data, val_treatments, dmso_label, include_compound_negatives)
        
        return train_dataset, val_dataset

