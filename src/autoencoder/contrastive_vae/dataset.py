"""
Dataset for Contrastive VAE that groups replicates.
"""

import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import defaultdict


class ContrastiveGeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for gene expression with contrastive learning.
    
    Groups samples by treatment to enable contrastive learning on replicates.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        treatments: pd.Series,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Gene expression data [samples x genes]
        treatments : pd.Series
            Treatment labels (same length as data)
        transform : callable, optional
            Transform to apply to data
        """
        self.data = torch.FloatTensor(data.values)
        self.treatments = treatments.values
        self.transform = transform
        
        # Create treatment encoder
        unique_treatments = pd.Series(self.treatments).unique()
        self.treatment_to_idx = {t: i for i, t in enumerate(unique_treatments)}
        self.idx_to_treatment = {i: t for t, i in self.treatment_to_idx.items()}
        
        # Encode treatments as integers
        self.treatment_labels = torch.LongTensor([
            self.treatment_to_idx[t] for t in self.treatments
        ])
        
        # Group samples by treatment (for analysis)
        self.treatment_groups = defaultdict(list)
        for idx, treatment in enumerate(self.treatments):
            self.treatment_groups[treatment].append(idx)
        
        # Store metadata
        self.num_treatments = len(unique_treatments)
        self.samples_per_treatment = pd.Series(self.treatments).value_counts().to_dict()
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get sample by index.
        
        Returns:
        --------
        sample : torch.Tensor
            Gene expression data
        treatment_label : torch.Tensor
            Encoded treatment label (for grouping replicates)
        treatment_name : str
            Original treatment name
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        treatment_label = self.treatment_labels[idx]
        treatment_name = self.treatments[idx]
        
        return sample, treatment_label, treatment_name
    
    def get_replicate_indices(self, treatment: str) -> List[int]:
        """Get indices of all replicates for a given treatment."""
        return self.treatment_groups.get(treatment, [])
    
    def get_treatment_name(self, label_idx: int) -> str:
        """Convert encoded label back to treatment name."""
        return self.idx_to_treatment[label_idx]
    
    @staticmethod
    def create_train_val_split(
        data: pd.DataFrame,
        treatments: pd.Series,
        val_split: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple['ContrastiveGeneExpressionDataset', 'ContrastiveGeneExpressionDataset']:
        """
        Create train/validation split.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Gene expression data
        treatments : pd.Series
            Treatment labels
        val_split : float
            Fraction for validation
        stratify : bool
            If True, ensures each treatment is in both train and val
            (important for contrastive learning!)
        random_state : int
            Random seed
        
        Returns:
        --------
        train_dataset : ContrastiveGeneExpressionDataset
            Training dataset
        val_dataset : ContrastiveGeneExpressionDataset
            Validation dataset
        """
        np.random.seed(random_state)
        
        if stratify:
            # Stratified split: keep some replicates of each treatment in both sets
            train_indices = []
            val_indices = []
            
            for treatment in treatments.unique():
                treatment_mask = treatments == treatment
                treatment_indices = np.where(treatment_mask)[0]
                
                # Shuffle
                np.random.shuffle(treatment_indices)
                
                # Split
                n_val = max(1, int(len(treatment_indices) * val_split))
                val_indices.extend(treatment_indices[:n_val])
                train_indices.extend(treatment_indices[n_val:])
            
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
        else:
            # Random split
            indices = np.random.permutation(len(data))
            split_idx = int(len(data) * (1 - val_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
        
        # Create datasets
        train_data = data.iloc[train_indices].reset_index(drop=True)
        val_data = data.iloc[val_indices].reset_index(drop=True)
        
        train_treatments = treatments.iloc[train_indices].reset_index(drop=True)
        val_treatments = treatments.iloc[val_indices].reset_index(drop=True)
        
        train_dataset = ContrastiveGeneExpressionDataset(train_data, train_treatments)
        val_dataset = ContrastiveGeneExpressionDataset(val_data, val_treatments)
        
        return train_dataset, val_dataset


class TreatmentBalancedSampler(Sampler):
    """
    Sampler that ensures each batch has multiple samples per treatment.
    This improves contrastive learning by having more positive pairs per batch.
    """
    
    def __init__(
        self,
        dataset: ContrastiveGeneExpressionDataset,
        batch_size: int,
        samples_per_treatment: int = 2
    ):
        """
        Initialize sampler.
        
        Parameters:
        -----------
        dataset : ContrastiveGeneExpressionDataset
            Dataset to sample from
        batch_size : int
            Batch size
        samples_per_treatment : int
            Target number of samples per treatment in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_treatment = samples_per_treatment
        
        # Group indices by treatment
        self.treatment_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            treatment_label = dataset.treatment_labels[idx].item()
            self.treatment_to_indices[treatment_label].append(idx)
        
        self.num_batches = len(dataset) // batch_size
    
    def __iter__(self):
        """Generate batches with balanced treatments."""
        all_indices = []
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample treatments for this batch
            treatments = list(self.treatment_to_indices.keys())
            np.random.shuffle(treatments)
            
            # For each treatment, sample a few replicates
            for treatment in treatments:
                available_indices = self.treatment_to_indices[treatment]
                n_samples = min(self.samples_per_treatment, len(available_indices))
                sampled = np.random.choice(available_indices, size=n_samples, replace=False)
                batch_indices.extend(sampled)
                
                if len(batch_indices) >= self.batch_size:
                    break
            
            # Trim to batch size
            batch_indices = batch_indices[:self.batch_size]
            
            # Shuffle batch
            np.random.shuffle(batch_indices)
            
            all_indices.extend(batch_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        """Return total number of samples."""
        return self.num_batches * self.batch_size

