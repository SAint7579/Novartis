"""
Dataset class for gene expression data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for gene expression data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Gene expression data [samples x genes]
        labels : pd.Series, optional
            Sample labels (e.g., treatment conditions)
        transform : callable, optional
            Transform to apply to data
        """
        self.data = torch.FloatTensor(data.values)
        self.labels = labels
        self.transform = transform
        
        if labels is not None:
            # Convert labels to categorical indices
            self.label_encoder = {label: idx for idx, label in enumerate(labels.unique())}
            self.encoded_labels = torch.LongTensor([
                self.label_encoder[label] for label in labels
            ])
        else:
            self.encoded_labels = None
            self.label_encoder = None
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get sample by index.
        
        Parameters:
        -----------
        idx : int
            Sample index
        
        Returns:
        --------
        sample : torch.Tensor
            Gene expression data for sample
        label : torch.Tensor or None
            Label for sample (if available)
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.encoded_labels is not None:
            return sample, self.encoded_labels[idx]
        else:
            return sample, torch.tensor(-1)  # Dummy label
    
    def get_original_label(self, encoded_label: int) -> str:
        """Convert encoded label back to original label."""
        if self.label_encoder is None:
            return None
        
        reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        return reverse_encoder.get(encoded_label)
    
    @staticmethod
    def create_train_val_split(
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        val_split: float = 0.2,
        random_state: int = 42
    ) -> Tuple['GeneExpressionDataset', 'GeneExpressionDataset']:
        """
        Create train/validation split.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Gene expression data
        labels : pd.Series, optional
            Sample labels
        val_split : float
            Fraction of data to use for validation
        random_state : int
            Random seed
        
        Returns:
        --------
        train_dataset : GeneExpressionDataset
            Training dataset
        val_dataset : GeneExpressionDataset
            Validation dataset
        """
        np.random.seed(random_state)
        
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - val_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = data.iloc[train_indices].reset_index(drop=True)
        val_data = data.iloc[val_indices].reset_index(drop=True)
        
        train_labels = labels.iloc[train_indices].reset_index(drop=True) if labels is not None else None
        val_labels = labels.iloc[val_indices].reset_index(drop=True) if labels is not None else None
        
        train_dataset = GeneExpressionDataset(train_data, train_labels)
        val_dataset = GeneExpressionDataset(val_data, val_labels)
        
        return train_dataset, val_dataset

