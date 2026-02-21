"""
Training module for LSTM model.

This module provides functionality to train the AttnLSTM model
with DataLoader support.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential data.
    
    This dataset creates sequences of fixed length from the input data
    for LSTM training.
    """
    
    def __init__(
        self,
        df,
        feature_cols: list,
        target_col: str,
        seq_len: int
    ):
        """
        Initialize the SequenceDataset.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Name of the target column
            seq_len: Length of the input sequence
        """
        self.X = df[feature_cols].values
        self.y = df[target_col].values
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.
        
        Args:
            idx: Index of the sequence
        
        Returns:
            Tuple of (input_sequence, target)
        """
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def create_data_loaders(
    train_df,
    test_df,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Name of the target column
        seq_len: Length of the input sequence
        batch_size: Batch size for DataLoader
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_ds = SequenceDataset(train_df, feature_cols, target_col, seq_len)
    test_ds = SequenceDataset(test_df, feature_cols, target_col, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    gradient_clip_value: Optional[float] = None
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        data_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip_value: Gradient clipping value (None = no clipping)
    
    Returns:
        float: Average training loss
    """
    model.train()
    losses = []
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(x).squeeze(-1)  # Shape: (batch,) to match target shape
        loss = criterion(preds, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_value
            )
        
        # Update weights
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    y_scaler: Optional[object] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on test data.
    
    Args:
        model: PyTorch model
        data_loader: Test data loader
        device: Device to evaluate on
        y_scaler: Optional scaler to inverse transform predictions
    
    Returns:
        Tuple of (predictions, true_values)
    """
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # Inverse scale if scaler provided
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        trues = y_scaler.inverse_transform(trues.reshape(-1, 1)).flatten()
    
    return preds, trues


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    gradient_clip_value: float = 1.0,
    y_scaler: Optional[object] = None,
    verbose: bool = True
) -> Tuple[nn.Module, dict]:
    """
    Train the model for multiple epochs.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        gradient_clip_value: Gradient clipping value
        y_scaler: Optional scaler for inverse transformation
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    history = {
        "train_loss": [],
        "test_loss": [],
    }
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, gradient_clip_value
        )
        
        # Evaluate
        preds, trues = evaluate(model, test_loader, device, y_scaler)
        test_loss = np.mean((preds - trues) ** 2)
        
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}: Train MSE = {train_loss:.4f}")
    
    return model, history
