"""
Model save and load module.

This module provides functionality to save and load the LSTM model
with all associated artifacts.
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import torch.nn as nn


def infer_model_hyperparameters(model: nn.Module) -> Dict[str, int]:
    """
    Infer hidden_size and num_layers from a model's LSTM layer.
    
    Args:
        model: PyTorch model with LSTM layer
    
    Returns:
        Dictionary containing 'hidden_size' and 'num_layers'
    """
    # Find the LSTM layer in the model
    lstm_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            lstm_layer = module
            break
    
    if lstm_layer is None:
        # Default values if LSTM not found
        return {"hidden_size": 32, "num_layers": 1}
    
    hidden_size = lstm_layer.hidden_size
    num_layers = lstm_layer.num_layers
    
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers
    }


def save_model(
    model: nn.Module,
    model_path: str,
    feature_cols: list,
    seq_len: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    model_type: str = "AttnLSTM",
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save the model with all artifacts.
    
    Args:
        model: PyTorch model
        model_path: Path to save the model
        feature_cols: List of feature column names
        seq_len: Sequence length
        x_mean: Mean of training features
        x_std: Standard deviation of training features
        model_type: Type of the model
        hidden_size: LSTM hidden size (optional, will be inferred if not provided)
        num_layers: Number of LSTM layers (optional, will be inferred if not provided)
        additional_info: Additional information to save
    """
    # Infer hyperparameters from model if not provided
    if hidden_size is None or num_layers is None:
        inferred = infer_model_hyperparameters(model)
        hidden_size = hidden_size or inferred["hidden_size"]
        num_layers = num_layers or inferred["num_layers"]
    
    # Create artifact dictionary
    artifact = {
        "model_state": model.state_dict(),
        "features": feature_cols,
        "seq_len": seq_len,
        "x_mean": x_mean,
        "x_std": x_std,
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    
    # Add additional info if provided
    if additional_info:
        artifact.update(additional_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    torch.save(artifact, model_path)
    print(f"Model saved to {model_path}")


def load_model(
    model: nn.Module,
    model_path: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load the model with all artifacts.
    
    Args:
        model: PyTorch model (architecture must match saved model)
        model_path: Path to the saved model
        device: Device to load the model to
    
    Returns:
        Dictionary containing model artifacts
    """
    # Load the artifact
    # Note: weights_only=False is required for models saved with numpy arrays (x_mean, x_std)
    # This is safe since we trust the model file source
    artifact = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(artifact["model_state"])
    model.eval()
    
    return artifact


def create_model_artifact(
    model: nn.Module,
    feature_cols: list,
    seq_len: int,
    x_train: np.ndarray,
    model_type: str = "AttnLSTM",
    version: str = "v1.0",
    x_scaler: Optional[Any] = None,
    y_scaler: Optional[Any] = None,
    metrics: Dict[str, float] = None,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a model artifact dictionary.
    
    Args:
        model: PyTorch model
        feature_cols: List of feature column names
        seq_len: Sequence length
        x_train: Training features for computing mean/std
        model_type: Type of the model
        version: Model version
        x_scaler: Feature scaler to save
        y_scaler: Target scaler to save
        metrics: Metrics to save (wf, max_drawdown, win_rate, strategy_vol, correlation)
        hidden_size: LSTM hidden size (optional, will be inferred if not provided)
        num_layers: Number of LSTM layers (optional, will be inferred if not provided)
    
    Returns:
        Dictionary containing model artifacts
    """
    # Validate metrics is provided (required)
    if metrics is None:
        raise ValueError("metrics is required and must be provided")
    
    # Infer hyperparameters from model if not provided
    if hidden_size is None or num_layers is None:
        inferred = infer_model_hyperparameters(model)
        hidden_size = hidden_size or inferred["hidden_size"]
        num_layers = num_layers or inferred["num_layers"]
    
    artifact = {
        "model_state": model.state_dict(),
        "features": feature_cols,
        "seq_len": seq_len,
        "x_mean": x_train.mean(axis=0),
        "x_std": x_train.std(axis=0),
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "version": version,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    
    # Add scalers if provided
    if x_scaler is not None:
        artifact["x_scaler"] = x_scaler
    if y_scaler is not None:
        artifact["y_scaler"] = y_scaler
    
    # Always save metrics (required)
    artifact["metrics"] = metrics
    
    return artifact


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a saved model without loading the full state dict.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Dictionary containing model information
    """
    # Load only the metadata
    # Note: weights_only=False is required for models saved with numpy arrays (x_mean, x_std)
    # This is safe since we trust the model file source
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    
    info = {
        "model_type": artifact.get("model_type", "Unknown"),
        "features": artifact.get("features", []),
        "seq_len": artifact.get("seq_len", 0),
        "trained_at": artifact.get("trained_at", "Unknown"),
        "version": artifact.get("version", "Unknown"),
        "hidden_size": artifact.get("hidden_size", 32),
        "num_layers": artifact.get("num_layers", 1),
    }
    
    if "metrics" in artifact:
        info["metrics"] = artifact["metrics"]
    
    return info
