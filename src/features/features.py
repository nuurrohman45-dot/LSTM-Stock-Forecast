"""
Feature engineering module for stock data.

This module provides functionality to create technical indicators
and features for the LSTM model.
"""

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Compute log returns for the given price column.
    
    Args:
        df: Input DataFrame
        price_col: Name of the price column
    
    Returns:
        pd.DataFrame: DataFrame with log returns added
    """
    df = df.copy()
    df["log_return"] = np.log(df[price_col]).diff()
    return df


def compute_target(df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
    """
    Compute the target variable (next day return).
    
    Args:
        df: Input DataFrame
        return_col: Name of the return column
    
    Returns:
        pd.DataFrame: DataFrame with target added
    """
    df = df.copy()
    df["target"] = df[return_col].shift(-1)
    return df


def compute_volatility_features(
    df: pd.DataFrame,
    return_col: str = "log_return",
    windows: list = [10, 30]
) -> pd.DataFrame:
    """
    Compute rolling volatility features.
    
    Args:
        df: Input DataFrame
        return_col: Name of the return column
        windows: List of rolling window sizes
    
    Returns:
        pd.DataFrame: DataFrame with volatility features added
    """
    df = df.copy()
    for window in windows:
        df[f"volatility_{window}"] = df[return_col].rolling(window).std()
    return df


def compute_momentum_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    windows: list = [10, 30]
) -> pd.DataFrame:
    """
    Compute momentum features (price rate of change).
    
    Args:
        df: Input DataFrame
        price_col: Name of the price column
        windows: List of rolling window sizes
    
    Returns:
        pd.DataFrame: DataFrame with momentum features added
    """
    df = df.copy()
    for window in windows:
        df[f"momentum_{window}"] = df[price_col].pct_change(window)
    return df


def compute_volume_features(
    df: pd.DataFrame,
    volume_col: str = "Volume",
    window: int = 30
) -> pd.DataFrame:
    """
    Compute volume features (z-score of volume).
    
    Args:
        df: Input DataFrame
        volume_col: Name of the volume column
        window: Rolling window size
    
    Returns:
        pd.DataFrame: DataFrame with volume features added
    """
    df = df.copy()
    df["volume_z"] = (
        df[volume_col] - df[volume_col].rolling(window).mean()
    ) / df[volume_col].rolling(window).std()
    return df


def create_features(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "target"
) -> pd.DataFrame:
    """
    Create all features for the LSTM model.
    
    Args:
        df: Input DataFrame with raw stock data
        feature_cols: List of feature column names
        target_col: Name of the target column
    
    Returns:
        pd.DataFrame: DataFrame with all features
    """
    # Compute log returns
    df = compute_log_returns(df)
    
    # Compute target (next day return)
    df = compute_target(df)
    
    # Compute volatility features
    df = compute_volatility_features(df)
    
    # Compute momentum features
    df = compute_momentum_features(df)
    
    # Compute volume features
    df = compute_volume_features(df)
    
    # Drop NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df


def get_feature_columns() -> list:
    """
    Get the default list of feature column names.
    
    Returns:
        list: List of feature column names
    """
    return [
        "volatility_10",
        "volatility_30",
        "momentum_10",
        "momentum_30",
        "volume_z",
    ]
