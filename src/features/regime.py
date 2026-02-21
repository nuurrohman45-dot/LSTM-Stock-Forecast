"""
Regime detection module for market regime identification.

This module provides functionality to detect market regimes based on
volatility and trend strength.
"""

import numpy as np
import pandas as pd


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Args:
        df: Input DataFrame
        window: Rolling window size
    
    Returns:
        pd.DataFrame: DataFrame with volatility added
    """
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["vol_20"] = df["ret"].rolling(window).std()
    return df


def compute_volatility_regime(
    df: pd.DataFrame,
    vol_col: str = "vol_20",
    median_window: int = 60,
    threshold_multiplier: float = 1.2
) -> pd.DataFrame:
    """
    Compute volatility regime (normal vs high volatility).
    
    Args:
        df: Input DataFrame
        vol_col: Name of the volatility column
        median_window: Window for median volatility calculation
        threshold_multiplier: Multiplier for median volatility
    
    Returns:
        pd.DataFrame: DataFrame with volatility regime added
    """
    df = df.copy()
    df[f"vol_{median_window}_med"] = df[vol_col].rolling(median_window).median()
    df["vol_ok"] = df[vol_col] < threshold_multiplier * df[f"vol_{median_window}_med"]
    return df


def compute_trend_regime(
    df: pd.DataFrame,
    price_col: str = "Close",
    ma_window: int = 50,
    strength_threshold: float = 0.0075
) -> pd.DataFrame:
    """
    Compute trend regime (trending vs ranging).
    
    Args:
        df: Input DataFrame
        price_col: Name of the price column
        ma_window: Moving average window size
        strength_threshold: Threshold for trend strength
    
    Returns:
        pd.DataFrame: DataFrame with trend regime added
    """
    df = df.copy()
    close = df[price_col]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    df["ma50"] = close.rolling(ma_window).mean()
    df["trend_strength"] = (close - df["ma50"]).abs() / df["ma50"]
    df["trend_ok"] = df["trend_strength"] > strength_threshold
    return df


def compute_regime(
    df: pd.DataFrame,
    vol_window: int = 20,
    vol_median_window: int = 60,
    vol_threshold_multiplier: float = 1.2,
    ma_window: int = 50,
    trend_strength_threshold: float = 0.0075
) -> pd.DataFrame:
    """
    Compute overall market regime.
    
    Args:
        df: Input DataFrame
        vol_window: Rolling window for volatility
        vol_median_window: Window for median volatility
        vol_threshold_multiplier: Volatility threshold multiplier
        ma_window: Moving average window
        trend_strength_threshold: Trend strength threshold
    
    Returns:
        pd.DataFrame: DataFrame with regime features added
    """
    df = compute_volatility(df, vol_window)
    df = compute_volatility_regime(
        df,
        vol_col="vol_20",
        median_window=vol_median_window,
        threshold_multiplier=vol_threshold_multiplier
    )
    df = compute_trend_regime(
        df,
        price_col="Close",
        ma_window=ma_window,
        strength_threshold=trend_strength_threshold
    )
    df["regime_ok"] = df["vol_ok"] & df["trend_ok"]
    return df


def add_regime_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    vol_median_window: int = 60,
    vol_threshold_multiplier: float = 1.2,
    ma_window: int = 50,
    trend_strength_threshold: float = 0.0075
) -> pd.DataFrame:
    """
    Add regime features to the DataFrame (alias for compute_regime).
    
    Args:
        df: Input DataFrame
        vol_window: Rolling window for volatility
        vol_median_window: Window for median volatility
        vol_threshold_multiplier: Volatility threshold multiplier
        ma_window: Moving average window
        trend_strength_threshold: Trend strength threshold
    
    Returns:
        pd.DataFrame: DataFrame with regime features
    """
    return compute_regime(
        df,
        vol_window=vol_window,
        vol_median_window=vol_median_window,
        vol_threshold_multiplier=vol_threshold_multiplier,
        ma_window=ma_window,
        trend_strength_threshold=trend_strength_threshold
    )
