"""
Portfolio position sizing module.

This module provides functionality to calculate position sizes
based on model predictions and risk management rules.
"""

import numpy as np
import pandas as pd
from typing import Optional


def scale_position_tanh(
    predictions: np.ndarray,
    vol_target: float = 0.01
) -> np.ndarray:
    """
    Scale predictions to position sizes using tanh function.
    
    This provides a smooth, bounded position sizing that naturally
    clips extreme predictions.
    
    Args:
        predictions: Model predictions (returns)
        vol_target: Target volatility for scaling
    
    Returns:
        Position sizes in range [-1, 1]
    """
    return np.tanh(predictions / vol_target)


def apply_confidence_gating(
    positions: np.ndarray,
    predictions: np.ndarray,
    percentile: float = 60
) -> np.ndarray:
    """
    Apply confidence gating to positions.
    
    Only take positions when the prediction magnitude exceeds
    a certain percentile threshold.
    
    Args:
        positions: Current position sizes
        predictions: Model predictions
        percentile: Percentile threshold for gating
    
    Returns:
        Gated position sizes
    """
    thresh = np.nanpercentile(np.abs(predictions), percentile)
    gate = np.abs(predictions) > thresh
    return positions * gate.astype(float)


def apply_volatility_targeting(
    positions: np.ndarray,
    realized_vol: np.ndarray,
    vol_target: float = 0.01,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply volatility targeting to positions.
    
    Scale positions based on realized volatility relative to target.
    
    Args:
        positions: Current position sizes
        realized_vol: Realized volatility
        vol_target: Target volatility
        epsilon: Small value to avoid division by zero
    
    Returns:
        Volatility-targeted position sizes
    """
    # Handle NaN and infinite values
    realized_vol = np.nan_to_num(
        realized_vol,
        nan=vol_target,
        posinf=vol_target,
        neginf=vol_target
    )
    realized_vol = np.maximum(realized_vol, epsilon)
    
    # Scale positions
    scaled_positions = positions * (vol_target / realized_vol)
    
    # Clip to valid range
    return np.clip(scaled_positions, -1, 1)


def apply_regime_filter(
    positions: np.ndarray,
    regime_ok: np.ndarray
) -> np.ndarray:
    """
    Apply regime filter to positions.
    
    Zero out positions when market regime is not favorable.
    
    Args:
        positions: Current position sizes
        regime_ok: Boolean array indicating favorable regime
    
    Returns:
        Regime-filtered position sizes
    """
    # Handle NaN values
    regime_ok = np.nan_to_num(regime_ok, nan=0.0)
    return positions * regime_ok.astype(float)


def smooth_predictions(
    predictions: np.ndarray,
    window: int = 3
) -> np.ndarray:
    """
    Apply moving average smoothing to predictions.
    
    Args:
        predictions: Raw predictions
        window: Smoothing window size
    
    Returns:
        Smoothed predictions
    """
    return pd.Series(predictions).rolling(window, min_periods=1).mean().values


def calculate_turnover(positions: np.ndarray) -> np.ndarray:
    """
    Calculate turnover (position changes).
    
    Args:
        positions: Position sizes
    
    Returns:
        Turnover (absolute position changes)
    """
    return np.abs(np.diff(positions, prepend=0))


def calculate_transaction_costs(
    positions: np.ndarray,
    cost_per_trade: float = 0.0005
) -> np.ndarray:
    """
    Calculate transaction costs based on turnover.
    
    Args:
        positions: Position sizes
        cost_per_trade: Cost per unit of turnover
    
    Returns:
        Transaction costs
    """
    turnover = calculate_turnover(positions)
    return cost_per_trade * turnover


def calculate_returns(
    positions: np.ndarray,
    returns: np.ndarray,
    costs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate strategy returns.
    
    Args:
        positions: Position sizes
        returns: Asset returns
        costs: Optional transaction costs
    
    Returns:
        Net strategy returns
    """
    gross_returns = positions * returns
    
    if costs is not None:
        return gross_returns - costs
    
    return gross_returns


def calculate_positions(
    predictions: np.ndarray,
    realized_vol: Optional[np.ndarray] = None,
    regime_ok: Optional[np.ndarray] = None,
    vol_target: float = 0.01,
    cost_per_trade: float = 0.0005,
    epsilon: float = 1e-8,
    smoothing_window: int = 3,
    confidence_percentile: float = 60,
    apply_smoothing: bool = True,
    enable_confidence_gating: bool = True,
    apply_vol_targeting: bool = True,
    apply_regime_filter: bool = True
) -> dict:
    """
    Calculate final positions with all risk management rules.
    
    Args:
        predictions: Model predictions
        realized_vol: Realized volatility (optional)
        regime_ok: Regime indicator (optional)
        vol_target: Target volatility
        cost_per_trade: Cost per trade
        epsilon: Small value to avoid division by zero
        smoothing_window: Prediction smoothing window
        confidence_percentile: Confidence gate percentile
        apply_smoothing: Whether to apply prediction smoothing
        enable_confidence_gating: Whether to apply confidence gating
        apply_vol_targeting: Whether to apply volatility targeting
        apply_regime_filter: Whether to apply regime filtering
    
    Returns:
        Dictionary containing positions and intermediate results
    """
    preds = predictions.copy()
    
    # 1. Prediction smoothing
    if apply_smoothing:
        preds = smooth_predictions(preds, smoothing_window)
    
    # 2. Raw position from predictions
    positions = scale_position_tanh(preds, vol_target)
    
    # 3. Confidence gating
    if enable_confidence_gating:
        positions = apply_confidence_gating(positions, preds, confidence_percentile)
    
    # 4. Volatility targeting
    if apply_vol_targeting and realized_vol is not None:
        positions = apply_volatility_targeting(
            positions, realized_vol, vol_target, epsilon
        )
    
    # 5. Regime filter
    if apply_regime_filter and regime_ok is not None:
        positions = apply_regime_filter(positions, regime_ok)
    
    # Calculate turnover and costs
    turnover = calculate_turnover(positions)
    costs = calculate_transaction_costs(positions, cost_per_trade)
    
    return {
        "positions": positions,
        "turnover": turnover,
        "costs": costs,
        "predictions": preds,
    }
