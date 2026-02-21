"""
Performance metrics module for backtesting.

This module provides functionality to calculate various performance
metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def sharpe_ratio(
    returns: np.ndarray,
    annualize: bool = True,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0
    
    sr = returns.mean() / returns.std()
    
    if annualize:
        return sr * np.sqrt(periods_per_year)
    
    return sr


def sortino_ratio(
    returns: np.ndarray,
    annualize: bool = True,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (using downside deviation).
    
    Args:
        returns: Array of returns
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods per year
        target_return: Target return threshold
    
    Returns:
        Sortino ratio
    """
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    excess_return = returns.mean() - target_return
    
    sr = excess_return / downside_std
    
    if annualize:
        return sr * np.sqrt(periods_per_year)
    
    return sr


def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
    
    Returns:
        Maximum drawdown (negative value)
    """
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - running_max
    return drawdown.min()


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * periods_per_year
    dd = abs(max_drawdown(returns))
    
    if dd == 0:
        return 0.0
    
    return annual_return / dd


def calc_rmse(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        actual: Actual values
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((predictions - actual) ** 2))


def calc_mae(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        actual: Actual values
    
    Returns:
        MAE
    """
    return np.mean(np.abs(predictions - actual))


def calc_mape(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        actual: Actual values
    
    Returns:
        MAPE (as percentage)
    """
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100


def calc_correlation(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate correlation between predictions and actual values.
    
    Args:
        predictions: Predicted values
        actual: Actual values
    
    Returns:
        Correlation coefficient
    """
    return np.corrcoef(predictions, actual)[0, 1]


def calc_direction_accuracy(
    positions: np.ndarray,
    returns: np.ndarray
) -> float:
    """
    Calculate direction accuracy (percentage of correct direction predictions).
    
    Args:
        positions: Position sizes
        returns: Actual returns
    
    Returns:
        Direction accuracy (0-1)
    """
    return np.mean(np.sign(positions) == np.sign(returns))


def calc_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
    
    Returns:
        Information ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    ir = excess_returns.mean() / tracking_error
    
    return ir * np.sqrt(periods_per_year)


def calc_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Array of returns
    
    Returns:
        Win rate (0-1)
    """
    return np.mean(returns > 0)


def calc_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profits / gross losses).
    
    Args:
        returns: Array of returns
    
    Returns:
        Profit factor
    """
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    
    if gross_losses == 0:
        return np.inf if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses


def calculate_all_metrics(
    predictions: np.ndarray,
    actual: np.ndarray,
    positions: np.ndarray,
    returns: np.ndarray,
    gross_returns: Optional[np.ndarray] = None,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Args:
        predictions: Model predictions
        actual: Actual returns
        positions: Position sizes
        returns: Net strategy returns
        gross_returns: Gross strategy returns (without costs)
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Prediction metrics
    metrics["rmse"] = calc_rmse(predictions, actual)
    metrics["mae"] = calc_mae(predictions, actual)
    metrics["correlation"] = calc_correlation(predictions, actual)
    metrics["direction_accuracy"] = calc_direction_accuracy(positions, actual)
    
    # Return metrics
    metrics["total_return"] = returns.sum()
    metrics["annual_return"] = returns.mean() * periods_per_year
    metrics["annual_volatility"] = returns.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    metrics["sharpe_ratio"] = sharpe_ratio(returns, annualize=True, periods_per_year=periods_per_year)
    metrics["sortino_ratio"] = sortino_ratio(returns, annualize=True, periods_per_year=periods_per_year)
    metrics["max_drawdown"] = max_drawdown(returns)
    metrics["calmar_ratio"] = calmar_ratio(returns, periods_per_year=periods_per_year)
    
    # Trade statistics
    metrics["win_rate"] = calc_win_rate(returns)
    metrics["profit_factor"] = calc_profit_factor(returns)
    
    # Gross returns (if provided)
    if gross_returns is not None:
        metrics["sharpe_ratio_gross"] = sharpe_ratio(
            gross_returns, annualize=True, periods_per_year=periods_per_year
        )
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for each line (e.g., "Gross: ", "Net: ")
    """
    print(f"\n=== {prefix}PERFORMANCE ===")
    
    # Prediction metrics
    print(f"RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"Correlation: {metrics.get('correlation', 0):.4f}")
    print(f"Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}")
    
    # Return metrics
    print(f"Total Return: {metrics.get('total_return', 0):.4f}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.4f}")
    print(f"Annual Volatility: {metrics.get('annual_volatility', 0):.4f}")
    
    # Risk-adjusted metrics
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
    print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
    
    # Trade statistics
    print(f"Win Rate: {metrics.get('win_rate', 0):.4f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}")
