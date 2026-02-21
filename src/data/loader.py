"""
Data loading module for stock price data.

This module provides functionality to download and load historical
stock data using yfinance.
"""

import pandas as pd
import yfinance as yf
from typing import Optional, Tuple


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (None = present)
    
    Returns:
        pd.DataFrame: DataFrame with historical stock data
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


def load_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None
) -> pd.DataFrame:

    df = download_stock_data(ticker, start_date, end_date)

    # FORCE datetime index (NO ASSUMPTION)
    if not isinstance(df.index, pd.DatetimeIndex):

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        else:
            # fallback: index assumed to be date-like
            df.index = pd.to_datetime(df.index, errors="raise")

    df = df.sort_index()

    # HARD FAIL if still wrong
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(
            f"load_data FAILED: index type = {type(df.index)}"
        )

    return df



def get_train_test_split(
    df: pd.DataFrame,
    train_split: float = 0.7,
    seq_len: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        train_split: Ratio of training data (0-1)
        seq_len: Sequence length for LSTM
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx + seq_len:]
    return train_df, test_df
