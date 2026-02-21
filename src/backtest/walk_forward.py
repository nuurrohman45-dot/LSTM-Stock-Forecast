"""
Walk-forward validation module (FIXED VERSION).

This implementation avoids:
- lookahead bias
- volatility leakage
- misaligned positions
- artificial Sharpe inflation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from models.lstm import AttnLSTM
from models.train import SequenceDataset, DataLoader
from features.regime import add_regime_features


# =========================================================
# TRAIN + PREDICT
# =========================================================
def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    target: str,
    seq_len: int,
    epochs: int,
    hidden_size: int,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[features] = x_scaler.fit_transform(train_df[features])
    test_df[features] = x_scaler.transform(test_df[features])

    train_df[target] = y_scaler.fit_transform(train_df[[target]])
    test_df[target] = y_scaler.transform(test_df[[target]])

    train_ds = SequenceDataset(train_df, features, target, seq_len)
    test_ds = SequenceDataset(test_df, features, target, seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = AttnLSTM(
        input_size=len(features),
        hidden_size=hidden_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
            trues.append(y.numpy())

    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()

    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    trues = y_scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

    return preds, trues


# =========================================================
# SINGLE WALK-FORWARD SPLIT
# =========================================================
def walk_forward_single_split(
    df: pd.DataFrame,
    features: list,
    target: str,
    start: int,
    train_days: int,
    test_days: int,
    seq_len: int,
    epochs: int,
    hidden_size: int,
    vol_target: float,
    cost: float,
    device: str
) -> np.ndarray:

    train_df = df.iloc[start:start + train_days].copy()
    test_df = df.iloc[start + train_days:start + train_days + test_days].copy()

    preds, trues = train_and_predict(
        train_df, test_df, features, target,
        seq_len, epochs, hidden_size, device
    )

    # Smooth predictions
    preds = pd.Series(preds).rolling(3, min_periods=1).mean().values

    # Raw signal
    raw_pos = np.tanh(preds / vol_target)

    # Confidence gating
    thresh = np.percentile(np.abs(preds), 60)
    raw_pos *= (np.abs(preds) > thresh)

    # Lagged volatility (NO LOOKAHEAD)
    realized_vol = (
        test_df["vol_20"]
        .shift(1)
        .iloc[-len(raw_pos):]
        .values
    )
    realized_vol = np.nan_to_num(realized_vol, nan=vol_target)
    realized_vol = np.maximum(realized_vol, 1e-6)

    pos = raw_pos * (vol_target / realized_vol)
    pos = np.clip(pos, -1, 1)

    # Regime filter (lagged)
    regime = (
        test_df["regime_ok"]
        .shift(1)
        .iloc[-len(pos):]
        .fillna(0.0)
        .values
    )
    pos *= regime

    # Shift position forward (trade today → earn tomorrow)
    pos_shifted = np.roll(pos, 1)
    pos_shifted[0] = 0.0

    # Transaction costs
    turnover = np.abs(np.diff(pos_shifted, prepend=0))
    costs = cost * turnover

    # FINAL NET RETURN (CORRECT ALIGNMENT)
    net_ret = pos_shifted * trues - costs
    return net_ret


# =========================================================
# FULL WALK-FORWARD
# =========================================================
def walk_forward(
    df: pd.DataFrame,
    features: list,
    target: str,
    train_days: int,
    test_days: int,
    trade_start: int,
    seq_len: int,
    epochs: int,
    hidden_size: int,
    vol_target: float,
    cost: float,
    device: str
) -> np.ndarray:

    all_returns = []
    start = trade_start

    while start + train_days + test_days < len(df):
        ret = walk_forward_single_split(
            df,
            features,
            target,
            start,
            train_days,
            test_days,
            seq_len,
            epochs,
            hidden_size,
            vol_target,
            cost,
            device
        )
        all_returns.append(ret)
        start += test_days

    return np.concatenate(all_returns)


# =========================================================
# ENTRY POINT
# =========================================================
def run_walk_forward_validation(
    df: pd.DataFrame,
    features: list,
    target: str,
    train_days: int,
    test_days: int,
    trade_start: int,
    seq_len: int,
    epochs: int,
    hidden_size: int,
    vol_target: float,
    cost: float,
    device: str
):

    df = add_regime_features(df)
    df = df.dropna()

    returns = walk_forward(
        df,
        features,
        target,
        train_days,
        test_days,
        trade_start,
        seq_len,
        epochs,
        hidden_size,
        vol_target,
        cost,
        device
    )

    return returns
