"""
Final training experiment entry point.

This module provides the main entry point for training the LSTM model
with full experiment tracking and validation.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data import loader
from features import features, regime
from models import lstm, train, save_load
from portfolio import position
from backtest import metrics, walk_forward
from mlflow_utils import tracking


def main():
    """
    Main training function.
    """
    print("=" * 60)
    print("LSTM Stock Prediction - Final Training")
    print("=" * 60)
    
    # Get configuration
    cfg = config.get_config()
    
    # Initialize MLflow tracker
    tracker = tracking.MLflowTracker(
        experiment_name="lstm_stock_prediction_final"
    )
    
    with tracker.start_run(run_name="final_training"):
        # Log configuration
        tracker.log_params({
            "ticker": cfg["ticker"],
            "start_date": cfg["start_date"],
            "seq_len": cfg["seq_len"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "hidden_size": cfg["hidden_size"],
            "model_type": cfg["model_type"],
        })
        
        # Step 1: Load data
        print("\n[1/6] Loading data...")
        df = loader.load_data(
            ticker=cfg["ticker"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"]
        )
        print(f"Loaded {len(df)} rows of data")
        
        # Step 2: Feature engineering
        print("\n[2/6] Creating features...")
        df = features.create_features(df)
        print(f"Created {len(cfg['feature_cols'])} features")
        
        # Step 3: Train/test split
        print("\n[3/6] Splitting data...")
        train_df, test_df = loader.get_train_test_split(
            df,
            train_split=cfg["train_split"],
            seq_len=cfg["seq_len"]
        )
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Step 4: Scale features
        from sklearn.preprocessing import StandardScaler
        
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()
        
        train_df_scaled[cfg["feature_cols"]] = x_scaler.fit_transform(
            train_df[cfg["feature_cols"]]
        )
        test_df_scaled[cfg["feature_cols"]] = x_scaler.transform(
            test_df[cfg["feature_cols"]]
        )
        
        train_df_scaled[cfg["target_col"]] = y_scaler.fit_transform(
            train_df[[cfg["target_col"]]]
        )
        test_df_scaled[cfg["target_col"]] = y_scaler.transform(
            test_df[[cfg["target_col"]]]
        )
        
        # Step 5: Create data loaders
        print("\n[4/6] Creating data loaders...")
        train_loader, test_loader = train.create_data_loaders(
            train_df_scaled,
            test_df_scaled,
            cfg["feature_cols"],
            cfg["target_col"],
            cfg["seq_len"],
            cfg["batch_size"]
        )
        
        # Step 6: Create and train model
        print("\n[5/6] Training model...")
        model = lstm.create_model(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"]
        )
        
        trained_model, history = train.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            device=cfg["device"],
            gradient_clip_value=cfg["gradient_clip_value"],
            y_scaler=y_scaler,
            verbose=True
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        preds, trues = train.evaluate(
            trained_model,
            test_loader,
            cfg["device"],
            y_scaler
        )
        
        # Calculate positions
        pos_result = position.calculate_positions(
            predictions=preds,
            vol_target=cfg["vol_target"],
            cost_per_trade=cfg["cost_per_trade"],
            epsilon=cfg["epsilon"],
            smoothing_window=cfg["prediction_smoothing_window"],
            confidence_percentile=cfg["confidence_gate_percentile"],
            apply_smoothing=True,
            enable_confidence_gating=True,
            apply_vol_targeting=False,
            apply_regime_filter=False
        )
        
        # Calculate returns
        costs = position.calculate_transaction_costs(
            pos_result["positions"],
            cfg["cost_per_trade"]
        )
        net_returns = position.calculate_returns(
            pos_result["positions"],
            trues,
            costs
        )
        
        # Calculate metrics
        test_metrics = metrics.calculate_all_metrics(
            predictions=preds,
            actual=trues,
            positions=pos_result["positions"],
            returns=net_returns,
            gross_returns=pos_result["positions"] * trues
        )
        
        # Print metrics
        metrics.print_metrics(test_metrics, prefix="Test ")
        
        # Log metrics to MLflow
        tracker.log_metrics(test_metrics)
        
        # Step 7: Walk-forward validation
        print("\n[6/6] Running walk-forward validation...")
        
        # Add regime features for walk-forward
        df_wf = regime.add_regime_features(
            df.copy(),
            vol_window=cfg["vol_window"],
            vol_median_window=cfg["vol_median_window"],
            vol_threshold_multiplier=cfg["vol_threshold_multiplier"],
            ma_window=cfg["ma_window"],
            trend_strength_threshold=cfg["trend_strength_threshold"]
        )
        df_wf = df_wf[cfg["wf_trade_start"]:].dropna().reset_index(drop=True)
        
        wf_returns = walk_forward.walk_forward(
            df_wf,
            cfg["feature_cols"],
            cfg["target_col"],
            train_days=cfg["wf_train_days"],
            test_days=cfg["wf_test_days"],
            trade_start=0,
            seq_len=cfg["seq_len"],
            epochs=15,  # Fewer epochs for WF
            hidden_size=cfg["hidden_size"],
            vol_target=cfg["vol_target"],
            cost=cfg["cost_per_trade"],
            device=cfg["device"]
        )
        
        wf_sharpe = metrics.sharpe_ratio(wf_returns)
        wf_max_dd = metrics.max_drawdown(wf_returns)
        wf_sortino = metrics.sortino_ratio(wf_returns)
        wf_calmar = metrics.calmar_ratio(wf_returns)
        wf_win_rate = metrics.calc_win_rate(wf_returns)
        wf_profit_factor = metrics.calc_profit_factor(wf_returns)
        wf_annual_return = wf_returns.mean() * 252
        wf_annual_vol = wf_returns.std() * np.sqrt(252)
        
        print(f"\n=== Walk-Forward Results ===")
        print(f"Sharpe: {wf_sharpe:.3f}")
        print(f"Max Drawdown: {wf_max_dd:.4f}")
        print(f"Sortino: {wf_sortino:.3f}")
        print(f"Calmar: {wf_calmar:.3f}")
        print(f"Win Rate: {wf_win_rate:.2%}")
        print(f"Profit Factor: {wf_profit_factor:.3f}")
        
        # Log walk-forward metrics
        tracker.log_metrics({
            "wf_sharpe": wf_sharpe,
            "wf_max_drawdown": wf_max_dd,
            "wf_mean_return": wf_returns.mean(),
            "wf_std_return": wf_returns.std(),
            "wf_sortino_ratio": wf_sortino,
            "wf_calmar_ratio": wf_calmar,
            "wf_win_rate": wf_win_rate,
            "wf_profit_factor": wf_profit_factor,
            "wf_annual_return": wf_annual_return,
            "wf_annual_volatility": wf_annual_vol,
        })
        
        # Save model
        print("\nSaving model...")
        os.makedirs(cfg["model_dir"], exist_ok=True)
        model_path = os.path.join(cfg["model_dir"], cfg["model_filename"])
        
        # Create comprehensive metrics dictionary with all test metrics and walk-forward metrics
        all_metrics = {
            # Test metrics
            "test_sharpe": test_metrics.get("sharpe_ratio", 0),
            "test_rmse": test_metrics.get("rmse", 0),
            "test_mae": test_metrics.get("mae", 0),
            "test_correlation": test_metrics.get("correlation", 0),
            "test_direction_accuracy": test_metrics.get("direction_accuracy", 0),
            "test_total_return": test_metrics.get("total_return", 0),
            "test_annual_return": test_metrics.get("annual_return", 0),
            "test_annual_volatility": test_metrics.get("annual_volatility", 0),
            "test_sortino_ratio": test_metrics.get("sortino_ratio", 0),
            "test_max_drawdown": test_metrics.get("max_drawdown", 0),
            "test_calmar_ratio": test_metrics.get("calmar_ratio", 0),
            "test_win_rate": test_metrics.get("win_rate", 0),
            "test_profit_factor": test_metrics.get("profit_factor", 0),
            # Walk-forward metrics
            "wf_sharpe": wf_sharpe,
            "wf_max_drawdown": wf_max_dd,
            "wf_sortino_ratio": wf_sortino,
            "wf_calmar_ratio": wf_calmar,
            "wf_win_rate": wf_win_rate,
            "wf_profit_factor": wf_profit_factor,
            "wf_mean_return": wf_returns.mean(),
            "wf_std_return": wf_returns.std(),
            "wf_annual_return": wf_annual_return,
            "wf_annual_volatility": wf_annual_vol,
        }
        
        # Create artifact with scalers and all metrics
        artifact = save_load.create_model_artifact(
            trained_model,
            cfg["feature_cols"],
            cfg["seq_len"],
            train_df[cfg["feature_cols"]].values,
            cfg["model_type"],
            version="v1.0",
            metrics=all_metrics,
            x_scaler=x_scaler,
            y_scaler=y_scaler
        )
        
        torch.save(artifact, model_path)
        print(f"Model saved to {model_path}")
        
        # Log model to MLflow
        tracker.log_model(
            trained_model,
            artifact_path="model",
            registered_model_name="AttnLSTM"
        )
        
        # Log configuration as artifact
        tracker.log_dict(cfg, "config.json")
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        
        return trained_model, test_metrics


if __name__ == "__main__":
    main()
