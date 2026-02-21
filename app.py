"""
Flask API for LSTM Stock Forecast.

This module provides a REST API for stock price forecasting using
an LSTM model with attention mechanism.
"""

import os
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

# =========================================================
# FLASK APP SETUP
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# MODEL LOADING
# =========================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "src/artifacts/models/lstm_v1.pt")

# Global variables to store model and artifact
_model = None
_artifact = None
_device = None


def load_model():
    """
    Load the LSTM model and artifacts.
    
    Returns:
        tuple: (model, artifact, device)
    """
    from src.models.lstm import AttnLSTM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load artifact
    artifact = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Get hyperparameters from artifact
    hidden_size = artifact.get("hidden_size", 64)
    num_layers = artifact.get("num_layers", 2)
    
    # Create model
    model = AttnLSTM(
        input_size=len(artifact["features"]),
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    model.load_state_dict(artifact["model_state"])
    model.to(device)
    model.eval()
    
    return model, artifact, device


def load_and_prepare_data(ticker, start_date, features, seq_len):
    """
    Load stock data and prepare features.
    
    Args:
        ticker: Stock symbol
        start_date: Start date for data
        features: List of feature column names
        seq_len: Sequence length
    
    Returns:
        tuple: (df, last_seq, last_price) or (None, None, None) if insufficient data
    """
    from src.data.loader import load_data
    from src.features.features import create_features
    
    df = load_data(ticker=ticker, start_date=start_date)
    
    if df is None or len(df) < seq_len + 5:
        return None, None, None
    
    df = create_features(df)
    
    # Get the last sequence for prediction
    last_price = float(df["Close"].iloc[-1])
    last_seq = df[features].iloc[-seq_len:].values
    
    return df, last_seq, last_price


def forecast_n_days(model, last_seq, n_days, artifact, device, current_price):
    """
    Generate n-day price forecast.
    
    Args:
        model: LSTM model
        last_seq: Last sequence of features
        n_days: Number of days to forecast
        artifact: Model artifact containing scalers
        device: Device to run inference on
        current_price: Current stock price
    
    Returns:
        tuple: (prices, returns)
    """
    # Get scalers from artifact
    x_scaler = artifact.get("x_scaler")
    y_scaler = artifact.get("y_scaler")
    
    # If scalers not saved, create simple ones
    if x_scaler is None or y_scaler is None:
        x_mean = artifact.get("x_mean", np.zeros(last_seq.shape[1]))
        x_std = artifact.get("x_std", np.ones(last_seq.shape[1]))
        
        class SimpleScaler:
            def transform(self, data):
                return (data - x_mean) / (x_std + 1e-8)
            
            def inverse_transform(self, data):
                return data * (x_std + 1e-8) + x_mean
        
        x_scaler = SimpleScaler()
        y_scaler = SimpleScaler()
    
    preds = []
    seq = last_seq.copy()
    
    for _ in range(n_days):
        # Scale input
        seq_scaled = x_scaler.transform(seq)
        X = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(X).item()
        
        # Inverse transform prediction
        pred = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred))
        
        # Roll sequence and update with prediction
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 0] = pred
    
    returns = np.array(preds)
    # Convert returns to actual prices using current price
    prices = current_price * np.exp(np.cumsum(returns))
    
    return prices, returns


# =========================================================
# API ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def index():
    """
    API info endpoint.
    
    Returns:
        JSON: API information
    """
    return jsonify({
        "name": "LSTM Stock Forecast API",
        "version": "1.0.0",
        "description": "REST API for LSTM-based stock price forecasting",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Get stock price forecast",
            "/model/info": "Model information"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON: Health status
    """
    global _model, _artifact, _device
    
    status = {
        "status": "healthy",
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if _artifact:
        status["model_info"] = {
            "features": _artifact.get("features", []),
            "seq_len": _artifact.get("seq_len", 0),
            "version": _artifact.get("version", "unknown")
        }
    
    return jsonify(status)


@app.route("/model/info", methods=["GET"])
def model_info():
    """
    Get model information.
    
    Returns:
        JSON: Model information
    """
    global _artifact
    
    if _artifact is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": _artifact.get("model_type", "AttnLSTM"),
        "features": _artifact.get("features", []),
        "seq_len": _artifact.get("seq_len", 0),
        "hidden_size": _artifact.get("hidden_size", 64),
        "num_layers": _artifact.get("num_layers", 2),
        "version": _artifact.get("version", "unknown"),
        "trained_at": _artifact.get("trained_at", "unknown"),
        "metrics": _artifact.get("metrics", {})
    })


@app.route("/predict", methods=["GET"])
def predict():
    """
    Get stock price forecast.
    
    Query Parameters:
        ticker: Stock symbol (default: AAPL)
        n_days: Number of days to forecast (default: 7, max: 30)
    
    Returns:
        JSON: Forecast data
    """
    global _model, _artifact, _device
    
    if _model is None:
        return jsonify({"error": "Model not loaded. Please try again later."}), 500
    
    # Get parameters
    ticker = request.args.get("ticker", "AAPL").upper()
    n_days = min(int(request.args.get("n_days", 7)), 30)
    n_days = max(n_days, 1)
    
    try:
        # Load data
        df, last_seq, last_price = load_and_prepare_data(
            ticker=ticker,
            start_date="2022-01-01",
            features=_artifact["features"],
            seq_len=_artifact["seq_len"]
        )
        
        if df is None:
            return jsonify({
                "error": f"Insufficient data for ticker {ticker}",
                "ticker": ticker
            }), 400
        
        # Generate forecast
        prices, returns = forecast_n_days(
            _model,
            last_seq,
            n_days,
            _artifact,
            _device,
            last_price
        )
        
        # Build response
        forecast_dates = [
            (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(n_days)
        ]
        
        forecast = []
        for i in range(n_days):
            forecast.append({
                "date": forecast_dates[i],
                "day": i + 1,
                "price": round(prices[i], 2),
                "return": round(returns[i], 6),
                "return_pct": round(returns[i] * 100, 4)
            })
        
        return jsonify({
            "ticker": ticker,
            "current_price": round(last_price, 2),
            "n_days": n_days,
            "forecast": forecast,
            "model_version": _artifact.get("version", "unknown")
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "ticker": ticker
        }), 500


# =========================================================
# APP INITIALIZATION
# =========================================================

def init_app():
    """
    Initialize the Flask app by loading the model.
    """
    global _model, _artifact, _device
    
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        _model, _artifact, _device = load_model()
        print(f"Model loaded successfully on device: {_device}")
        print(f"Model features: {_artifact.get('features', [])}")
        print(f"Sequence length: {_artifact.get('seq_len', 0)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Initialize on startup
init_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
