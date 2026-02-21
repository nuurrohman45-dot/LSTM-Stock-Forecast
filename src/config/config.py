"""
Configuration file for LSTM-based stock prediction model.

This module contains all hyperparameters and configuration settings
used across the project modules.
"""

# =============================================================================
# Data Configuration
# =============================================================================
TICKER = "AAPL"  # Stock symbol
START_DATE = "2022-01-01"  # Start date for historical data
END_DATE = None  # End date (None = present)

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
FEATURE_COLS = [
    "volatility_10",
    "volatility_30",
    "momentum_10",
    "momentum_30",
    "volume_z",
]

TARGET_COL = "target"

# Sequence length for LSTM input
SEQ_LEN = 60

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_TYPE = "AttnLSTM"  # Model architecture
INPUT_SIZE = len(FEATURE_COLS)  # Number of input features
HIDDEN_SIZE = 64  # LSTM hidden size
NUM_LAYERS = 2  # Number of LSTM layers

# =============================================================================
# Training Configuration
# =============================================================================
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_VALUE = 1.0

# =============================================================================
# Data Split Configuration
# =============================================================================
TRAIN_SPLIT = 0.7  # Train/test split ratio

# =============================================================================
# Walk-Forward Validation Configuration
# =============================================================================
WF_TRAIN_DAYS = 504  # Training window (in days)
WF_TEST_DAYS = 63  # Test window (in days)
WF_TRADE_START = 60  # Starting point for trading

# =============================================================================
# Portfolio Configuration
# =============================================================================
VOL_TARGET = 0.01  # Target volatility (1% daily)
COST_PER_TRADE = 0.0005  # Transaction cost per trade (5 bps)
EPSILON = 1e-8  # Small value to avoid division by zero

# Position sizing
PREDICTION_SMOOTHING_WINDOW = 3
CONFIDENCE_GATE_PERCENTILE = 60

# =============================================================================
# Regime Configuration
# =============================================================================
VOL_WINDOW = 20  # Rolling window for volatility
VOL_MEDIAN_WINDOW = 60  # Window for median volatility
VOL_THRESHOLD_MULTIPLIER = 1.2  # Volatility threshold
MA_WINDOW = 50  # Moving average window
TREND_STRENGTH_THRESHOLD = 0.0075  # Trend strength threshold

# =============================================================================
# Paths Configuration
# =============================================================================
MODEL_DIR = "artifacts/models"
MODEL_FILENAME = "lstm_v1.pt"
MLRUNS_DIR = "mlruns"

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def get_config():
    """
    Returns a dictionary of all configuration settings.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "seq_len": SEQ_LEN,
        "model_type": MODEL_TYPE,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_value": GRADIENT_CLIP_VALUE,
        "train_split": TRAIN_SPLIT,
        "wf_train_days": WF_TRAIN_DAYS,
        "wf_test_days": WF_TEST_DAYS,
        "wf_trade_start": WF_TRADE_START,
        "vol_target": VOL_TARGET,
        "cost_per_trade": COST_PER_TRADE,
        "epsilon": EPSILON,
        "prediction_smoothing_window": PREDICTION_SMOOTHING_WINDOW,
        "confidence_gate_percentile": CONFIDENCE_GATE_PERCENTILE,
        "vol_window": VOL_WINDOW,
        "vol_median_window": VOL_MEDIAN_WINDOW,
        "vol_threshold_multiplier": VOL_THRESHOLD_MULTIPLIER,
        "ma_window": MA_WINDOW,
        "trend_strength_threshold": TREND_STRENGTH_THRESHOLD,
        "model_dir": MODEL_DIR,
        "model_filename": MODEL_FILENAME,
        "mlruns_dir": MLRUNS_DIR,
        "device": DEVICE,
    }
