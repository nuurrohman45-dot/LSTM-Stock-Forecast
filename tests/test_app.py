"""Basic tests for the LSTM Stock Forecast application."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_import():
    """Test that config module can be imported."""
    from src.config import config
    assert config.TICKER is not None
    assert config.SEQ_LEN > 0


def test_models_import():
    """Test that models module can be imported."""
    from src.models.lstm import AttnLSTM
    assert AttnLSTM is not None


def test_data_loader_import():
    """Test that data loader can be imported."""
    from src.data.loader import load_data
    assert load_data is not None


def test_features_import():
    """Test that features module can be imported."""
    from src.features.features import create_features
    assert create_features is not None


def test_config_values():
    """Test that config values are valid."""
    from src.config import config
    
    assert config.SEQ_LEN >= 10
    assert config.HIDDEN_SIZE > 0
    assert config.NUM_LAYERS > 0
    assert config.EPOCHS > 0
    assert config.BATCH_SIZE > 0
    assert config.LEARNING_RATE > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
