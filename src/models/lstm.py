"""
LSTM model module with Attention mechanism.

This module contains the AttnLSTM model architecture for stock prediction.
"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism for LSTM.
    
    This attention mechanism learns to weight different time steps
    in the LSTM output sequence.
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize the Temporal Attention layer.
        
        Args:
            hidden_size: Size of the LSTM hidden state
        """
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.
        
        Args:
            lstm_out: LSTM output of shape (batch, seq_len, hidden)
        
        Returns:
            Context vector of shape (batch, hidden)
        """
        # Compute attention scores
        scores = self.score(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        # Compute attention weights (softmax)
        weights = torch.softmax(scores, dim=1)  # stable softmax
        
        # Compute context vector as weighted sum
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        
        return context


class AttnLSTM(nn.Module):
    """
    LSTM model with Temporal Attention for stock prediction.
    
    This model uses an LSTM encoder followed by an attention mechanism
    to produce a context-aware representation of the input sequence.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        """
        Initialize the AttnLSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Temporal Attention layer
        self.attn = TemporalAttention(hidden_size=hidden_size)
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttnLSTM model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Output tensor of shape (batch,)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Apply attention mechanism
        context = self.attn(lstm_out)  # (batch, hidden)
        
        # Final prediction - keep as 2D tensor (batch, 1) to avoid shape issues
        output = self.fc(context)  # (batch, 1)
        
        return output


def create_model(input_size: int, hidden_size: int = 32, num_layers: int = 1) -> AttnLSTM:
    """
    Factory function to create an AttnLSTM model.
    
    Args:
        input_size: Number of input features
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
    
    Returns:
        AttnLSTM: The created model
    """
    return AttnLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
