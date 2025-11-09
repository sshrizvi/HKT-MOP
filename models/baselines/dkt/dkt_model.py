"""
Deep Knowledge Tracing (DKT) Model Implementation
Based on Piech et al., 2015

Author: Syed Shujaat Haider
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from preprocessing.dkt_format_adapter import DKTDataset

class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing model using LSTM.
    
    Architecture:
    - Input: One-hot encoded (exercise_id, outcome) tuples
    - LSTM: Processes sequential interactions
    - Output: Probability distribution over outcomes for each exercise
    
    Args:
        input_dim: Dimension of input (num_exercises * num_outcomes)
        hidden_dim: LSTM hidden state dimension
        num_exercises: Number of unique exercises
        num_outcomes: Number of outcome classes (2 for binary, K for multiclass)
        num_layers: Number of LSTM layers
        dropout: Dropout probability between LSTM layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_exercises: int,
        num_outcomes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super(DKTModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_exercises = num_exercises
        self.num_outcomes = num_outcomes
        self.num_layers = num_layers
        
        # LSTM layer for sequential modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer: maps LSTM hidden state to outcome predictions
        # Output shape: (batch, seq_len, num_exercises * num_outcomes)
        self.fc = nn.Linear(hidden_dim, num_exercises * num_outcomes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through DKT model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            seq_lens: Actual sequence lengths (batch_size,)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, num_exercises, num_outcomes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pack padded sequences if sequence lengths provided
        if seq_lens is not None:
            # Sort by sequence length (required by pack_padded_sequence)
            seq_lens_cpu = seq_lens.cpu()
            sorted_lens, sorted_idx = seq_lens_cpu.sort(descending=True)
            x_sorted = x[sorted_idx]
            
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x_sorted, 
                sorted_lens.clamp(min=1),  # Ensure at least length 1
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM forward pass
            packed_output, (h_n, c_n) = self.lstm(packed_input)
            
            # Unpack sequences
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
            
            # Unsort to original order
            _, unsorted_idx = sorted_idx.sort()
            lstm_out = lstm_out[unsorted_idx]
        else:
            # Standard LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Map to output predictions
        output = self.fc(lstm_out)  # (batch, seq_len, num_exercises * num_outcomes)
        
        # Reshape to separate exercises and outcomes
        output = output.view(batch_size, seq_len, self.num_exercises, self.num_outcomes)
        
        return output
    
    def predict_proba(
        self,
        x: torch.Tensor,
        target_exercises: torch.Tensor,
        seq_lens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get probability predictions for specific target exercises.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            target_exercises: One-hot encoded target exercises (batch_size, seq_len, num_exercises)
            seq_lens: Actual sequence lengths (batch_size,)
            
        Returns:
            Probability distribution over outcomes (batch_size, seq_len, num_outcomes)
        """
        # Forward pass
        output = self.forward(x, seq_lens)  # (batch, seq_len, num_exercises, num_outcomes)
        
        # Apply softmax over outcome dimension
        probs = F.softmax(output, dim=-1)  # (batch, seq_len, num_exercises, num_outcomes)
        
        # Select probabilities for target exercises
        # target_exercises: (batch, seq_len, num_exercises) one-hot
        # Expand for broadcasting: (batch, seq_len, num_exercises, 1)
        target_mask = target_exercises.unsqueeze(-1)
        
        # Multiply and sum over exercise dimension
        target_probs = (probs * target_mask).sum(dim=2)  # (batch, seq_len, num_outcomes)
        
        return target_probs
