"""
Hierarchical Knowledge Tracing (HKT) Model Implementation
Two-level hierarchical prediction: Compilation → Execution
Author: Syed Shujaat Haider
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class HierarchicalKTModel(nn.Module):
    """
    Hierarchical Knowledge Tracing model with two-level prediction structure.
    
    Architecture:
    - Shared LSTM Encoder: Processes student interaction sequences
    - Level 1 Head (Compilation): Binary classification (CE vs Compiled)
    - Level 2 Head (Execution): Multi-class classification (AC, WA, TLE, MLE, RE, PE, OE)
    
    The execution prediction is conditioned on successful compilation,
    reflecting the natural programming evaluation pipeline.
    
    Args:
        num_exercises: Number of unique programming exercises
        num_compilation_classes: Number of compilation outcomes (2: CE, COMPILED)
        num_execution_classes: Number of execution outcomes (7: AC, WA, TLE, MLE, RE, PE, OE)
        embedding_dim: Dimension of exercise and outcome embeddings
        hidden_dim: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_attention: Whether to use attention mechanism over LSTM outputs
    """
    
    def __init__(
        self,
        num_exercises: int,
        num_compilation_classes: int = 2,
        num_execution_classes: int = 7,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = False
    ):
        super(HierarchicalKTModel, self).__init__()
        
        self.num_exercises = num_exercises
        self.num_compilation_classes = num_compilation_classes
        self.num_execution_classes = num_execution_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Exercise embedding layer
        self.exercise_embedding = nn.Embedding(
            num_embeddings=num_exercises,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Compilation outcome embedding (2 classes: CE=0, COMPILED=1)
        self.compilation_embedding = nn.Embedding(
            num_embeddings=num_compilation_classes,
            embedding_dim=embedding_dim // 2
        )
        
        # Execution outcome embedding (7 classes + 1 for N/A when CE)
        self.execution_embedding = nn.Embedding(
            num_embeddings=num_execution_classes + 1,  # +1 for padding/invalid
            embedding_dim=embedding_dim // 2,
            padding_idx=num_execution_classes  # Use last index as padding
        )
        
        # Input dimension: exercise + compilation + execution embeddings
        input_dim = embedding_dim + (embedding_dim // 2) + (embedding_dim // 2)
        
        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Level 1: Compilation Head (Binary Classification)
        self.compilation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_exercises * num_compilation_classes)
        )
        
        # Level 2: Execution Head (Multi-class Classification)
        # Takes LSTM output + compilation prediction as input
        self.execution_head = nn.Sequential(
            nn.Linear(hidden_dim + num_compilation_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_exercises * num_execution_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    
    def forward(
        self,
        exercise_ids: torch.Tensor,
        compilation_labels: torch.Tensor,
        execution_labels: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.
        
        Args:
            exercise_ids: Exercise indices (batch_size, seq_len)
            compilation_labels: Compilation outcomes at time t (batch_size, seq_len)
            execution_labels: Execution outcomes at time t (batch_size, seq_len)
                             -1 for CE cases (not applicable)
            seq_lens: Actual sequence lengths (batch_size,)
        
        Returns:
            Dictionary containing:
            - 'compilation_logits': (batch, seq_len, num_exercises, num_compilation_classes)
            - 'execution_logits': (batch, seq_len, num_exercises, num_execution_classes)
            - 'lstm_hidden': (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = exercise_ids.shape
        
        # Embed inputs
        exercise_emb = self.exercise_embedding(exercise_ids)  # (batch, seq_len, emb_dim)
        compilation_emb = self.compilation_embedding(compilation_labels)  # (batch, seq_len, emb_dim//2)
        
        # Handle execution labels: replace -1 with padding index
        execution_labels_valid = execution_labels.clone()
        execution_labels_valid[execution_labels == -1] = self.num_execution_classes
        execution_emb = self.execution_embedding(execution_labels_valid)  # (batch, seq_len, emb_dim//2)
        
        # Concatenate embeddings
        combined_emb = torch.cat([exercise_emb, compilation_emb, execution_emb], dim=-1)
        
        # LSTM encoding
        if seq_lens is not None:
            # Pack padded sequences
            seq_lens_cpu = seq_lens.cpu()
            sorted_lens, sorted_idx = seq_lens_cpu.sort(descending=True)
            combined_emb_sorted = combined_emb[sorted_idx]
            
            packed_input = nn.utils.rnn.pack_padded_sequence(
                combined_emb_sorted,
                sorted_lens.clamp(min=1),
                batch_first=True,
                enforce_sorted=True
            )
            
            packed_output, (h_n, c_n) = self.lstm(packed_input)
            
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
            
            # Unsort to original order
            _, unsorted_idx = sorted_idx.sort()
            lstm_out = lstm_out[unsorted_idx]
        else:
            lstm_out, (h_n, c_n) = self.lstm(combined_emb)
        
        # Optional attention mechanism
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attended_out  # Residual connection
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Level 1: Compilation Prediction
        compilation_logits = self.compilation_head(lstm_out)  # (batch, seq_len, num_ex * num_comp_classes)
        compilation_logits = compilation_logits.view(
            batch_size, seq_len, self.num_exercises, self.num_compilation_classes
        )
        
        # Get compilation probabilities for conditioning
        compilation_probs = F.softmax(compilation_logits, dim=-1)  # (batch, seq_len, num_ex, num_comp_classes)
        
        # Aggregate compilation probabilities across exercises (weighted by attention if needed)
        # For simplicity, we take max pool across exercises
        compilation_probs_agg, _ = torch.max(compilation_probs, dim=2)  # (batch, seq_len, num_comp_classes)
        
        # Level 2: Execution Prediction (conditioned on compilation)
        # Concatenate LSTM output with compilation probabilities
        execution_input = torch.cat([lstm_out, compilation_probs_agg], dim=-1)
        execution_logits = self.execution_head(execution_input)  # (batch, seq_len, num_ex * num_exec_classes)
        execution_logits = execution_logits.view(
            batch_size, seq_len, self.num_exercises, self.num_execution_classes
        )
        
        return {
            'compilation_logits': compilation_logits,
            'execution_logits': execution_logits,
            'lstm_hidden': lstm_out
        }
    
    
    def predict(
        self,
        exercise_ids: torch.Tensor,
        compilation_labels: torch.Tensor,
        execution_labels: torch.Tensor,
        target_exercises: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions for specific target exercises.
        
        Args:
            exercise_ids: Exercise indices (batch_size, seq_len)
            compilation_labels: Compilation outcomes (batch_size, seq_len)
            execution_labels: Execution outcomes (batch_size, seq_len)
            target_exercises: Target exercise indices (batch_size, seq_len)
            seq_lens: Actual sequence lengths (batch_size,)
        
        Returns:
            Dictionary containing:
            - 'compilation_probs': (batch, seq_len, num_compilation_classes)
            - 'execution_probs': (batch, seq_len, num_execution_classes)
        """
        # Forward pass
        outputs = self.forward(exercise_ids, compilation_labels, execution_labels, seq_lens)
        
        compilation_logits = outputs['compilation_logits']  # (batch, seq_len, num_ex, num_comp_classes)
        execution_logits = outputs['execution_logits']  # (batch, seq_len, num_ex, num_exec_classes)
        
        batch_size, seq_len = target_exercises.shape
        
        # Create one-hot encoding for target exercises
        target_exercises_one_hot = F.one_hot(
            target_exercises,
            num_classes=self.num_exercises
        ).float()  # (batch, seq_len, num_ex)
        
        # Select logits for target exercises
        target_mask = target_exercises_one_hot.unsqueeze(-1)  # (batch, seq_len, num_ex, 1)
        
        compilation_logits_target = (compilation_logits * target_mask).sum(dim=2)  # (batch, seq_len, num_comp)
        execution_logits_target = (execution_logits * target_mask).sum(dim=2)  # (batch, seq_len, num_exec)
        
        # Convert to probabilities
        compilation_probs = F.softmax(compilation_logits_target, dim=-1)
        execution_probs = F.softmax(execution_logits_target, dim=-1)
        
        return {
            'compilation_probs': compilation_probs,
            'execution_probs': execution_probs
        }
    
    
    def get_hierarchical_prediction(
        self,
        compilation_probs: torch.Tensor,
        execution_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Combine hierarchical predictions into final outcome probabilities.
        
        P(outcome) = P(compile) * P(outcome | compile) if outcome != CE
                   = P(CE) if outcome == CE
        
        Args:
            compilation_probs: (batch, seq_len, 2) - [P(CE), P(COMPILED)]
            execution_probs: (batch, seq_len, 7) - [P(AC), P(WA), ...]
        
        Returns:
            Dictionary with combined probabilities
        """
        # P(CE) = first column of compilation_probs
        prob_ce = compilation_probs[..., 0:1]  # (batch, seq_len, 1)
        
        # P(COMPILED) = second column of compilation_probs
        prob_compiled = compilation_probs[..., 1:2]  # (batch, seq_len, 1)
        
        # P(execution_outcome | compiled) * P(compiled)
        prob_execution_outcomes = execution_probs * prob_compiled  # (batch, seq_len, 7)
        
        # Combine: [P(CE), P(AC|compiled)*P(compiled), P(WA|compiled)*P(compiled), ...]
        combined_probs = torch.cat([prob_ce, prob_execution_outcomes], dim=-1)  # (batch, seq_len, 8)
        
        return {
            'combined_probs': combined_probs,
            'prob_ce': prob_ce,
            'prob_compiled': prob_compiled
        }


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss function for two-level predictions.
    
    Combines compilation loss and execution loss with adaptive weighting.
    Execution loss is only computed for samples that compiled successfully.
    """
    
    def __init__(
        self,
        lambda_compilation: float = 1.0,
        lambda_execution: float = 1.0,
        adaptive_weighting: bool = True
    ):
        """
        Initialize hierarchical loss.
        
        Args:
            lambda_compilation: Weight for compilation loss
            lambda_execution: Weight for execution loss
            adaptive_weighting: Whether to adapt weights based on class imbalance
        """
        super(HierarchicalLoss, self).__init__()
        
        self.lambda_compilation = lambda_compilation
        self.lambda_execution = lambda_execution
        self.adaptive_weighting = adaptive_weighting
        
        self.compilation_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.execution_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    
    def forward(
        self,
        compilation_logits: torch.Tensor,
        execution_logits: torch.Tensor,
        compilation_targets: torch.Tensor,
        execution_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss.
        
        Args:
            compilation_logits: (batch, seq_len, num_compilation_classes)
            execution_logits: (batch, seq_len, num_execution_classes)
            compilation_targets: (batch, seq_len)
            execution_targets: (batch, seq_len) - -1 for CE cases
            mask: (batch, seq_len) - boolean mask for valid positions
        
        Returns:
            Dictionary with total loss and component losses
        """
        # Flatten tensors
        compilation_logits_flat = compilation_logits.view(-1, compilation_logits.size(-1))
        execution_logits_flat = execution_logits.view(-1, execution_logits.size(-1))
        compilation_targets_flat = compilation_targets.view(-1)
        execution_targets_flat = execution_targets.view(-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            compilation_logits_flat = compilation_logits_flat[mask_flat]
            execution_logits_flat = execution_logits_flat[mask_flat]
            compilation_targets_flat = compilation_targets_flat[mask_flat]
            execution_targets_flat = execution_targets_flat[mask_flat]
        
        # Compilation loss (all samples)
        compilation_loss = self.compilation_criterion(
            compilation_logits_flat,
            compilation_targets_flat
        )
        
        # Execution loss (only for compiled samples)
        # execution_targets == -1 for CE cases
        execution_loss = self.execution_criterion(
            execution_logits_flat,
            execution_targets_flat
        )
        
        # Adaptive weighting based on number of valid samples
        if self.adaptive_weighting:
            num_compiled = (execution_targets_flat != -1).sum().item()
            total_samples = len(execution_targets_flat)
            
            if num_compiled > 0 and total_samples > 0:
                # Weight execution loss based on proportion of compiled samples
                adaptive_exec_weight = self.lambda_execution * (total_samples / num_compiled)
            else:
                adaptive_exec_weight = self.lambda_execution
        else:
            adaptive_exec_weight = self.lambda_execution
        
        # Total loss
        total_loss = (
            self.lambda_compilation * compilation_loss +
            adaptive_exec_weight * execution_loss
        )
        
        return {
            'total_loss': total_loss,
            'compilation_loss': compilation_loss,
            'execution_loss': execution_loss,
            'lambda_compilation': self.lambda_compilation,
            'lambda_execution': adaptive_exec_weight
        }


# Helper function to create model from config
def create_hierarchical_model(config: dict, vocab_data: dict) -> HierarchicalKTModel:
    """
    Create hierarchical model from configuration and vocabulary.
    
    Args:
        config: Model configuration dictionary
        vocab_data: Vocabulary data from hierarchical format adapter
    
    Returns:
        HierarchicalKTModel instance
    """
    model = HierarchicalKTModel(
        num_exercises=vocab_data['num_exercises'],
        num_compilation_classes=vocab_data['num_compilation_classes'],
        num_execution_classes=vocab_data['num_execution_classes'],
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        use_attention=config.get('use_attention', False)
    )
    
    return model
