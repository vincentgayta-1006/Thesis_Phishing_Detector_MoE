"""
Neural network models
"""

import torch
import torch.nn as nn

class EnhancedGatingNetwork(nn.Module):
    """Improved gating network with better feature processing"""
    
    def __init__(self, input_size=14, hidden_size=128, num_experts=2, dropout=0.3):
        super(EnhancedGatingNetwork, self).__init__()
        
        # Feature processor with batch normalization
        self.feature_processor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts),
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """Forward pass"""
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        features = self.feature_processor(x)
        gate_output = self.gate(features)
        weights = self.softmax(gate_output)
        return weights