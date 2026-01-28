"""
Configuration management
"""

from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    """Configuration for model paths and hyperparameters"""
    
    # Model paths
    url_model_path: str = "models/url_expert.pkl"
    text_model_path: str = "models/distilbert_phishing_model"
    gating_network_path: str = "models/gating_network.pth"
    
    # Hyperparameters
    max_text_length: int = 128
    phishing_threshold: float = 0.6
    confidence_threshold: float = 0.8
    device: str = "cpu"
    
    def __post_init__(self):
        """Validate paths after initialization"""
        self._validate_paths()
    
    def _validate_paths(self):
        """Check if model files exist"""
        if not os.path.exists(self.url_model_path):
            print(f"Warning: URL model not found at {self.url_model_path}")
        
        if not os.path.exists(self.gating_network_path):
            print(f"Warning: Gating network not found at {self.gating_network_path}")
        
        if not os.path.exists(self.text_model_path):
            print(f"Warning: Text model directory not found at {self.text_model_path}")