"""
Utility functions
"""

import yaml
import json
import logging
from typing import Dict

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract and flatten config
        model_paths = config.get('model_paths', {})
        parameters = config.get('parameters', {})
        
        # Combine into single dict with correct field names
        model_config = {
            'url_model_path': model_paths.get('url_expert', 'models/url_expert.pkl'),
            'text_model_path': model_paths.get('text_model', 'models/distilbert_phishing_model'),
            'gating_network_path': model_paths.get('gating_network', 'models/gating_network.pth'),
            'max_text_length': parameters.get('max_text_length', 128),
            'phishing_threshold': parameters.get('phishing_threshold', 0.6),
            'confidence_threshold': parameters.get('confidence_threshold', 0.8),
            'device': parameters.get('device', 'cpu'),
        }
        
        return model_config
        
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return {
            'url_model_path': 'models/url_expert.pkl',
            'text_model_path': 'models/distilbert_phishing_model',
            'gating_network_path': 'models/gating_network.pth',
            'max_text_length': 128,
            'phishing_threshold': 0.6,
            'confidence_threshold': 0.8,
            'device': 'cpu'
        }
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        raise

def setup_logging(log_file: str = "phishing_detector.log") -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("phishing_detector")
    
    # Remove existing handlers
    logger.handlers = []
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def save_results(results: Dict, output_file: str = "results.json"):
    """Save prediction results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")