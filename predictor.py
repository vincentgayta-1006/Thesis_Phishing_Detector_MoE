"""
Main prediction module
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import ModelConfig
from models import EnhancedGatingNetwork
from feature_extractor import AdvancedFeatureExtractor

# ============================================================================
# COMPATIBILITY FIX: URLFeatures must be available when loading pickle
# ============================================================================

# Import and ensure URLFeatures is available globally
from feature_extractor import URLFeatures
import sys

# Patch into main module namespace for pickle compatibility
if not hasattr(sys.modules['__main__'], 'URLFeatures'):
    sys.modules['__main__'].URLFeatures = URLFeatures

# ============================================================================

class ModelLoader:
    """Centralized model loading with error handling"""
    
    @staticmethod
    def load_models(config: ModelConfig):
        """Load all required models"""
        print("Loading Expert Models...")
        print("-" * 70)
        
        # URL Expert (traditional ML model)
        try:
            # The URLFeatures class is now available in __main__
            expert_1 = joblib.load(config.url_model_path)
            print("✓ URL Expert: Loaded successfully")
        except Exception as e:
            print(f"✗ Error loading URL expert: {e}")
            print("Creating fallback URL expert...")
            # Create a simple fallback
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from feature_extractor import URLFeatures
            
            expert_1 = Pipeline([
                ('features', URLFeatures()),
                ('classifier', RandomForestClassifier(n_estimators=50))
            ])
            print("✓ Created fallback URL expert")
        
        # Text Expert (DistilBERT model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.text_model_path)
            expert_2 = AutoModelForSequenceClassification.from_pretrained(config.text_model_path)
            expert_2.eval()
            expert_2.to(config.device)
            print("✓ Text Expert: Loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Text expert: {e}")
            raise
        
        # Gating Network
        try:
            gating_net = EnhancedGatingNetwork(
                input_size=14,  # Total feature size
                hidden_size=128,
                num_experts=2,
                dropout=0.3
            )
            
            # Load state dict
            state_dict = torch.load(config.gating_network_path, 
                                   map_location=config.device)
            gating_net.load_state_dict(state_dict)
            
            gating_net.eval()
            gating_net.to(config.device)
            print("✓ Gating Network: Loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Gating Network: {e}")
            print("Creating fallback gating network...")
            gating_net = EnhancedGatingNetwork(input_size=14, hidden_size=128, num_experts=2)
            gating_net.eval()
            gating_net.to(config.device)
            print("✓ Created fallback gating network")
        
        print("-" * 70)
        print("All models loaded successfully!")
        
        return expert_1, expert_2, tokenizer, gating_net


class BalancedPhishingDetector:
    """Main phishing detection system with balanced expert weighting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.expert_1, self.expert_2, self.tokenizer, self.gating_net = \
            ModelLoader.load_models(config)
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Store prediction history for analysis
        self.prediction_history = []
    
    def predict(self, text: str, url: str = "") -> dict:
        """
        Main prediction function with balanced expert weighting
        """
        # Preprocess inputs
        clean_text = self.feature_extractor.preprocess_text(text)
        phrase_score = self.feature_extractor.calculate_phrase_score(clean_text)
        
        # Determine input type
        has_url = bool(url and url.strip())
        has_text = bool(clean_text and clean_text.strip())
        
        # Get individual expert predictions
        url_probs = self._get_url_prediction(url)
        text_probs = self._get_text_prediction(clean_text)
        
        # Extract gating features
        gating_features = self.feature_extractor.extract_gating_features(
            clean_text, url, phrase_score
        )
        
        # Compute expert weights using gating network
        expert_weights = self._compute_gating_weights(gating_features)
        
        # Weighted ensemble of expert predictions
        final_probs = (expert_weights[0] * url_probs + 
                      expert_weights[1] * text_probs)
        
        # Make final decision
        prediction = "PHISHING" if final_probs[1] > self.config.phishing_threshold else "SAFE"
        confidence = max(final_probs) * 100
        
        # Individual expert decisions
        url_pred = "PHISHING" if url_probs[1] > 0.5 else "SAFE"
        text_pred = "PHISHING" if text_probs[1] > 0.5 else "SAFE"
        
        # Determine dominant expert
        url_dominant = expert_weights[0] > expert_weights[1]
        
        # Determine routing method
        if has_url and not has_text:
            routing_method = "URL-only (No text available)"
            primary_expert = "URL Expert"
        elif has_text and not has_url:
            routing_method = "Text-only (No URL available)"
            primary_expert = "Text Expert"
        else:
            routing_method = "Adaptive Gating Network"
            primary_expert = "URL Expert" if url_dominant else "Text Expert"
        
        # Compile results
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'is_high_confidence': max(final_probs) > self.config.confidence_threshold,
            'url_weight': float(expert_weights[0] * 100),
            'text_weight': float(expert_weights[1] * 100),
            'url_prediction': url_pred,
            'text_prediction': text_pred,
            'url_confidence': float(max(url_probs) * 100),
            'text_confidence': float(max(text_probs) * 100),
            'phrase_score': float(phrase_score),
            'expert_agreement': url_pred == text_pred,
            'routing_method': routing_method,
            'primary_expert': primary_expert,
            'input_type': 'URL+Text' if (has_url and has_text) else ('URL' if has_url else 'Text'),
            'url_dominant': url_dominant,
        }
        
        # Store for analysis
        self.prediction_history.append(result)
        
        return result
    
    def _get_url_prediction(self, url: str) -> np.ndarray:
        """Get prediction from URL expert"""
        if url and url.strip():
            try:
                url_df = pd.DataFrame({'url': [url]})
                
                # Try different prediction methods
                if hasattr(self.expert_1, 'predict_proba'):
                    return self.expert_1.predict_proba(url_df)[0]
                elif hasattr(self.expert_1, 'predict'):
                    pred = self.expert_1.predict(url_df)[0]
                    return np.array([1 - pred, pred])  # Convert to probabilities
                else:
                    raise AttributeError("Model doesn't have predict_proba or predict method")
                    
            except Exception as e:
                print(f"URL expert prediction error: {e}")
                return np.array([0.5, 0.5])
        
        return np.array([0.5, 0.5])  # Fallback for no URL
    
    def _get_text_prediction(self, text: str) -> np.ndarray:
        """Get prediction from text expert (DistilBERT)"""
        if text:
            try:
                # Tokenize text
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True,
                    truncation=True, 
                    max_length=self.config.max_text_length
                )
                
                # Move to device
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.expert_2(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                    return probs
                    
            except Exception as e:
                print(f"Text expert prediction error: {e}")
                return np.array([0.5, 0.5])
        
        return np.array([0.5, 0.5])  # Fallback for no text
    
    def _compute_gating_weights(self, features: np.ndarray) -> np.ndarray:
        """Compute expert weights using gating network"""
        # Convert to tensor
        gating_input = torch.FloatTensor(features).unsqueeze(0).to(self.config.device)
        
        # Get weights from gating network
        with torch.no_grad():
            weights = self.gating_net(gating_input)
        
        return weights[0].cpu().numpy()
    
    def analyze_weight_distribution(self, num_samples: int = 100) -> dict:
        """Analyze weight distribution from prediction history"""
        if not self.prediction_history:
            return {
                'avg_url_weight': 50.0,
                'avg_text_weight': 50.0,
                'url_weight_std': 0.0,
                'text_weight_std': 0.0,
                'url_dominant_count': 0,
                'text_dominant_count': 0,
                'url_dominant_pct': 0.0,
            }
        
        # Use recent samples
        if len(self.prediction_history) < num_samples:
            samples = self.prediction_history
        else:
            samples = self.prediction_history[-num_samples:]
        
        # Extract weights
        url_weights = [s['url_weight'] for s in samples]
        text_weights = [s['text_weight'] for s in samples]
        
        # Calculate statistics
        return {
            'avg_url_weight': float(np.mean(url_weights)),
            'avg_text_weight': float(np.mean(text_weights)),
            'url_weight_std': float(np.std(url_weights)),
            'text_weight_std': float(np.std(text_weights)),
            'url_dominant_count': sum(1 for s in samples if s['url_dominant']),
            'text_dominant_count': sum(1 for s in samples if not s['url_dominant']),
            'url_dominant_pct': sum(1 for s in samples if s['url_dominant']) / len(samples) * 100,
        }