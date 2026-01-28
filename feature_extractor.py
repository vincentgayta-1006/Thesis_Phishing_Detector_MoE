"""
Feature extraction utilities
"""

import re
import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

# ============================================================================
# URLFeatures class - MUST match exactly what was used in the original notebook
# ============================================================================

class URLFeatures(BaseEstimator, TransformerMixin):
    """Feature extractor for URL-based phishing detection"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, urls):
        """Transform URLs to feature matrix"""
        urls = np.array(urls).reshape(-1)
        feats = np.array([self._extract_features(u) for u in urls])
        return csr_matrix(feats)
    
    def _extract_features(self, url: str) -> List[float]:
        """Extract URL features"""
        if not url or pd.isna(url):
            return [0] * 8
        
        try:
            return [
                len(url),
                url.count('-'),
                url.count('@'),
                url.count('?'),
                url.count('='),
                url.count('.'),
                int(url.startswith("https")),
                int(url.count("//") > 1),
            ]
        except Exception:
            return [0] * 8

# ============================================================================

class AdvancedFeatureExtractor:
    """Enhanced feature extraction with balanced input representation"""
    
    # Phishing phrases with weights
    PHRASE_DICT = {
        'urgent': 0.6, 'immediately': 0.6, 'act now': 0.6, 'limited time': 0.4,
        'expires today': 0.6, 'last chance': 0.6,
        'verify account': 0.7, 'suspended': 0.6, 'confirm your': 0.6,
        'update account': 0.6, 'security alert': 0.7, 'unusual activity': 0.6,
        'verify identity': 0.7, 'locked': 0.6, 'restricted': 0.5,
        'congratulations': 0.5, 'winner': 0.7, 'claim': 0.6, 'prize': 0.5,
        'free money': 0.6, 'cash prize': 0.6, 'refund': 0.4, 'bonus': 0.3,
        'click here': 0.5, 'click now': 0.5, 'download': 0.3, 'open attachment': 0.4,
        'confirm': 0.3, 'validate': 0.4, 'reactivate': 0.5,
        'free': 0.5, 'offer': 0.2, 'deal': 0.2, 'discount': 0.2,
    }
    
    # Suspicious TLDs
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click', '.link']
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        return text
    
    @classmethod
    def calculate_phrase_score(cls, text: str) -> float:
        """Calculate phishing phrase score"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Sum weights for matching phrases
        for phrase, weight in cls.PHRASE_DICT.items():
            if phrase in text_lower:
                score += weight
        
        # Normalize by text length
        words = text_lower.split()
        if len(words) > 0:
            score = score / (1 + np.log(len(words)))
        
        return min(score, 1.0)
    
    @classmethod
    def extract_url_quality_features(cls, url: str) -> List[float]:
        """Extract URL quality indicators"""
        if not url or pd.isna(url) or url == "":
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            url_lower = url.lower()
            
            # Feature 1: URL length (normalized)
            url_length = min(len(url) / 100.0, 1.0)
            
            # Feature 2: Suspicious patterns
            suspicious_patterns = sum([
                bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)),  # IP address
                any(tld in url_lower for tld in cls.SUSPICIOUS_TLDS),
                url.count('@') > 0,  # Email in URL
                url.count('-') > 3,   # Many hyphens
            ]) / 4.0
            
            # Feature 3: Legitimate structure
            has_protocol = url_lower.startswith('http')
            has_domain = bool(re.search(r'\.(com|org|net|edu|gov)', url_lower))
            looks_legit = float(has_protocol and has_domain)
            
            # Feature 4: Structure complexity
            dot_count = min(url.count('.') / 5.0, 1.0)
            slash_count = min(url.count('/') / 10.0, 1.0)
            structure_complexity = (dot_count + slash_count) / 2.0
            
            return [url_length, suspicious_patterns, looks_legit, structure_complexity]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    @classmethod
    def extract_text_quality_features(cls, text: str) -> List[float]:
        """Extract text quality indicators"""
        if not text or pd.isna(text) or text == "":
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        try:
            # Feature 1: Text length (normalized)
            text_length = min(len(text) / 500.0, 1.0)
            
            # Feature 2: Word count (normalized)
            words = text.split()
            word_count = min(len(words) / 50.0, 1.0)
            
            # Feature 3: Vocabulary richness
            unique_words = len(set(words))
            vocab_richness = unique_words / max(len(words), 1)
            
            # Feature 4: Special character density
            special_chars = len(re.findall(r'[^\w\s]', text))
            special_density = min(special_chars / max(len(text), 1), 0.5)
            
            # Feature 5: Capitalization ratio
            capital_count = sum(1 for c in text if c.isupper())
            capital_ratio = min(capital_count / max(len(text), 1), 0.5)
            
            # Feature 6: Average word length (coherence)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            coherence = min(avg_word_length / 10.0, 1.0)
            
            return [text_length, word_count, vocab_richness, 
                   special_density, capital_ratio, coherence]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    @classmethod
    def extract_gating_features(cls, text: str, url: str, phrase_score: float) -> np.ndarray:
        """
        Extract features for gating network decision
        Total features: 14
          2 binary (has_url, has_text)
          1 phrase score
          1 url-text ratio
          6 text features
          4 url features
        """
        # Text quality features
        text_features = cls.extract_text_quality_features(text)
        
        # URL quality features
        url_features = cls.extract_url_quality_features(url)
        
        # Presence flags
        has_url = 1.0 if (url and not pd.isna(url) and url.strip()) else 0.0
        has_text = 1.0 if (text and not pd.isna(text) and text.strip()) else 0.0
        
        # URL-text length ratio
        url_text_ratio = 0.0
        if has_url and has_text:
            url_length = len(url) if url else 0
            text_length = len(text) if text else 0
            total_length = url_length + text_length
            if total_length > 0:
                url_text_ratio = url_length / total_length
        
        # Combine all features
        features = np.array([
            has_url,                    # Binary: URL present
            has_text,                   # Binary: Text present
            phrase_score,               # Phishing phrase score
            url_text_ratio,             # URL-text length ratio
            *text_features,             # 6 text quality features
            *url_features,              # 4 URL quality features
        ], dtype=np.float32)
        
        return features