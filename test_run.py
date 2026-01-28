#!/usr/bin/env python3
"""
Test script to verify the fix
"""

import sys
import os

# Test 1: Check if URLFeatures is properly defined
print("Test 1: Checking URLFeatures definition...")
from feature_extractor import URLFeatures
print(f"✓ URLFeatures class found: {URLFeatures}")

# Test 2: Check if it's patched into __main__
print("\nTest 2: Checking __main__ namespace...")
import __main__
if hasattr(__main__, 'URLFeatures'):
    print(f"✓ URLFeatures patched into __main__: {__main__.URLFeatures}")
else:
    print("✗ URLFeatures not in __main__")

# Test 3: Try to load the model
print("\nTest 3: Testing model loading...")
try:
    import joblib
    
    # Patch URLFeatures into __main__ first
    from feature_extractor import URLFeatures
    __main__.URLFeatures = URLFeatures
    
    # Now try to load
    if os.path.exists('models/url_expert.pkl'):
        model = joblib.load('models/url_expert.pkl')
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Model file not found (expected if you haven't placed it yet)")
        
except Exception as e:
    print(f"✗ Error loading model: {e}")

print("\nTest complete!")