#!/usr/bin/env python3
"""
Main entry point that ensures proper module loading for pickle compatibility
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: Patch URLFeatures into __main__ before importing anything else
# This is required for pickle to find the class when loading the model
from feature_extractor import URLFeatures
import __main__
__main__.URLFeatures = URLFeatures

# Now import and run the main application
from main import main

if __name__ == "__main__":
    main()