"""
Simple wrapper to extract GC features from the features/ directory.
This provides a simple interface without needing the external Genre_Classifier library.
"""

from .gc_extractor import add_gc_features, get_gc_extractor

__all__ = ['add_gc_features', 'get_gc_extractor']
