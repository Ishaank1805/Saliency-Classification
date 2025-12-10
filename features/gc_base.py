"""
Base classes for Genre Classifier features.
Simplified version without external dependencies.
"""

import warnings
import logging
import os

warnings.filterwarnings('ignore')

try:
    import stanza
except ImportError:
    stanza = None

try:
    import spacy
except ImportError:
    spacy = None


class BaseFeatureExtractor:
    """Base class for all GC feature extractors"""
    
    def __init__(self):
        self.nlp_stanza = None
        self.nlp_spacy = None
        self._initialized = False
    
    def initialize_nlp(self):
        """Initialize NLP models (lazy loading)"""
        if not self._initialized:
            if stanza is not None:
                try:
                    logging.getLogger('stanza').setLevel(logging.ERROR)
                    os.environ['STANZA_RESOURCES_DIR'] = os.path.expanduser('~/stanza_resources')
                    self.nlp_stanza = stanza.Pipeline(
                        'en', 
                        processors='tokenize,pos,lemma,depparse',
                        download_method=None,
                        verbose=False,
                        logging_level='ERROR'
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize Stanza: {e}")
                    self.nlp_stanza = None
            
            if spacy is not None:
                try:
                    self.nlp_spacy = spacy.load('en_core_web_sm')
                except Exception as e:
                    print(f"Warning: Could not initialize spaCy: {e}")
                    self.nlp_spacy = None
            
            self._initialized = True
    
    def extract(self, text):
        """Extract features from text. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement extract method")
