"""
Main Genre Classifier Feature Extractor
Orchestrates all GC feature extractors without external Genre_Classifier dependency
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm.auto import tqdm

# Import all feature extractors from features directory
from .gc_basic import BasicFeatureExtractor
from .gc_char_diversity import CharDiversityFeatureExtractor
from .gc_pos import POSFeatureExtractor
from .gc_syntax import SyntaxFeatureExtractor
from .gc_dialogue import DialogueFeatureExtractor
from .gc_pronouns import PronounFeatureExtractor
from .gc_temporal import TemporalFeatureExtractor
from .gc_narrative import NarrativeFeatureExtractor
from .gc_academic import AcademicFeatureExtractor
from .gc_punctuation import PunctuationFeatureExtractor
from .gc_discourse import DiscourseFeatureExtractor
from .gc_readability import ReadabilityFeatureExtractor
from .gc_polarity import PolarityFeatureExtractor
from .gc_concreteness import ConcretenessFeatureExtractor


class GCFeatureExtractor:
    """Main feature extractor that combines all GC feature groups"""
    
    # Map of feature group names to extractor classes
    EXTRACTORS = {
        'basic': BasicFeatureExtractor,
        'char_diversity': CharDiversityFeatureExtractor,
        'pos': POSFeatureExtractor,
        'syntax': SyntaxFeatureExtractor,
        'dialogue': DialogueFeatureExtractor,
        'pronouns': PronounFeatureExtractor,
        'temporal': TemporalFeatureExtractor,
        'narrative': NarrativeFeatureExtractor,
        'academic': AcademicFeatureExtractor,
        'punctuation': PunctuationFeatureExtractor,
        'discourse': DiscourseFeatureExtractor,
        'readability': ReadabilityFeatureExtractor,
        'polarity': PolarityFeatureExtractor,
        'concreteness': ConcretenessFeatureExtractor,
    }
    
    def __init__(self, feature_groups='all', 
                 polarity_lexicon_path=None,
                 concreteness_lexicon_path=None):
        """
        Initialize the feature extractor with specified feature groups.
        
        Parameters:
        -----------
        feature_groups : list or str
            List of feature groups to extract. Use 'all' for all features.
            Available groups: basic, char_diversity, pos, syntax, dialogue,
                            pronouns, temporal, narrative, academic, punctuation,
                            discourse, readability, polarity, concreteness
        polarity_lexicon_path : str, optional
            Path to polarity lexicon file (not required - features will be 0 if missing)
        concreteness_lexicon_path : str, optional
            Path to concreteness lexicon file (not required - features will be 0 if missing)
        """
        # Determine which groups to use
        if feature_groups == 'all':
            self.feature_groups = list(self.EXTRACTORS.keys())
        else:
            self.feature_groups = feature_groups if isinstance(feature_groups, list) else [feature_groups]
        
        # Initialize extractors for selected groups
        self.extractors = {}
        for group in self.feature_groups:
            if group not in self.EXTRACTORS:
                print(f"Warning: Unknown feature group '{group}', skipping")
                continue
            
            # Special handling for lexicon-based extractors
            if group == 'polarity' and polarity_lexicon_path:
                self.extractors[group] = PolarityFeatureExtractor(lexicon_path=polarity_lexicon_path)
            elif group == 'concreteness' and concreteness_lexicon_path:
                self.extractors[group] = ConcretenessFeatureExtractor(lexicon_path=concreteness_lexicon_path)
            else:
                # Use default constructor
                self.extractors[group] = self.EXTRACTORS[group]()
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Extract features from a DataFrame containing text.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with text column
        text_column : str
            Name of the column containing text
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted features (one row per input text)
        """
        texts = df[text_column].tolist()
        
        print(f"Extracting features from {len(texts)} texts...")
        print(f"Selected feature groups: {self.feature_groups}")
        
        # Extract features for each text
        all_features = []
        
        for text in tqdm(texts, desc="Genre features"):
            text_features = {}
            
            # Extract from each extractor
            for group_name, extractor in self.extractors.items():
                try:
                    group_features = extractor.extract(text)
                    text_features.update(group_features)
                except Exception as e:
                    print(f"Warning: Error extracting {group_name} features: {e}")
                    # Add zeros for failed extraction
                    pass
            
            all_features.append(text_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Fill any NaN values with 0
        features_df = features_df.fillna(0)
        
        print(f"Extracted {len(features_df.columns)} features from {len(texts)} texts")
        
        return features_df


# Global cache for extractors
_EXTRACTOR_CACHE = {}


def get_gc_extractor(groups=None):
    """Get or create a cached GC feature extractor for given groups."""
    key = tuple(sorted(groups)) if groups else ("__ALL__",)
    if key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[key] = GCFeatureExtractor(feature_groups=(groups or "all"))
    return _EXTRACTOR_CACHE[key]


def add_gc_features(df: pd.DataFrame, text_col: str = "scene_text", groups=None):
    """
    Add Genre Classifier features to dataframe.
    
    Args:
        df: DataFrame with text column
        text_col: Name of text column
        groups: List of feature groups or None for all
        
    Returns:
        DataFrame with gc_* columns added
    """
    df = df.copy()
    extractor = get_gc_extractor(groups)
    
    # Run extractor on text
    temp = pd.DataFrame({"text": df[text_col].astype(str).tolist()})
    feats = extractor.extract_features_from_dataframe(temp, text_column="text")
    
    # Prefix columns to avoid collisions
    feats = feats.add_prefix("gc_")
    
    # Align lengths; fill missing
    feats = feats.reindex(range(len(df))).fillna(0)
    
    # Concatenate side-by-side
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return df
