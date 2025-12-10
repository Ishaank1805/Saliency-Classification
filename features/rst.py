#!/usr/bin/env python3
"""
RST (Rhetorical Structure Theory) Feature Extractor

Provides feature extraction functions compatible with the MENSA pipeline.
These wrappers extract only scene-level features from the RST extractor.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List
import torch

warnings.filterwarnings('ignore')


def add_rst_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Add RST (Rhetorical Structure Theory) structural features.
    
    This extracts only scene-level features. For word-level (EDU) data,
    use RSTFeatureExtractor directly.
    
    Args:
        df: DataFrame with scene texts
        text_col: Column containing text
        
    Returns:
        DataFrame with added rst_* columns
    """
    from feature_extractors import RSTFeatureExtractor
    
    print("  Initializing RST parser...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = RSTFeatureExtractor(device=device, preprocess=True)
    
    print(f"  Extracting RST features from {len(df)} scenes...")
    features_list = []
    
    from tqdm.auto import tqdm
    for text in tqdm(df[text_col], desc="  RST", leave=False):
        result = extractor.extract(text)
        # Extract only scene-level features
        scene_features = result['scene_level']
        features_list.append(scene_features)
    
    # Convert to dataframe and concatenate
    features_df = pd.DataFrame(features_list)
    result = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    return result


def extract_word_level_data(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Extract word-level EDU data from scenes.
    
    This is separate from the main pipeline and produces word-level DataFrames
    suitable for psycholinguistic analysis.
    
    Args:
        df: DataFrame with scene texts and metadata
        text_col: Column containing text
        
    Returns:
        DataFrame with EDU-level data (one row per EDU)
    """
    from feature_extractors import RSTFeatureExtractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = RSTFeatureExtractor(device=device, preprocess=True)
    
    all_rows = []
    from tqdm.auto import tqdm
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting RST EDU-level"):
        text = row[text_col]
        if not text or not text.strip():
            continue
        
        result = extractor.extract(text)
        word_level_data = result['word_level']
        
        # Add scene metadata to each EDU
        for word_item in word_level_data:
            word_row = {
                'movie_id': row.get('movie_id', idx),
                'scene_index': row.get('scene_index', 0),
                'saliency_score': row.get('label', None),
            }
            word_row.update(word_item)
            all_rows.append(word_row)
    
    return pd.DataFrame(all_rows)
