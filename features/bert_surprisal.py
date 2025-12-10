#!/usr/bin/env python3
"""
BERT Surprisal Feature Extractor

Provides feature extraction functions compatible with the MENSA pipeline.
These wrappers extract only scene-level features from BERT surprisal.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List
import torch

warnings.filterwarnings('ignore')


def add_bert_surprisal_features(df: pd.DataFrame, text_col: str = "scene_text") -> pd.DataFrame:
    """
    Add BERT-large surprisal features.
    
    This extracts only scene-level features. For word-level surprisal data,
    use BERTSurprisalExtractor directly.
    
    Args:
        df: DataFrame with scene texts
        text_col: Column containing text
        
    Returns:
        DataFrame with added bert_* columns
    """
    from feature_extractors import BERTSurprisalExtractor
    
    print("  Loading BERT-large model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = device == 'cuda'
    extractor = BERTSurprisalExtractor(device=device, use_fp16=use_fp16, preprocess=True)
    
    print(f"  Computing BERT surprisal for {len(df)} scenes...")
    features_list = []
    
    from tqdm.auto import tqdm
    for text in tqdm(df[text_col], desc="  BERT Surprisal", leave=False):
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
    Extract word-level surprisal data from scenes.
    
    This is separate from the main pipeline and produces word-level DataFrames
    suitable for psycholinguistic analysis.
    
    Args:
        df: DataFrame with scene texts and metadata
        text_col: Column containing text
        
    Returns:
        DataFrame with word-level data (one row per word)
    """
    from feature_extractors import BERTSurprisalExtractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = BERTSurprisalExtractor(device=device, preprocess=True)
    
    all_rows = []
    from tqdm.auto import tqdm
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting BERT word-level"):
        text = row[text_col]
        if not text or not text.strip():
            continue
        
        result = extractor.extract(text)
        word_level_data = result['word_level']
        
        # Add scene metadata to each word
        for word_item in word_level_data:
            word_row = {
                'movie_id': row.get('movie_id', idx),
                'scene_index': row.get('scene_index', 0),
                'saliency_score': row.get('label', None),
            }
            word_row.update(word_item)
            all_rows.append(word_row)
    
    return pd.DataFrame(all_rows)
