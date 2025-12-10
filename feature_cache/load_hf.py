"""
Load cached features from Hugging Face Hub.

This module provides functions to load pre-extracted features from Hugging Face Hub.

Usage:
    from feature_cache.load_hf import load_groups
    
    # Load from HuggingFace
    X, y = load_groups(["base", "gc_polarity"], split="train", 
                       hf_repo="username/screenplay-features")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


# Default Hugging Face repo
DEFAULT_HF_REPO = os.environ.get("HF_FEATURE_REPO", "Ishaank18/screenplay-features")

# Feature group shortcuts
GROUP_SHORTCUTS = {
    "gc_sentiment": ["gc_polarity", "gc_pronouns"],
    "all_gc": [
        "gc_academic", "gc_basic", "gc_char_diversity", "gc_concreteness",
        "gc_dialogue", "gc_discourse", "gc_narrative", "gc_polarity",
        "gc_pos", "gc_pronouns", "gc_punctuation", "gc_readability",
        "gc_syntax", "gc_temporal"
    ],
}


def load_groups(
    groups: List[str],
    split: str = "train",
    hf_repo: Optional[str] = None,
    include_label: bool = True,
    verbose: bool = True,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.Series]]:
    """
    Load and merge multiple feature groups from Hugging Face Hub.
    
    Args:
        groups: List of group names to load
        split: Dataset split (train, validation, test)
        hf_repo: Hugging Face repo name (e.g., "username/screenplay-features")
                If None, uses DEFAULT_HF_REPO
        include_label: Whether to include label column
        verbose: Print loading information
        
    Returns:
        If include_label=True: tuple of (X, y) where X is features and y is labels
        If include_label=False: DataFrame with features only
        
    Examples:
        >>> X, y = load_groups(["base", "gc_polarity"], split="train")
        >>> X = load_groups(["gc_sentiment"], split="test", include_label=False)
    """
    # Determine source
    if hf_repo is None:
        hf_repo = DEFAULT_HF_REPO
    
    if not hf_repo:
        raise ValueError("hf_repo must be specified. Provide a Hugging Face repo like 'username/screenplay-features'")
    
    # Expand shortcuts
    expanded_groups = []
    for g in groups:
        if g in GROUP_SHORTCUTS:
            expanded_groups.extend(GROUP_SHORTCUTS[g])
        else:
            expanded_groups.append(g)
    
    # Remove duplicates while preserving order
    expanded_groups = list(dict.fromkeys(expanded_groups))
    
    if verbose:
        print(f"Loading {len(expanded_groups)} feature groups for {split} split...")
    
    # Load from HuggingFace
    dfs = _load_from_hf(expanded_groups, split, hf_repo, verbose)
    
    if len(dfs) == 0:
        raise ValueError("No groups loaded")
    
    # Merge dataframes
    if len(dfs) == 1:
        result = dfs[0]
    else:
        result = _merge_dataframes(dfs, verbose)
    
    if verbose:
        print(f"✓ Loaded {len(result)} samples with {len(result.columns)-3} features")
    
    # Handle label
    if include_label:
        # Check if we have labels
        if 'label' not in result.columns:
            # Load labels from MENSA data
            if verbose:
                print("  Loading labels from MENSA dataset...")
            
            try:
                from data import load_mensa_dataframe
                mensa_df = load_mensa_dataframe(split)
                label_df = mensa_df[["movie_id", "scene_index", "label"]]
                
                # Merge labels
                result = result.merge(label_df, on=["movie_id", "scene_index"], how="left")
                
                if verbose:
                    print(f"  ✓ Loaded labels for {len(result)} samples")
            except Exception as e:
                raise ValueError(f"Label column not found in data and failed to load from MENSA: {e}")
        
        # Return (X, y)
        feature_cols = [c for c in result.columns if c not in ["movie_id", "scene_index", "label"]]
        X = result[feature_cols]
        y = result["label"]
        return X, y
    else:
        # Drop label if present and return features only
        if 'label' in result.columns:
            result = result.drop(columns=['label'])
        
        feature_cols = [c for c in result.columns if c not in ["movie_id", "scene_index"]]
        return result[feature_cols]


def _load_from_hf(
    groups: List[str],
    split: str,
    hf_repo: str,
    verbose: bool = True,
) -> List[pd.DataFrame]:
    """Load feature groups from Hugging Face Hub."""
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required for HuggingFace loading. Install with: pip install datasets")
    
    if verbose:
        print(f"  Source: Hugging Face ({hf_repo})")
    
    dfs = []
    for group in groups:
        try:
            # Construct file path
            file_path = f"{split}/{group}.parquet"
            
            # Load from HF
            ds = load_dataset(hf_repo, data_files=file_path, split="train")
            df = ds.to_pandas()
            dfs.append(df)
            
            if verbose:
                n_features = len([c for c in df.columns if c not in ["movie_id", "scene_index", "label"]])
                print(f"    ✓ {group}: {len(df)} samples, {n_features} features")
                
        except Exception as e:
            print(f"    ✗ {group}: Error - {e}")
            raise
    
    return dfs


def _merge_dataframes(dfs: List[pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """Merge multiple feature dataframes on movie_id + scene_index."""
    
    result = dfs[0]
    
    # Keep track of label from first df
    has_label = 'label' in result.columns
    if has_label:
        label_col = result['label'].copy()
        result = result.drop(columns=['label'])
    
    # Merge remaining dataframes
    for i, df in enumerate(dfs[1:], 1):
        # Drop label column to avoid duplicates
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        
        # Check for overlapping feature columns
        overlap = set(result.columns) & set(df.columns)
        overlap.discard('movie_id')
        overlap.discard('scene_index')
        
        if overlap:
            if verbose:
                print(f"  Warning: Dropping {len(overlap)} duplicate columns from group {i}")
            df = df.drop(columns=list(overlap))
        
        # Merge
        result = result.merge(df, on=["movie_id", "scene_index"], how="inner")
    
    # Add label back
    if has_label:
        result['label'] = label_col
    
    return result


def list_available_groups(
    split: str = "train",
    hf_repo: Optional[str] = None,
) -> List[str]:
    """
    List all available feature groups.
    
    Args:
        split: Dataset split
        hf_repo: Hugging Face repo (if None, uses DEFAULT_HF_REPO)
        
    Returns:
        List of available group names
        
    Note:
        This returns a hardcoded list of known groups. 
        For dynamic listing, you would need to use the HuggingFace Hub API.
    """
    # Return all known feature groups
    return [
        "base",
        "bert_surprisal",
        "character_arcs",
        "emotional",
        "gc_academic",
        "gc_basic",
        "gc_char_diversity",
        "gc_concreteness",
        "gc_dialogue",
        "gc_discourse",
        "gc_narrative",
        "gc_polarity",
        "gc_pos",
        "gc_pronouns",
        "gc_punctuation",
        "gc_readability",
        "gc_syntax",
        "gc_temporal",
        "ngram",
        "ngram_surprisal",
        "plot_shifts",
        "rst",
        "structure",
        "surprisal",
    ]


def print_cache_info(hf_repo: Optional[str] = None) -> None:
    """
    Print information about available cache.
    
    Args:
        hf_repo: Hugging Face repo (if None, uses DEFAULT_HF_REPO)
    """
    if hf_repo is None:
        hf_repo = DEFAULT_HF_REPO
    
    print(f"\n{'='*80}")
    print(f"FEATURE CACHE INFORMATION")
    print(f"{'='*80}")
    print(f"Source: Hugging Face Hub")
    print(f"Repo: {hf_repo}")
    print(f"{'='*80}\n")
    
    groups = list_available_groups()
    print(f"Available groups: {len(groups)}")
    print(f"  {', '.join(groups[:10])}")
    if len(groups) > 10:
        print(f"  ... and {len(groups) - 10} more")
    print(f"\n{'='*80}\n")


# Convenience function for backward compatibility
def load_feature_matrix(
    groups: List[str],
    split: str = "train",
    hf_repo: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature groups and return (X, y).
    
    This is an alias for load_groups with include_label=True.
    """
    return load_groups(groups, split, hf_repo, include_label=True)
