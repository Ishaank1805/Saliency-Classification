"""
Training script for sequence-based scene saliency classification.

Supports all ablation experiments with comprehensive tracking.
"""

import argparse
import os
import json
import pickle
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from sequence_model import SceneSaliencyWithLinguistic

# SHAP imports (optional - only used if --shap-analysis is enabled)
try:
    import shap
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
        sns = None
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    plt = None
    sns = None
    SEABORN_AVAILABLE = False


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and improving precision.
    
    Focal loss focuses on hard examples rather than just positive examples,
    which helps reduce false positives and improve precision.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Weighting factor for positive class
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t (probability of correct class)
        pt = torch.exp(-bce_loss)  # pt = p if target=1, else 1-p
        
        # Compute focal weight: (1-pt)^gamma
        # This downweights easy examples (high pt) and focuses on hard examples (low pt)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting for positive class
        # alpha_t = alpha if target=1, else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
from experiment_config import (
    get_experiment_config,
    get_experiment_group,
    list_all_experiments,
    list_all_groups
)


# Global cache for TextRank model (to avoid reloading for each movie)
_textrank_model_cache = {}
_textrank_tokenizer_cache = {}

# Top 20 linguistic features from ensemble model analysis (run_lr.out)
# These are the most important features identified by the ensemble model
# Based on base_lr coefficients (highest magnitude)
TOP_20_LINGUISTIC_FEATURES = [
    'gc_cd_ttr',                    # Character diversity: Type-Token Ratio
    'pos_act',                       # Position: Act
    'gc_conc_mean',                  # Concreteness mean
    'bert_text_length',              # BERT text length
    'gc_pol_negative_sum',          # Polarity: negative sum
    'gc_pol_positive_sum',           # Polarity: positive sum
    'gc_dep_big_ADP_before_PUNCT',  # Dependency: ADP before PUNCT
    'rst_mean_depth',                # RST mean depth
    'gc_lex_den',                    # Lexical density
    'bert_avg_token_log_prob',       # BERT avg token log probability
    'bert_avg_token_surprisal',      # BERT avg token surprisal
    'pos_edge_proximity',            # Position: edge proximity
    'scene_index_norm',               # Scene index (normalized)
    'ngram_unigram_diversity',       # N-gram unigram diversity
    'gc_cd_mtld_ma',                 # Character diversity: MTLD moving average
    'gc_cd_mtld',                    # Character diversity: MTLD
    'gc_dep_big_DET_before_NOUN',    # Dependency: DET before NOUN
    'bert_num_tokens',                # BERT number of tokens
    'gc_pol_neutral_mean',            # Polarity: neutral mean
    'gc_dep_big_PART_after_ADP',     # Dependency: PART after ADP
]

# Top 20 linguistic features from ensemble model analysis (run_lr.out)
# These are the most important features identified by the ensemble model
TOP_20_LINGUISTIC_FEATURES = [
    'gc_cd_ttr',                    # Character diversity: Type-Token Ratio
    'pos_act',                       # Position: Act
    'gc_conc_mean',                  # Concreteness mean
    'bert_text_length',              # BERT text length
    'gc_pol_negative_sum',          # Polarity: negative sum
    'gc_pol_positive_sum',           # Polarity: positive sum
    'gc_dep_big_ADP_before_PUNCT',  # Dependency: ADP before PUNCT
    'rst_mean_depth',                # RST mean depth
    'gc_lex_den',                    # Lexical density
    'bert_avg_token_log_prob',       # BERT avg token log probability
    'bert_avg_token_surprisal',      # BERT avg token surprisal
    'pos_edge_proximity',            # Position: edge proximity
    'scene_index_norm',               # Scene index (normalized)
    'ngram_unigram_diversity',       # N-gram unigram diversity
    'gc_cd_mtld_ma',                 # Character diversity: MTLD moving average
    'gc_cd_mtld',                    # Character diversity: MTLD
    'gc_dep_big_DET_before_NOUN',    # Dependency: DET before NOUN
    'bert_num_tokens',                # BERT number of tokens
    'gc_pol_neutral_mean',            # Polarity: neutral mean
    'gc_dep_big_PART_after_ADP',     # Dependency: PART after ADP
]


def compute_textrank_centrality(
    scene_texts: List[str],
    movie_id: int,
    split: str,
    scene_model_name: str,
    embeddings_cache_dir: Optional[str] = None,
    lambda1: float = 0.7,
    lambda2: float = 0.3,
    device: str = "cpu",
    batch_size: int = 16
) -> np.ndarray:
    """
    Compute TextRank centrality scores for scenes using pre-computed embeddings if available.
    
    Args:
        scene_texts: List of scene text strings
        movie_id: Movie ID for loading pre-computed embeddings
        split: Data split (train/validation/test) for loading pre-computed embeddings
        scene_model_name: Pre-trained model name for scene embeddings (used if cache unavailable)
        embeddings_cache_dir: Directory with pre-computed embeddings (optional)
        lambda1: Weight for forward-looking edges (j > i)
        lambda2: Weight for backward-looking edges (j < i)
        device: Device to run computation on
        batch_size: Batch size for encoding scenes (only used if cache unavailable)
    
    Returns:
        Centrality scores as numpy array [num_scenes]
    """
    num_scenes = len(scene_texts)
    if num_scenes == 0:
        return np.array([])
    
    # Try to load pre-computed embeddings first
    scene_embeddings = None
    if embeddings_cache_dir is not None:
        try:
            import pickle
            model_name_safe = scene_model_name.replace('/', '_').replace('-', '_')
            cache_file = os.path.join(
                embeddings_cache_dir,
                f"{model_name_safe}_{split}.pkl"
            )
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                embeddings_dict = cache_data.get('embeddings', {})
                if movie_id in embeddings_dict:
                    scene_embeddings = torch.tensor(embeddings_dict[movie_id], dtype=torch.float32)
                    # Verify shape matches
                    if scene_embeddings.shape[0] == num_scenes:
                        scene_embeddings = scene_embeddings.cpu()
                    else:
                        # Shape mismatch, will compute on-the-fly
                        scene_embeddings = None
                else:
                    # Movie not in cache, will compute on-the-fly
                    scene_embeddings = None
        except Exception as e:
            # If loading fails, fall back to computing embeddings
            scene_embeddings = None
    
    # Compute embeddings on-the-fly if cache unavailable
    if scene_embeddings is None:
        # Use cached model if available
        if scene_model_name not in _textrank_model_cache:
            model = AutoModel.from_pretrained(scene_model_name)
            tokenizer = AutoTokenizer.from_pretrained(scene_model_name)
            model.eval()
            model.to(device)
            _textrank_model_cache[scene_model_name] = model
            _textrank_tokenizer_cache[scene_model_name] = tokenizer
        else:
            model = _textrank_model_cache[scene_model_name]
            tokenizer = _textrank_tokenizer_cache[scene_model_name]
        
        # Encode all scenes to embeddings
        scene_embeddings_list = []
        with torch.no_grad():
            for i in range(0, num_scenes, batch_size):
                batch = scene_texts[i:i+batch_size]
                encoded = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                outputs = model(**encoded)
                
                # Use [CLS] token or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    scene_emb = outputs.pooler_output
                else:
                    scene_emb = outputs.last_hidden_state.mean(dim=1)
                
                scene_embeddings_list.append(scene_emb.cpu())
        
        # Concatenate all embeddings
        scene_embeddings = torch.cat(scene_embeddings_list, dim=0)  # [num_scenes, hidden_dim]
    
    # Normalize embeddings for cosine similarity
    scene_embeddings = F.normalize(scene_embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(scene_embeddings, scene_embeddings.t())  # [num_scenes, num_scenes]
    
    # Compute centrality for each scene
    centrality_scores = []
    for i in range(num_scenes):
        # Forward-looking edges (j > i)
        forward_sum = similarity_matrix[i, i+1:].sum().item() if i < num_scenes - 1 else 0.0
        
        # Backward-looking edges (j < i)
        backward_sum = similarity_matrix[i, :i].sum().item() if i > 0 else 0.0
        
        # Centrality = λ1 * forward + λ2 * backward
        centrality = lambda1 * forward_sum + lambda2 * backward_sum
        centrality_scores.append(centrality)
    
    return np.array(centrality_scores, dtype=np.float32)


# Feature group to column prefix/pattern mapping
# Patterns are matched as: exact match OR column starts with pattern + '_' OR column starts with pattern
FEATURE_GROUP_PATTERNS = {
    'surprisal': [
        'bert_surprisal', 'bert_var_surprisal', 'bert_q90_surprisal', 'bert_avg_token_surprisal',
        'bert_avg_token_log_prob', 'bert_text_length', 'bert_num_tokens',
        'surprisal', 'gpt2_char_surprisal', 'ngram_surprisal', 'ngram_char_surprisal',
        'psychformers'
    ],
    'morphosyntactic': [
        'gc_syntax', 'gc_dep_', 'gc_pos', 'gc_temporal', 'gc_tense'
    ],
    'lexical': [
        'gc_basic', 'gc_readability', 'gc_char_diversity', 'gc_cd_', 'gc_lex_',
        'ngram_', 'gc_narrative', 'gc_punctuation'
    ],
    'semantic': [
        'gc_academic', 'gc_conc_', 'gc_concreteness'
    ],
    'discourse_pragmatic': [
        'gc_discourse', 'rst_', 'textrank_centrality'
    ],
    'dialogic': [
        'gc_dialogue', 'gc_pronouns', 'unique_PERSON_count', 'char_'
    ],
    'emotional': [
        'emotional', 'gc_pol_', 'gc_polarity'
    ],
    'narrative_structural': [
        'structure', 'plot_shifts', 'character_arcs', 'pos_', 'position', 
        'scene_index_norm', 'graph_num_central_chars', 'graph_'
    ],
    'saxena_keller': [
        'saxena_keller'
    ],
    'genre': [
        'genre'
    ]
}


def filter_features_by_groups(df: pd.DataFrame, feature_groups: List[str]) -> pd.DataFrame:
    """
    Filter dataframe columns to only include features from specified groups.
    
    Args:
        df: DataFrame with all features
        feature_groups: List of feature group names to include
    
    Returns:
        DataFrame with only columns from specified groups (plus metadata columns)
    """
    if feature_groups is None or len(feature_groups) == 0:
        return df
    
    # Metadata columns to always keep
    metadata_cols = ['movie_id', 'scene_index', 'scene_text', 'label', 'labels', 
                     'movie_title', 'name', 'title']
    
    # Get all column names
    all_cols = set(df.columns)
    
    # Start with metadata columns
    selected_cols = [col for col in metadata_cols if col in all_cols]
    
    # For each feature group, find matching columns
    for group in feature_groups:
        group_lower = group.lower()
        if group_lower not in FEATURE_GROUP_PATTERNS:
            print(f"  ⚠ Warning: Unknown feature group '{group}', skipping...")
            continue
        
        patterns = FEATURE_GROUP_PATTERNS[group_lower]
        group_cols = []
        
        for col in all_cols:
            # Skip metadata columns
            if col in metadata_cols:
                continue
            
            # Check if column matches any pattern in this group
            col_lower = col.lower()
            matched = False
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # Exact match
                if col_lower == pattern_lower:
                    matched = True
                    break
                # Starts with pattern + '_' (e.g., 'gc_cd_' matches 'gc_cd_ttr')
                elif pattern_lower.endswith('_') and col_lower.startswith(pattern_lower):
                    matched = True
                    break
                # Starts with pattern + '_' (e.g., 'gc_pol' matches 'gc_pol_positive_sum')
                elif not pattern_lower.endswith('_') and col_lower.startswith(pattern_lower + '_'):
                    matched = True
                    break
            
            if matched and col not in selected_cols:
                group_cols.append(col)
                selected_cols.append(col)
        
        if len(group_cols) > 0:
            print(f"  ✓ Feature group '{group}': selected {len(group_cols)} columns")
        else:
            print(f"  ⚠ Warning: Feature group '{group}': no matching columns found")
    
    # Filter dataframe to only selected columns
    filtered_df = df[selected_cols].copy()
    
    # Count feature columns (exclude metadata)
    feature_cols = [c for c in filtered_df.columns if c not in metadata_cols]
    print(f"  ✓ Total features selected: {len(feature_cols)} (from {len(feature_groups)} groups)")
    
    return filtered_df


def load_features_from_huggingface(
    feature_groups: List[str],
    split: str,
    hf_repo: str = "Ishaank18/screenplay-features",
    include_label: bool = True
) -> pd.DataFrame:
    """
    Load and merge feature groups from Hugging Face datasets.
    
    Args:
        feature_groups: List of feature group names to load (e.g., ["base", "gc_polarity", "rst"])
        split: Data split ("train", "validation", or "test")
        hf_repo: Hugging Face repository name
        include_label: Whether to include labels in the merged dataframe
    
    Returns:
        Merged DataFrame with all features from specified groups
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print(f"Loading {len(feature_groups)} feature groups from Hugging Face: {feature_groups}")
    
    # Map split name (Hugging Face uses "validation" but we might pass "val")
    split_map = {"val": "validation", "test": "test", "train": "train"}
    hf_split = split_map.get(split, split)
    
    # Load each feature group separately and merge them
    df_list = []
    base_df = None
    
    for group in feature_groups:
        try:
            # Load the specific group file
            group_ds = load_dataset(hf_repo, data_files=f"{hf_split}/{group}.parquet")
            # The dataset might have a different split name, try both
            if hf_split in group_ds:
                group_df = group_ds[hf_split].to_pandas()
            elif split in group_ds:
                group_df = group_ds[split].to_pandas()
            else:
                # Try the first available split
                available_splits = list(group_ds.keys())
                if len(available_splits) > 0:
                    group_df = group_ds[available_splits[0]].to_pandas()
                else:
                    raise ValueError(f"No splits available in dataset for group '{group}'")
            
            if base_df is None:
                # First group: use as base
                base_df = group_df.copy()
                print(f"  ✓ Loaded base group '{group}': {len(group_df)} scenes, {len(group_df.columns)} columns")
            else:
                # Merge with existing dataframe on movie_id and scene_index
                merge_cols = ['movie_id', 'scene_index']
                
                # Check which columns to merge (exclude metadata columns that might duplicate)
                cols_to_merge = [c for c in group_df.columns if c not in merge_cols]
                
                # If scene_text exists in both, keep from base
                if 'scene_text' in base_df.columns and 'scene_text' in group_df.columns:
                    group_df = group_df.drop(columns=['scene_text'])
                
                # If label exists in both, keep from base
                if 'label' in base_df.columns and 'label' in group_df.columns:
                    group_df = group_df.drop(columns=['label'])
                
                # Merge on movie_id and scene_index
                base_df = base_df.merge(
                    group_df,
                    on=merge_cols,
                    how='outer',
                    suffixes=('', f'_{group}')
                )
                print(f"  ✓ Merged group '{group}': added {len([c for c in cols_to_merge if c in group_df.columns])} features")
        except Exception as e:
            print(f"  ✗ Warning: Could not load feature group '{group}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if base_df is None:
        raise ValueError(f"No feature groups could be loaded for split '{split}'")
    
    # Remove duplicate columns (keep first occurrence)
    base_df = base_df.loc[:, ~base_df.columns.duplicated()]
    
    # Ensure we have required columns
    required_cols = ['movie_id', 'scene_index']
    missing_cols = [c for c in required_cols if c not in base_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    if include_label and 'label' not in base_df.columns:
        print("  ⚠ Warning: 'label' column not found. Labels may not be available.")
    
    # Sort by movie_id and scene_index
    base_df = base_df.sort_values(['movie_id', 'scene_index']).reset_index(drop=True)
    
    # Count feature columns (exclude metadata)
    metadata_cols = ['movie_id', 'scene_index', 'scene_text', 'label']
    feature_cols = [c for c in base_df.columns if c not in metadata_cols]
    
    print(f"  ✓ Final merged dataset: {len(base_df)} scenes from {len(base_df['movie_id'].unique())} movies")
    print(f"  ✓ Total feature columns: {len(feature_cols)}")
    
    return base_df


class MovieDataset(Dataset):
    """Dataset for movies with scenes."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        linguistic_cols: List[str] = None,
        feature_columns: List[str] = None, # New argument to enforce consistent columns
        use_textrank: bool = True,  # Whether to compute TextRank centrality
        textrank_model: str = None,  # Model for TextRank (will use scene_model_name from config if None)
        embeddings_cache_dir: str = None,  # Directory with pre-computed embeddings
        textrank_lambda1: float = 0.7,  # Forward-looking weight
        textrank_lambda2: float = 0.3,  # Backward-looking weight
        use_top_features: bool = False,  # Use only top 20 features from ensemble analysis
        top_features_list: List[str] = None,  # Custom list of top features (default: TOP_20_LINGUISTIC_FEATURES)
        use_huggingface: bool = False,  # Load features from Hugging Face datasets
        hf_repo: str = "Ishaank18/screenplay-features",  # Hugging Face repository name
        feature_groups: List[str] = None,  # List of feature groups to load (e.g., ["base", "gc_polarity", "rst"])
        mensa_repo: str = "rohitsaxena/MENSA",  # Hugging Face repository for MENSA dataset (for scene_text and labels)
    ):
        self.split = split  # Store split for embedding loading
        self.embeddings_cache_dir = embeddings_cache_dir
        self.use_textrank = use_textrank
        self.textrank_model = textrank_model
        self.textrank_lambda1 = textrank_lambda1
        self.textrank_lambda2 = textrank_lambda2
        self.use_top_features = use_top_features
        self.top_features_list = top_features_list
        """
        Args:
            data_path: Path to parquet file or directory (ignored if use_huggingface=True)
            split: "train", "validation", or "test"
            linguistic_cols: List of linguistic feature column names
            feature_columns: Optional list of columns to enforce consistent features across splits
            use_huggingface: If True, load features from Hugging Face datasets
            hf_repo: Hugging Face repository name (default: "Ishaank18/screenplay-features")
            feature_groups: List of feature groups to load (e.g., ["base", "gc_polarity", "rst"])
        """
        # Load data from Hugging Face or local files
        if use_huggingface:
            if feature_groups is None or len(feature_groups) == 0:
                raise ValueError("feature_groups must be specified when use_huggingface=True")
            
            # Load from Hugging Face
            self.df = load_features_from_huggingface(
                feature_groups=feature_groups,
                split=split,
                hf_repo=hf_repo,
                include_label=True
            )
        else:
            # Load from local files (original behavior)
            if os.path.isdir(data_path):
                file_path = os.path.join(data_path, f"{split}.parquet")
            else:
                file_path = data_path
            
            self.df = pd.read_parquet(file_path)
            
            # Filter features by groups if specified (for ablation studies)
            if feature_groups is not None and len(feature_groups) > 0:
                print(f"\nFiltering features by groups: {feature_groups}")
                self.df = filter_features_by_groups(self.df, feature_groups)
        
        # Load scene_text and labels from MENSA dataset if not already present
        # (This is a fallback - scene_text should already be in parquet files from merge_all_features.py)
        needs_mensa = False
        if 'scene_text' not in self.df.columns:
            needs_mensa = True
        elif self.df['scene_text'].isna().all() or (self.df['scene_text'].astype(str).str.strip() == '').all():
            # All scene_text values are null or empty strings
            needs_mensa = True
        
        if needs_mensa:
            print(f"  ⚠ Warning: scene_text not found in parquet file. Loading from MENSA dataset ({mensa_repo}) as fallback...")
            try:
                from datasets import load_dataset
                
                # Map split name for MENSA
                split_map = {"val": "validation", "test": "test", "train": "train"}
                mensa_split = split_map.get(split, split)
                
                # Try loading MENSA dataset
                try:
                    mensa_ds = load_dataset(mensa_repo, split=mensa_split)
                    mensa_df = mensa_ds.to_pandas()
                except:
                    # Try with data_files if split-based loading doesn't work
                    mensa_ds = load_dataset(mensa_repo, data_files=f"{mensa_split}.parquet")
                    if mensa_split in mensa_ds:
                        mensa_df = mensa_ds[mensa_split].to_pandas()
                    else:
                        available_splits = list(mensa_ds.keys())
                        if len(available_splits) > 0:
                            mensa_df = mensa_ds[available_splits[0]].to_pandas()
                        else:
                            raise ValueError("No splits available in MENSA dataset")
                
                # Check what columns MENSA has
                print(f"    MENSA dataset columns: {list(mensa_df.columns)}")
                
                # MENSA dataset structure: it has 'scenes' (list) and 'labels' (list) per movie
                # We need to check if it's in list format or already flattened
                if 'scenes' in mensa_df.columns and 'labels' in mensa_df.columns:
                    # MENSA has lists - need to explode/flatten them
                    print(f"    MENSA has 'scenes' and 'labels' lists - flattening...")
                    
                    # Create a list to store exploded rows
                    exploded_rows = []
                    
                    # Get movie identifier - check if we have movie_id or need to use name/index
                    has_movie_id = 'movie_id' in mensa_df.columns
                    has_name = 'name' in mensa_df.columns
                    
                    # Try to create a mapping from MENSA to feature data movie_ids
                    # First, check if feature data has movie names we can match
                    name_to_id_map = {}
                    if has_name and 'movie_title' in self.df.columns:
                        # Create mapping from movie_title to movie_id in feature data
                        for movie_id, title in self.df[['movie_id', 'movie_title']].drop_duplicates().values:
                            name_to_id_map[title] = movie_id
                    elif has_name:
                        # If no movie_title, try to match by order (risky but might work)
                        # Get unique movie_ids from feature data in order
                        unique_movie_ids = self.df['movie_id'].unique()
                        for idx, name in enumerate(mensa_df['name']):
                            if idx < len(unique_movie_ids):
                                name_to_id_map[name] = unique_movie_ids[idx]
                    
                    # Explode scenes and labels
                    for idx, row in mensa_df.iterrows():
                        scenes_list = row['scenes'] if isinstance(row['scenes'], list) else []
                        labels_list = row['labels'] if isinstance(row['labels'], list) else []
                        
                        # Determine movie_id
                        if has_movie_id:
                            movie_id_val = row['movie_id']
                        elif has_name and row['name'] in name_to_id_map:
                            movie_id_val = name_to_id_map[row['name']]
                        else:
                            # Fallback: use row index (will need manual alignment)
                            movie_id_val = idx
                            if has_name:
                                print(f"    ⚠ Warning: Could not map movie '{row['name']}' to movie_id, using index {idx}")
                        
                        # Explode scenes and labels
                        for scene_idx, (scene_text, label) in enumerate(zip(scenes_list, labels_list)):
                            exploded_rows.append({
                                'movie_id': movie_id_val,
                                'scene_index': scene_idx,
                                'scene_text': scene_text if isinstance(scene_text, str) else str(scene_text),
                                'label': label if isinstance(label, (int, float)) else int(label) if label else 0
                            })
                    
                    mensa_df = pd.DataFrame(exploded_rows)
                    print(f"    ✓ Flattened MENSA: {len(mensa_df)} scenes from {len(mensa_df['movie_id'].unique())} movies")
                    
                    merge_cols = ['movie_id', 'scene_index']
                    
                else:
                    # MENSA is already in flattened format - try to find merge keys
                    merge_key_mensa = None
                    scene_key_mensa = None
                    
                    # Check for movie_id variations (in order of preference)
                    for col in ['movie_id', 'movie', 'id', 'movieId', 'movie_id']:
                        if col in mensa_df.columns:
                            merge_key_mensa = col
                            break
                    
                    # Check for scene_index variations (in order of preference)
                    for col in ['scene_index', 'scene', 'index', 'sceneIndex', 'scene_idx', 'scene_id']:
                        if col in mensa_df.columns:
                            scene_key_mensa = col
                            break
                    
                    if merge_key_mensa is None or scene_key_mensa is None:
                        # If we can't find merge keys, raise a helpful error
                        available_cols = list(mensa_df.columns)
                        raise ValueError(
                            f"Cannot find merge keys in MENSA dataset.\n"
                            f"  Expected: 'movie_id' (or 'movie', 'id') and 'scene_index' (or 'scene', 'index')\n"
                            f"  Or expected: 'scenes' and 'labels' lists to flatten\n"
                            f"  Found columns: {available_cols}\n"
                            f"  Please check the MENSA dataset structure."
                        )
                    
                    print(f"    Using merge keys: '{merge_key_mensa}' -> 'movie_id', '{scene_key_mensa}' -> 'scene_index'")
                    
                    # Select only the columns we need from MENSA
                    mensa_cols = [merge_key_mensa, scene_key_mensa]
                    
                    # Find scene_text column (check various names)
                    scene_text_col = None
                    for col in ['scene_text', 'text', 'scene', 'content', 'script']:
                        if col in mensa_df.columns:
                            # Check if it's actually text (string type)
                            if pd.api.types.is_string_dtype(mensa_df[col]) or pd.api.types.is_object_dtype(mensa_df[col]):
                                scene_text_col = col
                                break
                    
                    if scene_text_col:
                        mensa_cols.append(scene_text_col)
                    
                    if 'label' in mensa_df.columns and 'label' not in self.df.columns:
                        mensa_cols.append('label')
                    
                    # Only select columns that exist
                    mensa_cols = [c for c in mensa_cols if c in mensa_df.columns]
                    mensa_df = mensa_df[mensa_cols].copy()
                    
                    # Rename columns to match our expected names
                    rename_dict = {}
                    if merge_key_mensa != 'movie_id':
                        rename_dict[merge_key_mensa] = 'movie_id'
                    if scene_key_mensa != 'scene_index':
                        rename_dict[scene_key_mensa] = 'scene_index'
                    if scene_text_col and scene_text_col != 'scene_text':
                        rename_dict[scene_text_col] = 'scene_text'
                    
                    if rename_dict:
                        mensa_df = mensa_df.rename(columns=rename_dict)
                    
                    # Now use standard column names for merge
                    merge_cols = ['movie_id', 'scene_index']
                
                # Merge MENSA data with feature data
                self.df = self.df.merge(
                    mensa_df,
                    on=merge_cols,
                    how='left',  # Left join to keep all feature rows
                    suffixes=('', '_mensa')
                )
                
                # Drop duplicate columns from MENSA merge
                for col in self.df.columns:
                    if col.endswith('_mensa'):
                        original_col = col.replace('_mensa', '')
                        if original_col in self.df.columns:
                            # Keep original, drop mensa version
                            self.df = self.df.drop(columns=[col])
                        else:
                            # Rename mensa version to original name
                            self.df = self.df.rename(columns={col: original_col})
                
                # Check what we got from MENSA
                if 'scene_text' in self.df.columns:
                    non_null_scenes = self.df['scene_text'].notna().sum()
                    print(f"  ✓ Merged scene_text from MENSA: {non_null_scenes}/{len(self.df)} scenes have text")
                if 'label' in self.df.columns:
                    non_null_labels = self.df['label'].notna().sum()
                    print(f"  ✓ Merged labels from MENSA: {non_null_labels}/{len(self.df)} scenes have labels")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load MENSA dataset ({mensa_repo}): {e}")
                print(f"  → Continuing without scene_text from MENSA")
                if "not found" not in str(e).lower() and "does not exist" not in str(e).lower():
                    import traceback
                    traceback.print_exc()
        
        # Group by movie
        self.movies = []
        
        # Determine feature columns if not provided
        self.feature_columns = feature_columns
        
        # Check if we should use only top 20 features
        if self.use_top_features and self.feature_columns is None:
            # Use only top 20 features (if available in dataset)
            if self.top_features_list is None:
                self.top_features_list = TOP_20_LINGUISTIC_FEATURES
            
            # Filter to only features that exist in the dataset
            available_top_features = [f for f in self.top_features_list if f in self.df.columns]
            if len(available_top_features) < len(self.top_features_list):
                missing = set(self.top_features_list) - set(available_top_features)
                print(f"Warning: {len(missing)} top features not found in dataset: {list(missing)[:5]}...")
            self.feature_columns = available_top_features
            if self.split == "train":  # Only print once for train
                print(f"Using top {len(self.feature_columns)} features (from ensemble analysis) + TextRank centrality")
        elif self.feature_columns is None and linguistic_cols is None:
             # Auto-detect linguistic features (exclude metadata columns)
            exclude_cols = [
                'movie_id', 'scene_index',
                'scene_text',
                'label', 'labels', 'label_x', 'label_y',
                'movie_title', 'movie_title_x', 'movie_title_y',
                'name', 'title',
            ]

            # Only include numeric columns (int, float)
            self.feature_columns = [
                col for col in self.df.columns 
                if col not in exclude_cols 
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]
        elif self.feature_columns is None:
             self.feature_columns = [col for col in linguistic_cols if col in self.df.columns]
        
        # If feature_columns were passed (e.g. from train set), ensure they exist and are in order
        if feature_columns:
            # Reindex ensures columns exist (filling with 0.0) and are in the correct order
            # This is critical for matching model input dimensions
            # We only reindex the feature columns part of the dataframe for memory efficiency
            # But here we need to keep metadata, so we'll just handle it during extraction
            pass

        for movie_id in self.df['movie_id'].unique():
            movie_df = self.df[self.df['movie_id'] == movie_id].sort_values('scene_index')
            
            # Handle scene_text - check for different possible column names
            scene_texts = None
            if 'scene_text' in movie_df.columns:
                scene_texts = movie_df['scene_text'].astype(str).tolist()
            elif 'scene_texts' in movie_df.columns:
                scene_texts = movie_df['scene_texts'].astype(str).tolist()
            elif 'text' in movie_df.columns:
                scene_texts = movie_df['text'].astype(str).tolist()
            else:
                # If no scene text column, create empty strings (for SHAP analysis, scene text is needed)
                # But for training, we can proceed without it
                scene_texts = [''] * len(movie_df)
                if self.split == "train":
                    print(f"Warning: No 'scene_text' column found in data. Using empty strings.")
            
            movie_data = {
                'movie_id': movie_id,
                'scene_texts': scene_texts,
                'labels': movie_df['label'].astype(int).tolist() if 'label' in movie_df.columns else None,
                'scene_indices': movie_df['scene_index'].tolist(),
            }
            
            # Extract linguistic features using the determined columns
            if self.feature_columns:
                # Use reindex to strictly enforce column presence and order
                # fill_value=0.0 handles any missing columns in this split
                features_df = movie_df.reindex(columns=self.feature_columns, fill_value=0.0)
                features = features_df.values.astype(np.float32)
                
                # Replace NaN/Inf with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values to prevent numerical instability
                # Clip to [-10, 10] (sigmoid range) instead of [-100, 100]
                features = np.clip(features, -10.0, 10.0)
                
                movie_data['linguistic_features'] = features
            else:
                movie_data['linguistic_features'] = None
            
            # Compute TextRank centrality if enabled
            if self.use_textrank:
                try:
                    textrank_centrality = compute_textrank_centrality(
                        scene_texts=movie_data['scene_texts'],
                        movie_id=movie_id,
                        split=self.split,
                        scene_model_name=self.textrank_model,
                        embeddings_cache_dir=self.embeddings_cache_dir,
                        lambda1=self.textrank_lambda1,
                        lambda2=self.textrank_lambda2,
                        device="cpu",  # Use CPU during data loading to avoid GPU memory issues
                        batch_size=16
                    )
                    
                    # Normalize centrality scores to [0, 1] range
                    if len(textrank_centrality) > 0:
                        if textrank_centrality.max() - textrank_centrality.min() > 1e-6:
                            textrank_centrality = (textrank_centrality - textrank_centrality.min()) / (
                                textrank_centrality.max() - textrank_centrality.min() + 1e-6
                            )
                        else:
                            # If all values are the same, set to 0.5
                            textrank_centrality = np.full_like(textrank_centrality, 0.5)
                    
                    # Append to linguistic features
                    centrality_col = textrank_centrality.reshape(-1, 1)
                    if movie_data['linguistic_features'] is not None:
                        movie_data['linguistic_features'] = np.concatenate(
                            [movie_data['linguistic_features'], centrality_col],
                            axis=1
                        )
                    else:
                        # If no linguistic features, create a feature array with just centrality
                        movie_data['linguistic_features'] = centrality_col
                except Exception as e:
                    print(f"Warning: Could not compute TextRank centrality for movie {movie_id}: {e}")
                    # Add zero centrality if computation fails
                    zero_centrality = np.zeros((len(movie_data['scene_texts']), 1), dtype=np.float32)
                    if movie_data['linguistic_features'] is not None:
                        movie_data['linguistic_features'] = np.concatenate(
                            [movie_data['linguistic_features'], zero_centrality],
                            axis=1
                        )
                    else:
                        movie_data['linguistic_features'] = zero_centrality
            
            self.movies.append(movie_data)
        
        print(f"Loaded {len(self.movies)} movies for {split} split")
        if self.movies and self.movies[0]['linguistic_features'] is not None:
            print(f"  Linguistic features: {self.movies[0]['linguistic_features'].shape[1]} dimensions")
            
    
    def __len__(self):
        return len(self.movies)
    
    def __getitem__(self, idx):
        return self.movies[idx]


def collate_movies(batch: List[Dict]) -> Dict:
    """Collate function for movie batches."""
    return batch  # Return as-is, will process in training loop


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray, optimize_for_precision: bool = False, min_macro_f1: float = 0.67, min_threshold: float = 0.0) -> Tuple[float, float, float, float]:
    """
    Find optimal classification threshold that maximizes Binary F1 (salient class).
    
    Optimizes for binary F1 to maximize performance on the salient class.
    This prioritizes recall for identifying important scenes.
    
    Strategy:
    1. Test all thresholds from max(0.01, min_threshold) to 0.99
    2. Select the threshold that maximizes binary F1
    3. If multiple thresholds have the same binary F1, prefer the one with higher macro F1
    
    Tests thresholds from max(0.01, min_threshold) to 0.99 in steps of 0.01.
    
    Args:
        probs: Probability predictions
        targets: Ground truth labels
        optimize_for_precision: If True, optimize for precision (deprecated)
        min_macro_f1: Minimum required macro F1 score (deprecated, kept for compatibility)
        min_threshold: Minimum threshold value to consider (default: 0.0, no constraint)
    
    Returns:
        best_threshold: Optimal threshold value
        best_f1: Binary F1 score at optimal threshold
        best_precision: Precision at optimal threshold
        best_recall: Recall at optimal threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Start from max(0.01, min_threshold) to enforce minimum threshold constraint
    start_threshold = max(0.01, min_threshold)
    thresholds = np.arange(start_threshold, 0.99, 0.01)
    best_threshold = max(0.5, min_threshold)  # Default to at least min_threshold
    best_binary_f1 = 0.0
    best_macro_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    # Check if we have valid predictions
    if len(probs) == 0 or len(targets) == 0:
        return max(0.5, min_threshold), 0.0, 0.0, 0.0
    
    # Check if probs are valid (not all NaN/Inf)
    if np.isnan(probs).all() or np.isinf(probs).all():
        return max(0.5, min_threshold), 0.0, 0.0, 0.0
    
    # Find threshold that maximizes binary F1 (only considering thresholds >= min_threshold)
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        # Check if we have both classes in predictions
        if len(np.unique(preds)) < 2:
            continue
        
        # Compute binary F1 (what we optimize for) and macro F1 (tiebreaker)
        binary_f1 = f1_score(targets, preds, average='binary', zero_division=0)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        
        # Update best if this threshold has higher binary F1, or same binary F1 with higher macro F1
        if binary_f1 > best_binary_f1 or (binary_f1 == best_binary_f1 and macro_f1 > best_macro_f1):
            best_threshold = thresh
            best_binary_f1 = binary_f1
            best_macro_f1 = macro_f1
            best_precision = precision
            best_recall = recall
    
    return best_threshold, best_binary_f1, best_precision, best_recall


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute classification metrics."""
    # Macro metrics (average across classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    # Binary F1 (positive class only) - important for imbalanced datasets
    f1_binary = f1_score(targets, predictions, average='binary', zero_division=0)
    
    accuracy = balanced_accuracy_score(targets, predictions)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    return {
        'macro_f1': float(f1_macro),
        'binary_f1': float(f1_binary),  # F1 for positive class (salient scenes)
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'balanced_accuracy': float(accuracy),
        'f1_class_0': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        'f1_class_1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        'precision_class_0': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        'precision_class_1': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        'recall_class_0': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        'recall_class_1': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
    }


def train_epoch(
    model: SceneSaliencyWithLinguistic,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    threshold: float = 0.5,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    warmup_scheduler=None,
    global_step=None,
) -> Tuple[float, Dict, int]:
    """Train for one epoch."""
    model.train()
    total_weighted_loss = 0.0  # Accumulate loss weighted by number of scenes
    total_scenes = 0  # Track total number of scenes processed
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    nan_count = 0
    
    # Track gradient accumulation and global step for warmup
    accumulation_counter = 0
    if global_step is None:
        current_step = 0
    else:
        current_step = global_step
    optimizer.zero_grad()  # Initialize gradients at start of epoch
    
    for batch_idx, batch in enumerate(progress_bar):
        for movie in batch:
            scene_texts = movie['scene_texts']
            labels = torch.tensor(movie['labels'], dtype=torch.float32).to(device)
            movie_id = movie.get('movie_id')
            num_scenes = len(labels)  # Number of scenes in this movie
            
            # DEBUG: Check labels
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print(f"\n[DEBUG] Movie {movie_id}: NaN/Inf in labels!")
                nan_count += 1
                continue
            
            # Prepare linguistic features
            linguistic_features = None
            if model.use_linguistic and movie.get('linguistic_features') is not None:
                linguistic_features = torch.tensor(
                    movie['linguistic_features'],
                    dtype=torch.float32
                ).to(device)
                
                # Replace any NaN/Inf with 0
                linguistic_features = torch.nan_to_num(linguistic_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values to prevent numerical instability
                linguistic_features = torch.clamp(linguistic_features, min=-10.0, max=10.0)
            
            # Forward pass (with FP16 support)
            split = getattr(dataloader.dataset, 'split', "train")
            
            try:
                if scaler is not None:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        logits = model(scene_texts, linguistic_features, movie_id=movie_id, split=split)  # [num_scenes, 1]
                        logits = logits.squeeze(-1)  # [num_scenes]
                else:
                    logits = model(scene_texts, linguistic_features, movie_id=movie_id, split=split)  # [num_scenes, 1]
                    logits = logits.squeeze(-1)  # [num_scenes]
                
                # DEBUG: Check logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"\n[DEBUG] Movie {movie_id}: NaN/Inf in logits!")
                    nan_count += 1
                    continue
                
                # Clip extreme logits to prevent numerical instability in BCEWithLogitsLoss
                logits_clipped = torch.clamp(logits, min=-10.0, max=10.0)
                
                loss = criterion(logits_clipped, labels)
                
                # DEBUG: Check loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[DEBUG] Movie {movie_id}: NaN/Inf loss!")
                    nan_count += 1
                    continue
                    
            except Exception as e:
                print(f"\n[DEBUG] Movie {movie_id}: Exception during forward pass! {e}")
                nan_count += 1
                continue
            
            # Backward pass (with FP16 support and gradient accumulation)
            # Scale loss by accumulation steps to get average gradient
            scaled_loss = loss / gradient_accumulation_steps
            
            try:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                accumulation_counter += 1
                
                # Only update weights after accumulating gradients for N steps
                if accumulation_counter % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Zero gradients after update
                    optimizer.zero_grad()
                    
                    # Update warmup scheduler if enabled
                    if warmup_scheduler is not None:
                        current_step += 1
                        warmup_scheduler.step()
                
                # Accumulate weighted loss: loss is mean over scenes, so multiply by num_scenes
                total_weighted_loss += loss.item() * num_scenes
                total_scenes += num_scenes
                
            except Exception as e:
                print(f"\n[DEBUG] Movie {movie_id}: Exception during backward pass! {e}")
                nan_count += 1
                continue
            
            # Predictions (use clipped logits for consistency)
            predictions = (torch.sigmoid(logits_clipped) > threshold).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
        
        # Update progress bar
        avg_loss = total_weighted_loss / total_scenes if total_scenes > 0 else 0.0
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'nan_count': nan_count,
            'accum': f"{accumulation_counter % gradient_accumulation_steps}/{gradient_accumulation_steps}"
        })
    
    # Handle remaining gradients at end of epoch
    if accumulation_counter % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_weighted_loss / total_scenes if total_scenes > 0 else 0.0
    metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    
    return avg_loss, metrics, current_step


def evaluate(
    model: SceneSaliencyWithLinguistic,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    threshold: float = 0.5,
    return_probs: bool = False,
    return_per_movie: bool = False,
) -> Tuple[float, Dict, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """
    Evaluate model.
    
    Args:
        return_probs: If True, also return probability predictions and targets for threshold optimization
        return_per_movie: If True, also return per-movie predictions for saving salient scenes
    
    Returns:
        avg_loss, metrics, (optional) probabilities, (optional) targets, (optional) per_movie_predictions
    """
    model.eval()
    total_weighted_loss = 0.0  # Accumulate loss weighted by number of scenes
    total_scenes = 0  # Track total number of scenes processed
    all_predictions: List[int] = []
    all_targets: List[float] = []
    # Always collect probabilities; we'll decide what to return at the end
    all_probs: List[float] = []
    
    # Store per-movie predictions if requested
    per_movie_predictions: Dict = {} if return_per_movie else {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for movie in batch:
                scene_texts = movie['scene_texts']
                labels = torch.tensor(movie['labels'], dtype=torch.float32).to(device)
                movie_id = movie.get('movie_id')
                scene_indices = movie.get('scene_indices', list(range(len(scene_texts))))
                num_scenes = len(labels)  # Number of scenes in this movie
                
                # Prepare linguistic features
                linguistic_features = None
                if model.use_linguistic and movie.get('linguistic_features') is not None:
                    linguistic_features = torch.tensor(
                        movie['linguistic_features'],
                        dtype=torch.float32
                    ).to(device)
                    
                    # Replace NaN/Inf with 0 and clip
                    linguistic_features = torch.nan_to_num(linguistic_features, nan=0.0, posinf=0.0, neginf=0.0)
                    linguistic_features = torch.clamp(linguistic_features, min=-10.0, max=10.0)
                
                # Determine split from dataloader
                split = getattr(dataloader.dataset, 'split', None)
                
                # Forward pass
                logits = model(scene_texts, linguistic_features, movie_id=movie_id, split=split)  # [num_scenes, 1]
                logits = logits.squeeze(-1)  # [num_scenes]
                
                # Clip logits
                logits = torch.clamp(logits, min=-10.0, max=10.0)
                
                # Loss
                loss = criterion(logits, labels)
                
                # Accumulate weighted loss: loss is mean over scenes, so multiply by num_scenes
                total_weighted_loss += loss.item() * num_scenes
                total_scenes += num_scenes
                
                # Predictions
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = (probs > threshold).astype(int)
                targets = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_probs.extend(probs)
                
                # Store per-movie predictions if requested
                if return_per_movie and movie_id is not None:
                    per_movie_predictions[movie_id] = {
                        'scene_texts': scene_texts,
                        'scene_indices': scene_indices,
                        'predictions': predictions.tolist(),
                        'probabilities': probs.tolist(),
                        'labels': targets.tolist(),
                    }
    
    # Compute metrics
    avg_loss = total_weighted_loss / total_scenes if total_scenes > 0 else 0.0
    metrics = compute_metrics(np.array(all_predictions), np.array(all_targets))
    
    # Prepare optional outputs
    if return_probs:
        probs_array: Optional[np.ndarray] = np.array(all_probs)
        targets_array: Optional[np.ndarray] = np.array(all_targets)
    else:
        probs_array = None
        targets_array = None
    
    # Return per-movie predictions if requested
    if return_per_movie:
        per_movie_dict = per_movie_predictions if per_movie_predictions else None
        # Return 5 values when return_per_movie is True
        return avg_loss, metrics, probs_array, targets_array, per_movie_dict
    else:
        # Return 4 values for backward compatibility when return_per_movie is False
        return avg_loss, metrics, probs_array, targets_array


def get_positive_weight(dataset: MovieDataset) -> float:
    """Calculate positive class weight for balanced loss."""
    all_labels = []
    for movie in dataset.movies:
        if movie['labels']:
            all_labels.extend(movie['labels'])
    
    if not all_labels:
        return 1.0
    
    ones = sum(all_labels)
    zeros = len(all_labels) - ones
    
    if ones == 0:
        return 1.0
    
    return zeros / ones


class ModelWrapperForSHAP(nn.Module):
    """
    Wrapper class to expose only linguistic features to SHAP analysis.
    
    This isolates the effect of linguistic features. To handle SHAP's batching (where
    input batch size != movie length), we use the MEAN of the movie's scene embeddings
    as the fixed background. This allows us to process arbitrary batch sizes of 
    linguistic features.
    """
    def __init__(self, model: SceneSaliencyWithLinguistic, fixed_scene_embeddings: torch.Tensor, device: str):
        super().__init__()
        self.model = model
        # Compute mean embedding to use as constant background [1, scene_dim]
        # This solves the shape mismatch when SHAP passes batches of random size
        self.mean_scene_embedding = fixed_scene_embeddings.mean(dim=0, keepdim=True)
        self.device = device
        self.model.eval()
    
    def forward(self, linguistic_features: torch.Tensor):
        # Convert numpy to torch if needed
        if isinstance(linguistic_features, np.ndarray):
            linguistic_features = torch.tensor(linguistic_features, dtype=torch.float32, requires_grad=True).to(self.device)
        elif not linguistic_features.requires_grad:
            # Ensure gradients are enabled for SHAP gradient-based explainers
            linguistic_features = linguistic_features.clone().detach().requires_grad_(True)
        
        # Determine input shape
        # SHAP might pass [batch_size, num_features] where batch_size is arbitrary
        
        # Expand mean embedding to match input batch size
        batch_size = linguistic_features.shape[0]
        scene_embeddings = self.mean_scene_embedding.expand(batch_size, -1)
        
        # Forward pass
        # Fuse mean scene embeddings with varying linguistic features
        fused = self.model.fuse_features(scene_embeddings, linguistic_features)
        
        # Add positional encoding
        # Since we treat these as independent samples/batch, we add PE to the batch
        # shape: [batch_size, hidden_dim] -> unsqueeze -> [1, batch_size, hidden_dim]?
        # No, sequence model expects [1, seq_len, hidden_dim] OR [batch, seq_len, hidden_dim]
        # Here we are treating the batch as a sequence of length 'batch_size'? 
        # OR treating it as a batch of sequences of length 1?
        # Let's treat it as a batch of independent scenes (seq_len=1) to avoid mixing them contextually
        
        # Reshape to [batch_size, 1, hidden_dim]
        fused = fused.unsqueeze(1) 
        
        if self.model.use_positional_encoding:
            # We assume position 0 for all (since they are independent/mean context)
            # Or we could just apply PE. 
            # Note: pos_encoder expects [batch, seq_len, dim]. 
            fused = self.model.pos_encoder(fused)
        
        # Apply sequence model
        # If we have [batch_size, 1, hidden_dim], transformer processes each independently
        sequence_output = self.model.apply_sequence_model(fused, mask=None)
        
        # Skip connection
        combined_features = torch.cat([sequence_output, fused], dim=-1)
        
        # Classify
        logits = self.model.classifier(combined_features)
        
        # Output shape: [batch_size, 1, 1] -> [batch_size, 1]
        # SHAP gradient-based explainers expect [batch_size, num_outputs] shape
        # For binary classification, we keep it as [batch_size, 1]
        logits = logits.squeeze(-1)  # [batch_size, 1, 1] -> [batch_size, 1]
        if logits.ndim == 1:
            # If somehow squeezed to 1D, add dimension back
            logits = logits.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        
        return logits


def compute_shap_values(
    model: SceneSaliencyWithLinguistic,
    test_dataset: MovieDataset,
    device: str,
    exp_output_dir: str,
    num_samples: int = 50,
    max_scenes_per_movie: int = 100,
) -> Dict:
    """
    Compute SHAP values for linguistic features to explain model predictions.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
        num_samples: Number of movies to analyze (for efficiency)
        max_scenes_per_movie: Maximum scenes per movie to analyze (for efficiency)
    
    Returns:
        Dictionary containing SHAP values, feature names, and analysis results
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Install with: pip install shap matplotlib")
        return {}
    
    print("\n" + "="*80)
    print("Computing SHAP Values for Interpretability Analysis")
    print("="*80)
    
    model.eval()
    all_shap_values = []
    all_linguistic_features = []
    all_predictions = []
    all_targets = []
    all_scene_texts = []
    all_shap_interaction_values = []
    feature_names = []
    
    # Get feature names from dataset
    if test_dataset.movies and test_dataset.movies[0].get('linguistic_features') is not None:
        num_features = test_dataset.movies[0]['linguistic_features'].shape[1]
        # Try to get feature names from dataset
        if hasattr(test_dataset, 'feature_columns') and test_dataset.feature_columns:
            # Use feature columns, but ensure we have enough names
            if len(test_dataset.feature_columns) >= num_features:
                feature_names = test_dataset.feature_columns[:num_features]
            else:
                # Pad with generic names if needed (e.g., for TextRank centrality added later)
                feature_names = list(test_dataset.feature_columns) + [f"Feature_{i}" for i in range(len(test_dataset.feature_columns), num_features)]
        else:
            # Fallback: generic names
            feature_names = [f"Feature_{i}" for i in range(num_features)]
    else:
        print("Warning: No linguistic features found in dataset. Skipping SHAP analysis.")
        return {}
    
    # Sample movies for analysis (for efficiency)
    sample_indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    print(f"Analyzing {len(sample_indices)} movies...")
    
    for idx in tqdm(sample_indices, desc="Computing SHAP"):
        movie = test_dataset[idx]
        scene_texts = movie['scene_texts']
        linguistic_features = movie.get('linguistic_features')
        
        if linguistic_features is None:
            continue
        
        # Limit scenes for efficiency
        if len(scene_texts) > max_scenes_per_movie:
            # Sample scenes
            scene_indices = np.random.choice(len(scene_texts), max_scenes_per_movie, replace=False)
            scene_texts = [scene_texts[i] for i in sorted(scene_indices)]
            linguistic_features = linguistic_features[sorted(scene_indices)]
        
        linguistic_features_tensor = torch.tensor(linguistic_features, dtype=torch.float32).to(device)
        
        # Pre-compute scene embeddings (fixed background)
        with torch.no_grad():
            scene_embeddings = model.encode_scenes(scene_texts).to(device)
        
        # Create wrapper
        wrapper = ModelWrapperForSHAP(model, scene_embeddings, device)
        
        # Create background for SHAP explainers
        # DeepExplainer/GradientExplainer work better with a single background sample
        # They handle batching internally
        num_scenes = linguistic_features_tensor.shape[0]
        linguistic_dim = linguistic_features_tensor.shape[1]
        background_mean = linguistic_features_tensor.mean(dim=0, keepdim=True)  # [1, linguistic_dim]
        background = background_mean.to(device)  # [1, linguistic_dim] - single sample for gradient-based explainers
        
        # Ensure input tensor has gradients enabled for gradient-based explainers
        linguistic_features_tensor_grad = linguistic_features_tensor.clone().detach().requires_grad_(True)
        
        # Try GradientExplainer first (more robust for complex architectures)
        # Then DeepExplainer, then KernelExplainer as fallback
        shap_values_raw = None
        explainer = None
        gradient_explainer_success = False
        
        try:
            # GradientExplainer is more robust and handles complex models better
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values_raw = explainer.shap_values(linguistic_features_tensor_grad)
            gradient_explainer_success = True
        except Exception as e1:
            # Fallback to DeepExplainer
            try:
                explainer = shap.DeepExplainer(wrapper, background)
                shap_values_raw = explainer.shap_values(linguistic_features_tensor_grad)
                gradient_explainer_success = True
            except Exception as e2:
                # Will fall through to KernelExplainer fallback below
                error_msg = str(e2)
                gradient_explainer_success = False
        
        # Process SHAP values if gradient-based explainers succeeded
        if gradient_explainer_success and shap_values_raw is not None:
            # Handle different output formats from SHAP
            # SHAP can return: numpy array, list, tuple, or nested structures
            shap_values = None
            
            # Debug: Log what SHAP returned (only for first movie to avoid spam)
            if idx == sample_indices[0]:
                print(f"DEBUG: SHAP returned type: {type(shap_values_raw)}")
                if hasattr(shap_values_raw, 'shape'):
                    print(f"DEBUG: SHAP shape: {shap_values_raw.shape}")
                elif hasattr(shap_values_raw, '__len__'):
                    print(f"DEBUG: SHAP length: {len(shap_values_raw)}")
                    if len(shap_values_raw) > 0:
                        print(f"DEBUG: First element type: {type(shap_values_raw[0])}")
                        if hasattr(shap_values_raw[0], 'shape'):
                            print(f"DEBUG: First element shape: {shap_values_raw[0].shape}")
            
            if shap_values_raw is None:
                raise ValueError("SHAP returned None")
            
            # If it's a list or tuple, handle nested structures
            if isinstance(shap_values_raw, (list, tuple)):
                if len(shap_values_raw) == 0:
                    raise ValueError("SHAP returned empty list/tuple")
                
                try:
                    # Check if elements are arrays/tensors
                    first_elem = shap_values_raw[0]
                    
                    # If first element is also a list/tuple, we might have nested structure
                    if isinstance(first_elem, (list, tuple)):
                        # Flatten nested structure: take the first non-empty element
                        for elem in shap_values_raw:
                            if isinstance(elem, (list, tuple)) and len(elem) > 0:
                                try:
                                    inner_elem = elem[0]
                                    if isinstance(inner_elem, np.ndarray) or hasattr(inner_elem, 'numpy'):
                                        shap_values = inner_elem
                                    else:
                                        shap_values = np.array(inner_elem)
                                    break
                                except (IndexError, TypeError):
                                    continue
                        if shap_values is None:
                            # Last resort: try to extract from nested structure
                            try:
                                last_elem = shap_values_raw[-1]
                                if isinstance(last_elem, (list, tuple)) and len(last_elem) > 0:
                                    shap_values = last_elem[0]
                                else:
                                    shap_values = last_elem
                            except (IndexError, TypeError):
                                raise ValueError(f"Could not extract SHAP values from nested structure: {type(shap_values_raw)}")
                    else:
                        # Simple list/tuple of arrays
                        # For binary classification, SHAP might return [values_for_class_0, values_for_class_1]
                        # We want the positive class (class 1), which is typically the last element
                        if len(shap_values_raw) == 1:
                            shap_values = shap_values_raw[0]
                        else:
                            # Multiple classes: use the last one (positive class for binary)
                            try:
                                shap_values = shap_values_raw[-1]
                            except IndexError:
                                # Fallback to first element if last doesn't exist
                                shap_values = shap_values_raw[0]
                except (IndexError, TypeError) as e:
                    raise ValueError(f"Error accessing SHAP tuple/list structure: {e}. Type: {type(shap_values_raw)}, Length: {len(shap_values_raw) if hasattr(shap_values_raw, '__len__') else 'N/A'}")
            elif isinstance(shap_values_raw, np.ndarray):
                shap_values = shap_values_raw
            else:
                # Try to convert to numpy
                if hasattr(shap_values_raw, 'numpy'):
                    shap_values = shap_values_raw.numpy()
                elif hasattr(shap_values_raw, 'cpu'):
                    shap_values = shap_values_raw.cpu().numpy()
                elif hasattr(shap_values_raw, 'detach'):
                    shap_values = shap_values_raw.detach().cpu().numpy()
                else:
                    try:
                        shap_values = np.array(shap_values_raw)
                    except Exception as e:
                        raise ValueError(f"Could not convert SHAP output to numpy array: {e}")
            
            # Final check: ensure we have a valid array
            if shap_values is None:
                raise ValueError("Failed to extract SHAP values from output")
            
            # Convert to numpy if not already
            if not isinstance(shap_values, np.ndarray):
                if hasattr(shap_values, 'numpy'):
                    shap_values = shap_values.numpy()
                elif hasattr(shap_values, 'cpu'):
                    shap_values = shap_values.cpu().numpy()
                elif hasattr(shap_values, 'detach'):
                    shap_values = shap_values.detach().cpu().numpy()
                else:
                    shap_values = np.array(shap_values)
            
            # Ensure shap_values is 2D: [num_scenes, num_features]
            if shap_values.ndim == 0:
                # Scalar: expand to match expected shape
                shap_values = np.full((num_scenes, linguistic_dim), shap_values.item())
            elif shap_values.ndim == 1:
                # If 1D, check if it's per-feature or per-scene
                if shap_values.shape[0] == linguistic_dim:
                    # Per-feature: repeat for each scene
                    shap_values = np.tile(shap_values, (num_scenes, 1))
                elif shap_values.shape[0] == num_scenes:
                    # Per-scene: expand to include features (unlikely but handle it)
                    shap_values = shap_values.reshape(-1, 1)
                    shap_values = np.tile(shap_values, (1, linguistic_dim))
                else:
                    # Unknown: try to reshape
                    shap_values = shap_values.reshape(1, -1)
            elif shap_values.ndim > 2:
                # If 3D+, flatten extra dimensions
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            # Verify shape matches linguistic features
            if shap_values.shape[0] != linguistic_features_tensor.shape[0]:
                # Try to fix by repeating or slicing
                if shap_values.shape[0] == 1:
                    shap_values = np.tile(shap_values, (linguistic_features_tensor.shape[0], 1))
                elif shap_values.shape[0] > linguistic_features_tensor.shape[0]:
                    shap_values = shap_values[:linguistic_features_tensor.shape[0]]
                else:
                    raise ValueError(f"SHAP shape mismatch: {shap_values.shape[0]} scenes vs {linguistic_features_tensor.shape[0]} expected")
            
            if shap_values.shape[1] != linguistic_features_tensor.shape[1]:
                # Try to fix by repeating or slicing
                if shap_values.shape[1] == 1:
                    shap_values = np.tile(shap_values, (1, linguistic_features_tensor.shape[1]))
                elif shap_values.shape[1] > linguistic_features_tensor.shape[1]:
                    shap_values = shap_values[:, :linguistic_features_tensor.shape[1]]
                else:
                    raise ValueError(f"SHAP feature dimension mismatch: {shap_values.shape[1]} features vs {linguistic_features_tensor.shape[1]} expected")
            
            # Get predictions for this movie
            with torch.no_grad():
                logits = wrapper(linguistic_features_tensor)
                # Handle both Tensor and numpy returns
                # Logits shape is [batch_size, 1], squeeze to [batch_size] for predictions
                if isinstance(logits, torch.Tensor):
                    if logits.ndim > 1:
                        logits = logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                    predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                else:
                    logits_np = np.array(logits)
                    if logits_np.ndim > 1:
                        logits_np = logits_np.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                    predictions = (torch.sigmoid(torch.tensor(logits_np)) > 0.5).numpy()
            
            # Get ground truth labels
            labels = movie.get('labels')
            if labels is not None:
                labels_array = np.array(labels[:len(predictions)])
            else:
                labels_array = np.zeros(len(predictions))
            
            # Try to compute SHAP interaction values (optional, can be slow)
            # Only works with gradient-based explainers (GradientExplainer/DeepExplainer)
            try:
                # Use the gradient-enabled tensor for interaction values
                shap_interaction = explainer.shap_interaction_values(linguistic_features_tensor_grad)
                if isinstance(shap_interaction, (list, tuple)):
                    if len(shap_interaction) > 0:
                        shap_interaction = shap_interaction[0]
                    else:
                        shap_interaction = None
                all_shap_interaction_values.append(shap_interaction)
            except Exception as e:
                # Interaction values may not be available or may be too slow
                all_shap_interaction_values.append(None)
            
            all_shap_values.append(shap_values)
            all_linguistic_features.append(linguistic_features)
            all_predictions.append(predictions)
            all_targets.append(labels_array)
            all_scene_texts.append(scene_texts)
        
        # If gradient-based explainers failed, try KernelExplainer
        if not gradient_explainer_success:
            error_msg = "GradientExplainer and DeepExplainer both failed"
            # Try KernelExplainer as fallback if GradientExplainer/DeepExplainer fails
            print(f"  → GradientExplainer/DeepExplainer failed, trying KernelExplainer for movie {movie.get('movie_id', idx)}...")
            try:
                # Use KernelExplainer with sufficient background samples to avoid convergence warnings
                # KernelExplainer expects numpy arrays and works better with multiple background samples
                # Create background from the actual linguistic features (not the single-sample one)
                num_background = min(50, max(20, num_scenes // 2))  # Use 20-50 samples from actual data
                # Sample random scenes for background
                if num_scenes > num_background:
                    bg_indices = np.random.choice(num_scenes, num_background, replace=False)
                    background_np = linguistic_features[bg_indices]  # Use actual feature values
                else:
                    background_np = linguistic_features  # Use all scenes if we have few
                
                # Ensure it's numpy array
                if isinstance(background_np, torch.Tensor):
                    background_np = background_np.cpu().numpy()
                elif not isinstance(background_np, np.ndarray):
                    background_np = np.array(background_np)
                
                # Define a prediction function that handles conversion
                def predict_fn(data_np):
                    # Convert numpy -> tensor
                    data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        out_tensor = wrapper(data_tensor)
                        # Output is [batch_size, 1], squeeze to [batch_size] for KernelExplainer
                        if out_tensor.ndim > 1:
                            out_tensor = out_tensor.squeeze(-1)
                    return out_tensor.cpu().numpy()
                
                # Create KernelExplainer with L1 regularization to avoid convergence warnings
                # Use feature_perturbation="interventional" for better stability
                explainer_kernel = shap.KernelExplainer(
                    predict_fn, 
                    background_np,
                    feature_perturbation="interventional"  # More stable than "tree_path_dependent"
                )
                # Increase nsamples and use L1 regularization to avoid convergence warnings
                # For 979 features, we need more samples: use min(200, 2*num_features) or at least 100
                num_features = linguistic_features_tensor.shape[1]
                nsamples = max(100, min(200, 2 * num_features // 10))  # Adaptive based on feature count
                shap_values_raw = explainer_kernel.shap_values(
                    linguistic_features_tensor.cpu().numpy(), 
                    nsamples=nsamples,
                    l1_reg="num_features(50)"  # L1 regularization: explain top 50 features
                )
                
                # Process the output (same logic as above)
                if isinstance(shap_values_raw, (list, tuple)) and len(shap_values_raw) > 0:
                    shap_values = shap_values_raw[-1] if len(shap_values_raw) > 1 else shap_values_raw[0]
                elif isinstance(shap_values_raw, np.ndarray):
                    shap_values = shap_values_raw
                else:
                    raise ValueError(f"KernelExplainer returned unexpected type: {type(shap_values_raw)}")
                
                # Convert to numpy and ensure correct shape
                if not isinstance(shap_values, np.ndarray):
                    if hasattr(shap_values, 'numpy'):
                        shap_values = shap_values.numpy()
                    elif hasattr(shap_values, 'cpu'):
                        shap_values = shap_values.cpu().numpy()
                    else:
                        shap_values = np.array(shap_values)
                
                # Ensure 2D shape
                if shap_values.ndim == 1:
                    if shap_values.shape[0] == linguistic_dim:
                        shap_values = np.tile(shap_values, (num_scenes, 1))
                    else:
                        shap_values = shap_values.reshape(1, -1)
                elif shap_values.ndim > 2:
                    shap_values = shap_values.reshape(shap_values.shape[0], -1)
                
                # Verify and fix shape
                if shap_values.shape[0] != num_scenes:
                    if shap_values.shape[0] == 1:
                        shap_values = np.tile(shap_values, (num_scenes, 1))
                    else:
                        shap_values = shap_values[:num_scenes]
                
                if shap_values.shape[1] != linguistic_dim:
                    if shap_values.shape[1] == 1:
                        shap_values = np.tile(shap_values, (1, linguistic_dim))
                    else:
                        shap_values = shap_values[:, :linguistic_dim]
                
                # Get predictions
                with torch.no_grad():
                    logits = wrapper(linguistic_features_tensor)
                    # Handle both Tensor and numpy returns
                    # Logits shape is [batch_size, 1], squeeze to [batch_size] for predictions
                    if isinstance(logits, torch.Tensor):
                        if logits.ndim > 1:
                            logits = logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                        predictions = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    else:
                        logits_np = np.array(logits)
                        if logits_np.ndim > 1:
                            logits_np = logits_np.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                        predictions = (torch.sigmoid(torch.tensor(logits_np)) > 0.5).numpy()
                
                # Get labels
                labels = movie.get('labels')
                labels_array = np.array(labels[:len(predictions)]) if labels is not None else np.zeros(len(predictions))
                
                # Append results
                all_shap_values.append(shap_values)
                all_linguistic_features.append(linguistic_features)
                all_predictions.append(predictions)
                all_targets.append(labels_array)
                all_scene_texts.append(scene_texts)
                all_shap_interaction_values.append(None)  # Skip interaction for KernelExplainer
                continue  # Success with KernelExplainer
            except Exception as e2:
                print(f"Warning: Could not compute SHAP for movie {movie.get('movie_id', idx)}: {error_msg} (KernelExplainer also failed: {e2})")
            continue
    
    if len(all_shap_values) == 0:
        print("Warning: No SHAP values computed. Skipping visualization.")
        return {}
    
    # Concatenate all SHAP values
    all_shap_array = np.vstack(all_shap_values)  # [total_scenes, num_features]
    all_ling_array = np.vstack(all_linguistic_features)  # [total_scenes, num_features]
    all_targets_array = np.concatenate(all_targets)  # [total_scenes]
    all_predictions_array = np.concatenate(all_predictions)  # [total_scenes]
    
    print(f"\nComputed SHAP values for {len(all_shap_array)} scenes across {len(all_shap_values)} movies")
    
    # Generate plots
    shap_output_dir = os.path.join(exp_output_dir, "shap_analysis")
    os.makedirs(shap_output_dir, exist_ok=True)
    
    # 1. Global Summary Plot (Feature Importance)
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    # Use explicit RNG to avoid FutureWarning about NumPy global RNG
    rng = np.random.default_rng(42)
    try:
        shap.summary_plot(
            all_shap_array,
            all_ling_array,
            feature_names=feature_names[:all_shap_array.shape[1]],
            show=False,
            max_display=20,  # Top 20 features
            rng=rng  # Pass explicit RNG to silence FutureWarning
        )
    except TypeError:
        # Older SHAP versions don't support rng parameter
        shap.summary_plot(
            all_shap_array,
            all_ling_array,
            feature_names=feature_names[:all_shap_array.shape[1]],
            show=False,
            max_display=20  # Top 20 features
        )
    plt.title("SHAP Summary: How Linguistic Features Drive Saliency Decisions", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'shap_summary.png')}")
    
    # 2. Bar Plot (Mean Absolute SHAP Values - Feature Importance)
    print("Generating SHAP feature importance plot...")
    mean_abs_shap = np.abs(all_shap_array).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_indices)), mean_abs_shap[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel("Mean |SHAP Value| (Feature Importance)")
    plt.title("Top 20 Most Important Linguistic Features for Saliency Prediction", fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, "shap_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'shap_feature_importance.png')}")
    
    # 3. Waterfall Plot for a specific scene (example: highest prediction)
    print("Generating SHAP waterfall plot for example scene...")
    all_pred_probs = []
    for preds in all_predictions:
        all_pred_probs.extend(preds)
    all_pred_probs = np.array(all_pred_probs)
    
    if len(all_pred_probs) > 0:
        # Find scene with highest prediction probability
        max_idx = np.argmax(all_pred_probs)
        scene_shap = all_shap_array[max_idx]
        scene_features = all_ling_array[max_idx]
        
        # Create waterfall plot (using Explanation object for newer SHAP versions)
        try:
            plt.figure(figsize=(12, 8))
            explanation = shap.Explanation(
                values=scene_shap,
                base_values=0.0,  # Baseline
                data=scene_features,
                feature_names=feature_names[:len(scene_shap)] if len(feature_names) >= len(scene_shap) else [f"Feature_{i}" for i in range(len(scene_shap))]
            )
            shap.waterfall_plot(explanation, show=False, max_display=15)
            plt.title(f"SHAP Waterfall: Example High-Saliency Scene (Prob: {all_pred_probs[max_idx]:.3f})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_output_dir, "shap_waterfall_example.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'shap_waterfall_example.png')}")
        except Exception as e:
            print(f"  Warning: Could not generate waterfall plot: {e}")
            print("  (This is okay - summary and importance plots are more important)")
    
    # Save SHAP values and analysis
    shap_results = {
        'shap_values': all_shap_array.tolist(),
        'linguistic_features': all_ling_array.tolist(),
        'feature_names': feature_names[:all_shap_array.shape[1]],
        'mean_abs_shap': mean_abs_shap.tolist(),
        'top_features': [feature_names[i] for i in top_indices],
        'num_scenes_analyzed': len(all_shap_array),
        'num_movies_analyzed': len(all_shap_values),
    }
    
    with open(os.path.join(shap_output_dir, "shap_results.json"), "w") as f:
        json.dump(shap_results, f, indent=2, default=str)
    
    # Additional interpretability analyses
    print("\n" + "="*80)
    print("Running Additional Interpretability Analyses")
    print("="*80)
    
    # 4. SHAP Interaction Values
    if any(x is not None for x in all_shap_interaction_values):
        print("\n4. Computing SHAP interaction values...")
        try:
            _analyze_shap_interactions(all_shap_interaction_values, feature_names, top_indices, shap_output_dir)
        except Exception as e:
            print(f"  Warning: Could not compute interaction values: {e}")
    
    # 5. Case Study Analysis (TP/FP/FN/TN breakdown)
    print("\n5. Analyzing predictions by category (TP/FP/FN/TN)...")
    try:
        _analyze_by_prediction_category(all_shap_array, all_predictions_array, all_targets_array, feature_names, shap_output_dir)
    except Exception as e:
        print(f"  Warning: Could not analyze by category: {e}")
    
    # 6. Feature Value Distribution vs SHAP Impact
    print("\n6. Analyzing feature value vs SHAP impact...")
    try:
        _plot_feature_value_vs_shap(all_shap_array, all_ling_array, feature_names, top_indices, shap_output_dir)
    except Exception as e:
        print(f"  Warning: Could not plot feature value vs SHAP: {e}")
    
    # 7. Statistical Summary Table
    print("\n7. Generating feature statistics table...")
    try:
        stats_df = _generate_feature_statistics_table(all_shap_array, all_ling_array, feature_names, shap_output_dir)
        shap_results['feature_statistics'] = stats_df.to_dict('records')
    except Exception as e:
        print(f"  Warning: Could not generate statistics table: {e}")
    
    # 8. Attention Visualization (if using attention/gated fusion)
    if model.fusion_method in ["attention", "gated"]:
        print("\n8. Visualizing attention/gate weights...")
        try:
            _visualize_attention_weights(model, test_dataset, device, shap_output_dir, num_samples=5)
        except Exception as e:
            print(f"  Warning: Could not visualize attention: {e}")
    
    # 9. Ablation Study (feature removal impact)
    print("\n9. Running ablation study (feature removal impact)...")
    try:
        ablation_results = _compute_ablation_study(model, test_dataset, device, feature_names, shap_output_dir, mean_abs_shap=mean_abs_shap, num_samples=10)
        shap_results['ablation_study'] = ablation_results
    except Exception as e:
        print(f"  Warning: Could not run ablation study: {e}")
    
    # 10. Compare BERT Attention vs Linguistic Features
    print("\n10. Comparing BERT Attention vs Linguistic Features...")
    try:
        # Flatten all_scene_texts for easy indexing
        flattened_scene_texts = []
        for movie_scenes in all_scene_texts:
            flattened_scene_texts.extend(movie_scenes)
        
        _compare_bert_attention_vs_linguistic_features(
            model, test_dataset, device, shap_output_dir,
            all_shap_array, all_ling_array, feature_names, flattened_scene_texts, num_examples=3
        )
    except Exception as e:
        print(f"  Warning: Could not compare BERT attention: {e}")
        import traceback
        traceback.print_exc()
    
    # 11. Feature Reliance Plot (Gate Values for Gated Fusion)
    if model.fusion_method == "gated":
        print("\n11. Generating Feature Reliance Plot (Gate Values)...")
        try:
            _plot_feature_reliance_gates(model, test_dataset, device, shap_output_dir, num_examples=5)
        except Exception as e:
            print(f"  Warning: Could not generate feature reliance plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Update results with additional data
    shap_results['all_targets'] = all_targets_array.tolist()
    shap_results['all_predictions'] = all_predictions_array.tolist()
    
    print(f"\n✓ SHAP analysis complete! Results saved to: {shap_output_dir}")
    print(f"\nGenerated Files:")
    print(f"  ✓ Summary plot: shap_summary.png")
    print(f"  ✓ Feature importance: shap_feature_importance.png")
    print(f"  ✓ Example waterfall: shap_waterfall_example.png")
    print(f"  ✓ Interaction values: shap_interactions.png")
    print(f"  ✓ Category analysis: shap_by_category.png")
    print(f"  ✓ Feature value analysis: feature_value_vs_shap.png")
    print(f"  ✓ Statistics table: feature_statistics.csv (and .tex)")
    if model.fusion_method in ["attention", "gated"]:
        print(f"  ✓ Attention/gate visualization: attention_weights.png or gate_strength_*.png")
    print(f"  ✓ Ablation study: ablation_study.png")
    print(f"  ✓ BERT vs Linguistic comparison: bert_vs_linguistic_comparison_example_*.png")
    if model.fusion_method == "gated":
        print(f"  ✓ Feature reliance plot: feature_reliance_gates_*.png")
    print(f"  ✓ Raw data: shap_results.json")
    
    return shap_results


def _analyze_shap_interactions(all_shap_interaction_values, feature_names, top_indices, shap_output_dir):
    """Analyze SHAP interaction values to show feature interactions."""
    if not SEABORN_AVAILABLE:
        print("  Skipping: seaborn not available")
        return
    
    # Filter to only non-None interaction values
    valid_interactions = [x for x in all_shap_interaction_values if x is not None]
    if len(valid_interactions) == 0:
        print("  No interaction values available")
        return
    
    # Average interaction values across all samples
    # Shape: [num_samples, num_scenes, num_features, num_features]
    # We want to average over samples and scenes
    all_interactions = np.stack(valid_interactions)  # [num_movies, num_scenes, num_features, num_features]
    mean_interactions = all_interactions.mean(axis=(0, 1))  # [num_features, num_features]
    
    # Focus on top features
    top_5_indices = top_indices[:5]
    interaction_matrix = mean_interactions[np.ix_(top_5_indices, top_5_indices)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix,
                xticklabels=[feature_names[i] for i in top_5_indices],
                yticklabels=[feature_names[i] for i in top_5_indices],
                annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'SHAP Interaction Value'})
    plt.title("SHAP Interaction Values: How Top Features Interact", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, "shap_interactions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'shap_interactions.png')}")


def _analyze_by_prediction_category(shap_values, predictions, targets, feature_names, shap_output_dir):
    """Analyze SHAP values separately for TP, FP, FN, TN."""
    tp_mask = (predictions == 1) & (targets == 1)
    fp_mask = (predictions == 1) & (targets == 0)
    fn_mask = (predictions == 0) & (targets == 1)
    tn_mask = (predictions == 0) & (targets == 0)
    
    categories = {
        'True Positives': tp_mask,
        'False Positives': fp_mask,
        'False Negatives': fn_mask,
        'True Negatives': tn_mask
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (category, mask) in enumerate(categories.items()):
        if mask.sum() > 0:
            category_shap = shap_values[mask].mean(axis=0)
            top_indices = np.argsort(np.abs(category_shap))[-10:][::-1]
            
            colors = ['green' if category_shap[i] > 0 else 'red' for i in top_indices]
            axes[idx].barh(range(len(top_indices)), category_shap[top_indices], color=colors, alpha=0.7)
            axes[idx].set_yticks(range(len(top_indices)))
            axes[idx].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
            axes[idx].set_xlabel('Mean SHAP Value', fontsize=10)
            axes[idx].set_title(f'{category} (n={mask.sum()})', fontsize=11, fontweight='bold')
            axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
        else:
            axes[idx].text(0.5, 0.5, f'No {category}', ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{category} (n=0)', fontsize=11)
    
    plt.suptitle('SHAP Values by Prediction Category', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, "shap_by_category.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'shap_by_category.png')}")


def _plot_feature_value_vs_shap(shap_values, feature_values, feature_names, top_indices, shap_output_dir):
    """Plot feature value distribution vs SHAP impact."""
    top_n = min(10, len(top_indices))
    top_feature_indices = top_indices[:top_n]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_feature_indices):
        feat_name = feature_names[feat_idx]
        feat_vals = feature_values[:, feat_idx]
        feat_shap = shap_values[:, feat_idx]
        
        # Create scatter plot
        axes[idx].scatter(feat_vals, feat_shap, alpha=0.3, s=10, c=feat_shap, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[idx].set_xlabel('Feature Value', fontsize=9)
        axes[idx].set_ylabel('SHAP Value', fontsize=9)
        axes[idx].set_title(feat_name[:30] + ('...' if len(feat_name) > 30 else ''), fontsize=9)
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
        axes[idx].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(feat_vals) > 1:
            corr = np.corrcoef(feat_vals, feat_shap)[0, 1]
            axes[idx].text(0.05, 0.95, f'r={corr:.2f}', transform=axes[idx].transAxes,
                          fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('Feature Value vs SHAP Impact (Top 10 Features)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, "feature_value_vs_shap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'feature_value_vs_shap.png')}")


def _generate_feature_statistics_table(shap_values, feature_values, feature_names, shap_output_dir):
    """Generate comprehensive statistics table for paper."""
    stats = []
    
    for idx, feat_name in enumerate(feature_names):
        feat_shap = shap_values[:, idx]
        feat_vals = feature_values[:, idx]
        
        # Compute statistics
        mean_abs_shap = np.abs(feat_shap).mean()
        std_shap = feat_shap.std()
        positive_shap_pct = (feat_shap > 0).mean() * 100
        
        mean_feat_val = feat_vals.mean()
        std_feat_val = feat_vals.std()
        
        # Correlation between feature value and SHAP
        if len(feat_vals) > 1 and feat_vals.std() > 1e-6:
            corr = np.corrcoef(feat_vals, feat_shap)[0, 1]
        else:
            corr = 0.0
        
        stats.append({
            'Feature': feat_name,
            'Mean_|SHAP|': mean_abs_shap,
            'Std_SHAP': std_shap,
            'Positive_SHAP_%': positive_shap_pct,
            'Mean_Feature_Value': mean_feat_val,
            'Std_Feature_Value': std_feat_val,
            'Correlation_Value_vs_SHAP': corr
        })
    
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values('Mean_|SHAP|', ascending=False)
    
    # Save as CSV
    df_stats.to_csv(os.path.join(shap_output_dir, "feature_statistics.csv"), index=False, float_format="%.4f")
    
    # Save as LaTeX table (top 20)
    df_stats_top20 = df_stats.head(20)
    latex_str = df_stats_top20.to_latex(index=False, float_format="%.4f", escape=False)
    with open(os.path.join(shap_output_dir, "feature_statistics.tex"), "w") as f:
        f.write(latex_str)
    
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'feature_statistics.csv')}")
    print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'feature_statistics.tex')}")
    
    return df_stats


def _compute_ablation_study(model, test_dataset, device, feature_names, shap_output_dir, mean_abs_shap=None, num_samples=10):
    """
    Compute ablation study by removing top features and measuring impact.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
        feature_names: List of feature names
        shap_output_dir: Output directory for plots
        mean_abs_shap: Mean absolute SHAP values (for feature ranking)
        num_samples: Number of movies to analyze
    
    Returns:
        Dictionary with ablation results
    """
    try:
        model.eval()
        sample_indices = np.random.choice(len(test_dataset.movies), min(num_samples, len(test_dataset.movies)), replace=False)
        
        ablation_results = {
            'removed_features': [],
            'performance_drops': [],
            'top_features_impact': {}
        }
        
        # Get baseline performance
        baseline_predictions = []
        baseline_targets = []
        
        for idx in sample_indices:
            movie = test_dataset.movies[idx]
            if movie.get('labels') is None:
                continue
            
            scene_texts = movie['scene_texts']
            labels = movie['labels']
            linguistic_features = movie.get('linguistic_features')
            
            if linguistic_features is None:
                continue
            
            # Get baseline predictions
            with torch.no_grad():
                linguistic_tensor = torch.tensor(linguistic_features, dtype=torch.float32).to(device)
                # Get embeddings
                scene_embeddings = test_dataset.get_scene_embeddings(movie['movie_id'], 'test')
                if scene_embeddings is None:
                    continue
                scene_embeddings = torch.tensor(scene_embeddings, dtype=torch.float32).to(device)
                
                logits = model(scene_texts=scene_texts, linguistic_features=linguistic_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                baseline_predictions.extend(probs.flatten())
                baseline_targets.extend(labels)
        
        if len(baseline_predictions) == 0:
            return ablation_results
        
        baseline_f1 = f1_score(baseline_targets, (np.array(baseline_predictions) > 0.5).astype(int))
        
        # Ablate top features one by one
        if mean_abs_shap is not None and len(mean_abs_shap) > 0:
            top_indices = np.argsort(mean_abs_shap)[-10:][::-1]  # Top 10 features
            
            for feat_idx in top_indices[:5]:  # Test top 5
                if feat_idx >= len(feature_names):
                    continue
                
                # Remove this feature and recompute
                ablated_predictions = []
                ablated_targets = []
                
                for idx in sample_indices:
                    movie = test_dataset.movies[idx]
                    if movie.get('labels') is None:
                        continue
                    
                    linguistic_features = movie.get('linguistic_features')
                    if linguistic_features is None:
                        continue
                    
                    # Create ablated features (set feature to 0)
                    ablated_features = linguistic_features.copy()
                    ablated_features[:, feat_idx] = 0
                    
                    with torch.no_grad():
                        linguistic_tensor = torch.tensor(ablated_features, dtype=torch.float32).to(device)
                        scene_texts = movie['scene_texts']
                        scene_embeddings = test_dataset.get_scene_embeddings(movie['movie_id'], 'test')
                        if scene_embeddings is None:
                            continue
                        scene_embeddings = torch.tensor(scene_embeddings, dtype=torch.float32).to(device)
                        
                        logits = model(scene_texts=scene_texts, linguistic_features=linguistic_tensor)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        ablated_predictions.extend(probs.flatten())
                        ablated_targets.extend(movie['labels'])
                
                if len(ablated_predictions) > 0:
                    ablated_f1 = f1_score(ablated_targets, (np.array(ablated_predictions) > 0.5).astype(int))
                    performance_drop = baseline_f1 - ablated_f1
                    
                    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
                    ablation_results['removed_features'].append(feat_name)
                    ablation_results['performance_drops'].append(performance_drop)
                    ablation_results['top_features_impact'][feat_name] = performance_drop
        
        # Save ablation results
        import pandas as pd
        if ablation_results['removed_features']:
            ablation_df = pd.DataFrame({
                'feature': ablation_results['removed_features'],
                'performance_drop': ablation_results['performance_drops']
            })
            ablation_df = ablation_df.sort_values('performance_drop', ascending=False)
            ablation_df.to_csv(os.path.join(shap_output_dir, "ablation_study.csv"), index=False)
            
            # Plot ablation results
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(ablation_df)), ablation_df['performance_drop'])
            plt.yticks(range(len(ablation_df)), ablation_df['feature'])
            plt.xlabel('Performance Drop (F1 Score)')
            plt.title('Ablation Study: Impact of Removing Top Features')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_output_dir, "ablation_study.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'ablation_study.png')}")
        
        return ablation_results
    except Exception as e:
        print(f"  Warning: Error in ablation study: {e}")
        import traceback
        traceback.print_exc()
        return {'removed_features': [], 'performance_drops': [], 'top_features_impact': {}}


def _visualize_attention_weights(model, test_dataset, device, shap_output_dir, num_samples=5):
    """Extract and visualize attention/gate weights from fusion layer."""
    model.eval()
    
    sample_indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    all_attention_maps = []
    
    for idx in sample_indices:
        movie = test_dataset[idx]
        scene_texts = movie['scene_texts']
        linguistic_features = movie.get('linguistic_features')
        
        if linguistic_features is None or len(scene_texts) == 0:
            continue
        
        # Limit to reasonable number of scenes
        if len(scene_texts) > 50:
            scene_texts = scene_texts[:50]
            linguistic_features = linguistic_features[:50]
        
        linguistic_features_tensor = torch.tensor(linguistic_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scene_emb = model.encode_scenes(scene_texts).to(device)
            
            if model.fusion_method == "attention":
                scene_proj = model.scene_proj(scene_emb)
                ling_proj = model.ling_proj(linguistic_features_tensor)
                
                # Get attention weights
                fused, attn_weights = model.attention_fusion(
                    scene_proj.unsqueeze(0),
                    ling_proj.unsqueeze(0),
                    ling_proj.unsqueeze(0),
                    average_attn_weights=False
                )
                
                # attn_weights shape: [num_heads, num_scenes, num_scenes]
                # Average over heads
                attn_avg = attn_weights.mean(dim=0).cpu().numpy()  # [num_scenes, num_scenes]
                all_attention_maps.append(attn_avg)
            
            elif model.fusion_method == "gated":
                # For gated fusion, visualize the gate values
                scene_proj = model.scene_proj(scene_emb)
                ling_proj = model.ling_proj(linguistic_features_tensor)
                
                # Compute gate
                gate_input = scene_proj + ling_proj
                gate = torch.sigmoid(model.gate(gate_input))  # [num_scenes, hidden_dim]
                
                # Average gate over hidden dimension to get per-scene gate strength
                gate_strength = gate.mean(dim=1).cpu().numpy()  # [num_scenes]
                
                # Create a simple visualization
                plt.figure(figsize=(12, 6))
                plt.plot(gate_strength, marker='o', linewidth=2, markersize=4)
                plt.xlabel('Scene Index')
                plt.ylabel('Gate Strength (Average)')
                plt.title(f'Gated Fusion: Gate Strength Across Scenes (Movie {movie.get("movie_id", idx)})')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_output_dir, f"gate_strength_movie_{movie.get('movie_id', idx)}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    # For attention, create average heatmap
    if model.fusion_method == "attention" and len(all_attention_maps) > 0:
        avg_attention = np.mean(all_attention_maps, axis=0)
        
        plt.figure(figsize=(12, 10))
        if SEABORN_AVAILABLE:
            sns.heatmap(avg_attention, cmap='Blues', cbar=True, square=True)
        else:
            plt.imshow(avg_attention, cmap='Blues', aspect='auto')
            plt.colorbar()
        plt.xlabel('Linguistic Feature Position')
        plt.ylabel('Scene Position')
        plt.title('Attention Weights: How Scenes Attend to Linguistic Features (Averaged)')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_output_dir, "attention_weights.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {os.path.join(shap_output_dir, 'attention_weights.png')}")
    elif model.fusion_method == "gated":
        print(f"  ✓ Saved: Gate strength plots for {len(sample_indices)} movies")


def _compare_bert_attention_vs_linguistic_features(
    model, 
    test_dataset, 
    device, 
    shap_output_dir,
    all_shap_array,
    all_ling_array,
    feature_names,
    all_scene_texts,
    num_examples=3
):
    """
    Compare BERT attention (black-box) vs Linguistic Features (interpretable).
    
    Shows side-by-side:
    - BERT Attention: Highlights random words like "the", "and", punctuation (hard to explain).
    - Linguistic Features: Highlights interpretable features like "Centrality Score" (easy to explain).
    """
    model.eval()
    
    # Find scenes with high saliency predictions and good SHAP values
    if len(all_scene_texts) == 0:
        print("  No scene texts available for comparison")
        return
    
    # Sample a few interesting scenes (prefer scenes with high SHAP values)
    if len(all_shap_array) > 0:
        # Find scenes with high absolute SHAP values (interesting predictions)
        shap_magnitudes = np.abs(all_shap_array).sum(axis=1)
        top_shap_indices = np.argsort(shap_magnitudes)[-min(num_examples * 3, len(all_shap_array)):][::-1]
        sample_indices = np.random.choice(top_shap_indices, min(num_examples, len(top_shap_indices)), replace=False)
    else:
        sample_indices = np.random.choice(len(all_scene_texts), min(num_examples, len(all_scene_texts)), replace=False)
    
    for example_idx, scene_idx in enumerate(sample_indices):
        if scene_idx >= len(all_scene_texts):
            continue
            
        scene_text = all_scene_texts[scene_idx]
        if scene_idx < len(all_shap_array):
            scene_shap = all_shap_array[scene_idx]
            scene_features = all_ling_array[scene_idx]
        else:
            continue
        
        # Get top SHAP features for this scene
        top_shap_indices = np.argsort(np.abs(scene_shap))[-10:][::-1]
        top_shap_features = [(feature_names[i] if i < len(feature_names) else f"Feature_{i}", 
                            scene_shap[i], scene_features[i]) for i in top_shap_indices]
        
        # Extract BERT attention weights
        try:
            with torch.no_grad():
                # Tokenize scene
                encoded = model.scene_tokenizer(
                    scene_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get attention weights from BERT
                # We need to set output_attentions=True
                outputs = model.scene_encoder(**encoded, output_attentions=True)
                
                # Get attention from all layers and heads
                # Shape: tuple of [batch, num_heads, seq_len, seq_len] for each layer
                attentions = outputs.attentions  # List of [batch, num_heads, seq_len, seq_len]
                
                # Average over all layers and heads
                # attentions is a tuple, each element is [1, num_heads, seq_len, seq_len]
                all_attentions = torch.stack([attn.squeeze(0) for attn in attentions])  # [num_layers, num_heads, seq_len, seq_len]
                avg_attention = all_attentions.mean(dim=(0, 1))  # Average over layers and heads: [seq_len, seq_len]
                
                # Get attention to [CLS] token (or average attention)
                # [CLS] is usually token 0
                cls_attention = avg_attention[0, :].cpu().numpy()  # Attention from [CLS] to all tokens
                
                # Get tokens
                token_ids = encoded['input_ids'][0].cpu().numpy()
                tokens = model.scene_tokenizer.convert_ids_to_tokens(token_ids)
                
                # Get top attended tokens (exclude [CLS] and [SEP] and padding)
                valid_indices = [i for i in range(len(tokens)) if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]'] and i < len(cls_attention)]
                if len(valid_indices) == 0:
                    continue
                    
                valid_attention = np.array([cls_attention[i] for i in valid_indices])
                valid_tokens = [tokens[i] for i in valid_indices]
                
                top_token_indices = np.argsort(valid_attention)[-15:][::-1]
                top_tokens = [(valid_tokens[i], valid_attention[i]) for i in top_token_indices if i < len(valid_tokens)]
                
        except Exception as e:
            print(f"  Warning: Could not extract BERT attention for example {example_idx + 1}: {e}")
            continue
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left: BERT Attention (Black-box)
        if len(top_tokens) > 0:
            ax1.barh(range(len(top_tokens)), [t[1] for t in top_tokens], color='red', alpha=0.7)
            ax1.set_yticks(range(len(top_tokens)))
            ax1.set_yticklabels([t[0][:20] + ('...' if len(t[0]) > 20 else '') for t in top_tokens], fontsize=10)
            ax1.set_xlabel('Attention Weight', fontsize=11, fontweight='bold')
            ax1.set_title('BERT Attention (Black-box)\nFocuses on: "the", "and", punctuation', 
                         fontsize=12, fontweight='bold', color='red')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Highlight common stopwords/punctuation
            stopwords_punct = ['the', 'and', 'a', 'an', 'to', 'of', 'in', 'for', 'is', 'on', 
                              '.', ',', '!', '?', '[CLS]', '[SEP]', 'Ġ', 'Ċ', 'ĉ']
            for i, (token, _) in enumerate(top_tokens):
                token_lower = token.lower().replace('Ġ', '').replace('Ċ', '').replace('ĉ', '')
                if any(sw in token_lower for sw in stopwords_punct) or len(token_lower.strip()) <= 1:
                    if i < len(ax1.patches):
                        ax1.patches[i].set_color('darkred')
        
        # Right: Linguistic Features (Interpretable)
        if len(top_shap_features) > 0:
            ax2.barh(range(len(top_shap_features)), [f[1] for f in top_shap_features], 
                    color='green', alpha=0.7)
            ax2.set_yticks(range(len(top_shap_features)))
            ax2.set_yticklabels([f[0][:40] + ('...' if len(f[0]) > 40 else '') for f in top_shap_features], 
                               fontsize=10)
            ax2.set_xlabel('SHAP Value (Feature Contribution)', fontsize=11, fontweight='bold')
            ax2.set_title('Linguistic Features (Interpretable)\nFocuses on: Centrality, RST, Emotion scores', 
                         fontsize=12, fontweight='bold', color='green')
            ax2.invert_yaxis()
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Color code: green for positive, red for negative
            for i, (_, shap_val, _) in enumerate(top_shap_features):
                if i < len(ax2.patches):
                    if shap_val > 0:
                        ax2.patches[i].set_color('green')
                    else:
                        ax2.patches[i].set_color('red')
        
        plt.suptitle(f'Interpretability Comparison: Example Scene {example_idx + 1}\n'
                    f'Why Our Method is More Explainable', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_output_dir, f"bert_vs_linguistic_comparison_example_{example_idx + 1}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a detailed token-level attention visualization
        if len(tokens) <= 50 and len(top_shap_features) > 0:  # Only for shorter scenes
            try:
                _visualize_bert_attention_tokens(tokens, cls_attention, top_shap_features, 
                                                shap_output_dir, example_idx)
            except Exception as e:
                print(f"    Warning: Could not create detailed visualization: {e}")
    
    print(f"  ✓ Saved: {len(sample_indices)} comparison plots to {shap_output_dir}")


def _plot_feature_reliance_gates(model, test_dataset, device, shap_output_dir, num_examples=5):
    """
    Generate Feature Reliance Plot showing gate values across scenes.
    
    Shows when the model relies on linguistic features (high gate) vs pure semantics (low gate).
    High gate = model trusts linguistic features (e.g., during transition scenes)
    Low gate = model relies on pure semantics (e.g., during high-dialogue scenes)
    """
    model.eval()
    
    sample_indices = np.random.choice(len(test_dataset), min(num_examples, len(test_dataset)), replace=False)
    
    print(f"  Analyzing {len(sample_indices)} movies for gate values...")
    
    for example_idx, movie_idx in enumerate(sample_indices):
        movie = test_dataset[movie_idx]
        scene_texts = movie['scene_texts']
        linguistic_features = movie.get('linguistic_features')
        scene_indices = movie.get('scene_indices', list(range(len(scene_texts))))
        movie_id = movie.get('movie_id', movie_idx)
        
        if linguistic_features is None or len(scene_texts) == 0:
            continue
        
        # Limit to reasonable number of scenes for visualization
        if len(scene_texts) > 100:
            # Sample scenes evenly
            step = len(scene_texts) // 100
            scene_texts = scene_texts[::step][:100]
            linguistic_features = linguistic_features[::step][:100]
            scene_indices = scene_indices[::step][:100]
        
        linguistic_features_tensor = torch.tensor(linguistic_features, dtype=torch.float32).to(device)
        
        try:
            with torch.no_grad():
                # Encode scenes
                scene_embeddings = model.encode_scenes(scene_texts).to(device)
                
                # Extract gate values from gated fusion
                h_scene = model.scene_proj(scene_embeddings)  # [num_scenes, hidden_dim]
                h_ling = model.ling_proj(linguistic_features_tensor)  # [num_scenes, hidden_dim]
                
                # Calculate gate
                gate_input = torch.cat([h_scene, h_ling], dim=-1)  # [num_scenes, hidden_dim * 2]
                g = model.gate(gate_input)  # [num_scenes, hidden_dim]
                
                # Average gate over hidden dimension to get per-scene gate strength
                gate_values = g.mean(dim=1).cpu().numpy()  # [num_scenes]
                
                # Get predictions for context
                fused = g * h_ling + (1 - g) * h_scene
                fused = model.fusion_norm(fused)
                fused = fused.unsqueeze(0)
                if model.use_positional_encoding:
                    fused = model.pos_encoder(fused)
                sequence_output = model.apply_sequence_model(fused, mask=None)
                combined_features = torch.cat([sequence_output, fused], dim=-1)
                logits = model.classifier(combined_features).squeeze(0).squeeze(-1)
                predictions = torch.sigmoid(logits).cpu().numpy()
                
        except Exception as e:
            print(f"  Warning: Could not extract gate values for movie {movie_id}: {e}")
            continue
        
        # Create Feature Reliance Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top plot: Gate values over scene indices
        ax1.plot(scene_indices, gate_values, marker='o', linewidth=2, markersize=4, 
                color='blue', alpha=0.7, label='Gate Value (Linguistic Feature Reliance)')
        ax1.fill_between(scene_indices, gate_values, alpha=0.3, color='blue')
        ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (0.5)')
        ax1.set_ylabel('Gate Value', fontsize=12, fontweight='bold')
        ax1.set_title(f'Feature Reliance Plot: Movie {movie_id}\n'
                     f'High Gate = Relies on Linguistic Features | Low Gate = Relies on Pure Semantics', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Add annotations for high/low gate regions
        high_gate_threshold = 0.7
        low_gate_threshold = 0.3
        
        high_gate_regions = [(i, g) for i, g in zip(scene_indices, gate_values) if g > high_gate_threshold]
        low_gate_regions = [(i, g) for i, g in zip(scene_indices, gate_values) if g < low_gate_threshold]
        
        if len(high_gate_regions) > 0:
            # Annotate a few high gate scenes
            for idx, (scene_idx, gate_val) in enumerate(high_gate_regions[:3]):
                ax1.annotate('High\nReliance', 
                           xy=(scene_idx, gate_val), 
                           xytext=(10, 20), 
                           textcoords='offset points',
                           fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if len(low_gate_regions) > 0:
            # Annotate a few low gate scenes
            for idx, (scene_idx, gate_val) in enumerate(low_gate_regions[:3]):
                ax1.annotate('Low\nReliance', 
                           xy=(scene_idx, gate_val), 
                           xytext=(10, -20), 
                           textcoords='offset points',
                           fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Bottom plot: Predictions over scene indices (for context)
        ax2.plot(scene_indices, predictions, marker='s', linewidth=2, markersize=4, 
                color='purple', alpha=0.7, label='Saliency Prediction')
        ax2.fill_between(scene_indices, predictions, alpha=0.3, color='purple')
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.5)')
        ax2.set_xlabel('Scene Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Saliency Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Saliency Predictions (Context)', fontsize=11, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # Add correlation annotation
        if len(gate_values) > 1:
            corr = np.corrcoef(gate_values, predictions)[0, 1]
            ax2.text(0.02, 0.98, f'Gate-Prediction Correlation: {corr:.3f}', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(shap_output_dir, f"feature_reliance_gates_movie_{movie_id}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a summary plot showing gate statistics
        _plot_gate_statistics_summary(gate_values, scene_indices, predictions, 
                                     shap_output_dir, movie_id)
    
    print(f"  ✓ Saved: Feature reliance plots for {len(sample_indices)} movies")


def _plot_gate_statistics_summary(gate_values, scene_indices, predictions, shap_output_dir, movie_id):
    """Create a summary plot showing gate value statistics and distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Gate value distribution
    axes[0].hist(gate_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(gate_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gate_values):.3f}')
    axes[0].axvline(x=np.median(gate_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(gate_values):.3f}')
    axes[0].set_xlabel('Gate Value', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title('Gate Value Distribution', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Middle: Gate vs Scene Index (scatter with trend)
    z = np.polyfit(scene_indices, gate_values, 1)
    p = np.poly1d(z)
    axes[1].scatter(scene_indices, gate_values, alpha=0.5, s=30, color='blue')
    axes[1].plot(scene_indices, p(scene_indices), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}x + {z[1]:.3f}')
    axes[1].set_xlabel('Scene Index', fontsize=10)
    axes[1].set_ylabel('Gate Value', fontsize=10)
    axes[1].set_title('Gate Values Over Movie Progression', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Right: Gate vs Prediction correlation
    axes[2].scatter(gate_values, predictions, alpha=0.5, s=30, color='purple')
    z2 = np.polyfit(gate_values, predictions, 1)
    p2 = np.poly1d(z2)
    axes[2].plot(gate_values, p2(gate_values), "r--", alpha=0.8, linewidth=2)
    corr = np.corrcoef(gate_values, predictions)[0, 1]
    axes[2].set_xlabel('Gate Value', fontsize=10)
    axes[2].set_ylabel('Saliency Prediction', fontsize=10)
    axes[2].set_title(f'Gate-Prediction Relationship\n(Correlation: {corr:.3f})', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Gate Statistics Summary: Movie {movie_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, f"gate_statistics_movie_{movie_id}.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_bert_attention_tokens(tokens, attention_weights, top_shap_features, shap_output_dir, example_idx):
    """Create a detailed visualization showing BERT attention at token level."""
    # Create a heatmap of attention weights
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Top: BERT attention heatmap (first 50 tokens)
    max_tokens = min(50, len(tokens), len(attention_weights))
    attention_subset = attention_weights[:max_tokens]
    
    # Filter out special tokens for display
    display_tokens = []
    display_attention = []
    for i in range(max_tokens):
        if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']:
            display_tokens.append(tokens[i])
            display_attention.append(attention_subset[i])
    
    if len(display_tokens) > 0:
        # Create a simple bar chart showing attention weights
        ax1.bar(range(len(display_tokens)), display_attention, color='red', alpha=0.7)
        ax1.set_xticks(range(0, len(display_tokens), max(1, len(display_tokens) // 10)))
        ax1.set_xticklabels([display_tokens[i] if i < len(display_tokens) else '' 
                            for i in range(0, len(display_tokens), max(1, len(display_tokens) // 10))], 
                           rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Attention Weight', fontsize=10)
        ax1.set_title('BERT Attention: Which Tokens Get Focused? (Hard to Interpret)', 
                     fontsize=11, fontweight='bold', color='red')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Top linguistic features
    feature_names = [f[0] for f in top_shap_features]
    shap_values = [f[1] for f in top_shap_features]
    colors = ['green' if v > 0 else 'red' for v in shap_values]
    
    ax2.barh(range(len(feature_names)), shap_values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([f[:35] + ('...' if len(f) > 35 else '') for f in feature_names], fontsize=9)
    ax2.set_xlabel('SHAP Value', fontsize=10)
    ax2.set_title('Linguistic Features: Interpretable Explanations (Easy to Understand)', 
                 fontsize=11, fontweight='bold', color='green')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(shap_output_dir, f"bert_attention_detailed_example_{example_idx + 1}.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()


def save_salient_scenes_for_summarization(
    per_movie_predictions: Dict,
    test_loader: DataLoader,
    threshold: float,
    output_dir: str,
    split: str = "test"
):
    """
    Save salient scenes in format expected by summarization script.
    
    Args:
        per_movie_predictions: Dict mapping movie_id to predictions
        test_loader: DataLoader to get reference summaries (if available)
        threshold: Classification threshold used
        output_dir: Directory to save pickle files
        split: "test", "val", or "train"
    """
    # Try to get reference summaries from the dataset
    dataset = test_loader.dataset
    summaries_dict = {}
    
    # Check if dataset has summary information
    if hasattr(dataset, 'df') and 'summary' in dataset.df.columns:
        for movie_id in dataset.df['movie_id'].unique():
            movie_df = dataset.df[dataset.df['movie_id'] == movie_id]
            # Get summary (assuming one summary per movie)
            summary = movie_df['summary'].iloc[0] if len(movie_df) > 0 else ""
            summaries_dict[movie_id] = summary
    
    # Create data in format expected by summarization script
    summarization_data = []
    
    for movie_id, pred_data in per_movie_predictions.items():
        scene_texts = pred_data['scene_texts']
        predictions = pred_data['predictions']
        scene_indices = pred_data['scene_indices']
        
        # Filter to only salient scenes (prediction == 1)
        salient_scenes = []
        for i, (scene_text, pred, scene_idx) in enumerate(zip(scene_texts, predictions, scene_indices)):
            if pred == 1:  # Salient scene
                salient_scenes.append((scene_idx, scene_text))
        
        # Sort by scene index to maintain order
        salient_scenes.sort(key=lambda x: x[0])
        
        # Concatenate salient scenes into script
        script = "\n\n".join([scene_text for _, scene_text in salient_scenes])
        
        # Get reference summary (if available)
        summary = summaries_dict.get(movie_id, "")
        
        summarization_data.append({
            'script': script,
            'summary': summary,
            'movie_id': movie_id,
            'num_salient_scenes': len(salient_scenes),
            'total_scenes': len(scene_texts),
        })
    
    # Save to pickle file
    output_path = os.path.join(output_dir, f"{split}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(summarization_data, f)
    
    print(f"\nSaved {len(summarization_data)} movies with salient scenes to: {output_path}")
    # Calculate total scenes from summarization_data or per_movie_predictions
    total_scenes = sum(data['total_scenes'] for data in summarization_data)
    total_salient = sum(data['num_salient_scenes'] for data in summarization_data)
    print(f"  Total scenes: {total_scenes}")
    if total_scenes > 0:
        print(f"  Salient scenes: {total_salient} ({100*total_salient/total_scenes:.1f}%)")
    else:
        print(f"  Salient scenes: {total_salient}")
    
    return output_path


def train_with_loss_function(
    loss_type: str,
    model: SceneSaliencyWithLinguistic,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    args,
    config: dict,
    exp_output_dir: str,
    positive_weight: float,
    scaler=None,
    test_dataset=None,
) -> Dict:
    """
    Train and evaluate model with a specific loss function.
    
    Args:
        loss_type: 'bce' or 'focal'
        model: Model to train
        train_loader, val_loader, test_loader: Data loaders
        device: Device to run on
        args: Command-line arguments
        config: Experiment configuration
        exp_output_dir: Output directory for this loss function
        positive_weight: Positive class weight for BCE
        scaler: GradScaler for FP16 (if enabled)
        test_dataset: Test dataset (for SHAP analysis)
    
    Returns:
        Dictionary with results
    """
    # Create loss function
    if loss_type == 'bce' or loss_type == 'weighted_bce':
        print(f"\n{'='*80}")
        print(f"TRAINING WITH WEIGHTED BCE LOSS")
        print(f"{'='*80}")
        print(f"Positive class weight: {positive_weight:.4f}")
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight).to(device), 
            reduction='mean'
        )
    elif loss_type == 'focal':
        print(f"\n{'='*80}")
        print(f"TRAINING WITH FOCAL LOSS")
        print(f"{'='*80}")
        focal_alpha = getattr(args, 'focal_alpha', 0.85)
        focal_gamma = getattr(args, 'focal_gamma', 2.5)
        print(f"  Alpha: {focal_alpha}, Gamma: {focal_gamma}")
        criterion = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            reduction='mean'
        ).to(device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer and scheduler
    learning_rate = args.lr if not args.fp16 else args.lr * 0.5
    weight_decay = getattr(args, 'weight_decay', 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with optional warmup
    from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
    
    warmup_steps = getattr(args, 'warmup_steps', 0)
    if warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
            min_lr=1e-6, threshold=0.001
        )
    else:
        warmup_scheduler = None
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
            min_lr=1e-6, threshold=0.001
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': [],
        'optimal_thresholds': [],
    }
    
    best_val_f1 = -1.0
    best_epoch = -1
    optimal_threshold = args.threshold
    global_step = 0
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, args.threshold, scaler,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            warmup_scheduler=warmup_scheduler,
            global_step=global_step
        )
        
        # Validate with threshold optimization
        val_result = evaluate(
            model, val_loader, criterion, device, optimal_threshold, 
            return_probs=True, return_per_movie=False
        )
        if len(val_result) == 4:
            val_loss, val_metrics, val_probs, val_targets = val_result
        else:
            val_loss, val_metrics, _ = val_result
            val_probs = None
            val_targets = None
        
        # Optimize threshold on validation set
        if getattr(args, 'fixed_threshold', None) is not None:
            optimal_threshold = args.fixed_threshold
            val_preds = (val_probs > optimal_threshold).astype(int) if val_probs is not None else None
            if val_preds is not None and val_targets is not None:
                val_metrics = compute_metrics(val_preds, val_targets)
        elif val_probs is not None and val_targets is not None and len(val_targets) == len(val_probs):
            new_threshold, threshold_f1, threshold_precision, threshold_recall = find_optimal_threshold(
                val_probs, val_targets, optimize_for_precision=False, min_macro_f1=0.67,
                min_threshold=getattr(args, 'min_threshold', 0.0)  # Optimize for binary F1 with minimum threshold constraint
            )
            optimal_threshold = new_threshold
            val_preds = (val_probs > optimal_threshold).astype(int)
            val_metrics = compute_metrics(val_preds, val_targets)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['binary_f1'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['optimal_thresholds'].append(optimal_threshold)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Binary F1: {train_metrics['binary_f1']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Binary F1: {val_metrics['binary_f1']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}, Threshold: {optimal_threshold:.3f}")
        
        # Save best model - use macro F1 for balanced class performance
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimal_threshold': optimal_threshold,
                'val_f1': best_val_f1,
                'val_binary_f1': val_metrics['binary_f1'],
                'val_macro_f1': val_metrics['macro_f1'],
                'config': config,
                'loss_type': loss_type,
            }, os.path.join(exp_output_dir, "best_model.pt"))
            print(f"  ✓ New best model (Macro F1: {best_val_f1:.4f}, Binary F1: {val_metrics['binary_f1']:.4f})")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation Binary F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    # Load best model and test
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(exp_output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if getattr(args, 'fixed_threshold', None) is not None:
        optimal_threshold = args.fixed_threshold
    else:
        optimal_threshold = checkpoint.get('optimal_threshold', args.threshold)
    
    test_result = evaluate(
        model, test_loader, criterion, device, optimal_threshold, 
        return_probs=True, return_per_movie=True
    )
    test_loss, test_metrics, test_probs, test_targets, test_per_movie = test_result
    
    print(f"Test - Loss: {test_loss:.4f}, Binary F1: {test_metrics['binary_f1']:.4f}, "
          f"Precision: {test_metrics['precision_class_1']:.4f}, "
          f"Recall: {test_metrics['recall_class_1']:.4f}, "
          f"Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Print confusion matrix for test set
    test_predictions = (test_probs > optimal_threshold).astype(int)
    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(test_targets, test_predictions)
    print("                Predicted")
    print("              Non-Salient  Salient")
    print(f"Actual Non-Salient    {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"       Salient         {cm[1][0]:6d}    {cm[1][1]:6d}")
    print(f"\nTrue Negatives (TN):  {cm[0][0]}")
    print(f"False Positives (FP): {cm[0][1]}")
    print(f"False Negatives (FN): {cm[1][0]}")
    print(f"True Positives (TP):  {cm[1][1]}")
    
    # SHAP Analysis (if enabled)
    # NOTE: Skip SHAP here if comparing loss functions - will run once after both complete
    shap_results = None
    skip_shap = getattr(args, 'loss_type', '').lower() == 'both' or getattr(args, 'compare_loss_functions', False)
    if getattr(args, 'shap_analysis', False) and model.use_linguistic and test_dataset is not None and not skip_shap:
        if SHAP_AVAILABLE:
            shap_results = compute_shap_values(
                model=model,
                test_dataset=test_dataset,
                device=device,
                exp_output_dir=exp_output_dir,
                num_samples=getattr(args, 'shap_num_samples', 50),
                max_scenes_per_movie=getattr(args, 'shap_max_scenes', 100),
            )
    
    # Prepare results
    results = {
        'loss_type': loss_type,
        'experiment': config.get('description', 'N/A'),
        'config': config,
        'optimal_threshold': optimal_threshold,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'history': history,
    }
    
    if shap_results:
        results['shap_analysis'] = {
            'top_features': shap_results.get('top_features', []),
            'num_scenes_analyzed': shap_results.get('num_scenes_analyzed', 0),
            'num_movies_analyzed': shap_results.get('num_movies_analyzed', 0),
        }
    
    # Save results
    with open(os.path.join(exp_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def compare_loss_results(bce_results: Dict, focal_results: Dict, output_dir: str):
    """Compare results from BCE and Focal Loss and generate comparison report."""
    print(f"\n{'='*80}")
    print("LOSS FUNCTION COMPARISON")
    print(f"{'='*80}")
    
    # Extract metrics
    metrics_to_compare = {
        'binary_f1': 'Binary F1',
        'precision_class_1': 'Precision (Salient)',
        'recall_class_1': 'Recall (Salient)',
        'macro_f1': 'Macro F1',
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_class_0': 'F1 (Non-Salient)',
        'f1_class_1': 'F1 (Salient)',
    }
    
    comparison_data = []
    
    for metric_key, metric_name in metrics_to_compare.items():
        bce_val = bce_results['test_metrics'].get(metric_key, 0)
        focal_val = focal_results['test_metrics'].get(metric_key, 0)
        
        improvement = focal_val - bce_val
        improvement_pct = (improvement / bce_val * 100) if bce_val > 0 else 0
        
        better = 'Focal Loss' if focal_val > bce_val else 'Weighted BCE'
        if abs(improvement) < 1e-6:
            better = 'Tie'
        
        comparison_data.append({
            'Metric': metric_name,
            'Weighted_BCE': f"{bce_val:.4f}",
            'Focal_Loss': f"{focal_val:.4f}",
            'Improvement': f"{improvement:+.4f}",
            'Improvement_Pct': f"{improvement_pct:+.2f}%",
            'Better': better
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "loss_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison saved to: {comparison_path}")
    
    # Generate LaTeX table
    latex_path = os.path.join(output_dir, "loss_comparison.tex")
    with open(latex_path, 'w') as f:
        f.write(comparison_df.to_latex(index=False, float_format="%.4f"))
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    focal_wins = sum(1 for row in comparison_data if row['Better'] == 'Focal Loss')
    bce_wins = sum(1 for row in comparison_data if row['Better'] == 'Weighted BCE')
    ties = sum(1 for row in comparison_data if row['Better'] == 'Tie')
    
    print(f"Metrics where Focal Loss is better: {focal_wins}")
    print(f"Metrics where Weighted BCE is better: {bce_wins}")
    print(f"Ties: {ties}")
    
    # Key metrics
    key_metrics = ['Binary F1', 'Precision (Salient)', 'Recall (Salient)', 'Macro F1']
    print(f"\nKey Metrics:")
    for metric_name in key_metrics:
        row = next((r for r in comparison_data if r['Metric'] == metric_name), None)
        if row:
            print(f"  {metric_name}:")
            print(f"    Weighted BCE: {row['Weighted_BCE']}")
            print(f"    Focal Loss:   {row['Focal_Loss']}")
            print(f"    Improvement:  {row['Improvement']} ({row['Improvement_Pct']})")
            print(f"    Winner:       {row['Better']}")
    
    # Overall recommendation
    print(f"\n{'='*80}")
    if focal_wins > bce_wins:
        print("RECOMMENDATION: Focal Loss performs better overall")
    elif bce_wins > focal_wins:
        print("RECOMMENDATION: Weighted BCE performs better overall")
    else:
        print("RECOMMENDATION: Both loss functions perform similarly")
    print(f"{'='*80}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Train sequence-based scene saliency model")
    
    # Data
    parser.add_argument("--train-path", type=str, default=None, help="Path to training data (required if not using Hugging Face)")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation data (required if not using Hugging Face)")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test data (required if not using Hugging Face)")
    parser.add_argument("--linguistic-cols", type=str, nargs="+", default=None,
                        help="List of linguistic feature column names (auto-detect if not provided)")
    
    # Hugging Face datasets
    parser.add_argument("--use-huggingface", action="store_true",
                        help="Load features from Hugging Face datasets instead of local files")
    parser.add_argument("--hf-repo", type=str, default="Ishaank18/screenplay-features",
                        help="Hugging Face repository name (default: Ishaank18/screenplay-features)")
    parser.add_argument("--mensa-repo", type=str, default="rohitsaxena/MENSA",
                        help="Hugging Face repository for MENSA dataset (for scene_text and labels, default: rohitsaxena/MENSA)")
    parser.add_argument("--feature-groups", type=str, nargs="+", default=None,
                        help="List of feature groups to load. Works with both Hugging Face and local parquet files. "
                             "Available 10 groups (for ablation studies, specify a single group): "
                             "1. surprisal (57 features - BERT/GPT-2/ngram surprisal, psychformers), "
                             "2. morphosyntactic (587 features - gc_syntax, gc_pos, gc_temporal), "
                             "3. lexical (44 features - gc_basic, gc_readability, gc_char_diversity, ngram, gc_narrative, gc_punctuation), "
                             "4. semantic (29 features - gc_academic, gc_concreteness), "
                             "5. discourse_pragmatic (26 features - gc_discourse, rst, textrank_centrality, gc_punctuation), "
                             "6. dialogic (20 features - gc_dialogue, gc_pronouns), "
                             "7. emotional (26 features - emotional, gc_polarity), "
                             "8. narrative_structural (18 features - structure, plot_shifts, character_arcs, position), "
                             "9. saxena_keller (3 features - prior salience predictions), "
                             "10. genre (27 features - film genre). "
                             "Example for ablation: --feature-groups surprisal (loads only surprisal features). "
                             "If not specified, all features from parquet files are used.")
    
    # Experiment
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from experiment_config.py")
    parser.add_argument("--experiment-group", type=str, default=None,
                        help="Run all experiments in a group")
    
    # Training
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (movies per batch)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (or use --fixed-threshold to override optimization)")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Use fixed threshold instead of optimizing (e.g., 0.40, 0.45, 0.50 for higher precision)")
    parser.add_argument("--min-threshold", type=float, default=0.0, help="Minimum threshold value to consider during optimization (default: 0.0, use 0.30-0.40 to prevent very low thresholds)")
    parser.add_argument("--pos-weight-scale", type=float, default=1.0, help="Scale factor for positive class weight (default: 1.0, use 0.5-0.7 to reduce from ~6.0 to 3.5-4.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer (default: 0.01, increase to 0.1 for stronger regularization)")
    parser.add_argument("--loss-type", type=str, default="focal", choices=["bce", "focal", "both"],
                        help="Loss function type: 'bce' for weighted BCE, 'focal' for Focal Loss, 'both' to run both sequentially and compare")
    parser.add_argument("--focal-alpha", type=float, default=0.85, help="Focal Loss alpha parameter (weight on positive class, default: 0.85)")
    parser.add_argument("--focal-gamma", type=float, default=2.5, help="Focal Loss gamma parameter (focus on hard examples, default: 2.5, higher = more focus)")
    parser.add_argument("--compare-loss-functions", action="store_true",
                        help="Run both loss functions (BCE and Focal) sequentially and compare results (same as --loss-type both)")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps for learning rate (default: 0, use 500-1000 to prevent early overfitting)")
    parser.add_argument("--test-thresholds", type=float, nargs="+", default=None,
                        help="Test multiple thresholds on validation set (e.g., --test-thresholds 0.39 0.45 0.50 0.55 0.60)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Memory optimizations for large models
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--small-batch-large-models", action="store_true",
                        help="Use batch_size=1 for large models (RoBERTa-large, BERT-large)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps (simulates larger batch size, default: 1)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./experiments",
                        help="Output directory for experiments")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Pre-computed embeddings
    parser.add_argument("--embeddings-cache-dir", type=str, default=None,
                        help="Directory with pre-computed embeddings (speeds up frozen encoders)")
    
    # TextRank centrality feature
    parser.add_argument("--use-textrank", action="store_true", default=True,
                        help="Compute and add TextRank centrality as a feature (default: True)")
    parser.add_argument("--no-textrank", dest="use_textrank", action="store_false",
                        help="Disable TextRank centrality computation")
    parser.add_argument("--textrank-model", type=str, default=None,
                        help="Model to use for TextRank scene embeddings (default: uses scene_model_name from experiment config)")
    parser.add_argument("--textrank-lambda1", type=float, default=0.7,
                        help="Weight for forward-looking edges in TextRank (λ1, default: 0.7)")
    parser.add_argument("--textrank-lambda2", type=float, default=0.3,
                        help="Weight for backward-looking edges in TextRank (λ2, default: 0.3)")
    
    # Feature selection
    parser.add_argument("--use-top-features", action="store_true",
                        help="Use only top 20 linguistic features (from ensemble analysis) + TextRank centrality instead of all features")
    parser.add_argument("--top-features-file", type=str, default=None,
                        help="Path to file with top features list (default: uses built-in top 20 from run_lr.out)")
    
    # SHAP Analysis
    parser.add_argument("--shap-analysis", action="store_true",
                        help="Run SHAP analysis for interpretability (requires shap package)")
    parser.add_argument("--shap-num-samples", type=int, default=50,
                        help="Number of movies to sample for SHAP analysis (default: 50)")
    parser.add_argument("--shap-max-scenes", type=int, default=100,
                        help="Maximum scenes per movie for SHAP analysis (default: 100)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Determine experiments to run
    if args.experiment_group:
        experiment_names = get_experiment_group(args.experiment_group)
        print(f"Running experiment group: {args.experiment_group} ({len(experiment_names)} experiments)")
    else:
        experiment_names = [args.experiment]
    
    # Run each experiment
    for exp_name in experiment_names:
        print(f"\n{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*80}")
        
        # Get experiment config
        config = get_experiment_config(exp_name)
        print(f"Description: {config.get('description', 'N/A')}")
        
        # Create output directory
        exp_output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(exp_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Load data
        print("\nLoading data...")
        
        # Check if using Hugging Face
        use_huggingface = getattr(args, 'use_huggingface', False)
        hf_repo = getattr(args, 'hf_repo', 'Ishaank18/screenplay-features')
        feature_groups = getattr(args, 'feature_groups', None)
        
        if use_huggingface:
            if feature_groups is None or len(feature_groups) == 0:
                raise ValueError("--feature-groups must be specified when --use-huggingface is enabled")
            print(f"Loading features from Hugging Face: {hf_repo}")
            print(f"  Feature groups: {feature_groups}")
            # train_path, val_path, test_path are not needed when using Hugging Face
            train_path = ""  # Dummy value, will be ignored
            val_path = ""
            test_path = ""
        else:
            # Validate that paths are provided
            if args.train_path is None or args.val_path is None or args.test_path is None:
                raise ValueError("--train-path, --val-path, and --test-path are required when not using --use-huggingface")
            train_path = args.train_path
            val_path = args.val_path
            test_path = args.test_path
            feature_groups = None  # Not used for local files
        
        # First load train dataset to determine consistent feature columns
        # Check if TextRank should be enabled (default: True)
        use_textrank = getattr(args, 'use_textrank', True)
        textrank_lambda1 = getattr(args, 'textrank_lambda1', 0.7)
        textrank_lambda2 = getattr(args, 'textrank_lambda2', 0.3)
        
        # Use the same model as the experiment config (or user-specified)
        textrank_model = getattr(args, 'textrank_model', None)
        if textrank_model is None:
            # Use the scene model from experiment config
            textrank_model = config.get('scene_model_name', 'bert-base-uncased')
        
        if use_textrank:
            print(f"TextRank centrality enabled (model: {textrank_model}, λ1={textrank_lambda1}, λ2={textrank_lambda2})")
            if args.embeddings_cache_dir:
                print(f"  Using pre-computed embeddings from: {args.embeddings_cache_dir}")
        
        # Get MENSA repo for scene_text loading
        mensa_repo = getattr(args, 'mensa_repo', 'rohitsaxena/MENSA')
        
        # Check if we should use top features
        use_top_features = getattr(args, 'use_top_features', False)
        top_features_list = None
        if use_top_features and hasattr(args, 'top_features_file') and args.top_features_file:
            # Load custom top features from file
            try:
                with open(args.top_features_file, 'r') as f:
                    top_features_list = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(top_features_list)} top features from: {args.top_features_file}")
            except Exception as e:
                print(f"Warning: Could not load top features from file: {e}")
                print("  Using built-in TOP_20_LINGUISTIC_FEATURES")
        
        train_dataset = MovieDataset(
            train_path, "train", args.linguistic_cols,
            use_textrank=use_textrank,
            textrank_model=textrank_model,
            embeddings_cache_dir=args.embeddings_cache_dir,
            textrank_lambda1=textrank_lambda1,
            textrank_lambda2=textrank_lambda2,
            use_top_features=use_top_features,
            top_features_list=top_features_list,
            use_huggingface=use_huggingface,
            hf_repo=hf_repo,
            feature_groups=feature_groups,  # Pass feature_groups for both HF and local files
            mensa_repo=mensa_repo
        )
        feature_columns = train_dataset.feature_columns
        
        # Then load val/test using the same columns
        val_dataset = MovieDataset(
            val_path, "validation", args.linguistic_cols, 
            feature_columns=feature_columns,
            use_textrank=use_textrank,
            textrank_model=textrank_model,
            embeddings_cache_dir=args.embeddings_cache_dir,
            textrank_lambda1=textrank_lambda1,
            textrank_lambda2=textrank_lambda2,
            use_top_features=use_top_features,
            top_features_list=top_features_list,
            use_huggingface=use_huggingface,
            hf_repo=hf_repo,
            feature_groups=feature_groups,  # Pass feature_groups for both HF and local files
            mensa_repo=mensa_repo
        )
        test_dataset = MovieDataset(
            test_path, "test", args.linguistic_cols, 
            feature_columns=feature_columns,
            use_textrank=use_textrank,
            textrank_model=textrank_model,
            embeddings_cache_dir=args.embeddings_cache_dir,
            textrank_lambda1=textrank_lambda1,
            textrank_lambda2=textrank_lambda2,
            use_top_features=use_top_features,
            top_features_list=top_features_list,
            use_huggingface=use_huggingface,
            hf_repo=hf_repo,
            feature_groups=feature_groups,  # Pass feature_groups for both HF and local files
            mensa_repo=mensa_repo
        )
        
        # Get linguistic dimension
        if train_dataset.movies and train_dataset.movies[0]['linguistic_features'] is not None:
            linguistic_dim = train_dataset.movies[0]['linguistic_features'].shape[1]
        else:
            linguistic_dim = 0
        
        # Update config with linguistic dimension
        config['linguistic_dim'] = linguistic_dim
        config['device'] = device
        
        # Add embeddings cache directory if provided
        if args.embeddings_cache_dir:
            config['embeddings_cache_dir'] = args.embeddings_cache_dir
        
        # Create model
        print("\nCreating model...")
        model = SceneSaliencyWithLinguistic(**{k: v for k, v in config.items() 
                                                if k not in ['description', 'device']})
        model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Adjust batch size for large models
        effective_batch_size = args.batch_size
        if args.small_batch_large_models:
            large_models = ['roberta-large', 'bert-large-uncased']
            if any(model_name in config.get('scene_model_name', '') for model_name in large_models):
                effective_batch_size = 1
                print(f"Using batch_size=1 for large model: {config.get('scene_model_name')}")
        
        # Data loaders (use effective batch size)
        train_loader = DataLoader(
            train_dataset, batch_size=effective_batch_size, shuffle=True,
            collate_fn=collate_movies, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=effective_batch_size, shuffle=False,
            collate_fn=collate_movies, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=effective_batch_size, shuffle=False,
            collate_fn=collate_movies, num_workers=0
        )
        
        # Memory optimizations
        scaler = None
        if args.fp16:
            from torch.amp import GradScaler
            scaler = GradScaler("cuda")
            print("Using mixed precision training (FP16)")
        
        # Gradient checkpointing (if supported by model)
        if args.gradient_checkpointing:
            # Enable gradient checkpointing for scene encoder if it's a transformer
            if hasattr(model.scene_encoder, 'gradient_checkpointing_enable'):
                model.scene_encoder.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
        
        # Loss and optimizer
        positive_weight = get_positive_weight(train_dataset)
        # Apply scaling factor to reduce positive weight (for better precision)
        positive_weight = positive_weight * getattr(args, 'pos_weight_scale', 1.0)
        # Clamp positive weight to reasonable range to avoid numerical issues
        positive_weight = max(0.1, min(10.0, positive_weight))
        print(f"Positive class weight: {positive_weight:.4f} (scaled by {getattr(args, 'pos_weight_scale', 1.0):.2f})")
        
        # Check if we should run both loss functions
        loss_type = getattr(args, 'loss_type', 'focal').lower()
        compare_losses = getattr(args, 'compare_loss_functions', False) or loss_type == 'both'
        
        if compare_losses:
            # Run both loss functions sequentially
            print(f"\n{'='*80}")
            print("RUNNING BOTH LOSS FUNCTIONS FOR COMPARISON")
            print(f"{'='*80}")
            print("This will train the model twice: once with Weighted BCE, once with Focal Loss")
            print("Both runs will use identical settings (seed, hyperparameters, data splits)")
            
            all_results = {}
            
            # Run Weighted BCE first
            print(f"\n{'='*80}")
            print("RUN 1: WEIGHTED BCE LOSS")
            print(f"{'='*80}")
            bce_output_dir = os.path.join(exp_output_dir, "bce_loss")
            os.makedirs(bce_output_dir, exist_ok=True)
            
            # Create a fresh model copy for BCE
            bce_model = SceneSaliencyWithLinguistic(**{k: v for k, v in config.items() 
                                                        if k not in ['description', 'device']})
            bce_model.to(device)
            
            bce_results = train_with_loss_function(
                loss_type='bce',
                model=bce_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                args=args,
                config=config,
                exp_output_dir=bce_output_dir,
                positive_weight=positive_weight,
                scaler=scaler,
                test_dataset=None,  # Don't pass test_dataset here - will run SHAP once after both complete
            )
            all_results['bce'] = bce_results
            
            # Clean up memory after BCE run
            del bce_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Run Focal Loss second
            print(f"\n{'='*80}")
            print("RUN 2: FOCAL LOSS")
            print(f"{'='*80}")
            focal_output_dir = os.path.join(exp_output_dir, "focal_loss")
            os.makedirs(focal_output_dir, exist_ok=True)
            
            # Create a fresh model copy for Focal Loss
            focal_model = SceneSaliencyWithLinguistic(**{k: v for k, v in config.items() 
                                                          if k not in ['description', 'device']})
            focal_model.to(device)
            
            focal_results = train_with_loss_function(
                loss_type='focal',
                model=focal_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                args=args,
                config=config,
                exp_output_dir=focal_output_dir,
                positive_weight=positive_weight,
                scaler=scaler,
                test_dataset=None,  # Don't pass test_dataset here - will run SHAP once after both complete
            )
            all_results['focal'] = focal_results
            
            # Clean up memory after Focal run
            del focal_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Compare results
            print(f"\n{'='*80}")
            print("COMPARING RESULTS")
            print(f"{'='*80}")
            comparison_df = compare_loss_results(bce_results, focal_results, exp_output_dir)
            
            # Determine which model performed better (for SHAP analysis)
            bce_test_f1 = bce_results.get('test_metrics', {}).get('binary_f1', 0.0)
            focal_test_f1 = focal_results.get('test_metrics', {}).get('binary_f1', 0.0)
            better_model_type = 'bce' if bce_test_f1 >= focal_test_f1 else 'focal'
            better_output_dir = bce_output_dir if better_model_type == 'bce' else focal_output_dir
            
            # SHAP Analysis (run once on the better performing model to save memory)
            shap_results = None
            if getattr(args, 'shap_analysis', False) and test_dataset is not None:
                if SHAP_AVAILABLE:
                    print(f"\n{'='*80}")
                    print(f"RUNNING SHAP ANALYSIS ON BETTER MODEL ({better_model_type.upper()})")
                    print(f"{'='*80}")
                    print(f"BCE Test Binary F1: {bce_test_f1:.4f}")
                    print(f"Focal Test Binary F1: {focal_test_f1:.4f}")
                    print(f"Using {better_model_type.upper()} model for SHAP analysis...")
                    
                    # Load the better model
                    better_model = SceneSaliencyWithLinguistic(**{k: v for k, v in config.items() 
                                                                  if k not in ['description', 'device']})
                    better_model.to(device)
                    checkpoint_path = os.path.join(better_output_dir, "best_model.pt")
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    better_model.load_state_dict(checkpoint['model_state_dict'])
                    better_model.eval()
                    
                    # Run SHAP with reduced parameters to save memory
                    shap_num_samples = min(getattr(args, 'shap_num_samples', 50), 25)  # Cap at 25
                    shap_max_scenes = min(getattr(args, 'shap_max_scenes', 100), 50)   # Cap at 50
                    print(f"  Using reduced SHAP parameters: {shap_num_samples} samples, {shap_max_scenes} max scenes per movie")
                    
                    shap_results = compute_shap_values(
                        model=better_model,
                        test_dataset=test_dataset,
                        device=device,
                        exp_output_dir=exp_output_dir,  # Save to main output dir
                        num_samples=shap_num_samples,
                        max_scenes_per_movie=shap_max_scenes,
                    )
                    
                    # Clean up
                    del better_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print("\nWarning: SHAP analysis requested but SHAP is not available.")
                    print("  Install with: pip install shap matplotlib")
            
            # Save combined results
            combined_results = {
                'experiment': exp_name,
                'config': config,
                'bce_results': bce_results,
                'focal_results': focal_results,
                'comparison': comparison_df.to_dict('records'),
                'better_model': better_model_type,
            }
            
            if shap_results:
                combined_results['shap_analysis'] = {
                    'top_features': shap_results.get('top_features', []),
                    'num_scenes_analyzed': shap_results.get('num_scenes_analyzed', 0),
                    'num_movies_analyzed': shap_results.get('num_movies_analyzed', 0),
                    'model_used': better_model_type,
                }
            
            with open(os.path.join(exp_output_dir, "loss_comparison_results.json"), "w") as f:
                json.dump(combined_results, f, indent=2, default=str)
            
            print(f"\n✓ All results saved to: {exp_output_dir}")
            print(f"  - BCE results: {bce_output_dir}/results.json")
            print(f"  - Focal results: {focal_output_dir}/results.json")
            print(f"  - Comparison: {exp_output_dir}/loss_comparison.csv")
            print(f"  - Combined: {exp_output_dir}/loss_comparison_results.json")
            if shap_results:
                print(f"  - SHAP analysis: {exp_output_dir}/shap_analysis/ (using {better_model_type} model)")
            
        else:
            # Single loss function (original behavior)
            if loss_type == 'bce' or loss_type == 'weighted_bce':
                # Weighted BCE Loss: Simpler, directly addresses class imbalance
                print(f"Using Weighted BCE Loss (pos_weight={positive_weight:.4f})...")
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight).to(device), reduction='mean')
            elif loss_type == 'focal':
                # Focal Loss: Focuses on hard examples, can improve precision
                print("Using Focal Loss to balance Precision and Recall...")
                focal_alpha = getattr(args, 'focal_alpha', 0.85)
                focal_gamma = getattr(args, 'focal_gamma', 2.5)
                print(f"  Focal Loss parameters: alpha={focal_alpha}, gamma={focal_gamma}")
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean').to(device)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}. Use 'bce', 'focal', or 'both'")
            # Use lower learning rate for stability, especially with FP16
            learning_rate = args.lr if not args.fp16 else args.lr * 0.5
            print(f"Using learning rate: {learning_rate:.2e}")
            weight_decay = getattr(args, 'weight_decay', 0.01)
            print(f"Using weight decay: {weight_decay:.4f}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Learning rate scheduler with optional warmup
            from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
            
            warmup_steps = getattr(args, 'warmup_steps', 0)
            if warmup_steps > 0:
                # Create warmup scheduler
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        # Linear warmup: lr = base_lr * (current_step / warmup_steps)
                        return float(current_step) / float(max(1, warmup_steps))
                    else:
                        # After warmup, maintain base LR (ReduceLROnPlateau will handle decay)
                        return 1.0
                
                warmup_scheduler = LambdaLR(optimizer, lr_lambda)
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3,
                    min_lr=1e-6, threshold=0.001
                )
                print(f"Using learning rate warmup: {warmup_steps} steps")
            else:
                warmup_scheduler = None
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3,
                    min_lr=1e-6, threshold=0.001
                )
            
            # Early stopping
            early_stopping_patience = 7
            early_stopping_counter = 0
            
            # DEBUG: Check initial model parameters
            print("\n[DEBUG] Checking initial model state...")
            param_nan_count = 0
            param_inf_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    nan_count = torch.isnan(param).sum().item()
                    inf_count = torch.isinf(param).sum().item()
                    if nan_count > 0:
                        print(f"  [WARNING] {name}: {nan_count} NaN values in initial parameters!")
                        param_nan_count += nan_count
                    if inf_count > 0:
                        print(f"  [WARNING] {name}: {inf_count} Inf values in initial parameters!")
                        param_inf_count += inf_count
            
            if param_nan_count == 0 and param_inf_count == 0:
                print("  ✓ All initial parameters are valid (no NaN/Inf)")
            else:
                print(f"  [ERROR] Found {param_nan_count} NaN and {param_inf_count} Inf in initial parameters!")
            
            # Training history
            history = {
                'train_loss': [],
                'train_metrics': [],
                'val_loss': [],
                'val_metrics': [],
                'optimal_thresholds': [],
            }
            
            best_val_f1 = -1.0
            best_epoch = -1
            optimal_threshold = args.threshold  # Start with default, optimize during training
            global_step = 0  # Track global step for warmup
            
            # Training loop
            print(f"\nTraining for {args.epochs} epochs...")
            start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, criterion, device, args.threshold, scaler,
                gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                warmup_scheduler=warmup_scheduler,
                global_step=global_step
            )
            
            # Validate with threshold optimization
            val_result = evaluate(
                model, val_loader, criterion, device, optimal_threshold, 
                return_probs=True, return_per_movie=False
            )
            if len(val_result) == 4:
                val_loss, val_metrics, val_probs, val_targets = val_result
            else:
                val_loss, val_metrics, _ = val_result
                val_probs = None
                val_targets = None
            
            # Optimize threshold on validation set (or use fixed threshold)
            if getattr(args, 'fixed_threshold', None) is not None:
                # Use fixed threshold (for higher precision)
                optimal_threshold = args.fixed_threshold
                val_preds = (val_probs > optimal_threshold).astype(int) if val_probs is not None else None
                if val_preds is not None and val_targets is not None:
                    val_metrics = compute_metrics(val_preds, val_targets)
                    print(f"  → Using fixed threshold: {optimal_threshold:.3f} (Binary F1: {val_metrics['binary_f1']:.4f}, Precision: {val_metrics['precision_class_1']:.4f}, Recall: {val_metrics['recall_class_1']:.4f})")
            elif val_probs is not None and val_targets is not None and len(val_targets) == len(val_probs):
                new_threshold, threshold_f1, threshold_precision, threshold_recall = find_optimal_threshold(
                    val_probs, val_targets, optimize_for_precision=False, min_macro_f1=0.67,
                    min_threshold=getattr(args, 'min_threshold', 0.0)  # Optimize for binary F1 with minimum threshold constraint
                )
                # Always use the optimal threshold (it should be better or equal)
                optimal_threshold = new_threshold
                # Recompute metrics with optimal threshold
                val_preds = (val_probs > optimal_threshold).astype(int)
                val_metrics = compute_metrics(val_preds, val_targets)
                # Optimize for binary F1 (recall-focused) with macro F1 >= 0.67 constraint
                print(f"  → Optimal threshold: {optimal_threshold:.3f} (Binary F1: {val_metrics['binary_f1']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}, Precision: {val_metrics['precision_class_1']:.4f}, Recall: {val_metrics['recall_class_1']:.4f})")
            
            # Update learning rate scheduler - use macro F1 for balanced class performance
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['macro_f1'])  # Optimize for macro F1
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            history['optimal_thresholds'].append(optimal_threshold)
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Binary F1: {train_metrics['binary_f1']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Binary F1: {val_metrics['binary_f1']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}, Threshold: {optimal_threshold:.3f}")
            
            # Print classification report for validation set
            print("\nValidation Classification Report:")
            print(classification_report(
                val_targets, 
                (val_probs > optimal_threshold).astype(int),
                target_names=['Non-Salient', 'Salient'],
                zero_division=0
            ))
            
            # Save best model (no early stopping) - use macro F1 for balanced class performance
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'optimal_threshold': optimal_threshold,
                    'val_f1': best_val_f1,
                    'val_binary_f1': val_metrics['binary_f1'],
                    'val_macro_f1': val_metrics['macro_f1'],
                    'config': config,
                }, os.path.join(exp_output_dir, "best_model.pt"))
                print(f"  ✓ New best model (Macro F1: {best_val_f1:.4f}, Binary F1: {val_metrics['binary_f1']:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                }, os.path.join(exp_output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Training loop completed - now do post-training evaluation and analysis
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation Binary F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        
        # Load best model and test
        print("\nEvaluating on test set...")
        checkpoint = torch.load(os.path.join(exp_output_dir, "best_model.pt"), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use fixed threshold if provided, otherwise use optimal threshold from best model
        if getattr(args, 'fixed_threshold', None) is not None:
            optimal_threshold = args.fixed_threshold
            print(f"Using fixed threshold: {optimal_threshold:.3f} (for higher precision)")
        else:
            optimal_threshold = checkpoint.get('optimal_threshold', args.threshold)
            print(f"Using optimal threshold: {optimal_threshold:.3f} (from best model)")
        
        test_result = evaluate(
            model, test_loader, criterion, device, optimal_threshold, 
            return_probs=True, return_per_movie=True
        )
        # When return_per_movie=True, evaluate returns 5 values
        test_loss, test_metrics, test_probs, test_targets, test_per_movie = test_result
        
        print(f"Test - Loss: {test_loss:.4f}, Binary F1: {test_metrics['binary_f1']:.4f}, Precision: {test_metrics['precision_class_1']:.4f}, Recall: {test_metrics['recall_class_1']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}")
        
        # Print classification report for test set
        print("\nTest Classification Report:")
        test_predictions = (test_probs > optimal_threshold).astype(int)
        print(classification_report(
            test_targets,
            test_predictions,
            target_names=['Non-Salient', 'Salient'],
            zero_division=0
        ))
        
        # Print confusion matrix for test set
        print("\nTest Confusion Matrix:")
        cm = confusion_matrix(test_targets, test_predictions)
        print("                Predicted")
        print("              Non-Salient  Salient")
        print(f"Actual Non-Salient    {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"       Salient         {cm[1][0]:6d}    {cm[1][1]:6d}")
        print(f"\nTrue Negatives (TN):  {cm[0][0]}")
        print(f"False Positives (FP): {cm[0][1]}")
        print(f"False Negatives (FN): {cm[1][0]}")
        print(f"True Positives (TP):  {cm[1][1]}")
        
        # Evaluate validation set with per-movie predictions for summarization
        print("\nEvaluating on validation set (for summarization)...")
        val_result = evaluate(
            model, val_loader, criterion, device, optimal_threshold, 
            return_probs=True, return_per_movie=True
        )
        # When return_per_movie=True, evaluate returns 5 values
        val_loss, val_metrics, val_probs, val_targets, val_per_movie = val_result
        
        # Save salient scenes for summarization (both test and validation)
        summarization_output_dir = os.path.join(exp_output_dir, "summarization_data")
        
        if test_per_movie:
            save_salient_scenes_for_summarization(
                test_per_movie,
                test_loader,
                optimal_threshold,
                summarization_output_dir,
                split="test"
            )
            print(f"\nTest salient scenes saved for summarization in: {summarization_output_dir}")
        
        if val_per_movie:
            save_salient_scenes_for_summarization(
                val_per_movie,
                val_loader,
                optimal_threshold,
                summarization_output_dir,
                split="val"
            )
            print(f"Validation salient scenes saved for summarization in: {summarization_output_dir}")
        
        # Test multiple thresholds if requested (for threshold sliding analysis)
        if getattr(args, 'test_thresholds', None) is not None:
            print("\n" + "="*80)
            print("THRESHOLD SLIDING ANALYSIS")
            print("="*80)
            print("Testing multiple thresholds to find precision-recall trade-off...")
            print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'Binary F1':<12} {'Macro F1':<12}")
            print("-" * 80)
            
            threshold_results = []
            for test_thresh in args.test_thresholds:
                test_preds = (test_probs > test_thresh).astype(int)
                test_metrics_thresh = compute_metrics(test_preds, test_targets)
                threshold_results.append({
                    'threshold': test_thresh,
                    'precision': test_metrics_thresh['precision_class_1'],
                    'recall': test_metrics_thresh['recall_class_1'],
                    'binary_f1': test_metrics_thresh['binary_f1'],
                    'macro_f1': test_metrics_thresh['macro_f1']
                })
                print(f"{test_thresh:<12.3f} {test_metrics_thresh['precision_class_1']:<12.4f} {test_metrics_thresh['recall_class_1']:<12.4f} {test_metrics_thresh['binary_f1']:<12.4f} {test_metrics_thresh['macro_f1']:<12.4f}")
            
            # Find best threshold for different objectives
            best_f1_thresh = max(threshold_results, key=lambda x: x['binary_f1'])
            best_macro_f1_thresh = max(threshold_results, key=lambda x: x['macro_f1'])
            best_precision_thresh = max(threshold_results, key=lambda x: x['precision'])
            
            print("\n" + "="*80)
            print("BEST THRESHOLDS BY METRIC:")
            print("="*80)
            print(f"Best Binary F1:     Threshold={best_f1_thresh['threshold']:.3f} → F1={best_f1_thresh['binary_f1']:.4f}, Precision={best_f1_thresh['precision']:.4f}, Recall={best_f1_thresh['recall']:.4f}")
            print(f"Best Macro F1:      Threshold={best_macro_f1_thresh['threshold']:.3f} → F1={best_macro_f1_thresh['macro_f1']:.4f}, Precision={best_macro_f1_thresh['precision']:.4f}, Recall={best_macro_f1_thresh['recall']:.4f}")
            print(f"Best Precision:     Threshold={best_precision_thresh['threshold']:.3f} → F1={best_precision_thresh['binary_f1']:.4f}, Precision={best_precision_thresh['precision']:.4f}, Recall={best_precision_thresh['recall']:.4f}")
            print("\nRecommendation: Use threshold that maximizes your target metric while maintaining 67+ macro F1")
            print("="*80)
            print(f"  Use this path with --file_path in summarize.py")
        
        # Initialize results dictionary (before SHAP analysis)
        results = {
            'experiment': exp_name,
            'config': config,
            'optimal_threshold': optimal_threshold,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'history': history,
        }
        
        # SHAP Analysis for Interpretability (if enabled) - runs after all epochs complete
        if getattr(args, 'shap_analysis', False) and model.use_linguistic:
            if not SHAP_AVAILABLE:
                print("\nWarning: SHAP analysis requested but SHAP is not available.")
                print("  Install with: pip install shap matplotlib")
            else:
                # Compute SHAP values
                shap_results = compute_shap_values(
                    model=model,
                    test_dataset=test_dataset,
                    device=device,
                    exp_output_dir=exp_output_dir,
                    num_samples=getattr(args, 'shap_num_samples', 50),
                    max_scenes_per_movie=getattr(args, 'shap_max_scenes', 100),
                )
                
                # Add SHAP results to final results
                if shap_results:
                    results['shap_analysis'] = {
                        'top_features': shap_results.get('top_features', []),
                        'num_scenes_analyzed': shap_results.get('num_scenes_analyzed', 0),
                        'num_movies_analyzed': shap_results.get('num_movies_analyzed', 0),
                    }
        
        # Save results
        with open(os.path.join(exp_output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {exp_output_dir}")
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
