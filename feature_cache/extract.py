"""
Per-group incremental feature extraction for MENSA scene saliency.

This module extracts features one group at a time and saves them separately,
allowing incremental additions without recomputing existing groups.

Architecture:
    features/
        train/
            base.parquet
            narrative.parquet
            polarity.parquet
            ...
        validation/
            base.parquet
            narrative.parquet
            ...
        test/
            base.parquet
            ...
"""

from __future__ import annotations

from typing import Callable, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from data import load_mensa_dataframe

# Import from organized features/ directory
from features.base import (
    add_ttr_features, 
    add_length_structure_features, 
    add_position_overlap_features,
    add_similarity_change_features,
    add_structural_position_features
)
from features.character_arcs import add_entity_character_features, add_character_arc_features
from features.surprisal import SurprisalComputer
from features.emotional import add_emotional_trajectory_features
from features.ngram import add_ngram_features
from features.rst import add_rst_features
from features.bert_surprisal import add_bert_surprisal_features
from features.ngram_surprisal import add_ngram_surprisal_features

# Genre Classifier features - now using features/gc_wrapper.py
from features.gc_wrapper import add_gc_features



def extract_base(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract base features: TTR, readability, length, structure, position, entity.
    
    Returns DataFrame with movie_id, scene_index, and feature columns.
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    # TTR
    temp = add_ttr_features(df, text_col="scene_text")
    result["ttr"] = temp["ttr"]
    
    # Readability
    temp = add_readability_features(df, text_col="scene_text")
    for col in ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index", "automated_readability_index"]:
        if col in temp.columns:
            result[col] = temp[col]
    
    # Length and structure
    temp = add_length_structure_features(df, text_col="scene_text")
    for col in ["sentence_count", "token_count", "avg_sentence_len", "var_sentence_len", 
                "exclaim_rate", "question_rate", "uppercase_ratio", "dialogue_ratio"]:
        if col in temp.columns:
            result[col] = temp[col]
    
    # Position and overlap
    temp = add_position_overlap_features(df)
    for col in ["scene_index_norm", "overlap_prev", "overlap_next"]:
        if col in temp.columns:
            result[col] = temp[col]
    
    # Entity features
    temp = add_entity_character_features(df, text_col="scene_text")
    for col in ["unique_PERSON_count", "top_character_mention_rate", "pronoun_ratio", "name_repetition_rate"]:
        if col in temp.columns:
            result[col] = temp[col]
    
    return result


def extract_gc_polarity(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier polarity features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["polarity"])
    # Direct extraction: just drop scene_text, keep everything else
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_concreteness(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier concreteness features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["concreteness"])
    # Direct extraction: just drop scene_text, keep everything else
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_narrative(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier narrative features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["narrative"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_temporal(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier temporal features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["temporal"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_dialogue(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier dialogue features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["dialogue"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_discourse(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier discourse features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["discourse"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_pronouns(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier pronoun features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["pronouns"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_syntax(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier syntax features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["syntax"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_pos(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier POS features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["pos"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_basic(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier basic features (avg_sen_len, std_sen_len, lex_den)."""
    temp = add_gc_features(df, text_col="scene_text", groups=["basic"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_char_diversity(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier character diversity features (TTR, MAAS, MSTTR, etc.)."""
    temp = add_gc_features(df, text_col="scene_text", groups=["char_diversity"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_punctuation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier punctuation features."""
    temp = add_gc_features(df, text_col="scene_text", groups=["punctuation"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_academic(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier academic features (citations, numbers, passive voice)."""
    temp = add_gc_features(df, text_col="scene_text", groups=["academic"])
    return temp.drop(columns=['scene_text']).copy()


def extract_gc_readability(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract Genre Classifier readability features (Flesch, Gunning Fog, etc.)."""
    temp = add_gc_features(df, text_col="scene_text", groups=["readability"])
    return temp.drop(columns=['scene_text']).copy()


def extract_surprisal(df: pd.DataFrame, lm_name: str = "distilgpt2", **kwargs) -> pd.DataFrame:
    """Extract surprisal features."""
    result = df[["movie_id", "scene_index"]].copy()
    
    surprisal = SurprisalComputer(model_name=lm_name)
    s_means, s_stds, s_cvs, s_p75s, s_maxs, s_slopes = [], [], [], [], [], []
    
    for text in tqdm(df["scene_text"].tolist(), desc="Surprisal"):
        feats = surprisal.scene_surprisal_features(text)
        s_means.append(feats.get("surprisal_mean", 0.0))
        s_stds.append(feats.get("surprisal_std", 0.0))
        s_cvs.append(feats.get("surprisal_cv", 0.0))
        s_p75s.append(feats.get("surprisal_p75", 0.0))
        s_maxs.append(feats.get("surprisal_max", 0.0))
        s_slopes.append(feats.get("surprisal_slope", 0.0))
    
    result["surprisal_mean"] = s_means
    result["surprisal_std"] = s_stds
    result["surprisal_cv"] = s_cvs
    result["surprisal_p75"] = s_p75s
    result["surprisal_max"] = s_maxs
    result["surprisal_slope"] = s_slopes
    
    return result


def extract_emotional(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Extract emotional trajectory features."""
    result = df[["movie_id", "scene_index"]].copy()
    temp = add_emotional_trajectory_features(df, text_col="scene_text")
    emo_cols = [c for c in temp.columns if c.startswith("emo_")]
    for col in emo_cols:
        result[col] = temp[col]
    return result


def extract_ngram(df: pd.DataFrame, n: int = 3, **kwargs) -> pd.DataFrame:
    """Extract n-gram features."""
    result = df[["movie_id", "scene_index"]].copy()
    temp = add_ngram_features(df, text_col="scene_text", n=n)
    ngram_cols = [c for c in temp.columns if c.startswith("ngram_")]
    for col in ngram_cols:
        result[col] = temp[col]
    return result



def extract_character_arcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract character transition features.
    
    Features:
    - char_new_introductions: First-time character appearances
    - char_returns: Characters returning after absence (payoff moments!)
    - char_callbacks: Characters from opening scene returning
    - char_turnover: Character lineup change rate
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    # Need both entity and arc features
    temp = add_entity_character_features(df, text_col="scene_text")
    temp = add_character_arc_features(temp, text_col="scene_text")
    
    arc_cols = ["char_new_introductions", "char_returns", "char_callbacks", "char_turnover"]
    for col in arc_cols:
        if col in temp.columns:
            result[col] = temp[col]
    
    return result


def extract_plot_shifts(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract similarity change features (plot shift detection).
    
    Features:
    - sim_change_magnitude: How different is current vs trend
    - vocab_novelty: New vocabulary appearing (reveals/twists)
    - dialogue_shift: Sudden dialogue amount changes
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    temp = add_similarity_change_features(df, text_col="scene_text")
    
    shift_cols = ["sim_change_magnitude", "vocab_novelty", "dialogue_shift"]
    for col in shift_cols:
        if col in temp.columns:
            result[col] = temp[col]
    
    return result


def extract_structure(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract structural position features.
    
    Features:
    - pos_edge_proximity: U-shaped importance (edges salient)
    - pos_act: Which act (1/2/3)
    - pos_within_act: Position within current act
    - callback_to_opening: Thematic callbacks to opening
    - callback_to_ending: Foreshadowing of ending
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    temp = add_structural_position_features(df, text_col="scene_text")
    
    struct_cols = [
        "pos_edge_proximity", "pos_act", "pos_within_act",
        "callback_to_opening", "callback_to_ending"
    ]
    for col in struct_cols:
        if col in temp.columns:
            result[col] = temp[col]
    
    return result


def extract_rst(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract RST (Rhetorical Structure Theory) structural features.
    
    Features:
    - Tree structure metrics (depth, nodes, branching)
    - Nuclearity distributions
    - Relation type frequencies (30 relations)
    - Complexity metrics (entropy, balance)
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    temp = add_rst_features(df, text_col="scene_text")
    
    rst_cols = [c for c in temp.columns if c.startswith("rst_")]
    for col in rst_cols:
        result[col] = temp[col]
    
    return result


def extract_bert_surprisal(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract BERT-large surprisal features.
    
    Features:
    - Surprisal statistics (mean, median, std, IQR, etc.)
    - Perplexity and extremes
    - Position-based surprisal
    - Tokenization metrics
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    temp = add_bert_surprisal_features(df, text_col="scene_text")
    
    # Get bert surprisal columns (not bert embeddings which start with bert_)
    bert_surp_cols = [c for c in temp.columns if c.startswith("bert_") and c not in result.columns]
    for col in bert_surp_cols:
        result[col] = temp[col]
    
    return result


def extract_ngram_surprisal(df: pd.DataFrame, model_paths: dict = None, max_order: int = 5, **kwargs) -> pd.DataFrame:
    """
    Extract n-gram language model surprisal features.
    
    Features:
    - For each n-gram order (1-5): 17 features
      - Sentence-level surprisal statistics (mean, median, std, var, cv)
      - Extremes (min, max, range, p25, p75, iqr)
      - Perplexity metrics
      - Position-based features (first/last quarter comparison)
      - Temporal trend (slope)
    - Total: 86 features (17 × 5 orders + 1 count)
    """
    result = df[["movie_id", "scene_index"]].copy()
    
    temp = add_ngram_surprisal_features(df, text_col="scene_text", model_paths=model_paths, max_order=max_order)
    
    # Get ngram surprisal columns
    ngram_surp_cols = [c for c in temp.columns if c.startswith("ngram_") and c not in result.columns]
    for col in ngram_surp_cols:
        result[col] = temp[col]
    
    return result


# Registry of all feature extraction functions
FEATURE_EXTRACTORS: Dict[str, Callable] = {
    "base": extract_base,
    # Genre Classifier groups (all 14) - ordered starting with syntax
    "gc_syntax": extract_gc_syntax,
    "gc_pos": extract_gc_pos,
    "gc_basic": extract_gc_basic,
    "gc_char_diversity": extract_gc_char_diversity,
    "gc_polarity": extract_gc_polarity,
    "gc_concreteness": extract_gc_concreteness,
    "gc_narrative": extract_gc_narrative,
    "gc_temporal": extract_gc_temporal,
    "gc_dialogue": extract_gc_dialogue,
    "gc_discourse": extract_gc_discourse,
    "gc_pronouns": extract_gc_pronouns,
    "gc_punctuation": extract_gc_punctuation,
    "gc_academic": extract_gc_academic,
    "gc_readability": extract_gc_readability,
    # Advanced features
    "surprisal": extract_surprisal,
    "emotional": extract_emotional,
    "ngram": extract_ngram,
    # NEW: Narrative structure features
    "character_arcs": extract_character_arcs,
    "plot_shifts": extract_plot_shifts,
    "structure": extract_structure,
    # NEW: Advanced linguistic features
    "rst": extract_rst,
    "bert_surprisal": extract_bert_surprisal,
    "ngram_surprisal": extract_ngram_surprisal,
}


# Shortcut groups
GROUP_SHORTCUTS: Dict[str, list[str]] = {
    "gc_sentiment": ["gc_polarity", "gc_concreteness"],
    "gc_structural": ["gc_syntax", "gc_pos"],
    "gc_narrative_all": ["gc_narrative", "gc_temporal", "gc_dialogue"],
    "gc_linguistic": ["gc_basic", "gc_char_diversity", "gc_pos", "gc_syntax"],
    "gc_style": ["gc_pronouns", "gc_punctuation", "gc_discourse"],
    "gc_all": [
        "gc_syntax", "gc_pos", "gc_basic", "gc_char_diversity",
        "gc_polarity", "gc_concreteness", "gc_narrative", "gc_temporal",
        "gc_dialogue", "gc_discourse", "gc_pronouns", "gc_punctuation", 
        "gc_academic", "gc_readability"
    ],
    "core": ["base"],
    # NEW: Narrative structure shortcuts
    "narrative": ["character_arcs", "plot_shifts", "structure"],
    "plot_structure": ["character_arcs", "plot_shifts", "structure", "emotional"],
    "all_narrative": ["base", "character_arcs", "plot_shifts", "structure", "emotional"],
    "advanced": ["rst", "bert_surprisal", "ngram_surprisal"],
    "surprisal_all": ["surprisal", "bert_surprisal", "ngram_surprisal"],
    "fast": ["base", "gc_all", "emotional", "ngram", "character_arcs", "plot_shifts", "structure"],
    "all": list(FEATURE_EXTRACTORS.keys()),
}


def extract_group(
    df: pd.DataFrame,
    group_name: str,
    cache_dir: str | Path,
    split: str,
    **kwargs
) -> pd.DataFrame:
    """
    Extract features for a single group and save to cache.
    
    Args:
        df: DataFrame with scene_text, movie_id, scene_index, label
        group_name: Name of feature group to extract
        cache_dir: Base cache directory
        split: Dataset split (train, validation, test)
        **kwargs: Additional arguments for extractor (e.g., lm_name, n)
        
    Returns:
        DataFrame with movie_id, scene_index, and feature columns
    """
    cache_dir = Path(cache_dir)
    split_dir = cache_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = split_dir / f"{group_name}.parquet"
    
    # Check if already exists
    if output_path.exists():
        print(f"⚠ {group_name} already exists at {output_path}")
        print(f"  Delete it first if you want to regenerate")
        existing_df = pd.read_parquet(output_path)
        # Show info about existing file
        feature_cols = [c for c in existing_df.columns if c not in ['movie_id', 'scene_index', 'label']]
        print(f"  Existing features: {len(feature_cols)}")
        return existing_df
    
    # Extract features
    if group_name not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list(FEATURE_EXTRACTORS.keys())}")
    
    print(f"\n{'='*80}")
    print(f"Extracting {group_name} for {split}...")
    print(f"{'='*80}")
    
    extractor = FEATURE_EXTRACTORS[group_name]
    features_df = extractor(df, **kwargs)
    
    # Get feature columns (exclude movie_id, scene_index, and label)
    feature_cols = [c for c in features_df.columns if c not in ['movie_id', 'scene_index', 'label']]
    
    # Print detailed feature information
    print(f"\n✓ Extraction complete for {group_name}")
    print(f"  Samples: {len(features_df)}")
    print(f"  Total columns: {len(features_df.columns)}")
    print(f"  Feature columns: {len(feature_cols)}")
    
    # Show feature list
    if len(feature_cols) > 0:
        print(f"\n  Features extracted ({len(feature_cols)}):")
        # Group features for better display
        if len(feature_cols) <= 20:
            for feat in feature_cols:
                print(f"    • {feat}")
        else:
            # Show first 15 and last 5
            for feat in feature_cols[:15]:
                print(f"    • {feat}")
            print(f"    ... ({len(feature_cols) - 20} more features)")
            for feat in feature_cols[-5:]:
                print(f"    • {feat}")
    
    # IMPORTANT: Drop label column before saving - it should not be in feature cache
    if 'label' in features_df.columns:
        features_df = features_df.drop(columns=['label'])
    
    # Save to cache
    features_df.to_parquet(output_path, index=False, compression="snappy")
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  💾 Saved to: {output_path}")
    print(f"  📊 File size: {size_mb:.2f} MB")
    print(f"{'='*80}\n")
    
    return features_df


def extract_multiple_groups(
    split: str,
    groups: list[str],
    cache_dir: str | Path = "/scratch/ishaan.karan/features",
    **kwargs
) -> None:
    """
    Extract multiple feature groups for a split.
    
    Args:
        split: Dataset split name
        groups: List of group names to extract
        cache_dir: Base cache directory
        **kwargs: Additional arguments for extractors
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING GROUPS FOR {split.upper()} SPLIT")
    print(f"{'='*80}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Cache dir: {cache_dir}/{split}/")
    print(f"{'='*80}\n")
    
    # Load data once
    print(f"Loading {split} data from MENSA...")
    df = load_mensa_dataframe(split)
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Extract each group
    for i, group in enumerate(groups, 1):
        print(f"[{i}/{len(groups)}] {group}")
        try:
            extract_group(df, group, cache_dir, split, **kwargs)
        except Exception as e:
            print(f"✗ Error extracting {group}: {e}")
        print()
    
    print(f"{'='*80}")
    print(f"EXTRACTION COMPLETE FOR {split.upper()}")
    print(f"{'='*80}\n")
