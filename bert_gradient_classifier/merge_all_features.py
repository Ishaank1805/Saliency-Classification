"""
Script to merge all feature groups from Hugging Face and save as parquet files.

This script loads all 30 feature groups, merges them, and saves to:
- data/train.parquet
- data/validation.parquet
- data/test.parquet

This avoids memory issues when loading from Hugging Face during training.
"""

import os
import sys
import pandas as pd
import numpy as np
import gc
from datasets import load_dataset
from tqdm import tqdm
from typing import List

# All 30 feature groups
ALL_FEATURE_GROUPS = [
    'base', 'bert_surprisal', 'character_arcs', 'emotional', 'gc_academic',
    'gc_basic', 'gc_char_diversity', 'gc_concreteness', 'gc_dialogue',
    'gc_discourse', 'gc_narrative', 'gc_polarity', 'gc_pos', 'gc_pronouns',
    'gc_punctuation', 'gc_readability', 'gc_syntax', 'gc_temporal', 'genre',
    'gpt2_char_surprisal', 'graph', 'ngram', 'ngram_char_surprisal',
    'ngram_surprisal', 'plot_shifts', 'psychformers', 'rst', 'saxena_keller',
    'structure', 'textrank_centrality'
]

HF_REPO = "Ishaank18/screenplay-features"
MENSA_REPO = "Ishaank18/mensa-dataset"  # MENSA dataset for scene_text and labels

# Split mapping
SPLIT_MAP = {"val": "validation", "test": "test", "train": "train"}


def merge_all_features_to_parquet(
    split: str = 'train',
    output_dir: str = 'data',
    feature_groups: List[str] = None,
    hf_repo: str = HF_REPO,
    mensa_repo: str = None
) -> str:
    """
    Merge all feature groups and save to a single parquet file.
    
    Args:
        split: Data split ('train', 'validation', or 'test')
        output_dir: Directory to save the parquet file
        feature_groups: List of feature groups to merge (default: all groups)
        hf_repo: Hugging Face repository name
    
    Returns:
        Path to the saved parquet file
    """
    if feature_groups is None:
        feature_groups = ALL_FEATURE_GROUPS
    
    # Map split name
    hf_split = SPLIT_MAP.get(split, split)
    
    print(f"\n{'='*80}")
    print(f"Merging {len(feature_groups)} feature groups for {split} split")
    print(f"{'='*80}")
    
    base_df = None
    base_group = None
    
    # Load and merge each group
    for i, group in enumerate(tqdm(feature_groups, desc=f"Loading {split} groups")):
        try:
            # Load the specific group file
            group_ds = load_dataset(hf_repo, data_files=f"{hf_split}/{group}.parquet")
            
            # Get the correct split
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
                base_group = group
                # Check if scene_text exists in base
                has_scene_text = 'scene_text' in base_df.columns
                print(f"  ✓ Loaded base group '{group}': {len(group_df)} scenes, {len(group_df.columns)} columns")
                if has_scene_text:
                    print(f"    → Found 'scene_text' column in base group")
                else:
                    print(f"    ⚠ Warning: No 'scene_text' column in base group '{group}'")
            else:
                # Prepare group_df for merging (drop duplicate metadata)
                group_df_clean = group_df.copy()
                
                # Drop metadata columns that will be duplicated (but preserve from base)
                # Only drop if base already has it
                if 'scene_text' in base_df.columns and 'scene_text' in group_df_clean.columns:
                    group_df_clean = group_df_clean.drop(columns=['scene_text'])
                elif 'scene_text' not in base_df.columns and 'scene_text' in group_df_clean.columns:
                    # If base doesn't have scene_text but this group does, keep it
                    print(f"    → Found 'scene_text' in group '{group}', adding to merged data")
                
                if 'label' in base_df.columns and 'label' in group_df_clean.columns:
                    group_df_clean = group_df_clean.drop(columns=['label'])
                
                # Count features before merge
                metadata_cols = ['movie_id', 'scene_index', 'scene_text', 'label']
                feature_count_before = len([c for c in base_df.columns if c not in metadata_cols])
                
                # Merge on movie_id and scene_index
                base_df = base_df.merge(
                    group_df_clean,
                    on=['movie_id', 'scene_index'],
                    how='outer',
                    suffixes=('', '_drop')
                )
                
                # Drop columns with '_drop' suffix
                base_df = base_df.loc[:, ~base_df.columns.str.endswith('_drop')]
                
                # Count features after merge
                feature_count_after = len([c for c in base_df.columns if c not in metadata_cols])
                features_added = feature_count_after - feature_count_before
                
                print(f"  ✓ Merged group '{group}': added {features_added} features")
            
            # Force garbage collection to free memory
            del group_df
            if 'group_df_clean' in locals():
                del group_df_clean
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Warning: Could not load feature group '{group}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if base_df is None:
        raise ValueError(f"No feature groups could be loaded for split '{split}'")
    
    # Load scene_text and labels from MENSA dataset
    if mensa_repo is None:
        mensa_repo = MENSA_REPO
    
    print(f"\n  Loading scene_text and labels from MENSA dataset ({mensa_repo})...")
    try:
        # Try loading MENSA dataset - it might be in different formats
        try:
            mensa_ds = load_dataset(mensa_repo, split=hf_split)
        except:
            # Try with data_files if split-based loading doesn't work
            mensa_ds = load_dataset(mensa_repo, data_files=f"{hf_split}.parquet")
            if hf_split in mensa_ds:
                mensa_df = mensa_ds[hf_split].to_pandas()
            else:
                # Get first available split
                available_splits = list(mensa_ds.keys())
                if len(available_splits) > 0:
                    mensa_df = mensa_ds[available_splits[0]].to_pandas()
                else:
                    raise ValueError("No splits available in MENSA dataset")
        else:
            mensa_df = mensa_ds.to_pandas()
        
        # Check MENSA dataset structure
        print(f"    MENSA dataset columns: {list(mensa_df.columns)}")
        print(f"    MENSA dataset shape: {mensa_df.shape}")
        
        # Debug: Check first row to understand data structure
        if len(mensa_df) > 0:
            first_row = mensa_df.iloc[0]
            if 'scenes' in mensa_df.columns:
                scenes_sample = first_row['scenes']
                
                # --- FIX: Safe NaN checking for the debug print ---
                is_sample_nan = False
                try:
                    if scenes_sample is None:
                        is_sample_nan = True
                    elif isinstance(scenes_sample, float) and pd.isna(scenes_sample):
                        is_sample_nan = True
                except:
                    pass # If it's an array, it's not NaN in the scalar sense

                print(f"    Sample 'scenes' type: {type(scenes_sample).__name__}, value type: {type(scenes_sample).__name__ if not is_sample_nan else 'NaN'}")
                
                # --- FIX: Safe length check ---
                try:
                    if not is_sample_nan and hasattr(scenes_sample, '__len__'):
                         print(f"    Sample 'scenes' length: {len(scenes_sample)}")
                except:
                    pass
        
        # MENSA dataset structure: it has 'scenes' (list) and 'labels' (list) per movie
        # We need to check if it's in list format or already flattened
        if 'scenes' in mensa_df.columns and 'labels' in mensa_df.columns:
            # MENSA has lists - need to explode/flatten them
            print(f"    MENSA has 'scenes' and 'labels' lists - flattening...")
            
            # Create a list to store exploded rows
            exploded_rows = []
            
            # MENSA doesn't have movie_id - we assign sequential IDs (0, 1, 2, ...)
            # based on row order in MENSA dataset
            has_name = 'name' in mensa_df.columns
            
            # Reset index to ensure sequential row numbers (0, 1, 2, ...)
            mensa_df_reset = mensa_df.reset_index(drop=True)
            
            # Explode scenes and labels
            for movie_id_val, (_, row) in enumerate(mensa_df_reset.iterrows()):
                # Handle different data types for scenes and labels
                scenes_raw = row['scenes']
                labels_raw = row['labels']
                
                # --- FIX: Safe list conversion function ---
                def to_list(value):
                    """Convert various types to a Python list."""
                    if value is None:
                        return []
                    # Check for scalar NaN safely (floats only)
                    if isinstance(value, float) and pd.isna(value):
                        return []
                    
                    # Check if it's already a list-like type
                    if isinstance(value, (list, tuple)):
                        return list(value)
                    
                    # Check if it's a numpy array or other sequence
                    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        try:
                            return list(value)
                        except:
                            return []
                    
                    # Try to parse as JSON string
                    if isinstance(value, str):
                        try:
                            import json
                            parsed = json.loads(value)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                            return []
                        except:
                            return []
                            
                    # Fallback: try to convert
                    try:
                        return [value] if value else []
                    except:
                        return []
                
                scenes_list = to_list(scenes_raw)
                labels_list = to_list(labels_raw)
                
                # Ensure both lists have the same length
                min_len = min(len(scenes_list), len(labels_list))
                if min_len == 0:
                    # Skip if no scenes
                    continue
                scenes_list = scenes_list[:min_len]
                labels_list = labels_list[:min_len]
                
                if has_name and movie_id_val < 5:  # Only print first 5 for debugging
                    movie_name = row.get('name', f'Movie_{movie_id_val}')
                    scenes_type = type(scenes_raw).__name__
                    try:
                        first_scene_preview = scenes_list[0][:50] + "..." if scenes_list and len(scenes_list[0]) > 50 else (scenes_list[0] if scenes_list else "")
                        print(f"    Movie {movie_id_val}: '{movie_name}' -> {len(scenes_list)} scenes (raw type: {scenes_type}, first scene preview: {first_scene_preview})")
                    except:
                        print(f"    Movie {movie_id_val}: '{movie_name}' -> {len(scenes_list)} scenes")
                
                # Explode scenes and labels
                for scene_idx, (scene_text, label) in enumerate(zip(scenes_list, labels_list)):
                    exploded_rows.append({
                        'movie_id': movie_id_val,
                        'scene_index': scene_idx,
                        'scene_text': scene_text if isinstance(scene_text, str) else str(scene_text),
                        'label': label if isinstance(label, (int, float)) else int(label) if label else 0
                    })
            
            mensa_df = pd.DataFrame(exploded_rows)
            if len(mensa_df) > 0:
                print(f"    ✓ Flattened MENSA: {len(mensa_df)} scenes from {len(mensa_df['movie_id'].unique())} movies")
                merge_cols = ['movie_id', 'scene_index']
            else:
                print(f"    ⚠ Warning: MENSA flattening resulted in 0 scenes - skipping merge")
                mensa_df = None
                merge_cols = None
            
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
            
            if 'label' in mensa_df.columns:
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
        
        # Merge MENSA data (scene_text and labels) with feature data
        # Only merge if we have valid MENSA data
        if mensa_df is not None and len(mensa_df) > 0 and merge_cols is not None:
            base_df = base_df.merge(
                mensa_df,
                on=merge_cols,
                how='left',  # Left join to keep all feature rows
                suffixes=('', '_mensa')
            )
            
            # Drop duplicate columns from MENSA merge (keep original if both exist)
            for col in base_df.columns:
                if col.endswith('_mensa'):
                    original_col = col.replace('_mensa', '')
                    if original_col in base_df.columns:
                        # Keep original, drop mensa version
                        base_df = base_df.drop(columns=[col])
                    else:
                        # Rename mensa version to original name
                        base_df = base_df.rename(columns={col: original_col})
            
            # Check what we got from MENSA
            if 'scene_text' in base_df.columns:
                non_null_scenes = base_df['scene_text'].notna().sum()
                print(f"    ✓ Merged scene_text from MENSA: {non_null_scenes}/{len(base_df)} scenes have text")
            else:
                print(f"    ⚠ Warning: scene_text not found in MENSA dataset")
            
            if 'label' in base_df.columns:
                non_null_labels = base_df['label'].notna().sum()
                print(f"    ✓ Merged labels from MENSA: {non_null_labels}/{len(base_df)} scenes have labels")
        else:
            print(f"    ⚠ Warning: Skipping MENSA merge - no valid data to merge")
    except Exception as e:
        print(f"    ⚠ Warning: Could not load MENSA dataset ({mensa_repo}): {e}")
        print(f"    → Continuing without scene_text from MENSA (may already be in feature groups)")
        # Don't print full traceback for expected errors (dataset not found)
        if "not found" not in str(e).lower() and "does not exist" not in str(e).lower():
            import traceback
            traceback.print_exc()
    
    # Remove duplicate columns (keep first occurrence)
    base_df = base_df.loc[:, ~base_df.columns.duplicated()]
    
    # Ensure we have required columns
    required_cols = ['movie_id', 'scene_index']
    missing_cols = [c for c in required_cols if c not in base_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Sort by movie_id and scene_index
    base_df = base_df.sort_values(['movie_id', 'scene_index']).reset_index(drop=True)
    
    # Count feature columns (exclude metadata)
    metadata_cols = ['movie_id', 'scene_index', 'scene_text', 'label']
    feature_cols = [c for c in base_df.columns if c not in metadata_cols]
    
    print(f"\n  ✓ Final merged dataset: {len(base_df)} scenes from {len(base_df['movie_id'].unique())} movies")
    print(f"  ✓ Total feature columns: {len(feature_cols)}")
    
    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    
    # Use 'validation' for filename but keep split name consistent
    filename = 'validation.parquet' if split == 'validation' else f"{split}.parquet"
    output_path = os.path.join(output_dir, filename)
    
    print(f"\n  Saving to {output_path}...")
    base_df.to_parquet(output_path, index=False, engine='pyarrow')
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved merged features to {output_path} ({file_size_mb:.1f} MB)")
    
    # Final cleanup
    del base_df
    gc.collect()
    
    return output_path


def main():
    """Main function to merge all splits."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge all feature groups from Hugging Face")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory to save parquet files (default: data)")
    parser.add_argument("--hf-repo", type=str, default=HF_REPO,
                        help="Hugging Face repository name")
    parser.add_argument("--splits", type=str, nargs="+", 
                        default=["train", "validation", "test"],
                        help="Splits to process (default: train validation test)")
    parser.add_argument("--feature-groups", type=str, nargs="+", default=None,
                        help="Feature groups to merge (default: all 30 groups)")
    parser.add_argument("--mensa-repo", type=str, default="rohitsaxena/MENSA",
                        help=f"Hugging Face repository for MENSA dataset (default: {MENSA_REPO})")
    
    args = parser.parse_args()
    
    # Use all feature groups if not specified
    feature_groups = args.feature_groups if args.feature_groups else ALL_FEATURE_GROUPS
    
    print(f"Will merge {len(feature_groups)} feature groups:")
    print(f"  {', '.join(feature_groups[:10])}...")
    print(f"  ... and {len(feature_groups) - 10} more")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Splits to process: {args.splits}")
    
    # Process each split
    output_paths = {}
    for split in args.splits:
        try:
            output_path = merge_all_features_to_parquet(
                split=split,
                output_dir=args.output_dir,
                feature_groups=feature_groups,
                hf_repo=args.hf_repo,
                mensa_repo=args.mensa_repo
            )
            output_paths[split] = output_path
        except Exception as e:
            print(f"\n✗ Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully created {len(output_paths)} parquet files:")
    for split, path in output_paths.items():
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {split}: {path} ({file_size_mb:.1f} MB)")
    
    print(f"\nYou can now use these files with:")
    print(f"  --train-path {args.output_dir}/train.parquet")
    print(f"  --val-path {args.output_dir}/validation.parquet")
    print(f"  --test-path {args.output_dir}/test.parquet")
    print(f"\n(Instead of --use-huggingface)")


if __name__ == "__main__":
    main()