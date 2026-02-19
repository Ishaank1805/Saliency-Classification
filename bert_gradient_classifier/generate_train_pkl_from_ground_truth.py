#!/usr/bin/env python3
"""
Script to generate train.pkl or val.pkl for summarization using GROUND TRUTH labels.

This script:
1. Loads the dataset (train or validation)
2. Uses ground truth labels (not model predictions) to identify salient scenes
3. Saves salient scenes in the same format as test.pkl

This is simpler and doesn't require a model checkpoint - it uses the actual labels
from your dataset. This is useful for training the summarization model on 
ground truth salient scenes, then evaluating on model predictions (test.pkl).
"""

import argparse
import os
import sys
import pickle

# Add parent directory to path to import from train_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_sequence import MovieDataset
from datasets import load_dataset
import pandas as pd


def generate_train_pkl_from_ground_truth(
    data_path: str,
    output_dir: str,
    split: str = "train",
    use_huggingface: bool = False,
    hf_repo: str = "Ishaank18/screenplay-features",
    feature_groups: list = None,
    mensa_repo: str = "rohitsaxena/MENSA",
    embeddings_cache_dir: str = None,
    use_textrank: bool = True,
):
    """
    Generate train.pkl or val.pkl using ground truth labels from the dataset.
    
    Args:
        data_path: Path to parquet file (train or validation)
        output_dir: Directory to save .pkl file
        split: Split name ("train" or "validation")
        use_huggingface: Whether to load from Hugging Face
        hf_repo: Hugging Face repository for features
        feature_groups: List of feature groups to load
        mensa_repo: Hugging Face repository for MENSA dataset
        embeddings_cache_dir: Cache directory (not used but kept for compatibility)
        use_textrank: Whether to use TextRank (not used but kept for compatibility)
    """
    # Map split for output filename
    output_split = "val" if split == "validation" else split
    
    print("=" * 80)
    print(f"Generating {output_split}.pkl from GROUND TRUTH labels")
    print("=" * 80)
    
    # Load dataset (we only need the data, not for training)
    print(f"\nLoading dataset from: {data_path}")
    dataset = MovieDataset(
        data_path, split, None,  # linguistic_cols will be determined from data
        use_textrank=False,  # Don't need TextRank for this
        embeddings_cache_dir=embeddings_cache_dir,
        use_huggingface=use_huggingface,
        hf_repo=hf_repo,
        feature_groups=feature_groups,
        mensa_repo=mensa_repo
    )
    
    print(f"  Loaded {len(dataset.movies)} movies")
    
    # Load reference summaries from MENSA dataset
    summaries_dict = {}
    try:
        print(f"\nLoading reference summaries from MENSA dataset ({mensa_repo})...")
        # Map split name for MENSA (MENSA uses "validation" not "val")
        mensa_split_map = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
        mensa_split = mensa_split_map.get(split, split)
        
        mensa_ds = load_dataset(mensa_repo, split=mensa_split)
        mensa_df = mensa_ds.to_pandas()
        
        # MENSA has 'name' (movie name) and 'summary' columns
        # We need to map movie_id to movie name, then to summary
        # First, try to get movie names from the dataset
        movie_id_to_name = {}
        
        # Debug: Check what columns are available
        if hasattr(dataset, 'df'):
            print(f"  Dataset columns: {list(dataset.df.columns)}")
            print(f"  Unique movie_ids: {len(dataset.df['movie_id'].unique())}")
        
        # Try multiple possible column names for movie name
        name_columns = ['movie_name', 'name', 'title', 'movie_title']
        found_name_col = None
        for col in name_columns:
            if hasattr(dataset, 'df') and col in dataset.df.columns:
                found_name_col = col
                print(f"  Found movie name column: '{col}'")
                for movie_id in dataset.df['movie_id'].unique():
                    movie_df = dataset.df[dataset.df['movie_id'] == movie_id]
                    movie_name = movie_df[col].iloc[0] if len(movie_df) > 0 else None
                    if movie_name:
                        movie_id_to_name[movie_id] = str(movie_name).strip()
                break
        
        # Create name to summary mapping from MENSA (only if we have movie names)
        if len(movie_id_to_name) > 0 and 'name' in mensa_df.columns and 'summary' in mensa_df.columns:
            name_to_summary = {}
            for _, row in mensa_df.iterrows():
                name = str(row['name']).strip() if pd.notna(row['name']) else None
                summary = row.get('summary', '')
                if name and summary:
                    name_to_summary[name] = summary
            
            # Try exact match first
            for movie_id, movie_name in movie_id_to_name.items():
                if movie_name in name_to_summary:
                    summaries_dict[movie_id] = name_to_summary[movie_name]
            
            # Try case-insensitive and normalized matching
            if len(summaries_dict) < len(movie_id_to_name):
                print(f"  → Trying case-insensitive and normalized matching...")
                name_to_summary_lower = {k.lower().strip(): v for k, v in name_to_summary.items()}
                for movie_id, movie_name in movie_id_to_name.items():
                    if movie_id not in summaries_dict:
                        movie_name_lower = movie_name.lower().strip()
                        # Try exact lowercase match
                        if movie_name_lower in name_to_summary_lower:
                            summaries_dict[movie_id] = name_to_summary_lower[movie_name_lower]
                        else:
                            # Try partial match (in case of formatting differences)
                            for mensa_name, summary in name_to_summary_lower.items():
                                if movie_name_lower in mensa_name or mensa_name in movie_name_lower:
                                    summaries_dict[movie_id] = summary
                                    break
            
            print(f"  ✓ Loaded {len(name_to_summary)} summaries from MENSA")
            print(f"  ✓ Mapped {len(summaries_dict)} summaries to movies (out of {len(movie_id_to_name)} movies with names)")
        
        # If name-based matching didn't work or we don't have movie names, try index-based matching
        if len(summaries_dict) == 0 and 'summary' in mensa_df.columns:
            print(f"  → Name-based matching failed or no movie names found. Trying index-based matching...")
            # Match by index (assuming same order)
            # This works if MENSA and dataset have movies in the same order
            if len(mensa_df) == len(dataset.movies):
                print(f"  → MENSA has {len(mensa_df)} movies, dataset has {len(dataset.movies)} movies")
                print(f"  → Attempting index-based matching (assuming same order)")
                sorted_movie_ids = sorted(dataset.df['movie_id'].unique())
                for idx, movie_id in enumerate(sorted_movie_ids):
                    if idx < len(mensa_df):
                        summaries_dict[movie_id] = mensa_df.iloc[idx].get('summary', '')
                print(f"  ✓ Index-based mapping: {len(summaries_dict)} summaries mapped")
            else:
                print(f"  ⚠ Warning: MENSA has {len(mensa_df)} movies but dataset has {len(dataset.movies)} movies")
                print(f"    Cannot use index-based matching. Summaries will be empty.")
        else:
            print(f"  ⚠ Warning: MENSA dataset missing 'name' or 'summary' columns")
            print(f"    Available columns: {list(mensa_df.columns)}")
            
            # Fallback: try to get summaries from dataset.df if available
            if hasattr(dataset, 'df') and 'summary' in dataset.df.columns:
                for movie_id in dataset.df['movie_id'].unique():
                    movie_df = dataset.df[dataset.df['movie_id'] == movie_id]
                    summary = movie_df['summary'].iloc[0] if len(movie_df) > 0 else ""
                    if summary:
                        summaries_dict[movie_id] = summary
                print(f"  ✓ Fallback: Found {len(summaries_dict)} summaries in dataset.df")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load MENSA dataset: {e}")
        print(f"  → Trying fallback: checking dataset.df for summaries...")
        # Fallback: try to get summaries from dataset.df if available
        if hasattr(dataset, 'df') and 'summary' in dataset.df.columns:
            for movie_id in dataset.df['movie_id'].unique():
                movie_df = dataset.df[dataset.df['movie_id'] == movie_id]
                summary = movie_df['summary'].iloc[0] if len(movie_df) > 0 else ""
                if summary:
                    summaries_dict[movie_id] = summary
            print(f"  ✓ Fallback: Found {len(summaries_dict)} summaries in dataset.df")
        else:
            print(f"  ✗ No summaries found. Summaries will be empty strings.")
    
    # Create data in format expected by summarization script
    summarization_data = []
    total_scenes = 0
    total_salient = 0
    
    print(f"\nProcessing movies and extracting salient scenes from GROUND TRUTH labels...")
    
    for movie in dataset.movies:
        movie_id = movie.get('movie_id', None)
        scene_texts = movie.get('scene_texts', [])
        labels = movie.get('labels', [])
        
        if not scene_texts:
            continue
        
        # Use GROUND TRUTH labels (not model predictions)
        salient_scenes = []
        for i, (scene_text, label) in enumerate(zip(scene_texts, labels)):
            if label == 1:  # Ground truth salient scene
                salient_scenes.append((i, scene_text))
        
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
        
        total_scenes += len(scene_texts)
        total_salient += len(salient_scenes)
    
    # Save to pickle file
    output_path = os.path.join(output_dir, f"{output_split}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(summarization_data, f)
    
    print(f"\n✓ Saved {len(summarization_data)} movies with salient scenes to: {output_path}")
    print(f"  Total scenes: {total_scenes}")
    if total_scenes > 0:
        print(f"  Salient scenes: {total_salient} ({100*total_salient/total_scenes:.1f}%)")
    else:
        print(f"  Salient scenes: {total_salient}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate train.pkl or val.pkl from ground truth labels")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to parquet file (train or validation)")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train",
                        help="Split to generate (train or val)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save .pkl file (should be summarization_data directory)")
    
    # Optional arguments
    parser.add_argument("--embeddings-cache-dir", type=str, default=None,
                        help="Directory to cache BERT embeddings (not used but kept for compatibility)")
    
    # Hugging Face arguments (if using HF datasets)
    parser.add_argument("--use-huggingface", action="store_true",
                        help="Load features from Hugging Face datasets")
    parser.add_argument("--hf-repo", type=str, default="Ishaank18/screenplay-features",
                        help="Hugging Face repository for features")
    parser.add_argument("--feature-groups", type=str, nargs="+", default=None,
                        help="Feature groups to load from Hugging Face")
    parser.add_argument("--mensa-repo", type=str, default="rohitsaxena/MENSA",
                        help="Hugging Face repository for MENSA dataset")
    
    args = parser.parse_args()
    
    # Default feature groups if using Hugging Face
    if args.use_huggingface and args.feature_groups is None:
        args.feature_groups = [
            'base', 'bert_surprisal', 'character_arcs', 'emotional', 'gc_academic',
            'gc_basic', 'gc_char_diversity', 'gc_concreteness', 'gc_dialogue',
            'gc_discourse', 'gc_narrative', 'gc_polarity', 'gc_pos', 'gc_pronouns',
            'gc_punctuation', 'gc_readability', 'gc_syntax', 'gc_temporal', 'genre',
            'gpt2_char_surprisal', 'graph', 'ngram', 'ngram_char_surprisal',
            'ngram_surprisal', 'plot_shifts', 'psychformers', 'rst', 'saxena_keller',
            'structure', 'textrank_centrality'
        ]
    
    # Map split name for dataset loading (MovieDataset expects "validation" not "val")
    dataset_split = "validation" if args.split == "val" else args.split
    
    generate_train_pkl_from_ground_truth(
        data_path=args.data_path,
        output_dir=args.output_dir,
        split=dataset_split,
        use_huggingface=args.use_huggingface,
        hf_repo=args.hf_repo,
        feature_groups=args.feature_groups,
        mensa_repo=args.mensa_repo,
        embeddings_cache_dir=args.embeddings_cache_dir,
        use_textrank=False  # Not needed for this
    )
    
    output_filename = f"{args.split}.pkl"
    print(f"\n✓ Successfully generated {output_filename} from ground truth labels!")
    print(f"  Output: {os.path.join(args.output_dir, output_filename)}")


if __name__ == "__main__":
    main()

