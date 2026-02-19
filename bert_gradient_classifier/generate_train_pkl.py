#!/usr/bin/env python3
"""
Script to generate train.pkl for summarization from trained model predictions.

This script:
1. Loads a trained model checkpoint
2. Runs inference on the training set
3. Saves salient scenes in the same format as val.pkl and test.pkl
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

# Add parent directory to path to import from train_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_sequence import (
    MovieDataset,
    SceneSaliencyWithLinguistic,
    collate_movies,
    evaluate,
    save_salient_scenes_for_summarization,
    get_positive_weight
)
from experiment_config import get_experiment_config


def main():
    parser = argparse.ArgumentParser(description="Generate train.pkl for summarization")
    
    # Model checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., bert_large_frozen_linguistic_gated)")
    
    # Data arguments
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to train parquet file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save train.pkl (should be summarization_data directory)")
    
    # Optional arguments (same as train_sequence.py)
    parser.add_argument("--embeddings-cache-dir", type=str, default=None,
                        help="Directory to cache BERT embeddings")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detected if not specified")
    parser.add_argument("--no-textrank", action="store_true",
                        help="Disable TextRank centrality feature")
    
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
    
    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print("=" * 80)
    print(f"Generating train.pkl for experiment: {args.experiment}")
    print("=" * 80)
    
    # Load experiment config
    config = get_experiment_config(args.experiment)
    print(f"\nDescription: {config.get('description', 'N/A')}")
    
    # Load checkpoint
    print(f"\nLoading model from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
    print(f"  Optimal threshold from checkpoint: {optimal_threshold:.3f}")
    
    # Determine feature columns and other settings
    use_textrank = not args.no_textrank
    textrank_model = None
    textrank_lambda1 = 0.5
    textrank_lambda2 = 0.5
    use_top_features = False
    top_features_list = None
    
    # Hugging Face settings
    use_huggingface = args.use_huggingface
    hf_repo = args.hf_repo
    feature_groups = args.feature_groups
    mensa_repo = args.mensa_repo
    
    if use_huggingface and feature_groups is None:
        # Default to all feature groups if not specified
        feature_groups = [
            'base', 'bert_surprisal', 'character_arcs', 'emotional', 'gc_academic',
            'gc_basic', 'gc_char_diversity', 'gc_concreteness', 'gc_dialogue',
            'gc_discourse', 'gc_narrative', 'gc_polarity', 'gc_pos', 'gc_pronouns',
            'gc_punctuation', 'gc_readability', 'gc_syntax', 'gc_temporal', 'genre',
            'gpt2_char_surprisal', 'graph', 'ngram', 'ngram_char_surprisal',
            'ngram_surprisal', 'plot_shifts', 'psychformers', 'rst', 'saxena_keller',
            'structure', 'textrank_centrality'
        ]
    
    # Load train dataset
    print(f"\nLoading train dataset from: {args.train_path}")
    train_dataset = MovieDataset(
        args.train_path, "train", None,  # linguistic_cols will be determined from data
        use_textrank=use_textrank,
        textrank_model=textrank_model,
        embeddings_cache_dir=args.embeddings_cache_dir,
        textrank_lambda1=textrank_lambda1,
        textrank_lambda2=textrank_lambda2,
        use_top_features=use_top_features,
        top_features_list=top_features_list,
        use_huggingface=use_huggingface,
        hf_repo=hf_repo,
        feature_groups=feature_groups,
        mensa_repo=mensa_repo
    )
    feature_columns = train_dataset.feature_columns
    
    # Get linguistic dimension
    if train_dataset.movies and train_dataset.movies[0]['linguistic_features'] is not None:
        linguistic_dim = train_dataset.movies[0]['linguistic_features'].shape[1]
    else:
        linguistic_dim = 0
    
    print(f"  Loaded {len(train_dataset.movies)} movies")
    print(f"  Linguistic features: {linguistic_dim} dimensions")
    
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✓ Model loaded successfully")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_movies, num_workers=0
    )
    
    # Compute positive weight from dataset (for loss calculation, though not critical for inference)
    positive_weight = get_positive_weight(train_dataset)
    print(f"  Positive class weight: {positive_weight:.4f}")
    
    # Create criterion (not used for inference, but required by evaluate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight).to(device))
    
    # Run evaluation to get per-movie predictions
    print(f"\nRunning inference on training set...")
    print(f"  Using threshold: {optimal_threshold:.3f}")
    
    train_result = evaluate(
        model, train_loader, criterion, device, optimal_threshold,
        return_probs=True, return_per_movie=True
    )
    
    # Unpack results (5 values when return_per_movie=True)
    train_loss, train_metrics, train_probs, train_targets, train_per_movie = train_result
    
    print(f"\nTraining Set Metrics:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  Binary F1: {train_metrics['binary_f1']:.4f}")
    print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"  Precision: {train_metrics['precision_class_1']:.4f}")
    print(f"  Recall: {train_metrics['recall_class_1']:.4f}")
    print(f"  Processed {len(train_per_movie)} movies")
    
    # Save salient scenes for summarization
    print(f"\nSaving train.pkl to: {args.output_dir}")
    save_salient_scenes_for_summarization(
        train_per_movie,
        train_loader,
        optimal_threshold,
        args.output_dir,
        split="train"
    )
    
    print(f"\n✓ Successfully generated train.pkl!")
    print(f"  Output: {os.path.join(args.output_dir, 'train.pkl')}")


if __name__ == "__main__":
    main()

