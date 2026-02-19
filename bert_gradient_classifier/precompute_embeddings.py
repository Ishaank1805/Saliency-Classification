"""
Pre-compute scene embeddings for frozen encoders.

This significantly speeds up training by computing embeddings once
instead of recomputing them every epoch.

Usage:
    python3 precompute_embeddings.py \
        --model-name roberta-large \
        --train-path /tmp/ub/data/train.parquet \
        --val-path /tmp/ub/data/validation.parquet \
        --test-path /tmp/ub/data/test.parquet \
        --output-dir /tmp/ub/embeddings_cache
        
    python3 precompute_embeddings.py \
        --model-name bert-base-uncased \
        --train-path /tmp/ub/data/train.parquet \
        --val-path /tmp/ub/data/validation.parquet \
        --test-path /tmp/ub/data/test.parquet \
        --output-dir /tmp/ub/embeddings_cache
    
    python3 precompute_embeddings.py \
        --model-name bert-large-uncased \
        --train-path /tmp/ub/data/train.parquet \
        --val-path /tmp/ub/data/validation.parquet \
        --test-path /tmp/ub/data/test.parquet \
        --output-dir /tmp/ub/embeddings_cache
    
    python3 precompute_embeddings.py \
        --model-name roberta-base \
        --train-path /tmp/ub/data/train.parquet \
        --val-path /tmp/ub/data/validation.parquet \
        --test-path /tmp/ub/data/test.parquet \
        --output-dir /tmp/ub/embeddings_cache
    
"""

import argparse
import os
import pickle
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from typing import Dict, List


def load_movie_data(data_path: str) -> Dict:
    """Load movie data grouped by movie_id."""
    if os.path.isdir(data_path):
        # Assume it's a directory with train/val/test.parquet
        raise ValueError("Please specify full path to parquet file")
    
    df = pd.read_parquet(data_path)
    
    # Group by movie
    movies = {}
    for movie_id in df['movie_id'].unique():
        movie_df = df[df['movie_id'] == movie_id].sort_values('scene_index')
        movies[movie_id] = {
            'scene_texts': movie_df['scene_text'].astype(str).tolist(),
            'scene_indices': movie_df['scene_index'].tolist(),
        }
    
    return movies


def compute_embeddings(
    model_name: str,
    movies: Dict,
    device: str = None,
    batch_size: int = 16
) -> Dict:
    """Compute embeddings for all scenes in all movies."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Computing embeddings for {len(movies)} movies...")
    
    embeddings_dict = {}
    
    with torch.no_grad():
        for movie_id, movie_data in tqdm(movies.items(), desc="Processing movies"):
            scene_texts = movie_data['scene_texts']
            scene_embeddings = []
            
            # Process scenes in batches
            for i in range(0, len(scene_texts), batch_size):
                batch = scene_texts[i:i+batch_size]
                
                encoded = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                    # No max_length specified - uses model default (512 for RoBERTa-large)
                ).to(device)
                
                outputs = model(**encoded)
                
                # Use [CLS] token embedding (exact same as Saxena & Keller)
                batch_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                
                scene_embeddings.append(batch_emb.cpu())
            
            # Concatenate all scene embeddings for this movie
            movie_embeddings = torch.cat(scene_embeddings, dim=0)  # [num_scenes, hidden_dim]
            embeddings_dict[movie_id] = movie_embeddings.numpy()
    
    return embeddings_dict


def save_embeddings(embeddings_dict: Dict, output_path: str, model_name: str):
    """Save embeddings to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a dict with metadata
    save_data = {
        'embeddings': embeddings_dict,
        'model_name': model_name,
        'num_movies': len(embeddings_dict),
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved embeddings to: {output_path}")
    print(f"  Model: {model_name}")
    print(f"  Movies: {len(embeddings_dict)}")
    
    # Print embedding dimensions
    if embeddings_dict:
        first_movie = list(embeddings_dict.values())[0]
        print(f"  Embedding dim: {first_movie.shape[1]}")
        print(f"  Scenes per movie: {[emb.shape[0] for emb in list(embeddings_dict.values())[:3]]}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute scene embeddings")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name (e.g., roberta-large, bert-base-uncased)")
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to training data parquet file")
    parser.add_argument("--val-path", type=str, required=True,
                        help="Path to validation data parquet file")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to test data parquet file")
    parser.add_argument("--output-dir", type=str, default="./embeddings_cache",
                        help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding computation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each split
    splits = {
        'train': args.train_path,
        'validation': args.val_path,
        'test': args.test_path,
    }
    
    # Normalize model name for filename
    model_name_safe = args.model_name.replace('/', '_').replace('-', '_')
    
    for split_name, data_path in splits.items():
        print(f"\n{'='*80}")
        print(f"Processing {split_name} split")
        print(f"{'='*80}")
        
        # Load data
        movies = load_movie_data(data_path)
        print(f"Loaded {len(movies)} movies")
        
        # Compute embeddings
        embeddings = compute_embeddings(
            args.model_name,
            movies,
            batch_size=args.batch_size
        )
        
        # Save embeddings
        output_path = os.path.join(
            args.output_dir,
            f"{model_name_safe}_{split_name}.pkl"
        )
        save_embeddings(embeddings, output_path, args.model_name)
    
    print(f"\n{'='*80}")
    print("All embeddings pre-computed successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

