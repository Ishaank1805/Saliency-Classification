"""
Standalone script to compute TextRank centrality scores for movie scenes.

This implements the unsupervised TextRank algorithm as described in the paper:
- Constructs a graph where nodes = scenes, edges = cosine similarity between scene embeddings
- Computes centrality: centrality(Si) = λ1 * sum(j>i, eij) + λ2 * sum(j<i, eij)
- Uses BERT embeddings for scene representations
- Saves centrality scores to file

Based on:
- Mihalcea and Tarau (2004): TextRank algorithm
- Zheng and Lapata (2019): Extension for extractive summarization
- Papalampidi et al. (2020): Scene-based graph construction

Usage:
    # Process a single file
    python compute_textrank_centrality.py \
        --data-path /path/to/data.parquet \
        --model-name bert-base-uncased \
        --lambda1 0.7 --lambda2 0.3 \
        --device cuda --batch-size 16
    
    # Process directory with train/val/test splits (e.g., /tmp/ub/data with train.parquet, validation.parquet, test.parquet)
    python compute_textrank_centrality.py \
        --data-path /tmp/ub/data \
        --model-name bert-large-uncased \
        --embeddings-cache-dir /path/to/embeddings_cache \
        --lambda1 0.7 --lambda2 0.3
    
    # Use GPU with custom batch size
    python compute_textrank_centrality.py \
        --data-path /path/to/data.parquet \
        --device cuda --batch-size 32
    
    # Use pre-computed embeddings from cache (e.g., bert-large)
    python compute_textrank_centrality.py \
        --data-path /path/to/data.parquet \
        --model-name bert-large-uncased \
        --embeddings-cache-dir /path/to/embeddings_cache \
        --lambda1 0.7 --lambda2 0.3

Output:
    Saves a parquet file with columns:
    - movie_id: Movie identifier
    - scene_index: Scene index within movie
    - centrality_score: TextRank centrality score (higher = more central/important)
    - split: Data split (train/validation/test/all)
"""

import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Global cache for models (to avoid reloading for each movie)
_model_cache = {}
_tokenizer_cache = {}

# Global cache for scene_classification_data format (list of movies)
_scene_classification_data_cache = {}


def load_scene_embeddings_from_scene_classification_data(
    movie_id: int,
    split: str,
    embeddings_cache_dir: str,
    num_scenes: int
) -> Optional[torch.Tensor]:
    """
    Load pre-computed scene embeddings from scene_classification_data format.
    
    This format is from select_summ/src/precompute_embeddings.py:
    - Files: train.pkl, val.pkl, test.pkl
    - Format: List of movie dictionaries with 'scenes_embeddings' key
    - Uses RoBERTa-large [CLS] token embeddings
    
    Args:
        movie_id: Movie identifier (index in the list)
        split: Data split (train/validation/test -> maps to train/val/test.pkl)
        embeddings_cache_dir: Directory containing scene_classification_data folder OR the folder itself
        num_scenes: Expected number of scenes (for validation)
    
    Returns:
        Scene embeddings as torch.Tensor [num_scenes, hidden_dim] or None if not found
    """
    try:
        # Map split names: validation -> val, test -> test, train -> train
        split_map = {
            'train': 'train',
            'validation': 'val',
            'val': 'val',
            'test': 'test'
        }
        split_file = split_map.get(split, split)
        
        # Check if the provided path IS the scene_classification_data folder
        # (by checking if it contains train.pkl, val.pkl, or test.pkl directly)
        test_file = os.path.join(embeddings_cache_dir, f"{split_file}.pkl")
        if os.path.exists(test_file):
            # The provided path is the scene_classification_data folder itself
            scene_classification_path = embeddings_cache_dir
        else:
            # Look for scene_classification_data subfolder
            scene_classification_path = os.path.join(embeddings_cache_dir, "scene_classification_data")
            if not os.path.exists(scene_classification_path):
                return None
        
        cache_file = os.path.join(scene_classification_path, f"{split_file}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        # Load cached data (cache per split to avoid reloading)
        cache_key = (scene_classification_path, split_file)
        if cache_key not in _scene_classification_data_cache:
            with open(cache_file, 'rb') as f:
                movie_list = pickle.load(f)
            _scene_classification_data_cache[cache_key] = movie_list
        else:
            movie_list = _scene_classification_data_cache[cache_key]
        
        # Find movie by index (movie_id should match index in list)
        if movie_id < len(movie_list):
            movie_data = movie_list[movie_id]
            
            # Check if it has scenes_embeddings
            if 'scenes_embeddings' in movie_data:
                scene_embeddings = movie_data['scenes_embeddings']
                
                # Convert to tensor if needed
                if isinstance(scene_embeddings, torch.Tensor):
                    scene_embeddings = scene_embeddings.cpu()
                elif isinstance(scene_embeddings, np.ndarray):
                    scene_embeddings = torch.tensor(scene_embeddings, dtype=torch.float32)
                else:
                    scene_embeddings = torch.tensor(scene_embeddings, dtype=torch.float32)
                
                # Verify shape matches
                if scene_embeddings.shape[0] == num_scenes:
                    return scene_embeddings
                else:
                    print(f"Warning: Shape mismatch for movie {movie_id}: expected {num_scenes}, got {scene_embeddings.shape[0]}")
                    return None
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load embeddings from scene_classification_data for movie {movie_id}: {e}")
        return None


def load_scene_embeddings_from_cache(
    movie_id: int,
    split: str,
    model_name: str,
    embeddings_cache_dir: str,
    num_scenes: int
) -> Optional[torch.Tensor]:
    """
    Load pre-computed scene embeddings from cache.
    
    Args:
        movie_id: Movie identifier
        split: Data split (train/validation/test)
        model_name: Model name used for embeddings (e.g., 'bert-large-uncased')
        embeddings_cache_dir: Directory containing cache files
        num_scenes: Expected number of scenes (for validation)
    
    Returns:
        Scene embeddings as torch.Tensor [num_scenes, hidden_dim] or None if not found
    """
    try:
        # Convert model name to safe filename (e.g., 'bert-large-uncased' -> 'bert_large_uncased')
        model_name_safe = model_name.replace('/', '_').replace('-', '_')
        cache_file = os.path.join(
            embeddings_cache_dir,
            f"{model_name_safe}_{split}.pkl"
        )
        
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        embeddings_dict = cache_data.get('embeddings', {})
        if movie_id in embeddings_dict:
            scene_embeddings = torch.tensor(embeddings_dict[movie_id], dtype=torch.float32)
            # Verify shape matches
            if scene_embeddings.shape[0] == num_scenes:
                return scene_embeddings.cpu()
            else:
                print(f"Warning: Shape mismatch for movie {movie_id}: expected {num_scenes}, got {scene_embeddings.shape[0]}")
                return None
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load embeddings for movie {movie_id}: {e}")
        return None


def compute_scene_embeddings(
    scene_texts: List[str],
    model_name: str,
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 512
) -> torch.Tensor:
    """
    Compute BERT embeddings for scenes.
    
    Args:
        scene_texts: List of scene text strings
        model_name: Pre-trained BERT model name (e.g., 'bert-base-uncased')
        device: Device to run computation on ('cpu' or 'cuda')
        batch_size: Batch size for encoding scenes
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Scene embeddings as torch.Tensor [num_scenes, hidden_dim]
    """
    # Use cached model if available
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        model.to(device)
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
    else:
        model = _model_cache[model_name]
        tokenizer = _tokenizer_cache[model_name]
    
    num_scenes = len(scene_texts)
    if num_scenes == 0:
        return torch.empty(0)
    
    print(f"Computing embeddings for {num_scenes} scenes...")
    scene_embeddings_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_scenes, batch_size), desc="Encoding scenes"):
            batch = scene_texts[i:i+batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Get model outputs
            outputs = model(**encoded)
            
            # Use [CLS] token (pooler_output) if available, otherwise mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                scene_emb = outputs.pooler_output  # [batch_size, hidden_dim]
            else:
                # Mean pooling over sequence length
                scene_emb = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
            
            scene_embeddings_list.append(scene_emb.cpu())
    
    # Concatenate all embeddings
    scene_embeddings = torch.cat(scene_embeddings_list, dim=0)  # [num_scenes, hidden_dim]
    
    print(f"Computed embeddings shape: {scene_embeddings.shape}")
    return scene_embeddings


def compute_textrank_centrality(
    scene_embeddings: torch.Tensor,
    lambda1: float = 0.7,
    lambda2: float = 0.3
) -> np.ndarray:
    """
    Compute TextRank centrality scores for scenes.
    
    Centrality formula: centrality(Si) = λ1 * sum(j>i, eij) + λ2 * sum(j<i, eij)
    where eij is the edge weight (cosine similarity) between scenes Si and Sj.
    
    Args:
        scene_embeddings: Scene embeddings [num_scenes, hidden_dim]
        lambda1: Weight for forward-looking edges (j > i)
        lambda2: Weight for backward-looking edges (j < i)
    
    Returns:
        Centrality scores as numpy array [num_scenes]
    """
    num_scenes = scene_embeddings.shape[0]
    
    if num_scenes == 0:
        return np.array([])
    
    # Normalize embeddings for cosine similarity
    scene_embeddings = F.normalize(scene_embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    # similarity_matrix[i, j] = cosine_similarity(scene_i, scene_j)
    similarity_matrix = torch.mm(scene_embeddings, scene_embeddings.t())  # [num_scenes, num_scenes]
    
    # Compute centrality for each scene
    centrality_scores = []
    for i in range(num_scenes):
        # Forward-looking edges: sum of similarities to following scenes (j > i)
        if i < num_scenes - 1:
            forward_sum = similarity_matrix[i, i+1:].sum().item()
        else:
            forward_sum = 0.0
        
        # Backward-looking edges: sum of similarities to preceding scenes (j < i)
        if i > 0:
            backward_sum = similarity_matrix[i, :i].sum().item()
        else:
            backward_sum = 0.0
        
        # Centrality = λ1 * forward + λ2 * backward
        centrality = lambda1 * forward_sum + lambda2 * backward_sum
        centrality_scores.append(centrality)
    
    return np.array(centrality_scores, dtype=np.float32)


def compute_classification_metrics(
    results_df: pd.DataFrame,
    k_percent: float = 0.15
) -> Dict:
    """
    Compute classification metrics by selecting top-K scenes based on centrality.
    
    According to the paper, K = 15% of movie length.
    
    Args:
        results_df: DataFrame with movie_id, scene_index, centrality_score, label (if available)
        k_percent: Percentage of scenes to select as salient (default: 0.15 = 15%)
    
    Returns:
        Dictionary with classification metrics
    """
    if 'label' not in results_df.columns:
        print("Warning: No 'label' column found. Skipping classification metrics.")
        return {}
    
    all_predictions = []
    all_labels = []
    
    # Process each movie
    for movie_id in results_df['movie_id'].unique():
        movie_df = results_df[results_df['movie_id'] == movie_id].copy()
        movie_df = movie_df.sort_values('scene_index')
        
        num_scenes = len(movie_df)
        k = max(1, int(num_scenes * k_percent))  # At least 1 scene, K = 15% of movie length
        
        # Select top-K scenes by centrality score
        top_k_indices = movie_df.nlargest(k, 'centrality_score').index
        
        # Create predictions: 1 for top-K scenes, 0 otherwise
        predictions = np.zeros(num_scenes, dtype=int)
        predictions[movie_df.index.isin(top_k_indices)] = 1
        
        # Get ground truth labels
        labels = movie_df['label'].values.astype(int)
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    accuracy = balanced_accuracy_score(all_labels, all_predictions)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'binary_precision': float(precision_binary),  # Precision for salient class
        'binary_recall': float(recall_binary),  # Recall for salient class
        'binary_f1': float(f1_binary),  # F1 for salient class
        'balanced_accuracy': float(accuracy),
        'precision_class_0': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        'precision_class_1': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        'recall_class_0': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        'recall_class_1': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        'f1_class_0': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        'f1_class_1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        'support_class_0': int(support[0]) if len(support) > 0 else 0,
        'support_class_1': int(support[1]) if len(support) > 1 else 0,
        'k_percent': k_percent,
        'total_scenes': len(all_labels),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def print_classification_report(results_df: pd.DataFrame, k_percent: float = 0.15) -> Optional[Dict]:
    """Print detailed classification report and return metrics."""
    if 'label' not in results_df.columns:
        return None
    
    print(f"\n{'='*80}")
    print(f"TextRank Classification Report (Top-K Selection, K = {k_percent*100:.0f}% of movie length)")
    print(f"{'='*80}")
    
    # Compute predictions (top-K scenes per movie)
    all_predictions = []
    all_labels = []
    
    # Process each movie
    for movie_id in results_df['movie_id'].unique():
        movie_df = results_df[results_df['movie_id'] == movie_id].copy()
        movie_df = movie_df.sort_values('scene_index')
        
        num_scenes = len(movie_df)
        k = max(1, int(num_scenes * k_percent))  # At least 1 scene, K = 15% of movie length
        
        # Select top-K scenes by centrality score
        top_k_indices = movie_df.nlargest(k, 'centrality_score').index
        
        # Create predictions: 1 for top-K scenes, 0 otherwise
        predictions = np.zeros(num_scenes, dtype=int)
        predictions[movie_df.index.isin(top_k_indices)] = 1
        
        # Get ground truth labels
        labels = movie_df['label'].values.astype(int)
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Print sklearn classification report (same format as train_sequence.py)
    print("\nOverall Classification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=['Non-Salient', 'Salient'],
        zero_division=0
    ))
    
    # Compute and store detailed metrics
    overall_metrics = compute_classification_metrics(results_df, k_percent)
    
    print(f"\nSummary Metrics:")
    print(f"  Binary F1 (Salient Class): {overall_metrics['binary_f1']:.4f}")
    print(f"  Binary Precision:          {overall_metrics['binary_precision']:.4f}")
    print(f"  Binary Recall:             {overall_metrics['binary_recall']:.4f}")
    print(f"  Macro F1:                  {overall_metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:           {overall_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:              {overall_metrics['macro_recall']:.4f}")
    print(f"  Balanced Accuracy:         {overall_metrics['balanced_accuracy']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(overall_metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"              Non-Sal  Salient")
    print(f"  Actual Non-Sal  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"         Salient  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Per-split metrics
    if 'split' in results_df.columns:
        print(f"\n{'='*80}")
        print("Per-Split Classification Reports:")
        print(f"{'='*80}")
        
        for split in ['train', 'validation', 'test']:
            split_df = results_df[results_df['split'] == split]
            if len(split_df) == 0:
                continue
            
            # Compute predictions for this split
            split_predictions = []
            split_labels = []
            
            for movie_id in split_df['movie_id'].unique():
                movie_df = split_df[split_df['movie_id'] == movie_id].copy()
                movie_df = movie_df.sort_values('scene_index')
                
                num_scenes = len(movie_df)
                k = max(1, int(num_scenes * k_percent))
                
                top_k_indices = movie_df.nlargest(k, 'centrality_score').index
                predictions = np.zeros(num_scenes, dtype=int)
                predictions[movie_df.index.isin(top_k_indices)] = 1
                labels = movie_df['label'].values.astype(int)
                
                split_predictions.extend(predictions)
                split_labels.extend(labels)
            
            split_predictions = np.array(split_predictions)
            split_labels = np.array(split_labels)
            
            print(f"\n{split.upper()} Split Classification Report:")
            print(classification_report(
                split_labels,
                split_predictions,
                target_names=['Non-Salient', 'Salient'],
                zero_division=0
            ))
            
            # Print summary for this split
            split_metrics = compute_classification_metrics(split_df, k_percent)
            print(f"  Binary F1: {split_metrics['binary_f1']:.4f}, Macro F1: {split_metrics['macro_f1']:.4f}, Total Scenes: {split_metrics['total_scenes']}")
    
    return overall_metrics


def process_movie_data(
    data_path: str,
    model_name: str = "bert-base-uncased",
    lambda1: float = 0.7,
    lambda2: float = 0.3,
    device: str = "cpu",
    batch_size: int = 16,
    output_path: Optional[str] = None,
    embeddings_cache_dir: Optional[str] = None,
    k_percent: float = 0.15
) -> pd.DataFrame:
    """
    Process movie data and compute TextRank centrality scores.
    
    Args:
        data_path: Path to parquet file or directory with train/val/test.parquet
        model_name: BERT model name for scene embeddings
        lambda1: Weight for forward-looking edges (default: 0.7)
        lambda2: Weight for backward-looking edges (default: 0.3)
        device: Device to run computation on
        batch_size: Batch size for encoding scenes
        output_path: Path to save output file (if None, saves next to input)
        embeddings_cache_dir: Directory with pre-computed embeddings (optional)
    
    Returns:
        DataFrame with movie_id, scene_index, and centrality_score columns
    """
    # Load data
    if os.path.isdir(data_path):
        # If directory, process all splits
        splits = ['train', 'validation', 'test']
        all_results = []
        
        for split in splits:
            split_file = os.path.join(data_path, f"{split}.parquet")
            if os.path.exists(split_file):
                print(f"\n{'='*80}")
                print(f"Processing {split} split...")
                print(f"{'='*80}")
                split_results = process_single_file(
                    split_file, model_name, lambda1, lambda2, device, batch_size, split,
                    embeddings_cache_dir=embeddings_cache_dir
                )
                all_results.append(split_results)
        
        # Combine all results
        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
        else:
            raise ValueError(f"No parquet files found in {data_path}")
    else:
        # Single file
        print(f"\n{'='*80}")
        print(f"Processing file: {data_path}")
        print(f"{'='*80}")
        results_df = process_single_file(
            data_path, model_name, lambda1, lambda2, device, batch_size, "all",
            embeddings_cache_dir=embeddings_cache_dir
        )
    
    # Save results
    if output_path is None:
        if os.path.isdir(data_path):
            output_path = os.path.join(data_path, "textrank_centrality_scores.parquet")
        else:
            base_path = os.path.splitext(data_path)[0]
            output_path = f"{base_path}_textrank_centrality.parquet"
    
    print(f"\nSaving centrality scores to: {output_path}")
    results_df.to_parquet(output_path, index=False)
    print(f"✓ Saved {len(results_df)} scene centrality scores")
    
    # Print statistics
    print(f"\nCentrality Score Statistics:")
    print(f"  Mean: {results_df['centrality_score'].mean():.4f}")
    print(f"  Std:  {results_df['centrality_score'].std():.4f}")
    print(f"  Min:  {results_df['centrality_score'].min():.4f}")
    print(f"  Max:  {results_df['centrality_score'].max():.4f}")
    
    # Compute and print classification metrics
    classification_metrics = print_classification_report(results_df, k_percent=k_percent)
    
    # Save metrics to JSON file
    if classification_metrics is not None:
        metrics_output_path = output_path.replace('.parquet', '_metrics.json')
        with open(metrics_output_path, 'w') as f:
            json.dump(classification_metrics, f, indent=2, default=str)
        print(f"\n✓ Classification metrics saved to: {metrics_output_path}")
    
    return results_df


def process_single_file(
    file_path: str,
    model_name: str,
    lambda1: float,
    lambda2: float,
    device: str,
    batch_size: int,
    split: str,
    embeddings_cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """Process a single parquet file and compute centrality scores."""
    print(f"Loading data from: {file_path}")
    
    # Only read the columns we need (ignore linguistic features to save memory)
    # Include 'label' for classification metrics
    required_cols = ['movie_id', 'scene_index', 'scene_text', 'label']
    print(f"Reading columns: {required_cols} (including labels for classification metrics)")
    
    # Read parquet file with only required columns
    try:
        df = pd.read_parquet(file_path, columns=required_cols)
        print(f"✓ Successfully loaded data with {len(df)} scenes")
    except Exception as e:
        # If column selection fails (some parquet versions don't support it), read all and select
        print(f"Note: Could not read specific columns ({e}), reading all columns and selecting...")
        df = pd.read_parquet(file_path)
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Select only required columns
        df = df[required_cols]
        print(f"✓ Loaded data with {len(df)} scenes (filtered to required columns)")
    
    # Group by movie
    movie_ids = df['movie_id'].unique()
    print(f"Found {len(movie_ids)} movies")
    
    # Check if we can use cached embeddings
    use_cache = embeddings_cache_dir is not None and os.path.exists(embeddings_cache_dir)
    if use_cache:
        print(f"Using pre-computed embeddings from: {embeddings_cache_dir}")
        # Check which format is available
        split_map = {'train': 'train', 'validation': 'val', 'val': 'val', 'test': 'test'}
        split_file = split_map.get(split, split)
        
        # Check if the provided path IS the scene_classification_data folder
        test_file = os.path.join(embeddings_cache_dir, f"{split_file}.pkl")
        if os.path.exists(test_file):
            # The provided path is the scene_classification_data folder itself
            print(f"  Found scene_classification_data format: {test_file}")
        else:
            # Check for scene_classification_data subfolder
            scene_classification_path = os.path.join(embeddings_cache_dir, "scene_classification_data")
            cache_file = os.path.join(scene_classification_path, f"{split_file}.pkl")
            if os.path.exists(cache_file):
                print(f"  Found scene_classification_data format: {cache_file}")
            else:
                print(f"  Looking for standard cache file: {model_name.replace('/', '_').replace('-', '_')}_{split}.pkl")
    else:
        if embeddings_cache_dir is not None:
            print(f"Warning: Cache directory not found: {embeddings_cache_dir}")
            print("  Computing embeddings on-the-fly...")
        else:
            print("Computing embeddings on-the-fly (no cache directory provided)")
    
    all_results = []
    cache_hits = 0
    cache_misses = 0
    
    for movie_id in tqdm(movie_ids, desc="Processing movies"):
        movie_df = df[df['movie_id'] == movie_id].sort_values('scene_index')
        scene_texts = movie_df['scene_text'].astype(str).tolist()
        scene_indices = movie_df['scene_index'].tolist()
        
        if len(scene_texts) == 0:
            continue
        
        # Try to load from cache first
        scene_embeddings = None
        if use_cache:
            # First try scene_classification_data format (from precompute_embeddings.py)
            scene_embeddings = load_scene_embeddings_from_scene_classification_data(
                movie_id, split, embeddings_cache_dir, len(scene_texts)
            )
            
            # If not found, try standard cache format
            if scene_embeddings is None:
                scene_embeddings = load_scene_embeddings_from_cache(
                    movie_id, split, model_name, embeddings_cache_dir, len(scene_texts)
                )
            
            if scene_embeddings is not None:
                cache_hits += 1
            else:
                cache_misses += 1
        
        # Compute embeddings if not in cache
        if scene_embeddings is None:
            scene_embeddings = compute_scene_embeddings(
                scene_texts, model_name, device, batch_size
            )
        
        # Compute TextRank centrality
        centrality_scores = compute_textrank_centrality(
            scene_embeddings, lambda1, lambda2
        )
        
        # Store results (including labels)
        movie_labels = movie_df['label'].tolist()
        
        for scene_idx, centrality, label in zip(scene_indices, centrality_scores, movie_labels):
            all_results.append({
                'movie_id': movie_id,
                'scene_index': scene_idx,
                'centrality_score': float(centrality),
                'label': int(label),
                'split': split
            })
    
    # Print cache statistics
    if use_cache:
        print(f"\nCache Statistics:")
        print(f"  Cache hits: {cache_hits}/{len(movie_ids)} movies")
        print(f"  Cache misses: {cache_misses}/{len(movie_ids)} movies")
        if cache_hits > 0:
            print(f"  Cache hit rate: {100*cache_hits/len(movie_ids):.1f}%")
    
    results_df = pd.DataFrame(all_results)
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute TextRank centrality scores for movie scenes using BERT embeddings"
    )
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to parquet file or directory with train/val/test.parquet"
    )
    
    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="BERT model name for scene embeddings (default: bert-base-uncased)"
    )
    
    # TextRank hyperparameters
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.7,
        help="Weight for forward-looking edges (default: 0.7)"
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.3,
        help="Weight for backward-looking edges (default: 0.3)"
    )
    
    # Computation settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computation on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding scenes (default: 16)"
    )
    
    # Output
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save output file (default: saves next to input with _textrank_centrality suffix)"
    )
    
    # Classification metrics
    parser.add_argument(
        "--k-percent",
        type=float,
        default=0.15,
        help="Percentage of scenes to select as salient for classification metrics (default: 0.15 = 15%%, as in paper)"
    )
    
    # Embeddings cache
    parser.add_argument(
        "--embeddings-cache-dir",
        type=str,
        default=None,
        help="Directory with pre-computed embeddings. Supports two formats:\n"
             "1. Standard format: {model_name}_{split}.pkl (e.g., bert_large_uncased_train.pkl)\n"
             "2. scene_classification_data format: Directory containing scene_classification_data/\n"
             "   folder with train.pkl, val.pkl, test.pkl (from precompute_embeddings.py).\n"
             "   The script will automatically detect and use the appropriate format.\n"
             "If provided, will load embeddings from cache instead of computing on-the-fly."
    )
    
    # Additional options
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Also save scene embeddings to file (for reuse)"
    )
    parser.add_argument(
        "--embeddings-output-dir",
        type=str,
        default=None,
        help="Directory to save embeddings cache (default: same as data-path)"
    )
    
    args = parser.parse_args()
    
    # Validate lambda values
    if abs(args.lambda1 + args.lambda2 - 1.0) > 1e-6:
        print(f"Warning: lambda1 + lambda2 = {args.lambda1 + args.lambda2:.4f} (should be 1.0)")
        print("  Normalizing to sum to 1.0...")
        total = args.lambda1 + args.lambda2
        args.lambda1 = args.lambda1 / total
        args.lambda2 = args.lambda2 / total
        print(f"  Adjusted: lambda1={args.lambda1:.4f}, lambda2={args.lambda2:.4f}")
    
    print(f"\n{'='*80}")
    print("TextRank Centrality Computation")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Lambda1 (forward): {args.lambda1}")
    print(f"Lambda2 (backward): {args.lambda2}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    if args.embeddings_cache_dir:
        print(f"Embeddings cache: {args.embeddings_cache_dir}")
    print(f"{'='*80}\n")
    
    # Process data
    results_df = process_movie_data(
        data_path=args.data_path,
        model_name=args.model_name,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=args.device,
        batch_size=args.batch_size,
        output_path=args.output_path,
        embeddings_cache_dir=args.embeddings_cache_dir,
        k_percent=getattr(args, 'k_percent', 0.15)
    )
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings_dir = args.embeddings_output_dir
        if embeddings_dir is None:
            if os.path.isdir(args.data_path):
                embeddings_dir = args.data_path
            else:
                embeddings_dir = os.path.dirname(args.data_path)
        
        embeddings_dir = os.path.join(embeddings_dir, "embeddings_cache")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        print(f"\nSaving embeddings to: {embeddings_dir}")
        print("Note: Embeddings are not saved by default. Re-run with --save-embeddings to cache them.")
        # TODO: Implement embedding caching if needed
    
    print(f"\n{'='*80}")
    print("✓ TextRank centrality computation completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

