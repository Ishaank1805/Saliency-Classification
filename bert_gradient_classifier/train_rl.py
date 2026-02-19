"""
Reinforcement Learning (RL) Training Script for Scene Saliency.

This script implements Self-Critical Sequence Training (SCST), a form of RL 
that directly optimizes non-differentiable metrics (like F1 score).

Method:
1. Initialize with a pre-trained model (supervised learning).
2. For each movie:
   - Generate a "Greedy" prediction (argmax) -> Baseline Reward
   - Generate a "Sampled" prediction (probabilistic) -> Sample Reward
   - Loss = -(Sample_Reward - Baseline_Reward) * log_prob(Sample)
   
This encourages the model to generate sequences that have higher F1 scores 
than its current best guess.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import time
from typing import Dict, Tuple

# Import from existing codebase
from sequence_model import SceneSaliencyWithLinguistic
from train_sequence import MovieDataset, collate_movies, compute_metrics, evaluate
from experiment_config import get_experiment_config

def get_reward(predictions: torch.Tensor, targets: torch.Tensor, metric: str = 'f1_macro') -> float:
    """
    Compute reward (metric score) for a sequence of predictions.
    
    Args:
        predictions: Binary predictions [num_scenes]
        targets: Ground truth labels [num_scenes]
        metric: 'f1_macro', 'f1_binary' (positive class), or 'accuracy'
    """
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    if metric == 'f1_macro':
        return f1_score(targets_np, preds_np, average='macro', zero_division=0)
    elif metric == 'f1_binary':
        return f1_score(targets_np, preds_np, average='binary', zero_division=0)
    elif metric == 'accuracy':
        return (preds_np == targets_np).mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")

def train_epoch_rl(
    model: SceneSaliencyWithLinguistic,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    reward_metric: str = 'f1_macro',
    baseline_decay: float = 0.99
) -> Tuple[float, Dict]:
    """Train one epoch using Self-Critical Sequence Training (RL)."""
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    
    progress_bar = tqdm(dataloader, desc="RL Training")
    
    for batch in progress_bar:
        # RL works best with batch_size=1 (one movie episode at a time) usually, 
        # but we can do batched if we handle indices carefully.
        # For simplicity and safety with sequence lengths, we iterate through movies in the batch.
        
        batch_loss = 0.0
        optimizer.zero_grad()
        
        movies_processed = 0
        
        for movie in batch:
            scene_texts = movie['scene_texts']
            labels = torch.tensor(movie['labels'], dtype=torch.float32).to(device)
            movie_id = movie.get('movie_id')
            
            # Skip if no labels
            if len(labels) == 0:
                continue
                
            # Prepare linguistic features
            linguistic_features = None
            if model.use_linguistic and movie.get('linguistic_features') is not None:
                linguistic_features = torch.tensor(
                    movie['linguistic_features'],
                    dtype=torch.float32
                ).to(device)
                linguistic_features = torch.nan_to_num(linguistic_features, nan=0.0)
            
            # 1. Forward pass to get logits
            # Use split="train" for embedding loading if applicable
            logits = model(scene_texts, linguistic_features, movie_id=movie_id, split="train") # [num_scenes]
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits
            
            probs = torch.sigmoid(logits)
            
            # 2. Greedy Decoding (Baseline)
            # No gradient needed for baseline
            with torch.no_grad():
                greedy_preds = (probs > 0.5).float()
                greedy_reward = get_reward(greedy_preds, labels, reward_metric)
            
            # 3. Sampling (Exploration)
            # We need gradients here to update the policy
            m = torch.distributions.Bernoulli(probs)
            sample_preds = m.sample()
            
            # Compute log probability of the sampled sequence
            log_probs = m.log_prob(sample_preds)
            # Sum log probs over the sequence (episode)
            seq_log_prob = log_probs.sum()
            
            # 4. Compute Reward for Sample
            # Detach to ensure we don't backprop through reward calculation
            with torch.no_grad():
                sample_reward = get_reward(sample_preds, labels, reward_metric)
            
            # 5. RL Loss: -(Reward - Baseline) * LogProb
            # We want to maximize Reward, so minimize Negative Reward
            reward_diff = sample_reward - greedy_reward
            
            # Policy Gradient Loss
            # If sample is better than greedy (diff > 0), we increase prob of sample.
            # If sample is worse (diff < 0), we decrease prob of sample.
            loss = -(reward_diff) * seq_log_prob
            
            # Normalize by sequence length to keep gradients stable? 
            # Usually SCST doesn't, but for variable length movies it might help.
            # Let's stick to standard SCST first.
            
            loss.backward()
            
            batch_loss += loss.item()
            total_reward += sample_reward
            movies_processed += 1
        
        if movies_processed > 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss / movies_processed
        
        progress_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_reward': f'{total_reward / (progress_bar.n + 1):.4f}'
        })
        
    avg_loss = total_loss / len(dataloader)
    avg_reward = total_reward / len(dataloader.dataset)
    
    return avg_loss, {'avg_reward': avg_reward}

def main():
    parser = argparse.ArgumentParser(description="RL Fine-tuning for Scene Saliency")
    
    # Paths
    parser.add_argument("--checkpoint-path", type=str, required=True, 
                        help="Path to pre-trained model checkpoint (Supervised)")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test-path", type=str, required=True, help="Path to test data")
    
    # RL Config
    parser.add_argument("--epochs", type=int, default=5, help="Number of RL epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (usually smaller than SL)")
    parser.add_argument("--reward-metric", type=str, default="f1_macro", 
                        choices=["f1_macro", "f1_binary", "accuracy"],
                        help="Metric to optimize")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./experiments_rl",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Pre-trained Model Config & Weights
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create Output Dir
    exp_name = config.get('experiment_name', 'rl_experiment')
    output_dir = os.path.join(args.output_dir, f"{exp_name}_rl")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    # Helper to guess linguistic cols if not in config
    train_dataset = MovieDataset(args.train_path, "train")
    # We rely on the model config to know if linguistic features are used
    # But we need to make sure the dataset loads them if the model needs them
    
    # Re-initialize datasets ensuring correct feature columns from train are used
    feature_columns = train_dataset.feature_columns
    val_dataset = MovieDataset(args.val_path, "validation", feature_columns=feature_columns)
    test_dataset = MovieDataset(args.test_path, "test", feature_columns=feature_columns)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_movies)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_movies)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_movies)
    
    # Initialize Model
    print("Initializing model...")
    # Filter config keys that match the model constructor
    model_kwargs = {k: v for k, v in config.items() 
                   if k not in ['description', 'device', 'experiment_name']}
    
    model = SceneSaliencyWithLinguistic(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Optimizer
    # We might want to freeze the BERT encoder during RL to save memory/time
    # and only train the fusion/sequence/classifier layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initial Evaluation
    print("Evaluating pre-trained model...")
    criterion = nn.BCEWithLogitsLoss() # Only for evaluation tracking
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"Initial Validation F1: {val_metrics['macro_f1']:.4f}")
    
    best_val_f1 = val_metrics['macro_f1']
    
    # Training Loop
    print(f"\nStarting RL Training (Optimizing {args.reward_metric})...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # RL Train Step
        avg_loss, train_stats = train_epoch_rl(
            model, train_loader, optimizer, device, 
            reward_metric=args.reward_metric
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"RL Loss: {avg_loss:.4f}, Avg Reward: {train_stats['avg_reward']:.4f}")
        print(f"Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Save Best
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            print(f"  ✓ New best model!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'config': config
            }, os.path.join(output_dir, "best_model_rl.pt"))
            
    print("\nRL Training Complete.")
    
    # Test Best Model
    print("Evaluating best RL model on Test set...")
    checkpoint = torch.load(os.path.join(output_dir, "best_model_rl.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test F1: {test_metrics['macro_f1']:.4f}")
    
    # Save Results
    results = {
        'original_checkpoint': args.checkpoint_path,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'rl_config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'reward_metric': args.reward_metric
        }
    }
    
    with open(os.path.join(output_dir, "results_rl.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

