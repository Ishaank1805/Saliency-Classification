"""
Evaluation and comparison script.
Compares models with/without features.
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from extract_bert_gradients import extract_bert_features_for_dataframe, BERTGradientExtractor
from feature_fusion import FeatureFusion, load_linguistic_features
from classifiers import SVMClassifier, LlamaClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_configuration(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config_name: str,
    use_bert: bool,
    use_linguistic: bool,
    use_svm: bool = True,
    use_llama: bool = False,
    label_col: str = "label",
) -> Dict:
    """Evaluate a specific configuration."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {config_name}")
    print(f"{'='*80}")
    print(f"BERT features: {use_bert}")
    print(f"Linguistic features: {use_linguistic}")
    
    # Feature fusion
    fusion = FeatureFusion(use_pca=False)
    
    X_train = fusion.fuse_features(train_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=True)
    X_val = fusion.fuse_features(val_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=False)
    X_test = fusion.fuse_features(test_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=False)
    
    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values
    
    results = {
        "config": config_name,
        "use_bert": use_bert,
        "use_linguistic": use_linguistic,
        "feature_dim": X_train.shape[1],
    }
    
    # Train and evaluate SVM
    if use_svm:
        print("\nTraining SVM...")
        svm = SVMClassifier(tune_hyperparameters=True)
        svm.fit(X_train, y_train)
        
        val_metrics = svm.evaluate(X_val, y_val)
        test_metrics = svm.evaluate(X_test, y_test)
        
        results["svm"] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        
        print(f"SVM - Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}")
    
    # Train and evaluate Llama
    if use_llama:
        print("\nTraining Llama...")
        llama = LlamaClassifier()
        
        train_texts = train_df["scene_text"].astype(str).tolist()
        val_texts = val_df["scene_text"].astype(str).tolist()
        test_texts = test_df["scene_text"].astype(str).tolist()
        
        llama.fit(train_texts, y_train.tolist(), val_texts, y_val.tolist())
        
        val_metrics = llama.evaluate(val_texts, y_val.tolist())
        test_metrics = llama.evaluate(test_texts, y_test.tolist())
        
        results["llama"] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        
        print(f"Llama - Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}")
    
    return results


def compare_all_configurations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "comparison_results",
) -> Dict:
    """Compare all feature configurations."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    configurations = [
        ("BERT Only", True, False),
        ("Linguistic Only", False, True),
        ("BERT + Linguistic", True, True),
        ("No Features (Baseline)", False, False),
    ]
    
    all_results = []
    
    for config_name, use_bert, use_linguistic in configurations:
        results = evaluate_configuration(
            train_df,
            val_df,
            test_df,
            config_name,
            use_bert,
            use_linguistic,
            use_svm=True,
            use_llama=False,
        )
        all_results.append(results)
    
    # Save results
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    comparison_data = []
    for r in all_results:
        if "svm" in r:
            comparison_data.append({
                "Configuration": r["config"],
                "Features": f"BERT={r['use_bert']}, Ling={r['use_linguistic']}",
                "Feature Dim": r["feature_dim"],
                "Val Precision": r["svm"]["val"]["precision"],
                "Val Recall": r["svm"]["val"]["recall"],
                "Val F1": r["svm"]["val"]["f1"],
                "Test Precision": r["svm"]["test"]["precision"],
                "Test Recall": r["svm"]["test"]["recall"],
                "Test F1": r["svm"]["test"]["f1"],
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Create visualization
    create_comparison_plots(comparison_df, output_dir)
    
    return all_results


def create_comparison_plots(comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization plots for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F1 scores
    axes[0, 0].bar(comparison_df["Configuration"], comparison_df["Val F1"], label="Val F1", alpha=0.7)
    axes[0, 0].bar(comparison_df["Configuration"], comparison_df["Test F1"], label="Test F1", alpha=0.7)
    axes[0, 0].set_title("F1 Scores by Configuration")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision
    axes[0, 1].bar(comparison_df["Configuration"], comparison_df["Val Precision"], label="Val Precision", alpha=0.7)
    axes[0, 1].bar(comparison_df["Configuration"], comparison_df["Test Precision"], label="Test Precision", alpha=0.7)
    axes[0, 1].set_title("Precision by Configuration")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1, 0].bar(comparison_df["Configuration"], comparison_df["Val Recall"], label="Val Recall", alpha=0.7)
    axes[1, 0].bar(comparison_df["Configuration"], comparison_df["Test Recall"], label="Test Recall", alpha=0.7)
    axes[1, 0].set_title("Recall by Configuration")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Feature dimensions
    axes[1, 1].bar(comparison_df["Configuration"], comparison_df["Feature Dim"])
    axes[1, 1].set_title("Feature Dimensions by Configuration")
    axes[1, 1].set_ylabel("Number of Features")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plots.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison plots to {output_dir}/comparison_plots.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test-path", type=str, required=True, help="Path to test data")
    parser.add_argument("--compare-all", action="store_true", help="Compare all configurations")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(args.train_path) if args.train_path.endswith(".parquet") else pd.read_csv(args.train_path)
    val_df = pd.read_parquet(args.val_path) if args.val_path.endswith(".parquet") else pd.read_csv(args.val_path)
    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    
    # Extract features (assuming they're already extracted or will be extracted)
    # In practice, you'd extract features here similar to train.py
    
    if args.compare_all:
        compare_all_configurations(train_df, val_df, test_df, args.output_dir)
    else:
        # Single evaluation
        results = evaluate_configuration(
            train_df,
            val_df,
            test_df,
            "Default",
            use_bert=True,
            use_linguistic=True,
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

