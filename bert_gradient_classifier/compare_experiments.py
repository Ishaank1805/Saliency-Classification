"""
Compare results across all experiments and generate comparison reports.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_experiment_results(experiments_dir: str) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results = []
    
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue
        
        with open(results_file, "r") as f:
            result = json.load(f)
        
        # Extract key metrics
        row = {
            'experiment': result['experiment'],
            'best_val_f1': result['best_val_f1'],
            'test_f1': result['test_metrics']['macro_f1'],
            'test_precision': result['test_metrics']['macro_precision'],
            'test_recall': result['test_metrics']['macro_recall'],
            'test_accuracy': result['test_metrics']['balanced_accuracy'],
            'best_epoch': result['best_epoch'],
            'training_time': result['training_time'],
        }
        
        # Add config details
        config = result.get('config', {})
        row.update({
            'scene_model': config.get('scene_model_name', 'N/A'),
            'scene_finetune': config.get('scene_encoder_finetune', False),
            'use_linguistic': config.get('use_linguistic', False),
            'fusion_method': config.get('fusion_method', 'N/A'),
            'sequence_model': config.get('sequence_model_type', 'N/A'),
            'use_pos_encoding': config.get('use_positional_encoding', False),
            'classifier': config.get('classifier_type', 'N/A'),
        })
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def generate_comparison_report(df: pd.DataFrame, output_path: str):
    """Generate comprehensive comparison report."""
    
    report = []
    report.append("# Experiment Comparison Report\n")
    report.append(f"Total experiments: {len(df)}\n\n")
    
    # Overall best
    report.append("## Best Performing Models\n\n")
    report.append("### By Test F1 Score\n")
    best_f1 = df.nlargest(5, 'test_f1')[['experiment', 'test_f1', 'test_precision', 'test_recall']]
    report.append(best_f1.to_markdown(index=False))
    report.append("\n\n")
    
    # By dimension
    report.append("## Ablation Studies\n\n")
    
    # Linguistic features
    if 'use_linguistic' in df.columns:
        report.append("### Linguistic Features Impact\n")
        ling_comparison = df.groupby('use_linguistic').agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(ling_comparison.to_markdown())
        report.append("\n\n")
    
    # Fusion methods
    if 'fusion_method' in df.columns:
        report.append("### Fusion Method Comparison\n")
        fusion_comparison = df[df['use_linguistic'] == True].groupby('fusion_method').agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(fusion_comparison.to_markdown())
        report.append("\n\n")
    
    # Sequence models
    if 'sequence_model' in df.columns:
        report.append("### Sequence Model Comparison\n")
        seq_comparison = df.groupby('sequence_model').agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(seq_comparison.to_markdown())
        report.append("\n\n")
    
    # Scene encoders
    if 'scene_model' in df.columns:
        report.append("### Scene Encoder Comparison\n")
        scene_comparison = df.groupby(['scene_model', 'scene_finetune']).agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(scene_comparison.to_markdown())
        report.append("\n\n")
    
    # Classifiers
    if 'classifier' in df.columns:
        report.append("### Classifier Comparison\n")
        classifier_comparison = df.groupby('classifier').agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(classifier_comparison.to_markdown())
        report.append("\n\n")
    
    # Positional encoding
    if 'use_pos_encoding' in df.columns:
        report.append("### Positional Encoding Impact\n")
        pos_comparison = df.groupby('use_pos_encoding').agg({
            'test_f1': ['mean', 'std', 'count']
        }).round(4)
        report.append(pos_comparison.to_markdown())
        report.append("\n\n")
    
    # Full table
    report.append("## All Experiments\n\n")
    report.append(df.to_markdown(index=False))
    report.append("\n\n")
    
    # Write report
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    print(f"Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments-dir", type=str, default="./experiments",
                        help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default="./experiment_comparison.md",
                        help="Output path for comparison report")
    parser.add_argument("--csv", type=str, default="./experiment_comparison.csv",
                        help="Output path for CSV comparison")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading experiment results...")
    df = load_experiment_results(args.experiments_dir)
    
    if len(df) == 0:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Save CSV
    df.to_csv(args.csv, index=False)
    print(f"CSV saved to: {args.csv}")
    
    # Generate report
    generate_comparison_report(df, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nTest F1 Score:")
    print(f"  Mean: {df['test_f1'].mean():.4f}")
    print(f"  Std:  {df['test_f1'].std():.4f}")
    print(f"  Min:  {df['test_f1'].min():.4f}")
    print(f"  Max:  {df['test_f1'].max():.4f}")
    
    print(f"\nBest experiment: {df.loc[df['test_f1'].idxmax(), 'experiment']}")
    print(f"  Test F1: {df['test_f1'].max():.4f}")


if __name__ == "__main__":
    main()

