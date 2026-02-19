"""
Generate detailed analysis reports for all experiments.

Includes:
- Statistical significance tests
- Per-movie analysis
- Ablation study breakdowns
- Model comparison tables
- Visualization-ready data
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_results(experiments_dir: str) -> pd.DataFrame:
    """Load all experiment results with detailed metrics."""
    results = []
    
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, "r") as f:
                result = json.load(f)
            
            # Extract comprehensive metrics
            config = result.get('config', {})
            test_metrics = result.get('test_metrics', {})
            val_metrics = result.get('best_val_f1', 0)
            
            row = {
                'experiment': result.get('experiment', exp_dir.name),
                'description': config.get('description', 'N/A'),
                
                # Test metrics
                'test_f1_macro': test_metrics.get('macro_f1', 0),
                'test_precision_macro': test_metrics.get('macro_precision', 0),
                'test_recall_macro': test_metrics.get('macro_recall', 0),
                'test_accuracy': test_metrics.get('balanced_accuracy', 0),
                'test_f1_class_0': test_metrics.get('f1_class_0', 0),
                'test_f1_class_1': test_metrics.get('f1_class_1', 0),
                'test_precision_class_0': test_metrics.get('precision_class_0', 0),
                'test_precision_class_1': test_metrics.get('precision_class_1', 0),
                'test_recall_class_0': test_metrics.get('recall_class_0', 0),
                'test_recall_class_1': test_metrics.get('recall_class_1', 0),
                
                # Validation metrics
                'val_f1_macro': val_metrics,
                
                # Training info
                'best_epoch': result.get('best_epoch', -1),
                'training_time': result.get('training_time', 0),
                
                # Configuration
                'scene_model': config.get('scene_model_name', 'N/A'),
                'scene_finetune': config.get('scene_encoder_finetune', False),
                'use_linguistic': config.get('use_linguistic', False),
                'fusion_method': config.get('fusion_method', 'N/A'),
                'sequence_model': config.get('sequence_model_type', 'N/A'),
                'num_layers': config.get('num_transformer_layers', 0),
                'num_heads': config.get('num_heads', 0),
                'use_pos_encoding': config.get('use_positional_encoding', False),
                'classifier': config.get('classifier_type', 'N/A'),
                'hidden_dim': config.get('hidden_dim', 0),
            }
            
            results.append(row)
        except Exception as e:
            print(f"Error loading {exp_dir.name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df


def statistical_significance_test(df: pd.DataFrame, baseline: str, comparison: str) -> Dict:
    """Perform paired t-test between two experiments."""
    baseline_f1 = df[df['experiment'] == baseline]['test_f1_macro'].values
    comparison_f1 = df[df['experiment'] == comparison]['test_f1_macro'].values
    
    if len(baseline_f1) == 0 or len(comparison_f1) == 0:
        return None
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(comparison_f1, baseline_f1)
    
    # Effect size (Cohen's d)
    diff = comparison_f1 - baseline_f1
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    
    return {
        'baseline': baseline,
        'comparison': comparison,
        'baseline_mean': float(np.mean(baseline_f1)),
        'comparison_mean': float(np.mean(comparison_f1)),
        'improvement': float(np.mean(comparison_f1) - np.mean(baseline_f1)),
        'improvement_pct': float((np.mean(comparison_f1) - np.mean(baseline_f1)) / np.mean(baseline_f1) * 100),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
    }


def generate_ablation_analysis(df: pd.DataFrame, output_dir: str):
    """Generate detailed ablation study analysis."""
    
    report = []
    report.append("# Comprehensive Ablation Study Analysis\n\n")
    
    # 1. Overall best models
    report.append("## 1. Top Performing Models\n\n")
    top_models = df.nlargest(10, 'test_f1_macro')[
        ['experiment', 'description', 'test_f1_macro', 'test_precision_macro', 
         'test_recall_macro', 'scene_model', 'use_linguistic']
    ]
    report.append(top_models.to_markdown(index=False))
    report.append("\n\n")
    
    # 2. Fair comparison: RoBERTa-large with/without linguistic
    report.append("## 2. Fair Comparison: Linguistic Features Impact\n\n")
    roberta_large = df[df['scene_model'] == 'roberta-large'].copy()
    if len(roberta_large) > 0:
        comparison = roberta_large.groupby('use_linguistic').agg({
            'test_f1_macro': ['mean', 'std', 'count'],
            'test_precision_macro': 'mean',
            'test_recall_macro': 'mean',
        }).round(4)
        report.append(comparison.to_markdown())
        report.append("\n\n")
        
        # Statistical test
        baseline_exp = roberta_large[roberta_large['use_linguistic'] == False]['experiment'].iloc[0] if len(roberta_large[roberta_large['use_linguistic'] == False]) > 0 else None
        ling_exp = roberta_large[roberta_large['use_linguistic'] == True]['experiment'].iloc[0] if len(roberta_large[roberta_large['use_linguistic'] == True]) > 0 else None
        
        if baseline_exp and ling_exp:
            sig_test = statistical_significance_test(df, baseline_exp, ling_exp)
            if sig_test:
                report.append("### Statistical Significance Test\n\n")
                report.append(f"**Baseline**: {baseline_exp}\n")
                report.append(f"**With Linguistic**: {ling_exp}\n")
                report.append(f"**Improvement**: {sig_test['improvement']:.4f} ({sig_test['improvement_pct']:.2f}%)\n")
                report.append(f"**p-value**: {sig_test['p_value']:.4f}\n")
                report.append(f"**Significant**: {'Yes' if sig_test['significant'] else 'No'}\n")
                report.append(f"**Effect Size (Cohen's d)**: {sig_test['cohens_d']:.4f}\n\n")
    
    # 3. Model size comparison
    report.append("## 3. Model Size Comparison (with Linguistic Features)\n\n")
    with_ling = df[df['use_linguistic'] == True].copy()
    if len(with_ling) > 0:
        model_comparison = with_ling.groupby('scene_model').agg({
            'test_f1_macro': ['mean', 'std', 'count'],
            'test_precision_macro': 'mean',
            'test_recall_macro': 'mean',
            'training_time': 'mean',
        }).round(4)
        report.append(model_comparison.to_markdown())
        report.append("\n\n")
    
    # 4. Finetuning impact
    report.append("## 4. Finetuning Impact\n\n")
    finetune_comparison = df.groupby(['scene_model', 'scene_finetune']).agg({
        'test_f1_macro': ['mean', 'std'],
    }).round(4)
    report.append(finetune_comparison.to_markdown())
    report.append("\n\n")
    
    # 5. Fusion method comparison
    report.append("## 5. Fusion Method Comparison\n\n")
    fusion_comparison = df[df['use_linguistic'] == True].groupby('fusion_method').agg({
        'test_f1_macro': ['mean', 'std', 'count'],
    }).round(4)
    report.append(fusion_comparison.to_markdown())
    report.append("\n\n")
    
    # 6. Sequence model impact
    report.append("## 6. Sequence Model Impact\n\n")
    seq_comparison = df.groupby('sequence_model').agg({
        'test_f1_macro': ['mean', 'std', 'count'],
    }).round(4)
    report.append(seq_comparison.to_markdown())
    report.append("\n\n")
    
    # 7. Positional encoding impact
    report.append("## 7. Positional Encoding Impact\n\n")
    pos_comparison = df.groupby('use_pos_encoding').agg({
        'test_f1_macro': ['mean', 'std', 'count'],
    }).round(4)
    report.append(pos_comparison.to_markdown())
    report.append("\n\n")
    
    # 8. Classifier comparison
    report.append("## 8. Classifier Comparison\n\n")
    classifier_comparison = df.groupby('classifier').agg({
        'test_f1_macro': ['mean', 'std', 'count'],
    }).round(4)
    report.append(classifier_comparison.to_markdown())
    report.append("\n\n")
    
    # 9. Transformer architecture
    report.append("## 9. Transformer Architecture Analysis\n\n")
    
    # Layers
    layers_comparison = df[df['num_layers'] > 0].groupby('num_layers').agg({
        'test_f1_macro': ['mean', 'std'],
    }).round(4)
    report.append("### Number of Layers\n\n")
    report.append(layers_comparison.to_markdown())
    report.append("\n\n")
    
    # Heads
    heads_comparison = df[df['num_heads'] > 0].groupby('num_heads').agg({
        'test_f1_macro': ['mean', 'std'],
    }).round(4)
    report.append("### Number of Attention Heads\n\n")
    report.append(heads_comparison.to_markdown())
    report.append("\n\n")
    
    # 10. Complete experiment table
    report.append("## 10. Complete Experiment Results\n\n")
    full_table = df[[
        'experiment', 'description', 'test_f1_macro', 'test_precision_macro',
        'test_recall_macro', 'test_accuracy', 'scene_model', 'scene_finetune',
        'use_linguistic', 'fusion_method', 'sequence_model', 'classifier',
        'training_time'
    ]].sort_values('test_f1_macro', ascending=False)
    report.append(full_table.to_markdown(index=False))
    report.append("\n\n")
    
    # Write report
    report_path = os.path.join(output_dir, "detailed_ablation_analysis.md")
    with open(report_path, "w") as f:
        f.write("".join(report))
    
    print(f"Detailed analysis saved to: {report_path}")


def generate_comparison_tables(df: pd.DataFrame, output_dir: str):
    """Generate comparison tables for paper."""
    
    # Table 1: Fair comparison (RoBERTa-large)
    roberta_large = df[df['scene_model'] == 'roberta-large'].copy()
    if len(roberta_large) > 0:
        table1 = roberta_large[[
            'experiment', 'use_linguistic', 'fusion_method', 'scene_finetune',
            'test_f1_macro', 'test_precision_macro', 'test_recall_macro'
        ]].sort_values('test_f1_macro', ascending=False)
        
        table1_path = os.path.join(output_dir, "table_roberta_large_comparison.csv")
        table1.to_csv(table1_path, index=False)
        print(f"RoBERTa-large comparison table saved to: {table1_path}")
    
    # Table 2: Model size comparison
    with_ling = df[df['use_linguistic'] == True].copy()
    if len(with_ling) > 0:
        table2 = with_ling.groupby('scene_model').agg({
            'test_f1_macro': ['mean', 'std'],
            'test_precision_macro': 'mean',
            'test_recall_macro': 'mean',
        }).round(4)
        
        table2_path = os.path.join(output_dir, "table_model_size_comparison.csv")
        table2.to_csv(table2_path)
        print(f"Model size comparison table saved to: {table2_path}")
    
    # Table 3: Ablation summary
    ablation_summary = []
    
    # Linguistic features impact
    baseline = df[df['experiment'] == 'baseline_saxena']['test_f1_macro'].iloc[0] if len(df[df['experiment'] == 'baseline_saxena']) > 0 else None
    with_ling_best = df[df['use_linguistic'] == True]['test_f1_macro'].max() if len(df[df['use_linguistic'] == True]) > 0 else None
    
    if baseline and with_ling_best:
        ablation_summary.append({
            'Ablation': 'Linguistic Features',
            'Baseline': baseline,
            'With Feature': with_ling_best,
            'Improvement': with_ling_best - baseline,
            'Improvement %': (with_ling_best - baseline) / baseline * 100
        })
    
    # Finetuning impact
    frozen = df[(df['scene_model'] == 'roberta-large') & (df['scene_finetune'] == False) & (df['use_linguistic'] == True)]['test_f1_macro'].mean()
    finetuned = df[(df['scene_model'] == 'roberta-large') & (df['scene_finetune'] == True) & (df['use_linguistic'] == True)]['test_f1_macro'].mean()
    
    if not np.isnan(frozen) and not np.isnan(finetuned):
        ablation_summary.append({
            'Ablation': 'Finetuning',
            'Baseline': frozen,
            'With Feature': finetuned,
            'Improvement': finetuned - frozen,
            'Improvement %': (finetuned - frozen) / frozen * 100
        })
    
    if ablation_summary:
        table3 = pd.DataFrame(ablation_summary)
        table3_path = os.path.join(output_dir, "table_ablation_summary.csv")
        table3.to_csv(table3_path, index=False)
        print(f"Ablation summary table saved to: {table3_path}")


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate visualization plots."""
    
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # 1. Model size comparison
    with_ling = df[df['use_linguistic'] == True].copy()
    if len(with_ling) > 0:
        plt.figure(figsize=(10, 6))
        model_order = ['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large']
        model_data = []
        model_labels = []
        for model in model_order:
            model_df = with_ling[with_ling['scene_model'] == model]
            if len(model_df) > 0:
                model_data.append(model_df['test_f1_macro'].values)
                model_labels.append(model.replace('-uncased', '').replace('-', ' ').title())
        
        if model_data:
            plt.boxplot(model_data, labels=model_labels)
            plt.ylabel('Test F1 Score (Macro)')
            plt.title('Model Size Comparison (with Linguistic Features)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "model_size_comparison.png"), dpi=300)
            plt.close()
    
    # 2. Linguistic features impact
    roberta_large = df[df['scene_model'] == 'roberta-large'].copy()
    if len(roberta_large) > 0:
        plt.figure(figsize=(8, 6))
        ling_data = []
        ling_labels = []
        for use_ling in [False, True]:
            subset = roberta_large[roberta_large['use_linguistic'] == use_ling]
            if len(subset) > 0:
                ling_data.append(subset['test_f1_macro'].values)
                ling_labels.append('Without Linguistic' if not use_ling else 'With Linguistic')
        
        if len(ling_data) == 2:
            plt.boxplot(ling_data, labels=ling_labels)
            plt.ylabel('Test F1 Score (Macro)')
            plt.title('Linguistic Features Impact (RoBERTa-large)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "linguistic_impact.png"), dpi=300)
            plt.close()
    
    # 3. Finetuning impact
    finetune_data = []
    finetune_labels = []
    for finetune in [False, True]:
        subset = df[(df['scene_model'] == 'roberta-large') & (df['use_linguistic'] == True) & (df['scene_finetune'] == finetune)]
        if len(subset) > 0:
            finetune_data.append(subset['test_f1_macro'].values)
            finetune_labels.append('Frozen' if not finetune else 'Finetuned')
    
    if len(finetune_data) == 2:
        plt.figure(figsize=(8, 6))
        plt.boxplot(finetune_data, labels=finetune_labels)
        plt.ylabel('Test F1 Score (Macro)')
        plt.title('Finetuning Impact (RoBERTa-large + Linguistic)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "finetuning_impact.png"), dpi=300)
        plt.close()
    
    print(f"Visualizations saved to: {os.path.join(output_dir, 'plots')}")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed analysis reports")
    parser.add_argument("--experiments-dir", type=str, default="./experiments",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="./analysis",
                        help="Output directory for analysis reports")
    parser.add_argument("--generate-plots", action="store_true",
                        help="Generate visualization plots")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    df = load_all_results(args.experiments_dir)
    
    if len(df) == 0:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Save full CSV
    csv_path = os.path.join(args.output_dir, "all_experiments.csv")
    df.to_csv(csv_path, index=False)
    print(f"Full results CSV saved to: {csv_path}")
    
    # Generate detailed analysis
    print("\nGenerating detailed analysis...")
    generate_ablation_analysis(df, args.output_dir)
    
    # Generate comparison tables
    print("\nGenerating comparison tables...")
    generate_comparison_tables(df, args.output_dir)
    
    # Generate visualizations
    if args.generate_plots:
        print("\nGenerating visualizations...")
        generate_visualizations(df, args.output_dir)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nTest F1 Score (Macro):")
    print(f"  Mean: {df['test_f1_macro'].mean():.4f}")
    print(f"  Std:  {df['test_f1_macro'].std():.4f}")
    print(f"  Min:  {df['test_f1_macro'].min():.4f}")
    print(f"  Max:  {df['test_f1_macro'].max():.4f}")
    
    best_exp = df.loc[df['test_f1_macro'].idxmax()]
    print(f"\nBest experiment: {best_exp['experiment']}")
    print(f"  Test F1: {best_exp['test_f1_macro']:.4f}")
    print(f"  Description: {best_exp['description']}")
    
    # Linguistic features impact
    baseline = df[df['experiment'] == 'baseline_saxena']
    with_ling = df[(df['scene_model'] == 'roberta-large') & (df['use_linguistic'] == True)]
    
    if len(baseline) > 0 and len(with_ling) > 0:
        baseline_f1 = baseline['test_f1_macro'].iloc[0]
        best_ling_f1 = with_ling['test_f1_macro'].max()
        improvement = best_ling_f1 - baseline_f1
        improvement_pct = (improvement / baseline_f1) * 100
        
        print(f"\nLinguistic Features Impact (RoBERTa-large):")
        print(f"  Baseline (no linguistic): {baseline_f1:.4f}")
        print(f"  Best with linguistic: {best_ling_f1:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")


if __name__ == "__main__":
    main()

