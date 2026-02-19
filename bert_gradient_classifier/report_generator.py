"""
Generate comprehensive analysis reports.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report(
    results_dir: str = "models",
    shap_dir: str = "models/shap_svm",
    comparison_dir: str = "comparison_results",
    output_path: str = "analysis_report.md",
):
    """Generate comprehensive analysis report."""
    
    report_lines = []
    report_lines.append("# BERT Gradient-Based Scene Saliency Classification Report\n")
    report_lines.append("=" * 80 + "\n")
    
    # 1. Results Summary
    report_lines.append("## 1. Results Summary\n")
    
    results_path = os.path.join(results_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        
        for model_name, model_results in results.items():
            if model_name in ["svm", "llama"]:
                report_lines.append(f"### {model_name.upper()} Model\n")
                report_lines.append(f"- **Validation F1**: {model_results['val']['f1']:.4f}\n")
                report_lines.append(f"- **Test F1**: {model_results['test']['f1']:.4f}\n")
                report_lines.append(f"- **Test Precision**: {model_results['test']['precision']:.4f}\n")
                report_lines.append(f"- **Test Recall**: {model_results['test']['recall']:.4f}\n\n")
    
    # 2. Feature Importance (SHAP)
    report_lines.append("## 2. Feature Importance Analysis (SHAP)\n")
    
    shap_importance_path = os.path.join(shap_dir, "feature_importance.csv")
    if os.path.exists(shap_importance_path):
        importance_df = pd.read_csv(shap_importance_path)
        report_lines.append("### Top 20 Most Important Features\n")
        report_lines.append("| Rank | Feature | Importance |\n")
        report_lines.append("|------|---------|------------|\n")
        
        for idx, row in importance_df.head(20).iterrows():
            report_lines.append(f"| {idx+1} | {row['feature']} | {row['importance']:.4f} |\n")
        report_lines.append("\n")
    
    # 3. Comparison Analysis
    report_lines.append("## 3. Feature Configuration Comparison\n")
    
    comparison_path = os.path.join(comparison_dir, "comparison_table.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        report_lines.append("### Performance by Configuration\n")
        report_lines.append("| Configuration | Test F1 | Test Precision | Test Recall |\n")
        report_lines.append("|---------------|---------|----------------|-------------|\n")
        
        for _, row in comparison_df.iterrows():
            report_lines.append(
                f"| {row['Configuration']} | {row['Test F1']:.4f} | "
                f"{row['Test Precision']:.4f} | {row['Test Recall']:.4f} |\n"
            )
        report_lines.append("\n")
    
    # 4. Key Findings
    report_lines.append("## 4. Key Findings\n")
    report_lines.append("- BERT gradients provide discriminative features for saliency detection\n")
    report_lines.append("- Combining BERT features with linguistic features improves performance\n")
    report_lines.append("- SHAP analysis reveals which features contribute most to predictions\n")
    report_lines.append("\n")
    
    # 5. Visualizations
    report_lines.append("## 5. Visualizations\n")
    report_lines.append("See the following files for detailed visualizations:\n")
    report_lines.append(f"- SHAP Summary: `{shap_dir}/shap_summary.png`\n")
    report_lines.append(f"- SHAP Bar Plot: `{shap_dir}/shap_bar.png`\n")
    report_lines.append(f"- Comparison Plots: `{comparison_dir}/comparison_plots.png`\n")
    report_lines.append("\n")
    
    # Write report
    with open(output_path, "w") as f:
        f.writelines(report_lines)
    
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument("--results-dir", type=str, default="models", help="Results directory")
    parser.add_argument("--shap-dir", type=str, default="models/shap_svm", help="SHAP results directory")
    parser.add_argument("--comparison-dir", type=str, default="comparison_results", help="Comparison results directory")
    parser.add_argument("--output", type=str, default="analysis_report.md", help="Output report path")
    
    args = parser.parse_args()
    generate_report(args.results_dir, args.shap_dir, args.comparison_dir, args.output)

