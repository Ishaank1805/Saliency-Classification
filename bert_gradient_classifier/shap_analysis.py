"""
SHAP Analysis: Feature importance and model interpretability.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pickle
import os


class SHAPAnalyzer:
    """SHAP analysis for model interpretability."""
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        model_type: str = "svm",  # "svm" or "llama"
    ):
        """
        Args:
            model: Trained model (SVM or Llama)
            feature_names: Names of features
            model_type: Type of model ("svm" or "llama")
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(
        self,
        X_background: np.ndarray,
        X_sample: Optional[np.ndarray] = None,
        max_evals: int = 100,
    ):
        """
        Create SHAP explainer.
        
        Args:
            X_background: Background dataset for SHAP
            X_sample: Sample dataset to explain (if None, uses X_background)
            max_evals: Maximum evaluations for KernelExplainer
        """
        if X_sample is None:
            X_sample = X_background[:min(100, len(X_background))]  # Sample for efficiency
        
        if self.model_type == "svm":
            # For SVM, use KernelExplainer
            def model_predict(X):
                return self.model.predict_proba(X)[:, 1]
            
            self.explainer = shap.KernelExplainer(
                model_predict,
                X_background[:min(100, len(X_background))],  # Background sample
            )
            
        elif self.model_type == "llama":
            # For Llama, we'd need a different approach
            # This is a placeholder - Llama SHAP requires text-based explainers
            print("Warning: Llama SHAP analysis requires text-based explainers.")
            print("Consider using SHAP's TextExplainer or integrated gradients.")
            return
        
        print(f"Created SHAP explainer for {self.model_type} model")
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        max_evals: int = 100,
    ) -> np.ndarray:
        """
        Compute SHAP values.
        
        Args:
            X: Input features to explain
            max_evals: Maximum evaluations
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(
            X,
            nsamples=max_evals,
        )
        
        return self.shap_values
    
    def plot_summary(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
    ):
        """Plot SHAP summary plot."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names[:len(self.shap_values[0])],
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved summary plot to {save_path}")
        else:
            plt.show()
    
    def plot_bar(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
    ):
        """Plot SHAP bar plot (mean absolute SHAP values)."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names[:len(self.shap_values[0])],
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved bar plot to {save_path}")
        else:
            plt.show()
    
    def get_feature_importance(
        self,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap,
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df.head(top_k)
    
    def plot_waterfall(
        self,
        instance_idx: int,
        X: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot waterfall plot for a single instance."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=X[instance_idx],
                feature_names=self.feature_names[:len(self.shap_values[instance_idx])],
            ),
            show=False,
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved waterfall plot to {save_path}")
        else:
            plt.show()
    
    def generate_report(
        self,
        X: np.ndarray,
        output_dir: str = "shap_reports",
        top_k: int = 20,
    ):
        """Generate comprehensive SHAP analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute SHAP values if not already computed
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Feature importance
        importance_df = self.get_feature_importance(top_k=top_k)
        importance_df.to_csv(
            os.path.join(output_dir, "feature_importance.csv"),
            index=False,
        )
        print(f"Saved feature importance to {output_dir}/feature_importance.csv")
        
        # Plots
        self.plot_summary(
            max_display=top_k,
            save_path=os.path.join(output_dir, "shap_summary.png"),
        )
        self.plot_bar(
            max_display=top_k,
            save_path=os.path.join(output_dir, "shap_bar.png"),
        )
        
        # Waterfall plots for a few examples
        for i in range(min(5, len(X))):
            self.plot_waterfall(
                i,
                X,
                save_path=os.path.join(output_dir, f"waterfall_{i}.png"),
            )
        
        print(f"SHAP analysis report generated in {output_dir}/")


def analyze_model_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_type: str = "svm",
    output_dir: str = "shap_reports",
):
    """
    Convenience function to analyze a model with SHAP.
    
    Args:
        model: Trained model
        X_train: Training features (for background)
        X_test: Test features (to explain)
        feature_names: Feature names
        model_type: Model type ("svm" or "llama")
        output_dir: Output directory for reports
    """
    analyzer = SHAPAnalyzer(model, feature_names, model_type)
    analyzer.create_explainer(X_train)
    analyzer.compute_shap_values(X_test[:100])  # Sample for efficiency
    analyzer.generate_report(X_test[:100], output_dir)

