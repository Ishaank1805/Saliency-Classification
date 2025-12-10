"""
PCA Analysis utilities for MENSA scene saliency classification.

This module provides functions for:
- PCA-based dimensionality reduction
- Feature variance analysis
- Component loadings visualization
- Threshold optimization for macro F1
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


def fit_pca(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    n_components: int | float = 0.95,
    random_state: int = 0,
) -> Tuple[PCA, np.ndarray, np.ndarray | None]:
    """
    Fit PCA on training data and transform both train and validation sets.
    
    Args:
        X_train: Training feature matrix (samples x features)
        X_val: Optional validation feature matrix
        n_components: Number of components or variance ratio to preserve (default: 0.95)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (fitted PCA object, transformed train data, transformed val data)
    """
    # If n_components is a float between 0 and 1, it's interpreted as variance ratio
    # Otherwise use min of n_components and n_features
    if isinstance(n_components, float) and 0 < n_components < 1:
        pca = PCA(n_components=n_components, random_state=random_state)
    else:
        max_components = min(int(n_components), X_train.shape[1])
        pca = PCA(n_components=max_components, random_state=random_state)
    
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val) if X_val is not None else None
    
    return pca, X_train_pca, X_val_pca


def get_explained_variance_info(pca: PCA, target_variance: float = 0.95) -> dict:
    """
    Get information about explained variance from a fitted PCA.
    
    Args:
        pca: Fitted PCA object
        target_variance: Target cumulative variance ratio (default: 0.95)
        
    Returns:
        Dictionary with variance statistics
    """
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find number of components for target variance
    n_components_target = int(np.searchsorted(cumulative_var, target_variance) + 1)
    
    return {
        "explained_variance_ratio": explained_var,
        "cumulative_variance": cumulative_var,
        "n_components": len(explained_var),
        "n_components_for_target": n_components_target,
        "target_variance": target_variance,
        "variance_explained_first_10": explained_var[:10].tolist() if len(explained_var) >= 10 else explained_var.tolist(),
        "cumulative_variance_first_10": cumulative_var[:10].tolist() if len(cumulative_var) >= 10 else cumulative_var.tolist(),
    }


def get_component_loadings(
    pca: PCA,
    feature_names: List[str],
    n_components: int | None = None,
) -> pd.DataFrame:
    """
    Extract component loadings (feature contributions to each PC).
    
    Args:
        pca: Fitted PCA object
        feature_names: List of feature names
        n_components: Number of components to include (default: all)
        
    Returns:
        DataFrame with features as rows and components as columns
    """
    n_components = n_components or pca.n_components_
    n_components = min(n_components, pca.n_components_)
    
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    return loadings


def get_top_loadings(
    loadings: pd.DataFrame,
    n_top: int = 10,
    by_component: str | None = None,
) -> pd.DataFrame:
    """
    Get features with highest absolute loadings.
    
    Args:
        loadings: DataFrame from get_component_loadings
        n_top: Number of top features to return per component
        by_component: Specific component to analyze (e.g., "PC1"), or None for all
        
    Returns:
        DataFrame with top features and their loadings
    """
    if by_component:
        if by_component not in loadings.columns:
            raise ValueError(f"Component {by_component} not found in loadings")
        
        top = loadings[by_component].abs().sort_values(ascending=False).head(n_top)
        result = pd.DataFrame({
            "feature": top.index,
            "loading": loadings.loc[top.index, by_component].values,
            "abs_loading": top.values,
            "component": by_component,
        })
        return result.reset_index(drop=True)
    
    # Get top features for all components
    rows = []
    for pc in loadings.columns:
        top = loadings[pc].abs().sort_values(ascending=False).head(n_top)
        for feat in top.index:
            rows.append({
                "component": pc,
                "feature": feat,
                "loading": loadings.loc[feat, pc],
                "abs_loading": abs(loadings.loc[feat, pc]),
            })
    
    result = pd.DataFrame(rows)
    return result


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "macro",
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal probability threshold for classification by maximizing F1 score.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        metric: F1 averaging method ("macro", "binary", "weighted")
        n_thresholds: Number of thresholds to try
        
    Returns:
        Tuple of (optimal threshold, best F1 score)
    """
    # Try both uniform grid and actual probability values
    thresholds = np.unique(np.concatenate([
        np.linspace(0, 1, n_thresholds),
        y_proba
    ]))
    
    best_f1 = -1.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average=metric, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return float(best_threshold), float(best_f1)


def analyze_pca_classification(
    pca: PCA,
    X_train_pca: np.ndarray,
    X_val_pca: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_components_target: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive PCA analysis including variance, loadings, and classification performance.
    
    Args:
        pca: Fitted PCA object
        X_train_pca: PCA-transformed training data
        X_val_pca: PCA-transformed validation data
        y_train: Training labels
        y_val: Validation labels
        feature_names: Original feature names
        n_components_target: Number of components to use (default: all)
        verbose: Whether to print analysis results
        
    Returns:
        Dictionary with analysis results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    # Variance analysis
    variance_info = get_explained_variance_info(pca)
    
    # Use target components or all
    if n_components_target is None:
        n_components_target = variance_info["n_components_for_target"]
    n_components_target = min(n_components_target, X_train_pca.shape[1])
    
    # Train classifier on PCA features
    clf = LogisticRegression(max_iter=1000, class_weight=None, random_state=0)
    clf.fit(X_train_pca[:, :n_components_target], y_train)
    
    # Predictions
    y_pred = clf.predict(X_val_pca[:, :n_components_target])
    y_proba = clf.predict_proba(X_val_pca[:, :n_components_target])[:, 1]
    
    # Find optimal threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_val, y_proba, metric="macro")
    y_pred_opt = (y_proba >= opt_threshold).astype(int)
    
    # Component loadings
    loadings = get_component_loadings(pca, feature_names, n_components=10)
    top_loadings = get_top_loadings(loadings, n_top=10, by_component="PC1")
    
    results = {
        "variance_info": variance_info,
        "n_components_used": n_components_target,
        "classification_metrics": {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_binary": float(f1_score(y_val, y_pred, average="binary", zero_division=0)),
            "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
        },
        "optimal_threshold": {
            "threshold": opt_threshold,
            "f1_macro": opt_f1,
            "accuracy": float(accuracy_score(y_val, y_pred_opt)),
            "f1_binary": float(f1_score(y_val, y_pred_opt, average="binary", zero_division=0)),
        },
        "loadings": loadings,
        "top_features_pc1": top_loadings,
    }
    
    if verbose:
        print("\n" + "="*80)
        print("PCA ANALYSIS RESULTS")
        print("="*80)
        print(f"\nVariance explained by first 10 components:")
        for i, var in enumerate(variance_info["variance_explained_first_10"][:10], 1):
            cum_var = variance_info["cumulative_variance_first_10"][i-1]
            print(f"  PC{i}: {var:.4f} (cumulative: {cum_var:.4f})")
        
        print(f"\nComponents for 95% variance: {variance_info['n_components_for_target']}")
        print(f"Components used: {n_components_target}")
        
        print(f"\nClassification performance (default threshold=0.5):")
        metrics = results["classification_metrics"]
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  F1 (binary): {metrics['f1_binary']:.4f}")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        print(f"\nOptimal threshold analysis:")
        opt = results["optimal_threshold"]
        print(f"  Best threshold: {opt['threshold']:.4f}")
        print(f"  F1 (macro):  {opt['f1_macro']:.4f}")
        print(f"  F1 (binary): {opt['f1_binary']:.4f}")
        print(f"  Accuracy:    {opt['accuracy']:.4f}")
        
        print(f"\nTop 10 features for PC1 (by absolute loading):")
        for _, row in results["top_features_pc1"].head(10).iterrows():
            print(f"  {row['feature']}: {row['loading']:.4f}")
        
        print("="*80 + "\n")
    
    return results
