"""
Ensemble methods for MENSA scene saliency classification with PCA support.

This module provides ensemble learning utilities that combine multiple models
including both PCA-transformed and original feature spaces:
- Weighted soft voting ensemble
- Dirichlet-based weight search
- Stacking meta-learner with out-of-fold predictions
- Multiple model types (LR, SVM, RF, HGB, ExtraTrees, LightGBM)
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def create_base_models(
    random_state: int = 0,
    include_lgbm: bool = True,
    pos_weight_ratio: float | None = None,
) -> List[Tuple[str, Any]]:
    """
    Create a list of diverse base models for ensemble.
    
    Args:
        random_state: Random seed for reproducibility
        include_lgbm: Whether to include LightGBM (if available)
        pos_weight_ratio: Ratio for LightGBM scale_pos_weight (neg/pos counts)
        
    Returns:
        List of (name, model) tuples
    """
    models = [
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.3,
            random_state=random_state
        )),
        ("svc", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0, random_state=random_state),
            method="sigmoid",
            cv=2
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        )),
        ("hgb", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_leaf_nodes=31,
            max_iter=250,
            random_state=random_state
        )),
        ("extratrees", ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        )),
    ]
    
    if include_lgbm and HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight_ratio or 1.0,
            random_state=random_state,
            verbose=-1,
        )
        models.append(("lgbm", lgbm))
    
    return models


def weighted_ensemble_predict(
    model_probs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Make ensemble predictions using weighted soft voting.
    
    Args:
        model_probs: Array of shape (n_models, n_samples) with model probabilities
        weights: Array of shape (n_models,) with model weights (should sum to 1)
        
    Returns:
        Ensemble probabilities of shape (n_samples,)
    """
    return weights @ model_probs


def search_ensemble_weights_simplex(
    model_probs: np.ndarray,
    y_true: np.ndarray,
    threshold_finder_func,
    step: float = 0.1,
    metric: str = "macro",
) -> Dict[str, Any]:
    """
    Search for optimal ensemble weights using simplex grid search.
    
    Args:
        model_probs: Array of shape (n_models, n_samples)
        y_true: True labels
        threshold_finder_func: Function to find optimal threshold
        step: Grid step size (default: 0.1)
        metric: F1 averaging metric
        
    Returns:
        Dictionary with optimal weights, threshold, and scores
    """
    n_models = model_probs.shape[0]
    
    if n_models == 2:
        # Binary case: just vary w0, w1 = 1 - w0
        best = {"weights": None, "threshold": 0.5, "f1_macro": -1.0, "roc_auc": -1.0}
        for w0 in np.arange(0.0, 1.0 + step/2, step):
            w1 = 1.0 - w0
            weights = np.array([w0, w1])
            ens_probs = weighted_ensemble_predict(model_probs, weights)
            threshold, f1 = threshold_finder_func(y_true, ens_probs, metric=metric)
            auc = roc_auc_score(y_true, ens_probs)
            if f1 > best["f1_macro"] or (np.isclose(f1, best["f1_macro"]) and auc > best["roc_auc"]):
                best = {
                    "weights": weights,
                    "threshold": float(threshold),
                    "f1_macro": float(f1),
                    "roc_auc": float(auc),
                }
        return best
    
    elif n_models == 3:
        # Ternary case
        best = {"weights": None, "threshold": 0.5, "f1_macro": -1.0, "roc_auc": -1.0}
        for w0 in np.arange(0.0, 1.0 + step/2, step):
            for w1 in np.arange(0.0, 1.0 - w0 + step/2, step):
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                weights = np.array([w0, w1, max(0.0, w2)])
                if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                    continue
                ens_probs = weighted_ensemble_predict(model_probs, weights)
                threshold, f1 = threshold_finder_func(y_true, ens_probs, metric=metric)
                auc = roc_auc_score(y_true, ens_probs)
                if f1 > best["f1_macro"] or (np.isclose(f1, best["f1_macro"]) and auc > best["roc_auc"]):
                    best = {
                        "weights": weights,
                        "threshold": float(threshold),
                        "f1_macro": float(f1),
                        "roc_auc": float(auc),
                    }
        return best
    
    elif n_models == 4:
        # Quaternary case (from original code)
        best = {"weights": None, "threshold": 0.5, "f1_macro": -1.0, "roc_auc": -1.0}
        for w0 in np.arange(0.0, 1.0 + step/2, step):
            for w1 in np.arange(0.0, 1.0 - w0 + step/2, step):
                for w2 in np.arange(0.0, 1.0 - w0 - w1 + step/2, step):
                    w3 = 1.0 - (w0 + w1 + w2)
                    if w3 < -1e-9:
                        continue
                    weights = np.array([w0, w1, w2, max(0.0, w3)])
                    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                        continue
                    ens_probs = weighted_ensemble_predict(model_probs, weights)
                    threshold, f1 = threshold_finder_func(y_true, ens_probs, metric=metric)
                    auc = roc_auc_score(y_true, ens_probs)
                    if f1 > best["f1_macro"] or (np.isclose(f1, best["f1_macro"]) and auc > best["roc_auc"]):
                        best = {
                            "weights": weights,
                            "threshold": float(threshold),
                            "f1_macro": float(f1),
                            "roc_auc": float(auc),
                        }
        return best
    
    else:
        # Fall back to Dirichlet search for >4 models
        return search_ensemble_weights_dirichlet(
            model_probs, y_true, threshold_finder_func,
            n_trials=800, metric=metric, random_state=0
        )


def search_ensemble_weights_dirichlet(
    model_probs: np.ndarray,
    y_true: np.ndarray,
    threshold_finder_func,
    n_trials: int = 800,
    metric: str = "macro",
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Search for optimal ensemble weights using Dirichlet distribution sampling.
    
    Args:
        model_probs: Array of shape (n_models, n_samples)
        y_true: True labels
        threshold_finder_func: Function to find optimal threshold
        n_trials: Number of random weight combinations to try
        metric: F1 averaging metric
        random_state: Random seed
        
    Returns:
        Dictionary with optimal weights, threshold, and scores
    """
    n_models = model_probs.shape[0]
    rng = np.random.default_rng(random_state)
    
    best = {"weights": None, "threshold": 0.5, "f1_macro": -1.0, "roc_auc": -1.0}
    
    for _ in range(n_trials):
        weights = rng.dirichlet(np.ones(n_models))
        ens_probs = weighted_ensemble_predict(model_probs, weights)
        threshold, f1 = threshold_finder_func(y_true, ens_probs, metric=metric)
        auc = roc_auc_score(y_true, ens_probs)
        
        if f1 > best["f1_macro"] or (np.isclose(f1, best["f1_macro"]) and auc > best["roc_auc"]):
            best = {
                "weights": weights,
                "threshold": float(threshold),
                "f1_macro": float(f1),
                "roc_auc": float(auc),
            }
    
    return best


def train_pca_ensemble(
    X_train_std: np.ndarray,
    X_train_pca: np.ndarray,
    y_train: np.ndarray,
    X_val_std: np.ndarray,
    X_val_pca: np.ndarray,
    y_val: np.ndarray,
    threshold_finder_func,
    n_pca_components: int | None = None,
    random_state: int = 0,
    search_method: str = "dirichlet",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train ensemble combining models on both standard and PCA features.
    
    Args:
        X_train_std: Standardized training features
        X_train_pca: PCA-transformed training features
        y_train: Training labels
        X_val_std: Standardized validation features
        X_val_pca: PCA-transformed validation features
        y_val: Validation labels
        threshold_finder_func: Function to find optimal threshold (e.g., from pca_utils)
        n_pca_components: Number of PCA components to use (default: all)
        random_state: Random seed
        search_method: "dirichlet" or "simplex"
        verbose: Print progress
        
    Returns:
        Dictionary with ensemble results including models, weights, and metrics
    """
    if verbose:
        print("\n" + "="*80)
        print("TRAINING PCA ENSEMBLE")
        print("="*80)
    
    # Prepare PCA features
    if n_pca_components is not None:
        X_train_pca = X_train_pca[:, :n_pca_components]
        X_val_pca = X_val_pca[:, :n_pca_components]
    
    # Calculate class weights for LightGBM
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1.0, pos_count)
    
    # Create models
    if verbose:
        print(f"\nTraining {4 if not HAS_LGBM else 5} base models on standard features...")
    models_std = create_base_models(random_state, include_lgbm=True, pos_weight_ratio=pos_weight)
    
    if verbose:
        print(f"Training {4 if not HAS_LGBM else 5} base models on PCA features ({X_train_pca.shape[1]} components)...")
    models_pca = create_base_models(random_state, include_lgbm=True, pos_weight_ratio=pos_weight)
    
    # Train models
    probs_dict = {}
    model_dict = {}
    
    for name, model in models_std:
        model.fit(X_train_std, y_train)
        probs_dict[f"{name}_std"] = model.predict_proba(X_val_std)[:, 1]
        model_dict[f"{name}_std"] = model
    
    for name, model in models_pca:
        model.fit(X_train_pca, y_train)
        probs_dict[f"{name}_pca"] = model.predict_proba(X_val_pca)[:, 1]
        model_dict[f"{name}_pca"] = model
    
    # Stack probabilities
    model_names = list(probs_dict.keys())
    model_probs = np.vstack([probs_dict[name] for name in model_names])
    
    if verbose:
        print(f"\nSearching optimal weights for {len(model_names)} models using {search_method}...")
    
    # Search for optimal weights
    if search_method == "simplex" and len(model_names) <= 4:
        results = search_ensemble_weights_simplex(
            model_probs, y_val, threshold_finder_func,
            step=0.1, metric="macro"
        )
    else:
        results = search_ensemble_weights_dirichlet(
            model_probs, y_val, threshold_finder_func,
            n_trials=800, metric="macro", random_state=random_state
        )
    
    results["model_names"] = model_names
    results["models"] = model_dict
    results["n_models"] = len(model_names)
    
    # Compute validation predictions
    ens_probs = weighted_ensemble_predict(model_probs, results['weights'])
    y_pred = (ens_probs >= results['threshold']).astype(int)
    
    # Add validation metrics
    from sklearn.metrics import accuracy_score
    results["y_proba"] = ens_probs
    results["y_pred"] = y_pred
    results["accuracy"] = float(accuracy_score(y_val, y_pred))
    results["f1_binary"] = float(f1_score(y_val, y_pred, average='binary'))
    # f1_macro already exists from the search
    
    if verbose:
        print(f"\n{'='*80}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*80}")
        print(f"Models: {model_names}")
        print(f"Optimal weights: {np.round(results['weights'], 3).tolist()}")
        print(f"Optimal threshold: {results['threshold']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 (binary): {results['f1_binary']:.4f}")
        print(f"F1 (macro): {results['f1_macro']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"{'='*80}\n")
    
    return results


def train_stacking_ensemble(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    X_val_std: np.ndarray,
    y_val: np.ndarray,
    threshold_finder_func,
    n_folds: int = 5,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train stacking meta-learner using out-of-fold predictions.
    
    Args:
        X_train_std: Standardized training features
        y_train: Training labels
        X_val_std: Standardized validation features
        y_val: Validation labels
        threshold_finder_func: Function to find optimal threshold
        n_folds: Number of cross-validation folds for OOF predictions
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with stacking results
    """
    if verbose:
        print("\n" + "="*80)
        print("TRAINING STACKING ENSEMBLE")
        print("="*80)
    
    # Calculate class weights
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / max(1.0, pos_count)
    
    # Create base models
    base_models = create_base_models(random_state, include_lgbm=True, pos_weight_ratio=pos_weight)
    
    if verbose:
        print(f"\nGenerating out-of-fold predictions with {n_folds} folds...")
    
    # Generate OOF predictions
    oof_preds = np.zeros((len(y_train), len(base_models)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for model_idx, (name, model) in enumerate(base_models):
        if verbose:
            print(f"  {name}...", end=" ", flush=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_std, y_train)):
            model_clone = clone(model)
            model_clone.fit(X_train_std[train_idx], y_train[train_idx])
            
            if hasattr(model_clone, "predict_proba"):
                oof_preds[val_idx, model_idx] = model_clone.predict_proba(X_train_std[val_idx])[:, 1]
            else:
                # For models without predict_proba, use decision_function and scale
                scores = model_clone.decision_function(X_train_std[val_idx])
                scores = (scores - scores.min()) / (max(1e-9, scores.max() - scores.min()))
                oof_preds[val_idx, model_idx] = scores
        
        if verbose:
            print("✓")
    
    # Train meta-learner
    if verbose:
        print("\nTraining meta-learner (Logistic Regression)...")
    
    meta_model = LogisticRegression(max_iter=1000, class_weight=None, random_state=random_state)
    meta_model.fit(oof_preds, y_train)
    
    # Generate validation meta-features
    val_meta_features = np.zeros((len(y_val), len(base_models)))
    
    for model_idx, (name, model) in enumerate(base_models):
        # Refit on full training data
        model_clone = clone(model)
        model_clone.fit(X_train_std, y_train)
        
        if hasattr(model_clone, "predict_proba"):
            val_meta_features[:, model_idx] = model_clone.predict_proba(X_val_std)[:, 1]
        else:
            scores = model_clone.decision_function(X_val_std)
            scores = (scores - scores.min()) / (max(1e-9, scores.max() - scores.min()))
            val_meta_features[:, model_idx] = scores
    
    # Meta-model predictions
    ens_probs = meta_model.predict_proba(val_meta_features)[:, 1]
    threshold, f1 = threshold_finder_func(y_val, ens_probs, metric="macro")
    auc = roc_auc_score(y_val, ens_probs)
    y_pred = (ens_probs >= threshold).astype(int)
    
    results = {
        "meta_model": meta_model,
        "base_models": base_models,
        "threshold": float(threshold),
        "f1_macro": float(f1),
        "f1_binary": float(f1_score(y_val, y_pred, average='binary')),
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "y_pred": y_pred,
        "y_proba": ens_probs,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("STACKING RESULTS")
        print(f"{'='*80}")
        print(f"Base models: {[name for name, _ in base_models]}")
        print(f"Optimal threshold: {threshold:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 (binary): {results['f1_binary']:.4f}")
        print(f"F1 (macro): {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"{'='*80}\n")
    
    return results
