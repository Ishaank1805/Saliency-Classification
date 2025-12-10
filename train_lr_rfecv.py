#!/usr/bin/env python3
"""
Train Logistic Regression with RFECV (Recursive Feature Elimination with Cross-Validation)
for MENSA scene saliency classification.

Uses same data loading and balancing as train_pca_ensemble.py but with RFECV feature selection.
"""

from __future__ import annotations

import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

from feature_cache import load_feature_matrix
from pca_utils import find_optimal_threshold

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("⚠ Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")


def apply_custom_cluster_undersampling(X_train, y_train, n_clusters=50, random_state=42):
    """
    Custom cluster-based undersampling matching train.py logic.
    Clusters majority class and samples proportionally from each cluster.
    """
    print(f"\nApplying custom cluster-based undersampling (cluster_random method):")
    print(f"  Strategy: K-Means clustering ({n_clusters} clusters) + random sampling within clusters")
    
    num_pos = int((y_train == 1).sum())
    num_neg = int((y_train == 0).sum())
    
    print(f"  Original distribution: {num_neg} non-salient, {num_pos} salient")
    
    if num_pos == 0 or num_neg == 0:
        return X_train, y_train
    
    # Determine minority and majority
    if num_pos <= num_neg:
        minority_label = 1
        majority_label = 0
        target_majority = num_pos
    else:
        minority_label = 0
        majority_label = 1
        target_majority = num_neg
    
    maj_idx = np.where(y_train == majority_label)[0]
    min_idx = np.where(y_train == minority_label)[0]
    X_maj = X_train[maj_idx]
    
    print(f"  Target: Reduce majority class from {len(maj_idx)} to {target_majority} (match minority)")
    
    # Cluster majority class
    k = int(max(1, min(n_clusters, len(maj_idx), target_majority)))
    print(f"\n  Step 1: K-Means clustering of {len(maj_idx)} majority scenes into {k} clusters...")
    
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_maj)
        print(f"  ✓ Clustering complete")
    except Exception:
        labels = np.zeros(len(maj_idx), dtype=int)
        k = 1
        print(f"  ⚠ Clustering failed, using single cluster")
    
    # Compute sampling quota per cluster
    cluster_sizes = np.bincount(labels, minlength=k).astype(np.int64)
    total_maj = int(len(maj_idx))
    
    print(f"\n  Step 2: Compute proportional sampling quota for each cluster...")
    print(f"    Cluster sizes (min/mean/max): {cluster_sizes.min()}/{cluster_sizes.mean():.0f}/{cluster_sizes.max()}")
    
    if total_maj > 0:
        raw_quota = (cluster_sizes / max(1, total_maj)) * float(target_majority)
        base_quota = np.floor(raw_quota).astype(int)
        remainder = int(target_majority - int(base_quota.sum()))
        frac = raw_quota - base_quota
        order = np.argsort(-frac)
        for i in range(min(remainder, len(order))):
            base_quota[order[i]] += 1
        
        print(f"    Sampling quotas per cluster (min/mean/max): {base_quota.min()}/{base_quota.mean():.0f}/{base_quota.max()}")
        
        # Sample from each cluster
        print(f"\n  Step 3: Random sampling within each cluster...")
        chosen = []
        rng = np.random.RandomState(random_state)
        for c in range(k):
            members = np.where(labels == c)[0]
            take = int(min(base_quota[c], len(members)))
            if take > 0:
                chosen_members = rng.choice(members, size=take, replace=False)
                chosen.append(chosen_members)
        
        if len(chosen):
            chosen_idx_rel = np.concatenate(chosen, axis=0)
            chosen_idx_abs = maj_idx[chosen_idx_rel]
            keep_idx = np.concatenate([min_idx, chosen_idx_abs], axis=0)
            X_train = X_train[keep_idx]
            y_train = y_train[keep_idx]
            
            final_neg = (y_train == 0).sum()
            final_pos = (y_train == 1).sum()
            print(f"  ✓ Sampling complete: {final_neg} non-salient, {final_pos} salient")
            print(f"    Reduction: {len(maj_idx)} → {final_neg} ({100*final_neg/len(maj_idx):.1f}%)")
    
    return X_train, y_train


def apply_sampling(X_train, y_train, args):
    """
    Apply data balancing matching train.py exactly.
    
    For cluster_random: Undersample first (custom), then SMOTE
    For other methods: SMOTE first, then undersample (via pipeline behavior)
    """
    if not HAS_IMBLEARN:
        print("\n⚠ Cannot apply sampling: imbalanced-learn not installed")
        return X_train, y_train
    
    original_counts = np.bincount(y_train.astype(int))
    print(f"\nOriginal class distribution: {original_counts}")
    
    # Special handling for cluster_random (matches train.py)
    if args.undersample and args.undersample_method == "cluster_random":
        X_train, y_train = apply_custom_cluster_undersampling(
            X_train, y_train,
            n_clusters=args.undersample_clusters,
            random_state=args.random_state
        )
        after_under = np.bincount(y_train.astype(int))
        print(f"\n✓ After cluster_random undersampling: {after_under}")
        
        # Then optionally oversample
        if args.oversample:
            print(f"\nApplying SMOTE oversampling...")
            smote = SMOTE(random_state=args.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            after_over = np.bincount(y_train.astype(int))
            print(f"✓ After SMOTE: {after_over}")
    
    # For other methods: SMOTE first, then undersample (matches train.py pipeline)
    else:
        # Oversample first (if requested)
        if args.oversample:
            print(f"\nApplying SMOTE oversampling...")
            smote = SMOTE(random_state=args.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            after_over = np.bincount(y_train.astype(int))
            print(f"✓ After SMOTE: {after_over}")
        
        # Undersample second (if requested)
        if args.undersample:
            method_desc = {
                'random': 'Random undersampling (RandomUnderSampler)',
                'cluster': 'Cluster centroids (ClusterCentroids via K-Means)'
            }
            print(f"\nApplying undersampling (method: {args.undersample_method})...")
            print(f"  {method_desc.get(args.undersample_method, args.undersample_method)}")
            
            if args.undersample_method == "random":
                undersampler = RandomUnderSampler(random_state=args.random_state)
            elif args.undersample_method == "cluster":
                undersampler = ClusterCentroids(
                    estimator=KMeans(n_clusters=args.undersample_clusters, random_state=args.random_state, n_init=10),
                    random_state=args.random_state
                )
            else:
                print(f"⚠ Unknown undersample method: {args.undersample_method}, skipping")
                return X_train, y_train
            
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
            after_under = np.bincount(y_train.astype(int))
            print(f"✓ After undersampling: {after_under}")
    
    return X_train, y_train


def main():
    parser = argparse.ArgumentParser(description="Train LR with RFECV for scene saliency classification")
    
    # Feature groups
    parser.add_argument("--groups", nargs="+", required=True,
                       help="Feature groups to load")
    parser.add_argument("--hf_repo", default="Ishaank18/screenplay-features",
                       help="Hugging Face repo for features (default: Ishaank18/screenplay-features)")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output file for results")
    
    # RFECV parameters
    parser.add_argument("--rfecv_step", type=float, default=0.05,
                       help="Step size for RFECV (fraction of features to remove at each iteration)")
    parser.add_argument("--rfecv_cv", type=int, default=5,
                       help="Number of CV folds for RFECV")
    parser.add_argument("--rfecv_scoring", type=str, default="f1_macro",
                       help="Scoring metric for RFECV")
    parser.add_argument("--rfecv_min_features", type=int, default=10,
                       help="Minimum number of features to select")
    
    # Logistic Regression parameters
    parser.add_argument("--lr_C", type=float, default=0.3,
                       help="Inverse regularization strength for LR")
    parser.add_argument("--lr_max_iter", type=int, default=1000,
                       help="Maximum iterations for LR")
    
    # Sampling
    parser.add_argument("--oversample", action="store_true",
                       help="Use SMOTE oversampling")
    parser.add_argument("--undersample", action="store_true",
                       help="Use undersampling")
    parser.add_argument("--undersample_method", type=str, default="cluster_random",
                       choices=["random", "cluster", "cluster_random"],
                       help="Undersampling method")
    parser.add_argument("--undersample_clusters", type=int, default=50,
                       help="Number of clusters for cluster-based undersampling")
    
    # Feature preferences
    parser.add_argument("--prefer_gc_overlap", action="store_true",
                       help="Prefer Genre Classifier overlap features (for feature selection)")
    
    # Other
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Data source
    hf_repo = args.hf_repo
    data_source = f"Hugging Face ({hf_repo})"
    
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION WITH RFECV")
    print("="*80)
    print(f"Groups: {', '.join(args.groups)}")
    print(f"Source: {data_source}")
    print(f"RFECV: step={args.rfecv_step}, cv={args.rfecv_cv}, scoring={args.rfecv_scoring}")
    print(f"LR: C={args.lr_C}, max_iter={args.lr_max_iter}")
    print(f"Oversample: {args.oversample}")
    print(f"Undersample: {args.undersample} (method: {args.undersample_method})")
    if args.undersample:
        print(f"Undersample clusters: {args.undersample_clusters}")
    print(f"Prefer GC overlap: {args.prefer_gc_overlap}")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    X_train, y_train = load_feature_matrix(args.groups, "train", hf_repo)
    
    print("Loading validation data...")
    X_val, y_val = load_feature_matrix(args.groups, "validation", hf_repo)
    
    print("Loading test data...")
    X_test, y_test = load_feature_matrix(args.groups, "test", hf_repo)
    
    print(f"\n✓ Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Validation set: {len(X_val)} samples, {X_val.shape[1]} features")
    print(f"✓ Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    
    # Align features
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)
    common_cols = sorted(train_cols & val_cols & test_cols)
    
    if len(common_cols) < len(train_cols):
        print(f"\n⚠ Feature mismatch detected:")
        print(f"  Train: {len(train_cols)} features")
        print(f"  Val: {len(val_cols)} features")
        print(f"  Test: {len(test_cols)} features")
        print(f"  Common: {len(common_cols)} features")
        print(f"  Using only common features across all splits")
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]
        print(f"✓ Aligned to {len(common_cols)} common features")
    
    # Apply prefer_gc_overlap filter
    if args.prefer_gc_overlap:
        print("\nApplying prefer_gc_overlap filter...")
        # Identify duplicate features (same across gc_ prefixes)
        gc_overlap_features = set()
        base_to_gc = {}
        
        for col in common_cols:
            if col.startswith('gc_'):
                # Extract base name (e.g., gc_polarity_mean -> polarity_mean)
                parts = col.split('_', 2)
                if len(parts) >= 3:
                    base_name = parts[2]
                    if base_name not in base_to_gc:
                        base_to_gc[base_name] = []
                    base_to_gc[base_name].append(col)
        
        # For duplicates, keep only the first gc_ version
        features_to_remove = set()
        for base_name, gc_versions in base_to_gc.items():
            if len(gc_versions) > 1:
                # Keep first, remove others
                features_to_remove.update(gc_versions[1:])
        
        if features_to_remove:
            print(f"  Removing {len(features_to_remove)} duplicate GC features")
            common_cols = [c for c in common_cols if c not in features_to_remove]
            X_train = X_train[common_cols]
            X_val = X_val[common_cols]
            X_test = X_test[common_cols]
            print(f"  ✓ {len(common_cols)} features after deduplication")
    
    feature_names = common_cols
    
    # Original class distribution
    print("\nOriginal class distribution:")
    print(f"  Train: {np.bincount(y_train.astype(int))}")
    print(f"  Val:   {np.bincount(y_val.astype(int))}")
    print(f"  Test:  {np.bincount(y_test.astype(int))}")
    
    # Preprocessing (imputation + scaling)
    print("\nPreprocessing (imputation + scaling)...")
    preproc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    X_train_std = preproc.fit_transform(X_train)
    X_val_std = preproc.transform(X_val)
    X_test_std = preproc.transform(X_test)
    
    print("✓ Preprocessing complete")
    
    # Apply sampling
    X_train_std, y_train = apply_sampling(X_train_std, y_train, args)
    
    print(f"\nFinal training set: {len(X_train_std)} samples")
    print(f"Final class distribution: {np.bincount(y_train.astype(int))}")
    
    # Train LR with RFECV
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION WITH RFECV")
    print("="*80)
    
    print(f"\nBase model: LogisticRegression(C={args.lr_C}, max_iter={args.lr_max_iter}, class_weight='balanced')")
    print(f"RFECV settings:")
    print(f"  - CV folds: {args.rfecv_cv}")
    print(f"  - Scoring: {args.rfecv_scoring}")
    print(f"  - Step: {args.rfecv_step}")
    print(f"  - Min features: {args.rfecv_min_features}")
    
    # Create base estimator
    base_lr = LogisticRegression(
        C=args.lr_C,
        max_iter=args.lr_max_iter,
        class_weight="balanced",
        random_state=args.random_state
    )
    
    # Create RFECV
    rfecv = RFECV(
        estimator=base_lr,
        step=args.rfecv_step,
        cv=StratifiedKFold(n_splits=args.rfecv_cv, shuffle=True, random_state=args.random_state),
        scoring=args.rfecv_scoring,
        min_features_to_select=args.rfecv_min_features,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nRunning RFECV on {X_train_std.shape[1]} features...")
    print("This may take a while...\n")
    
    rfecv.fit(X_train_std, y_train)
    
    print(f"\n✓ RFECV complete!")
    print(f"  Optimal number of features: {rfecv.n_features_}")
    print(f"  Selected features: {rfecv.support_.sum()}")
    print(f"  Best CV score ({args.rfecv_scoring}): {rfecv.cv_results_['mean_test_score'].max():.4f}")
    
    # Get selected features
    selected_features = [f for f, selected in zip(feature_names, rfecv.support_) if selected]
    print(f"\nTop 20 selected features:")
    for i, feat in enumerate(selected_features[:20], 1):
        print(f"  {i:2d}. {feat}")
    
    # Transform data
    X_train_selected = rfecv.transform(X_train_std)
    X_val_selected = rfecv.transform(X_val_std)
    X_test_selected = rfecv.transform(X_test_std)
    
    # Get final model (refitted on all training data with selected features)
    final_lr = rfecv.estimator_
    
    # Find optimal threshold on validation set
    print("\nFinding optimal threshold on validation set...")
    val_proba = final_lr.predict_proba(X_val_selected)[:, 1]
    optimal_threshold, best_f1 = find_optimal_threshold(
        y_val, val_proba, metric="macro"
    )
    print(f"✓ Optimal threshold: {optimal_threshold:.4f} (macro F1: {best_f1:.4f})")
    
    # Validation metrics
    val_pred = (val_proba >= optimal_threshold).astype(int)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1_binary = f1_score(y_val, val_pred, average='binary')
    val_f1_macro = f1_score(y_val, val_pred, average='macro')
    val_roc_auc = roc_auc_score(y_val, val_proba)
    
    print("\n" + "="*80)
    print("VALIDATION SET RESULTS")
    print("="*80)
    print(f"Accuracy:    {val_accuracy:.4f}")
    print(f"F1 (binary): {val_f1_binary:.4f}")
    print(f"F1 (macro):  {val_f1_macro:.4f} ← OPTIMIZED")
    print(f"ROC-AUC:     {val_roc_auc:.4f}")
    
    # Test metrics
    print("\nEvaluating on test set...")
    test_proba = final_lr.predict_proba(X_test_selected)[:, 1]
    test_pred = (test_proba >= optimal_threshold).astype(int)
    
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_binary = f1_score(y_test, test_pred, average='binary')
    test_f1_macro = f1_score(y_test, test_pred, average='macro')
    test_roc_auc = roc_auc_score(y_test, test_proba)
    
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(f"Accuracy:    {test_accuracy:.4f}")
    print(f"F1 (binary): {test_f1_binary:.4f}")
    print(f"F1 (macro):  {test_f1_macro:.4f} ← OPTIMIZED")
    print(f"ROC-AUC:     {test_roc_auc:.4f}")
    
    # Detailed classification reports
    print("\n" + "-"*80)
    print("VALIDATION SET - DETAILED CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(y_val, val_pred, target_names=["Non-salient", "Salient"]))
    
    print("\n" + "-"*80)
    print("TEST SET - DETAILED CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(y_test, test_pred, target_names=["Non-salient", "Salient"]))
    
    # Feature coefficients
    print("\n" + "="*80)
    print("MODEL COEFFICIENTS (Selected Features)")
    print("="*80)
    
    coefs = final_lr.coef_[0]
    coef_abs = np.abs(coefs)
    top_idx = np.argsort(coef_abs)[::-1][:20]
    
    print("\nTop 20 features by absolute coefficient:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {selected_features[idx]:50s}: {coefs[idx]:+.6f}")
    
    # Save results
    results = {
        'rfecv': rfecv,
        'model': final_lr,
        'preprocessor': preproc,
        'selected_features': selected_features,
        'feature_names': feature_names,
        'n_features_selected': rfecv.n_features_,
        'support': rfecv.support_,
        'cv_results': rfecv.cv_results_,
        'threshold': optimal_threshold,
        'val_accuracy': val_accuracy,
        'val_f1_binary': val_f1_binary,
        'val_f1_macro': val_f1_macro,
        'val_roc_auc': val_roc_auc,
        'test_accuracy': test_accuracy,
        'test_f1_binary': test_f1_binary,
        'test_f1_macro': test_f1_macro,
        'test_roc_auc': test_roc_auc,
        'val_y_pred': val_pred,
        'val_y_proba': val_proba,
        'test_y_pred': test_pred,
        'test_y_proba': test_proba,
        'coefficients': dict(zip(selected_features, coefs)),
        'args': vars(args),
    }
    
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print("✓ Results saved")
    
    # Save selected features to text file
    features_output = args.output.replace('.pkl', '_selected_features.txt')
    with open(features_output, 'w') as f:
        f.write(f"# Selected Features (n={len(selected_features)})\n")
        f.write(f"# RFECV settings: step={args.rfecv_step}, cv={args.rfecv_cv}, scoring={args.rfecv_scoring}\n")
        f.write(f"# Best CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}\n\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"✓ Selected features saved to: {features_output}")
    
    # Save coefficients
    coef_output = args.output.replace('.pkl', '_coefficients.txt')
    with open(coef_output, 'w') as f:
        f.write("Feature\tCoefficient\n")
        for feat, coef in sorted(zip(selected_features, coefs), key=lambda x: abs(x[1]), reverse=True):
            f.write(f"{feat}\t{coef:+.8f}\n")
    print(f"✓ Coefficients saved to: {coef_output}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
