#!/usr/bin/env python
"""
PCA Ensemble Training with Full Feature Control

Combines:
- Cached feature loading
- Oversampling (SMOTE)
- Undersampling (cluster-based or random)
- PCA ensemble (voting or stacking)
- Feature selection and reporting

Usage:
    # Basic usage
    python train_pca_ensemble.py --groups base gc_all emotional ngram
    
    # With sampling
    python train_pca_ensemble.py \
        --groups base gc_all emotional ngram \
        --oversample --undersample --undersample_method cluster_random
    
    # Stacking instead of voting
    python train_pca_ensemble.py --groups base gc_all --method stacking
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

from feature_cache.load_hf import load_feature_matrix
from pca_utils import fit_pca, find_optimal_threshold
from pca_ensemble import search_ensemble_weights_dirichlet, search_ensemble_weights_simplex

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
    parser = argparse.ArgumentParser(
        description="Advanced PCA Ensemble Training with Cached Features"
    )
    # Feature groups
    parser.add_argument("--groups", nargs="+", required=True,
                       help="Feature groups to use (e.g., base gc_all emotional)")
    parser.add_argument("--hf_repo", default="Ishaank18/screenplay-features",
                       help="Hugging Face repo for features (default: Ishaank18/screenplay-features)")
    
    # Output
    parser.add_argument("--output", type=str,
                       default="/scratch/ishaan.karan/pca_ensemble_advanced.pkl",
                       help="Output file for results")
    
    # Ensemble method
    parser.add_argument("--method", type=str, default="voting",
                       choices=["voting", "stacking"],
                       help="Ensemble method: voting (weighted) or stacking (meta-learner)")
    
    # PCA settings
    parser.add_argument("--pca_n_components", type=float, default=0.95,
                       help="PCA components (float for variance ratio, int for exact number)")
    parser.add_argument("--search", type=str, default="dirichlet",
                       choices=["dirichlet", "simplex"],
                       help="Weight search method for voting ensemble")
    parser.add_argument("--stacking_folds", type=int, default=5,
                       help="Number of folds for stacking ensemble")
    
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
    
    # General settings
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--report_features", action="store_true",
                       help="Report feature importance")
    
    args = parser.parse_args()
    
    # Data source
    hf_repo = args.hf_repo
    data_source = f"Hugging Face ({hf_repo})"
    
    print("\n" + "="*80)
    print("ADVANCED PCA ENSEMBLE TRAINING")
    print("="*80)
    print(f"Groups: {', '.join(args.groups)}")
    print(f"Source: {data_source}")
    print(f"Method: {args.method}")
    print(f"PCA components: {args.pca_n_components}")
    print(f"Oversample: {args.oversample}")
    print(f"Undersample: {args.undersample} (method: {args.undersample_method})")
    if args.undersample:
        print(f"Undersample clusters: {args.undersample_clusters}")
    print(f"Prefer GC overlap: {args.prefer_gc_overlap}")
    print("="*80 + "\n")
    
    # Load cached features
    print("Loading training data...")
    X_train, y_train = load_feature_matrix(args.groups, "train", hf_repo)
    
    print("Loading validation data...")
    X_val, y_val = load_feature_matrix(args.groups, "validation", hf_repo)
    
    print("Loading test data...")
    X_test, y_test = load_feature_matrix(args.groups, "test", hf_repo)
    
    print(f"\n✓ Training set: {len(X_train)} samples, {len(X_train.columns)} features")
    print(f"✓ Validation set: {len(X_val)} samples, {len(X_val.columns)} features")
    print(f"✓ Test set: {len(X_test)} samples, {len(X_test.columns)} features")
    
    # Align features across splits (use intersection of columns)
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)
    
    common_cols = train_cols & val_cols & test_cols
    
    if len(common_cols) < len(train_cols):
        print(f"\n⚠ Feature mismatch detected:")
        print(f"  Train: {len(train_cols)} features")
        print(f"  Val: {len(val_cols)} features")
        print(f"  Test: {len(test_cols)} features")
        print(f"  Common: {len(common_cols)} features")
        print(f"  Using only common features across all splits")
        
        # Keep only common columns in same order
        common_cols = sorted(common_cols)
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]
        
        print(f"✓ Aligned to {len(common_cols)} common features")
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    
    # Remove duplicate features (GC versions that duplicate non-GC features)
    duplicate_gc_features = [
        'gc_flesch_reading_ease',      # Duplicate of flesch_reading_ease
        'gc_gunning_fog_index',         # Duplicate of gunning_fog
    ]
    
    features_before = len(feature_names)
    feature_names = [f for f in feature_names if f not in duplicate_gc_features]
    
    if features_before > len(feature_names):
        removed = features_before - len(feature_names)
        print(f"\n✓ Removed {removed} duplicate GC features")
        X_train = X_train[[f for f in X_train.columns if f not in duplicate_gc_features]]
        X_val = X_val[[f for f in X_val.columns if f not in duplicate_gc_features]]
        X_test = X_test[[f for f in X_test.columns if f not in duplicate_gc_features]]
    
    # Feature filtering based on prefer_gc_overlap
    if args.prefer_gc_overlap:
        print("\nApplying prefer_gc_overlap filter...")
        gc_overlap_cols = [c for c in feature_names if "overlap" in c.lower() and c.startswith("gc_")]
        if gc_overlap_cols:
            print(f"Found {len(gc_overlap_cols)} GC overlap features")
            # Keep GC overlap features and all non-GC features
            keep_cols = gc_overlap_cols + [c for c in feature_names if not c.startswith("gc_")]
            X_train = X_train[keep_cols]
            X_val = X_val[keep_cols]
            X_test = X_test[keep_cols]
            feature_names = keep_cols
            print(f"Filtered to {len(feature_names)} features")
    
    # Convert to numpy
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    
    print(f"\nOriginal class distribution:")
    print(f"  Train: {np.bincount(y_train.astype(int))}")
    print(f"  Val:   {np.bincount(y_val.astype(int))}")
    print(f"  Test:  {np.bincount(y_test.astype(int))}")
    
    # Preprocess
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
    
    if args.method == "voting":
        # Fit PCA
        print(f"\nFitting PCA (target: {args.pca_n_components})...")
        from sklearn.decomposition import PCA
        from sklearn.calibration import CalibratedClassifierCV
        
        pca = PCA(n_components=args.pca_n_components, random_state=args.random_state)
        X_train_pca = pca.fit_transform(X_train_std)
        X_val_pca = pca.transform(X_val_std)
        X_test_pca = pca.transform(X_test_std)
        
        print(f"✓ PCA fitted with {pca.n_components_} components")
        print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Train 4-model ensemble matching pca_analysis.py
        print(f"\nTraining 4-model ensemble (matching pca_analysis.py)...")
        print("=" * 80)
        print("Models:")
        print("  1. Logistic Regression (standard features, C=0.3)")
        print("  2. Logistic Regression (PCA features, C=0.1)")
        print("  3. Calibrated LinearSVC (standard features, C=1.0)")
        print("  4. Random Forest (standard features, 300 trees)")
        print("=" * 80)
        
        # Define 4 models (exactly as in pca_analysis.py)
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        
        base_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3, random_state=args.random_state)
        pca_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=args.random_state)
        svc = LinearSVC(class_weight="balanced", C=1.0, random_state=args.random_state)
        svc_cal = CalibratedClassifierCV(svc, method="sigmoid", cv=2)
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state,
        )
        
        # Train models
        print("\nTraining base_lr (standard features)...")
        base_lr.fit(X_train_std, y_train)
        
        print("Training pca_lr (PCA features)...")
        pca_lr.fit(X_train_pca, y_train)
        
        print("Training svc (calibrated, standard features)...")
        svc_cal.fit(X_train_std, y_train)
        
        print("Training rf (standard features)...")
        rf.fit(X_train_std, y_train)
        
        # Get validation probabilities
        print("\nComputing validation probabilities...")
        val_probs = {
            "base_lr": base_lr.predict_proba(X_val_std)[:, 1],
            "pca_lr": pca_lr.predict_proba(X_val_pca)[:, 1],
            "svc": svc_cal.predict_proba(X_val_std)[:, 1],
            "rf": rf.predict_proba(X_val_std)[:, 1],
        }
        
        model_names = list(val_probs.keys())
        P = np.vstack([val_probs[n] for n in model_names])
        
        # Search for optimal weights using specified method
        print(f"\nSearching optimal weights ({args.search} method)...")
        print(f"Optimizing for: macro F1 (both classes weighted equally)")
        
        if args.search == "dirichlet":
            search_results = search_ensemble_weights_dirichlet(
                P, y_val, 
                threshold_finder_func=find_optimal_threshold,
                n_trials=500, 
                metric="macro",
                random_state=args.random_state
            )
            best_weights = search_results['weights']
            best_threshold = search_results['threshold']
        else:  # simplex
            search_results = search_ensemble_weights_simplex(
                P, y_val,
                threshold_finder_func=find_optimal_threshold,
                step=0.1,
                metric="macro"
            )
            best_weights = search_results['weights']
            best_threshold = search_results['threshold']
        
        # Compute validation metrics with best weights
        val_proba = (best_weights @ P)
        val_pred = (val_proba >= best_threshold).astype(int)
        
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1_binary = f1_score(y_val, val_pred, average='binary')
        val_f1_macro = f1_score(y_val, val_pred, average='macro')
        val_roc_auc = roc_auc_score(y_val, val_proba)
        
        print("\n" + "=" * 80)
        print("ENSEMBLE RESULTS (OPTIMIZED FOR MACRO F1)")
        print("=" * 80)
        print(f"Models: {model_names}")
        print(f"Optimal weights: {[f'{w:.3f}' for w in best_weights]}")
        print(f"Optimal threshold: {best_threshold:.4f} (maximizes macro F1)")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"F1 (binary): {val_f1_binary:.4f}")
        print(f"F1 (macro): {val_f1_macro:.4f} ← OPTIMIZED FOR THIS")
        print(f"ROC-AUC: {val_roc_auc:.4f}")
        print("=" * 80 + "\n")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_probs = {
            "base_lr": base_lr.predict_proba(X_test_std)[:, 1],
            "pca_lr": pca_lr.predict_proba(X_test_pca)[:, 1],
            "svc": svc_cal.predict_proba(X_test_std)[:, 1],
            "rf": rf.predict_proba(X_test_std)[:, 1],
        }
        
        P_test = np.vstack([test_probs[n] for n in model_names])
        test_proba = (best_weights @ P_test)
        test_pred = (test_proba >= best_threshold).astype(int)
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1_binary = f1_score(y_test, test_pred, average='binary')
        test_f1_macro = f1_score(y_test, test_pred, average='macro')
        test_roc_auc = roc_auc_score(y_test, test_proba)
        
        # Package results
        results = {
            'models': {
                'base_lr': base_lr,
                'pca_lr': pca_lr,
                'svc': svc_cal,
                'rf': rf,
            },
            'model_names': model_names,
            'weights': best_weights,
            'threshold': best_threshold,
            'pca': pca,
            'method': 'voting',
            'search_method': args.search,
            'accuracy': val_accuracy,
            'f1_binary': val_f1_binary,
            'f1_macro': val_f1_macro,
            'roc_auc': val_roc_auc,
            'y_pred': val_pred,
            'y_proba': val_proba,
            'test_accuracy': test_accuracy,
            'test_f1_binary': test_f1_binary,
            'test_f1_macro': test_f1_macro,
            'test_roc_auc': test_roc_auc,
            'test_y_pred': test_pred,
            'test_y_proba': test_proba,
        }
    
    # Save results
    results["preprocessor"] = preproc
    results["feature_names"] = feature_names
    results["feature_groups"] = args.groups
    results["hf_repo"] = args.hf_repo
    results["sampling"] = {
        "oversample": args.oversample,
        "undersample": args.undersample,
        "undersample_method": args.undersample_method,
        "undersample_clusters": args.undersample_clusters,
    }
    results["args"] = vars(args)
    
    print(f"\nSaving results to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    
    print("✓ Results saved")
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Optimization: macro F1 (both classes weighted equally)")
    print(f"Optimal threshold: {results['threshold']:.4f}")
    
    print("\n" + "-"*80)
    print("VALIDATION SET RESULTS")
    print("-"*80)
    print(f"Accuracy:    {results.get('accuracy', 0.0):.4f}")
    print(f"F1 (binary): {results.get('f1_binary', 0.0):.4f}")
    print(f"F1 (macro):  {results.get('f1_macro', 0.0):.4f} ← OPTIMIZED")
    print(f"ROC-AUC:     {results.get('roc_auc', 0.0):.4f}")
    
    print("\n" + "-"*80)
    print("TEST SET RESULTS")
    print("-"*80)
    print(f"Accuracy:    {results.get('test_accuracy', 0.0):.4f}")
    print(f"F1 (binary): {results.get('test_f1_binary', 0.0):.4f}")
    print(f"F1 (macro):  {results.get('test_f1_macro', 0.0):.4f} ← OPTIMIZED")
    print(f"ROC-AUC:     {results.get('test_roc_auc', 0.0):.4f}")
    
    if args.method == "voting" and args.report_features:
        print("\n" + "-"*80)
        print("ENSEMBLE MODEL WEIGHTS")
        print("-"*80)
        for model_name, weight in zip(results['model_names'], results['weights']):
            print(f"  {model_name:30s}: {weight:.4f}")
    
    # Detailed classification reports
    print("\n" + "-"*80)
    print("VALIDATION SET - DETAILED CLASSIFICATION REPORT")
    print("-"*80)
    y_pred = results.get('y_pred')
    if y_pred is not None:
        print(classification_report(y_val, y_pred, target_names=["Non-salient", "Salient"]))
    else:
        print("Predictions not available")
    
    print("\n" + "-"*80)
    print("TEST SET - DETAILED CLASSIFICATION REPORT")
    print("-"*80)
    y_test_pred = results.get('test_y_pred')
    if y_test_pred is not None:
        print(classification_report(y_test, y_test_pred, target_names=["Non-salient", "Salient"]))
    else:
        print("Predictions not available")
    
    print("="*80)
    
    # Feature importance for PCA (if voting method)
    if args.method == "voting" and args.report_features and len(feature_names) <= 200:
        print("\nTop 20 Features by PCA Component Importance:")
        print("-" * 80)
        
        # Get top features from first few PCA components
        pca_obj = results.get("pca")
        if pca_obj is not None:
            components = pca_obj.components_[:min(5, pca_obj.n_components_)]
            
            # Calculate feature importance as sum of absolute loadings
            importance = np.abs(components).sum(axis=0)
            top_idx = np.argsort(importance)[-20:][::-1]
            
            for i, idx in enumerate(top_idx, 1):
                print(f"  {i:2d}. {feature_names[idx]:50s}: {importance[idx]:.4f}")
    
    # Extract and display model coefficients
    print("\n" + "="*80)
    print("MODEL COEFFICIENTS")
    print("="*80)
    
    if args.method == "voting":
        model_dict = results['models']
        model_names = results['model_names']
        weights = results['weights']
        pca_obj = results.get("pca")
        
        # Extract coefficients from each model
        for model_name, weight in zip(model_names, weights):
            model = model_dict[model_name]
            print(f"\n{model_name} (weight: {weight:.4f}):")
            print("-" * 80)
            
            if hasattr(model, 'coef_'):
                coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                
                # If PCA model, transform coefficients back to original feature space
                if '_pca' in model_name and pca_obj is not None:
                    # Transform PCA coefficients back to original space
                    # PCA inverse_transform needs 2D array, so reshape
                    coefs_2d = coefs.reshape(1, -1)
                    original_coefs = pca_obj.inverse_transform(coefs_2d).flatten()
                    
                    # Show top features by absolute coefficient value
                    top_idx = np.argsort(np.abs(original_coefs))[-20:][::-1]
                    print(f"Top 20 features (transformed from PCA space):")
                    for i, idx in enumerate(top_idx, 1):
                        print(f"  {i:2d}. {feature_names[idx]:50s}: {original_coefs[idx]:+.6f}")
                else:
                    # Direct feature space coefficients
                    top_idx = np.argsort(np.abs(coefs))[-20:][::-1]
                    print(f"Top 20 features:")
                    for i, idx in enumerate(top_idx, 1):
                        print(f"  {i:2d}. {feature_names[idx]:50s}: {coefs[idx]:+.6f}")
            else:
                print(f"  Model {model_name} does not have coefficients (non-linear model)")
        
        # Compute weighted ensemble coefficients (for linear models)
        print("\n" + "-"*80)
        print("WEIGHTED ENSEMBLE COEFFICIENTS")
        print("-"*80)
        
        ensemble_coefs = np.zeros(len(feature_names))
        total_weight = 0.0
        
        for model_name, weight in zip(model_names, weights):
            model = model_dict[model_name]
            if hasattr(model, 'coef_'):
                coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                
                if '_pca' in model_name and pca_obj is not None:
                    # Transform PCA coefficients back to original space
                    # PCA inverse_transform needs 2D array, so reshape
                    coefs_2d = coefs.reshape(1, -1)
                    original_coefs = pca_obj.inverse_transform(coefs_2d).flatten()
                    ensemble_coefs += weight * original_coefs
                else:
                    # Ensure coefs matches feature_names length
                    if len(coefs) == len(feature_names):
                        ensemble_coefs += weight * coefs
                    else:
                        print(f"  Warning: {model_name} has {len(coefs)} coefficients, expected {len(feature_names)}, skipping")
                
                total_weight += weight
        
        if total_weight > 0:
            # Show top features by absolute ensemble coefficient value
            top_idx = np.argsort(np.abs(ensemble_coefs))[-20:][::-1]
            print(f"Top 20 features by weighted ensemble coefficient:")
            for i, idx in enumerate(top_idx, 1):
                print(f"  {i:2d}. {feature_names[idx]:50s}: {ensemble_coefs[idx]:+.6f}")
            
            # Save ensemble coefficients to file
            # Add GPU/process ID prefix to avoid conflicts when running on multiple GPUs
            import os
            gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('GPU_ID', '0'))
            base_output = args.output.replace('.pkl', '')
            coef_output = f"{base_output}_gpu{gpu_id}_coefficients.txt"
            print(f"\nSaving all ensemble coefficients to: {coef_output}")
            with open(coef_output, 'w') as f:
                f.write("Feature\tCoefficient\n")
                for fname, coef in sorted(zip(feature_names, ensemble_coefs), 
                                         key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"{fname}\t{coef:+.8f}\n")
            print("✓ Coefficients saved")
            
            # Add to results dict
            results['ensemble_coefficients'] = dict(zip(feature_names, ensemble_coefs))
        else:
            print("No linear models in ensemble - cannot compute ensemble coefficients")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
