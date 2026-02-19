"""
Main training script for BERT gradient-based scene saliency classification.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import os

from extract_bert_gradients import extract_bert_features_for_dataframe, BERTGradientExtractor
from feature_fusion import FeatureFusion, load_linguistic_features
from classifiers import SVMClassifier, LlamaClassifier
from shap_analysis import analyze_model_shap


def load_data(
    train_path: str,
    val_path: str,
    test_path: str,
    text_col: str = "scene_text",
    label_col: str = "label",
    load_texts_from_mensa: bool = False,
) -> tuple:
    """
    Load train/val/test data.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        text_col: Column name for scene text
        label_col: Column name for labels
        load_texts_from_mensa: If True, load scene texts from MENSA dataset
    """
    print("Loading data...")
    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if val_path.endswith(".parquet") else pd.read_csv(val_path)
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # If scene_text is missing, try to load from MENSA dataset
    if load_texts_from_mensa or text_col not in train_df.columns:
        print(f"\n⚠ '{text_col}' column not found. Loading from MENSA dataset...")
        try:
            from datasets import load_dataset
            
            ds_train = load_dataset("rohitsaxena/MENSA", split="train")
            ds_val = load_dataset("rohitsaxena/MENSA", split="validation")
            ds_test = load_dataset("rohitsaxena/MENSA", split="test")
            
            # Create scene text mapping
            def create_text_mapping(ds):
                mapping = {}
                for item in ds:
                    title = item.get("name") or item.get("title", "")
                    scenes = item.get("scenes", [])
                    labels = item.get("labels", [])
                    for idx, scene in enumerate(scenes):
                        key = (title, idx)
                        mapping[key] = {
                            "scene_text": scene,
                            "label": int(labels[idx]) if idx < len(labels) else 0,
                        }
                return mapping
            
            train_mapping = create_text_mapping(ds_train)
            val_mapping = create_text_mapping(ds_val)
            test_mapping = create_text_mapping(ds_test)
            
            # Merge scene texts into dataframes
            def add_texts(df, mapping, split_name):
                if "movie_title" in df.columns and "scene_index" in df.columns:
                    texts = []
                    labels = []
                    for _, row in df.iterrows():
                        title = str(row.get("movie_title", row.get("title", "")))
                        idx = int(row.get("scene_index", 0))
                        key = (title, idx)
                        if key in mapping:
                            texts.append(mapping[key]["scene_text"])
                            labels.append(mapping[key]["label"])
                        else:
                            texts.append("")
                            labels.append(0)
                    df[text_col] = texts
                    if label_col not in df.columns:
                        df[label_col] = labels
                    print(f"  ✓ Added {len([t for t in texts if t])} scene texts to {split_name}")
                return df
            
            train_df = add_texts(train_df, train_mapping, "train")
            val_df = add_texts(val_df, val_mapping, "validation")
            test_df = add_texts(test_df, test_mapping, "test")
            
        except Exception as e:
            print(f"  ✗ Error loading from MENSA: {e}")
            print(f"  Please ensure '{text_col}' column exists in your data files")
    
    # Verify required columns
    if text_col not in train_df.columns:
        raise ValueError(f"Required column '{text_col}' not found. Available columns: {list(train_df.columns)[:10]}")
    
    if label_col not in train_df.columns:
        print(f"⚠ Warning: '{label_col}' column not found. Creating dummy labels.")
        train_df[label_col] = 0
        val_df[label_col] = 0
        test_df[label_col] = 0
    
    return train_df, val_df, test_df


def extract_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_gradients: bool = True,
    use_activations: bool = True,
    use_linguistic: bool = True,
    bert_model: str = "bert-base-uncased",
    text_col: str = "scene_text",
    label_col: str = "label",
    output_dir: str = None,
) -> tuple:
    """Extract all features."""
    import gc
    import torch
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION")
    print("="*80)
    
    # Extract linguistic features if needed
    if use_linguistic:
        print("\n[1/3] Loading linguistic features...")
        train_df = load_linguistic_features(train_df)
        val_df = load_linguistic_features(val_df)
        test_df = load_linguistic_features(test_df)
        print("✓ Linguistic features loaded")
    
    # Extract BERT features if needed
    if use_gradients or use_activations:
        print("\n[2/3] Extracting BERT gradients and activations...")
        
        # Train classifier first
        print("Training BERT classifier...")
        extractor = BERTGradientExtractor(model_name=bert_model)
        train_texts = train_df[text_col].astype(str).tolist()
        train_labels = train_df[label_col].astype(int).tolist()
        extractor.train_classifier(train_texts, train_labels)
        
        # Clear memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Extract features for train set
        print("Extracting features for train set...")
        train_df = extract_bert_features_for_dataframe(
            train_df,
            text_col=text_col,
            label_col=label_col,
            model_name=bert_model,
            train_classifier=False,  # Already trained
            extract_gradients=use_gradients,
            extract_activations=use_activations,
        )
        
        # Save train set immediately to free memory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_cache_path = os.path.join(output_dir, "train_with_bert_features.parquet")
            print(f"Saving train features to {train_cache_path}...")
            train_df.to_parquet(train_cache_path, index=False)
            print(f"✓ Saved {len(train_df)} train samples")
        
        # Clear memory before processing validation
        del train_texts, train_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleared after train extraction")
        
        # Extract features for val set
        print("Extracting features for val set...")
        val_df = extract_bert_features_for_dataframe(
            val_df,
            text_col=text_col,
            label_col=label_col,
            model_name=bert_model,
            train_classifier=False,
            extract_gradients=use_gradients,
            extract_activations=use_activations,
        )
        
        # Save val set immediately
        if output_dir:
            val_cache_path = os.path.join(output_dir, "val_with_bert_features.parquet")
            print(f"Saving val features to {val_cache_path}...")
            val_df.to_parquet(val_cache_path, index=False)
            print(f"✓ Saved {len(val_df)} val samples")
        
        # Clear memory before processing test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleared after val extraction")
        
        # Extract features for test set
        print("Extracting features for test set...")
        test_df = extract_bert_features_for_dataframe(
            test_df,
            text_col=text_col,
            label_col=label_col,
            model_name=bert_model,
            train_classifier=False,
            extract_gradients=use_gradients,
            extract_activations=use_activations,
        )
        
        # Save test set immediately
        if output_dir:
            test_cache_path = os.path.join(output_dir, "test_with_bert_features.parquet")
            print(f"Saving test features to {test_cache_path}...")
            test_df.to_parquet(test_cache_path, index=False)
            print(f"✓ Saved {len(test_df)} test samples")
        
        # Final cleanup
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✓ BERT features extracted and saved")
    
    return train_df, val_df, test_df


def train_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_svm: bool = True,
    use_llama: bool = False,
    use_bert: bool = True,
    use_linguistic: bool = True,
    output_dir: str = "models",
    label_col: str = "label",
) -> Dict:
    """Train SVM and/or Llama models."""
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Feature fusion
    print("\n[1/3] Fusing features...")
    fusion = FeatureFusion(use_pca=False)
    
    X_train = fusion.fuse_features(train_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=True)
    X_val = fusion.fuse_features(val_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=False)
    X_test = fusion.fuse_features(test_df, use_bert=use_bert, use_linguistic=use_linguistic, fit=False)
    
    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values
    
    feature_names = fusion.get_feature_names()
    print(f"✓ Fused features: {X_train.shape[1]} dimensions")
    
    # Train SVM
    if use_svm:
        print("\n[2/3] Training SVM classifier...")
        svm = SVMClassifier(tune_hyperparameters=True)
        svm.fit(X_train, y_train)
        
        # Evaluate
        val_metrics = svm.evaluate(X_val, y_val)
        test_metrics = svm.evaluate(X_test, y_test)
        
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        # Save
        svm.save(os.path.join(output_dir, "svm_model.pkl"))
        
        results["svm"] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        
        # SHAP analysis
        print("\n[3/3] Running SHAP analysis for SVM...")
        analyze_model_shap(
            svm.model,
            X_train[:1000],  # Sample for efficiency
            X_test[:100],
            feature_names,
            model_type="svm",
            output_dir=os.path.join(output_dir, "shap_svm"),
        )
    
    # Train Llama
    if use_llama:
        print("\n[2/3] Training Llama classifier...")
        llama = LlamaClassifier()
        
        train_texts = train_df["scene_text"].astype(str).tolist()
        val_texts = val_df["scene_text"].astype(str).tolist()
        test_texts = test_df["scene_text"].astype(str).tolist()
        
        llama.fit(train_texts, y_train.tolist(), val_texts, y_val.tolist())
        
        # Evaluate
        val_metrics = llama.evaluate(val_texts, y_val.tolist())
        test_metrics = llama.evaluate(test_texts, y_test.tolist())
        
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        # Save
        llama.save(os.path.join(output_dir, "llama_model"))
        
        results["llama"] = {
            "val": val_metrics,
            "test": test_metrics,
        }
    
    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train BERT gradient-based classifiers")
    
    # Data paths
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test-path", type=str, required=True, help="Path to test data")
    parser.add_argument("--text-col", type=str, default="scene_text", help="Column name for scene text")
    parser.add_argument("--label-col", type=str, default="label", help="Column name for labels")
    parser.add_argument("--load-texts-from-mensa", action="store_true", 
                        help="Load scene texts from MENSA dataset if not in feature files")
    
    # Feature options
    parser.add_argument("--use-gradients", action="store_true", help="Extract BERT gradients")
    parser.add_argument("--use-activations", action="store_true", help="Extract BERT activations")
    parser.add_argument("--use-linguistic", action="store_true", help="Use linguistic features")
    
    # Model options
    parser.add_argument("--use-svm", action="store_true", default=True, help="Train SVM classifier")
    parser.add_argument("--use-llama", action="store_true", help="Train Llama classifier")
    
    # BERT options
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased", help="BERT model name")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    train_df, val_df, test_df = load_data(
        args.train_path,
        args.val_path,
        args.test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        load_texts_from_mensa=args.load_texts_from_mensa,
    )
    
    # Extract features
    train_df, val_df, test_df = extract_features(
        train_df,
        val_df,
        test_df,
        use_gradients=args.use_gradients,
        use_activations=args.use_activations,
        use_linguistic=args.use_linguistic,
        bert_model=args.bert_model,
        text_col=args.text_col,
        label_col=args.label_col,
        output_dir=args.output_dir,  # Pass output_dir to save intermediate results
    )
    
    # Train models
    results = train_models(
        train_df,
        val_df,
        test_df,
        use_svm=args.use_svm,
        use_llama=args.use_llama,
        use_bert=(args.use_gradients or args.use_activations),
        use_linguistic=args.use_linguistic,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to {args.output_dir}/results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

