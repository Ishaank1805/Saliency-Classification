"""
Feature Fusion: Combine BERT gradients, activations, and linguistic features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureFusion:
    """Combine BERT features with linguistic features."""
    
    def __init__(
        self,
        use_pca: bool = False,
        pca_components: Optional[int] = None,
        pca_variance: float = 0.95,
    ):
        """
        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_components: Number of PCA components (None = auto)
            pca_variance: Variance to retain if pca_components is None
        """
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_variance = pca_variance
        self.scaler = StandardScaler()
        self.pca = PCA() if use_pca else None
        self.feature_names = []
        
    def get_bert_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract BERT-related features from DataFrame."""
        # Get all BERT features (gradients and activations)
        bert_cols = [c for c in df.columns if c.startswith("bert_")]
        if not bert_cols:
            return np.array([]).reshape(len(df), 0)
        return df[bert_cols].values
    
    def get_linguistic_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract linguistic features from DataFrame."""
        # Exclude BERT features and metadata
        exclude_prefixes = ["bert_", "movie_id", "scene_index", "scene_text", "label"]
        linguistic_cols = [
            c for c in df.columns
            if not any(c.startswith(prefix) for prefix in exclude_prefixes)
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        if not linguistic_cols:
            return np.array([]).reshape(len(df), 0)
        return df[linguistic_cols].values
    
    def fuse_features(
        self,
        df: pd.DataFrame,
        use_bert: bool = True,
        use_linguistic: bool = True,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Fuse BERT and linguistic features.
        
        Args:
            df: DataFrame with features
            use_bert: Include BERT features
            use_linguistic: Include linguistic features
            fit: Whether to fit scaler/PCA (True for train, False for test)
            
        Returns:
            Fused feature matrix
        """
        features_list = []
        feature_names_list = []
        
        if use_bert:
            bert_features = self.get_bert_features(df)
            if bert_features.size > 0:
                features_list.append(bert_features)
                bert_cols = [c for c in df.columns if c.startswith("bert_") or c.startswith("bert_grad_")]
                feature_names_list.extend(bert_cols)
        
        if use_linguistic:
            linguistic_features = self.get_linguistic_features(df)
            if linguistic_features.size > 0:
                features_list.append(linguistic_features)
                exclude_prefixes = ["bert_", "bert_grad_", "movie_id", "scene_index", "scene_text", "label"]
                linguistic_cols = [
                    c for c in df.columns
                    if not any(c.startswith(prefix) for prefix in exclude_prefixes)
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                ]
                feature_names_list.extend(linguistic_cols)
        
        if not features_list:
            raise ValueError("No features found! Enable use_bert or use_linguistic.")
        
        # Concatenate features
        fused = np.hstack(features_list)
        self.feature_names = feature_names_list
        
        # Handle NaN/Inf
        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if fit:
            fused = self.scaler.fit_transform(fused)
        else:
            fused = self.scaler.transform(fused)
        
        # Apply PCA if requested
        if self.use_pca:
            if fit:
                if self.pca_components is None:
                    # Determine components to retain variance
                    self.pca = PCA(n_components=self.pca_variance)
                    fused = self.pca.fit_transform(fused)
                    print(f"PCA: Retained {self.pca.n_components_} components "
                          f"({self.pca.explained_variance_ratio_.sum():.2%} variance)")
                else:
                    self.pca = PCA(n_components=self.pca_components)
                    fused = self.pca.fit_transform(fused)
            else:
                fused = self.pca.transform(fused)
        
        return fused
    
    def get_feature_names(self) -> List[str]:
        """Get names of features after fusion."""
        if self.use_pca and self.pca is not None:
            return [f"PC_{i}" for i in range(self.pca.n_components_)]
        return self.feature_names


def load_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load linguistic features from existing feature extraction pipeline.
    
    This function should integrate with your existing feature extraction code.
    """
    # This is a placeholder - integrate with your actual feature extraction
    # For example, from LLM's/lr_saliency/extract_all_features.py
    
    # You can import and use your existing feature extraction functions here
    # from LLM's.lr_saliency.extract_all_features import extract_all_features
    
    return df

