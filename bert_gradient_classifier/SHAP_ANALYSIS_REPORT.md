# In-Depth Analysis: `shap_analysis.py`

## Executive Summary

The `shap_analysis.py` module provides a **standalone SHAP (SHapley Additive exPlanations) analysis framework** for model interpretability in the BERT gradient-based scene saliency classification research. This analysis examines the module's architecture, design decisions, integration with the research pipeline, strengths, limitations, and recommendations for enhancement.

---

## 1. Research Context

### 1.1 Research Objectives
Your research focuses on **scene saliency classification** using:
- **BERT/RoBERTa/DeBERTa gradients and activations** as scene encoders
- **Linguistic features** (syntax, semantics, discourse, sentiment, etc.)
- **Sequence modeling** with Transformer encoders
- **Feature fusion methods** (concat, attention, gated)
- **Multiple model architectures** (BERT-base/large, RoBERTa-base/large, DeBERTa-v3-large)

### 1.2 Role of SHAP Analysis
SHAP analysis serves to:
1. **Identify which linguistic features** drive saliency predictions
2. **Quantify feature importance** for interpretability
3. **Compare feature contributions** across different model configurations
4. **Validate research hypotheses** about linguistic feature utility
5. **Provide explainability** for model decisions

### 1.3 Current Implementation Status
- **Standalone module** (`shap_analysis.py`): Generic SHAP wrapper for SVM/Llama models
- **Integrated implementation** (`train_sequence.py`): Sophisticated SHAP analysis specifically for neural sequence models with linguistic features

---

## 2. Code Architecture Analysis

### 2.1 Class Structure: `SHAPAnalyzer`

#### 2.1.1 Initialization
```python
def __init__(self, model, feature_names: List[str], model_type: str = "svm")
```

**Strengths:**
- Clean separation of concerns (model, features, type)
- Flexible model type parameter
- Stores explainer and SHAP values as instance variables

**Weaknesses:**
- **Limited model type support**: Only "svm" and "llama" (with llama being a placeholder)
- **No validation** of model compatibility
- **Missing error handling** for invalid model types

**Recommendation:**
```python
SUPPORTED_MODELS = {"svm", "llama", "neural", "transformer"}
if model_type not in SUPPORTED_MODELS:
    raise ValueError(f"Unsupported model_type: {model_type}. Supported: {SUPPORTED_MODELS}")
```

#### 2.1.2 Explainer Creation: `create_explainer()`

**Current Implementation:**
- Uses `KernelExplainer` for SVM (appropriate for non-differentiable models)
- Placeholder for Llama (not implemented)
- Background sampling limited to 100 samples

**Analysis:**

**Strengths:**
- Correct choice of `KernelExplainer` for SVM
- Background sampling for efficiency
- Handles background dataset preparation

**Critical Issues:**

1. **Llama Support is Incomplete**
   ```python
   elif self.model_type == "llama":
       print("Warning: Llama SHAP analysis requires text-based explainers.")
       return  # Just returns, doesn't create explainer
   ```
   - **Problem**: Silent failure - no explainer created, but no exception raised
   - **Impact**: Code will fail later with unclear error message
   - **Fix**: Raise `NotImplementedError` or implement proper text-based explainer

2. **Background Size Hardcoded**
   ```python
   X_background[:min(100, len(X_background))]  # Hardcoded 100
   ```
   - **Problem**: No parameter to control background size
   - **Impact**: May be too small for complex models, too large for simple ones
   - **Fix**: Make it a parameter with sensible default

3. **No Background Validation**
   - **Problem**: Doesn't check if background is representative
   - **Impact**: SHAP values may be biased if background is skewed
   - **Fix**: Add statistical checks (mean, std, distribution)

**Comparison with `train_sequence.py`:**
The integrated implementation uses:
- `DeepExplainer` for neural networks (much faster)
- `ModelWrapperForSHAP` to isolate linguistic features
- Mean-based background for handling variable sequence lengths
- Proper tensor/array conversion handling

**Recommendation:**
```python
def create_explainer(
    self,
    X_background: np.ndarray,
    X_sample: Optional[np.ndarray] = None,
    max_evals: int = 100,
    background_size: int = 100,  # NEW
    validate_background: bool = True,  # NEW
):
    if validate_background:
        # Check for NaN, inf, constant features
        if np.any(np.isnan(X_background)) or np.any(np.isinf(X_background)):
            raise ValueError("Background contains NaN or Inf values")
        # Check for constant features
        feature_stds = np.std(X_background, axis=0)
        if np.any(feature_stds < 1e-6):
            print("Warning: Some features have near-zero variance in background")
    
    background_sample = X_background[:min(background_size, len(X_background))]
    # ... rest of implementation
```

#### 2.1.3 SHAP Value Computation: `compute_shap_values()`

**Current Implementation:**
- Uses explainer's `shap_values()` method
- Configurable `max_evals` parameter
- Stores results as instance variable

**Analysis:**

**Strengths:**
- Simple, straightforward interface
- Stores values for reuse
- Returns values for immediate use

**Issues:**

1. **No Progress Tracking**
   - SHAP computation can be slow
   - No progress bar or logging
   - **Fix**: Add `tqdm` progress bar

2. **No Error Handling**
   - KernelExplainer can fail on large datasets
   - No timeout or memory management
   - **Fix**: Add try-except with informative errors

3. **No Shape Validation**
   - Doesn't verify input shape matches expected
   - **Fix**: Add shape checks before computation

**Comparison with `train_sequence.py`:**
The integrated version has:
- Extensive shape handling and validation
- Multiple fallback strategies (DeepExplainer → KernelExplainer)
- Detailed error messages with debugging info
- Handles nested SHAP output structures

**Recommendation:**
```python
def compute_shap_values(
    self,
    X: np.ndarray,
    max_evals: int = 100,
    show_progress: bool = True,
) -> np.ndarray:
    if self.explainer is None:
        raise ValueError("Explainer not created. Call create_explainer first.")
    
    # Validate input
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.shape[1] != len(self.feature_names):
        raise ValueError(f"Feature dimension mismatch: {X.shape[1]} vs {len(self.feature_names)}")
    
    print(f"Computing SHAP values for {len(X)} samples...")
    
    try:
        if show_progress:
            from tqdm import tqdm
            # For KernelExplainer, we can't easily add progress, but we can log
            print(f"  This may take a while (max_evals={max_evals})...")
        
        self.shap_values = self.explainer.shap_values(
            X,
            nsamples=max_evals,
        )
        
        # Convert to numpy if needed
        if not isinstance(self.shap_values, np.ndarray):
            self.shap_values = np.array(self.shap_values)
        
        print(f"✓ Computed SHAP values: shape {self.shap_values.shape}")
        return self.shap_values
        
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}") from e
```

### 2.2 Visualization Methods

#### 2.2.1 Summary Plot: `plot_summary()`

**Analysis:**
- Uses SHAP's built-in `summary_plot()`
- Configurable `max_display`
- Saves or displays plot

**Issues:**
1. **Feature Name Truncation**
   ```python
   feature_names=self.feature_names[:len(self.shap_values[0])]
   ```
   - **Problem**: Assumes `shap_values[0]` exists and has correct length
   - **Fix**: Handle edge cases (empty values, wrong shape)

2. **No Customization Options**
   - Can't customize colors, labels, titles
   - **Fix**: Add kwargs for matplotlib customization

**Recommendation:**
```python
def plot_summary(
    self,
    max_display: int = 20,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,  # Pass to shap.summary_plot
):
    if self.shap_values is None:
        raise ValueError("SHAP values not computed.")
    
    # Handle multi-dimensional shap_values
    if isinstance(self.shap_values, list):
        shap_vals = self.shap_values[0] if len(self.shap_values) > 0 else None
    else:
        shap_vals = self.shap_values
    
    if shap_vals is None or shap_vals.size == 0:
        raise ValueError("SHAP values are empty")
    
    # Ensure feature names match
    num_features = shap_vals.shape[-1] if shap_vals.ndim > 1 else len(shap_vals)
    feature_names_to_use = self.feature_names[:num_features]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        feature_names=feature_names_to_use,
        max_display=max_display,
        show=False,
        **kwargs
    )
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved summary plot to {save_path}")
    else:
        plt.show()
```

#### 2.2.2 Bar Plot: `plot_bar()`

**Analysis:**
- Similar to summary plot but with `plot_type="bar"`
- Shows mean absolute SHAP values
- Good for feature ranking

**Same issues as summary plot apply.**

#### 2.2.3 Waterfall Plot: `plot_waterfall()`

**Analysis:**
- Creates waterfall plot for single instance
- Uses `shap.Explanation` object
- Good for explaining individual predictions

**Issues:**

1. **Explanation Object Creation**
   ```python
   shap.Explanation(
       values=self.shap_values[instance_idx],
       base_values=self.explainer.expected_value,
       ...
   )
   ```
   - **Problem**: Assumes `expected_value` exists (may not for all explainers)
   - **Fix**: Handle missing expected_value gracefully

2. **No Instance Validation**
   - Doesn't check if `instance_idx` is valid
   - **Fix**: Add bounds checking

**Recommendation:**
```python
def plot_waterfall(
    self,
    instance_idx: int,
    X: np.ndarray,
    save_path: Optional[str] = None,
):
    if self.shap_values is None:
        raise ValueError("SHAP values not computed.")
    
    if instance_idx < 0 or instance_idx >= len(self.shap_values):
        raise ValueError(f"Invalid instance_idx: {instance_idx} (max: {len(self.shap_values)-1})")
    
    if instance_idx >= len(X):
        raise ValueError(f"Instance {instance_idx} not in X (len: {len(X)})")
    
    # Get expected value (handle different explainer types)
    if hasattr(self.explainer, 'expected_value'):
        base_value = self.explainer.expected_value
    else:
        # Fallback: mean of predictions
        base_value = np.mean(self.model.predict_proba(X)[:, 1])
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=base_value,
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
```

### 2.3 Feature Importance: `get_feature_importance()`

**Analysis:**
- Computes mean absolute SHAP values
- Returns sorted DataFrame
- Clean, simple implementation

**Strengths:**
- Correct statistical approach
- Returns structured data (DataFrame)
- Easy to use for downstream analysis

**Minor Issues:**
1. **No Confidence Intervals**: Only mean, no std or CI
2. **No Significance Testing**: Can't tell if differences are meaningful

**Enhancement:**
```python
def get_feature_importance(
    self,
    top_k: int = 20,
    include_stats: bool = True,
) -> pd.DataFrame:
    if self.shap_values is None:
        raise ValueError("SHAP values not computed.")
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
    
    # Additional statistics if requested
    if include_stats:
        std_abs_shap = np.abs(self.shap_values).std(axis=0)
        # Percentage of positive contributions
        positive_pct = (self.shap_values > 0).mean(axis=0) * 100
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap,
            "std": std_abs_shap,
            "positive_pct": positive_pct,
        })
    else:
        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap,
        })
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df.head(top_k)
```

### 2.4 Report Generation: `generate_report()`

**Analysis:**
- Comprehensive report generation
- Creates directory structure
- Generates multiple visualizations
- Saves feature importance CSV

**Strengths:**
- One-stop function for complete analysis
- Good file organization
- Includes multiple plot types

**Issues:**

1. **Hardcoded Sample Size**
   ```python
   analyzer.compute_shap_values(X_test[:100])  # Hardcoded 100
   ```
   - **Problem**: No parameter to control sample size
   - **Fix**: Add `num_samples` parameter

2. **No Configuration File**
   - Can't customize report without code changes
   - **Fix**: Add config dict parameter

3. **Limited Analysis Depth**
   - Only basic plots, no interaction analysis, no category-based analysis
   - **Comparison**: `train_sequence.py` has much more sophisticated analysis

**Recommendation:**
```python
def generate_report(
    self,
    X: np.ndarray,
    output_dir: str = "shap_reports",
    top_k: int = 20,
    num_samples: Optional[int] = None,
    num_waterfall_examples: int = 5,
    config: Optional[Dict] = None,
):
    """
    Generate comprehensive SHAP analysis report.
    
    Args:
        config: Optional dict with keys:
            - 'plot_dpi': int (default 300)
            - 'plot_format': str (default 'png')
            - 'include_interactions': bool (default False)
            - 'include_by_category': bool (default False)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample if needed
    if num_samples and len(X) > num_samples:
        sample_indices = np.random.choice(len(X), num_samples, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
    
    # Compute SHAP values if not already computed
    if self.shap_values is None:
        self.compute_shap_values(X_sample)
    
    # ... rest of implementation with config options
```

### 2.5 Convenience Function: `analyze_model_shap()`

**Analysis:**
- Simple wrapper for quick analysis
- Good for prototyping
- Limited customization

**Strengths:**
- Easy to use
- Good for quick experiments

**Weaknesses:**
- Hardcoded sample sizes
- No flexibility

---

## 3. Comparison: Standalone vs. Integrated Implementation

### 3.1 Feature Comparison

| Feature | `shap_analysis.py` | `train_sequence.py` |
|---------|-------------------|---------------------|
| Model Support | SVM, Llama (placeholder) | Neural sequence models |
| Explainer Type | KernelExplainer | DeepExplainer + KernelExplainer fallback |
| Feature Isolation | No | Yes (ModelWrapperForSHAP) |
| Shape Handling | Basic | Extensive validation |
| Error Handling | Minimal | Comprehensive |
| Progress Tracking | No | Yes (tqdm) |
| Interaction Analysis | No | Yes |
| Category-based Analysis | No | Yes |
| Feature Statistics | Basic | Detailed (mean, std, correlation) |
| Ablation Studies | No | Yes |
| BERT Attention Comparison | No | Yes |

### 3.2 Use Case Analysis

**When to use `shap_analysis.py`:**
- Quick prototyping with SVM models
- Simple feature importance analysis
- Standalone interpretability experiments
- When you need a generic, reusable SHAP wrapper

**When to use `train_sequence.py` implementation:**
- Production analysis of trained sequence models
- Deep interpretability research
- Feature interaction analysis
- Comparison with BERT attention
- Comprehensive reporting

### 3.3 Integration Gap

**Problem**: The standalone module doesn't integrate well with your main research pipeline because:
1. It's designed for SVM/Llama, not your neural sequence models
2. It doesn't handle the linguistic feature isolation you need
3. It lacks the sophisticated analysis your research requires

**Solution**: Either:
- **Option A**: Enhance `shap_analysis.py` to support neural models and feature isolation
- **Option B**: Extract the SHAP functionality from `train_sequence.py` into a reusable module
- **Option C**: Keep both - use `shap_analysis.py` for simple cases, `train_sequence.py` for research

---

## 4. Critical Issues and Recommendations

### 4.1 High Priority Issues

#### Issue 1: Incomplete Llama Support
**Severity**: High  
**Impact**: Silent failure, unclear errors  
**Fix**: Implement proper text-based explainer or raise NotImplementedError

#### Issue 2: No Neural Model Support
**Severity**: High  
**Impact**: Can't analyze your main research models  
**Fix**: Add support for PyTorch models with DeepExplainer

#### Issue 3: Limited Error Handling
**Severity**: Medium  
**Impact**: Unclear error messages, debugging difficulty  
**Fix**: Add comprehensive error handling with informative messages

#### Issue 4: No Feature Isolation
**Severity**: High  
**Impact**: Can't isolate linguistic feature contributions  
**Fix**: Add model wrapper functionality similar to `ModelWrapperForSHAP`

### 4.2 Medium Priority Enhancements

1. **Add Progress Tracking**: Use tqdm for long computations
2. **Add Configuration Support**: Allow customization via config dict
3. **Add Statistical Analysis**: Include std, confidence intervals, significance tests
4. **Add Interaction Analysis**: SHAP interaction values
5. **Add Category-based Analysis**: Analyze by prediction category
6. **Add Ablation Studies**: Feature removal experiments

### 4.3 Low Priority Improvements

1. **Better Documentation**: Docstrings with examples
2. **Type Hints**: Complete type annotations
3. **Unit Tests**: Test coverage
4. **Logging**: Structured logging instead of print statements
5. **Plot Customization**: More matplotlib customization options

---

## 5. Research-Specific Recommendations

### 5.1 For Your Research Context

Given your research focuses on:
- **Linguistic features** for scene saliency
- **Multiple model architectures** (BERT-base/large, RoBERTa, DeBERTa)
- **Feature fusion methods** (concat, attention, gated)
- **Sequence modeling**

**Recommended Enhancements:**

1. **Add Neural Model Support**
   ```python
   def create_explainer(
       self,
       X_background: np.ndarray,
       model_wrapper: Optional[Callable] = None,  # For neural models
       use_deep_explainer: bool = True,  # Use DeepExplainer for neural
   ):
       if self.model_type == "neural" and use_deep_explainer:
           if model_wrapper is None:
               raise ValueError("model_wrapper required for neural models")
           self.explainer = shap.DeepExplainer(model_wrapper, X_background)
       # ... rest
   ```

2. **Add Feature Group Analysis**
   ```python
   def analyze_feature_groups(
       self,
       feature_groups: Dict[str, List[str]],  # e.g., {"syntax": [...], "semantics": [...]}
   ):
       """Analyze SHAP contributions by feature group."""
       group_importance = {}
       for group_name, features in feature_groups.items:
           feature_indices = [self.feature_names.index(f) for f in features if f in self.feature_names]
           group_shap = self.shap_values[:, feature_indices]
           group_importance[group_name] = np.abs(group_shap).mean()
       return group_importance
   ```

3. **Add Fusion Method Comparison**
   ```python
   def compare_fusion_methods(
       self,
       shap_results_concat: np.ndarray,
       shap_results_attention: np.ndarray,
       shap_results_gated: np.ndarray,
   ):
       """Compare SHAP values across fusion methods."""
       # Statistical comparison
   ```

### 5.2 Integration with Experiment Pipeline

**Current State**: `shap_analysis.py` is not integrated with `experiment_config.py` or the main training pipeline.

**Recommendation**: Create a unified SHAP analysis interface:

```python
# In shap_analysis.py
def analyze_experiment_shap(
    experiment_name: str,
    model_path: str,
    test_data_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
):
    """Analyze SHAP for a specific experiment configuration."""
    from experiment_config import get_experiment_config
    
    exp_config = get_experiment_config(experiment_name)
    # Load model, data, features
    # Run SHAP analysis
    # Generate report
```

---

## 6. Code Quality Assessment

### 6.1 Strengths

1. **Clean Class Design**: Well-structured OOP approach
2. **Separation of Concerns**: Each method has a single responsibility
3. **Flexible Interface**: Easy to use for basic cases
4. **Good Defaults**: Sensible parameter defaults

### 6.2 Weaknesses

1. **Limited Model Support**: Only SVM fully supported
2. **Incomplete Error Handling**: Many edge cases not handled
3. **Hardcoded Values**: Sample sizes, background sizes hardcoded
4. **Limited Analysis Depth**: Basic analysis only
5. **No Integration**: Doesn't integrate with main research pipeline

### 6.3 Code Metrics

- **Lines of Code**: ~268 (reasonable size)
- **Cyclomatic Complexity**: Low (simple control flow)
- **Test Coverage**: Unknown (no tests visible)
- **Documentation**: Basic (docstrings present but could be more detailed)

---

## 7. Recommendations Summary

### 7.1 Immediate Actions (High Priority)

1. **Fix Llama Support**: Either implement or raise NotImplementedError
2. **Add Neural Model Support**: Critical for your research
3. **Add Error Handling**: Comprehensive try-except blocks
4. **Add Feature Isolation**: Model wrapper for linguistic features

### 7.2 Short-term Enhancements (Medium Priority)

1. **Add Progress Tracking**: tqdm integration
2. **Add Configuration Support**: Config dict parameter
3. **Enhance Statistical Analysis**: Std, CI, significance tests
4. **Add Shape Validation**: Comprehensive input validation

### 7.3 Long-term Improvements (Low Priority)

1. **Comprehensive Documentation**: Examples, tutorials
2. **Unit Tests**: Test coverage
3. **Integration**: Connect with experiment_config.py
4. **Advanced Analysis**: Interactions, categories, ablation

### 7.4 Research-Specific Enhancements

1. **Feature Group Analysis**: Analyze by linguistic feature category
2. **Fusion Method Comparison**: Compare concat vs attention vs gated
3. **Model Architecture Comparison**: Compare BERT-base vs large vs RoBERTa
4. **Sequence Model Analysis**: Analyze sequence model contributions

---

## 8. Conclusion

The `shap_analysis.py` module provides a **solid foundation** for SHAP-based interpretability analysis, but has **significant gaps** for your research needs:

**Strengths:**
- Clean, simple interface
- Good for basic SVM analysis
- Extensible design

**Critical Gaps:**
- No neural model support (your main research focus)
- No feature isolation (needed for linguistic feature analysis)
- Limited analysis depth (compared to your integrated implementation)

**Recommendation:**
1. **For immediate use**: Enhance `shap_analysis.py` with neural model support and feature isolation
2. **For research**: Continue using the sophisticated implementation in `train_sequence.py`
3. **For future**: Consider refactoring `train_sequence.py` SHAP code into a reusable module that combines the best of both

The module shows good software engineering practices but needs research-specific enhancements to be fully useful for your scene saliency classification research.

---

## Appendix: Code Examples

### Example 1: Enhanced Neural Model Support

```python
class SHAPAnalyzer:
    def create_explainer(
        self,
        X_background: np.ndarray,
        model_wrapper: Optional[Callable] = None,
        use_deep_explainer: bool = True,
    ):
        if self.model_type == "neural":
            if model_wrapper is None:
                raise ValueError("model_wrapper required for neural models")
            if use_deep_explainer:
                self.explainer = shap.DeepExplainer(
                    model_wrapper,
                    X_background[:min(100, len(X_background))],
                )
            else:
                # Fallback to KernelExplainer
                def predict_fn(X):
                    return model_wrapper(torch.tensor(X, dtype=torch.float32))
                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    X_background[:min(100, len(X_background))],
                )
        # ... rest
```

### Example 2: Feature Group Analysis

```python
def analyze_feature_groups(
    self,
    feature_groups: Dict[str, List[str]],
) -> pd.DataFrame:
    """Analyze SHAP contributions by feature group."""
    if self.shap_values is None:
        raise ValueError("SHAP values not computed.")
    
    results = []
    for group_name, features in feature_groups.items():
        # Find feature indices
        feature_indices = [
            i for i, name in enumerate(self.feature_names)
            if name in features
        ]
        if not feature_indices:
            continue
        
        # Compute group importance
        group_shap = self.shap_values[:, feature_indices]
        mean_abs = np.abs(group_shap).mean()
        std_abs = np.abs(group_shap).std()
        
        results.append({
            "group": group_name,
            "num_features": len(feature_indices),
            "mean_abs_shap": mean_abs,
            "std_abs_shap": std_abs,
        })
    
    return pd.DataFrame(results).sort_values("mean_abs_shap", ascending=False)
```

---

**End of Analysis Report**

