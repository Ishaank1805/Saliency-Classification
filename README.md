# MENSA Scene Saliency Classification

A comprehensive machine learning pipeline for classifying screenplay scene saliency using the MENSA dataset. This repository implements PCA ensemble methods and RFECV-based feature selection with extensive linguistic and narrative features.

## Overview

This project tackles the binary classification problem of identifying salient (memorable/important) scenes in movie screenplays. It extracts diverse linguistic, narrative, emotional, and structural features from screenplay text and employs advanced ensemble methods to achieve state-of-the-art performance.

## Features

### Feature Groups (30+ feature extractors)

The pipeline supports 30+ feature extraction groups organized into categories:

**Core Features:**
- `base`: Basic scene-level features (word count, sentence count, etc.)
- `structure`: Structural features (scene position, act structure)
- `character_arcs`: Character presence and interaction patterns
- `emotional`: Emotional trajectory features
- `plot_shifts`: Narrative shift detection

**Linguistic Complexity (GC Features):**
- `gc_basic`: Basic text statistics
- `gc_academic`: Academic vocabulary usage
- `gc_char_diversity`: Character and lexical diversity
- `gc_concreteness`: Concreteness ratings
- `gc_dialogue`: Dialogue-specific metrics
- `gc_discourse`: Discourse markers and connectives
- `gc_narrative`: Narrative tense and perspective
- `gc_polarity`: Sentiment and polarity
- `gc_pos`: Part-of-speech distributions
- `gc_pronouns`: Pronoun usage patterns
- `gc_punctuation`: Punctuation statistics
- `gc_readability`: Readability scores
- `gc_syntax`: Syntactic complexity
- `gc_temporal`: Temporal references

**Advanced NLP Features:**
- `ngram`: N-gram statistics
- `ngram_surprisal`: N-gram based surprisal scores
- `bert_surprisal`: BERT-based surprisal (requires GPU)
- `surprisal`: GPT-2 surprisal scores
- `rst`: Rhetorical Structure Theory features

All features are pre-extracted and cached on HuggingFace Hub for efficient loading.

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, for BERT/GPT features)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd lr_saliency
```

2. Create conda environment:
```bash
conda create -n saliency python=3.10
conda activate saliency
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download additional NLP models:
```bash
# Spacy model
python -m spacy download en_core_web_sm

# NLTK data (downloads automatically on first use)
# Stanza models (downloads automatically on first use)
```

## Data

The pipeline uses the MENSA dataset from HuggingFace:
- Dataset: `rohitsaxena/MENSA`
- Cached features: `Ishaank18/screenplay-features`

Features are automatically downloaded from HuggingFace Hub when running training scripts.

## Usage

### Training Scripts

The repository includes two main training approaches:

#### 1. PCA Ensemble (`train_pca_ensemble.py`)

Trains an ensemble of PCA-transformed feature groups with optimized voting or stacking.

**Key Parameters:**
- `--groups`: Feature groups to use
- `--method`: Ensemble method (`voting` or `stacking`)
- `--search`: Weight optimization (`dirichlet` or `simplex`)
- `--pca_n_components`: PCA variance threshold (0.0-1.0)
- `--oversample`: Enable SMOTE oversampling
- `--undersample`: Enable undersampling
- `--undersample_method`: Method (`cluster_random` or `random`)
- `--undersample_clusters`: Number of clusters for cluster-based undersampling
- `--prefer_gc_overlap`: Prioritize overlapping GC features
- `--report_features`: Print detailed feature importance

#### 2. RFECV Feature Selection (`train_lr_rfecv.py`)

Trains logistic regression with recursive feature elimination and cross-validation.

**Key Parameters:**
- `--groups`: Feature groups to use
- `--rfecv_step`: Feature elimination step size
- `--rfecv_cv`: Cross-validation folds
- `--rfecv_scoring`: Scoring metric for RFECV
- `--lr_C`: Logistic regression regularization
- `--oversample`, `--undersample`: Same as PCA ensemble

### Pre-configured Run Scripts

The repository includes 5 pre-configured SLURM scripts for different feature configurations:

#### 1. **Full Feature Set** (`run_lr.sh`)
Uses all 30 feature groups including BERT and GPT surprisal:
```bash
sbatch run_lr.sh
```

**Features:** base, bert_surprisal, character_arcs, emotional, all GC features, ngram, ngram_surprisal, plot_shifts, rst, structure, surprisal

**Use case:** Maximum performance with all available features

#### 2. **Base Features** (`run_lr_base.sh`)
Excludes neural surprisal features (BERT, GPT):
```bash
sbatch run_lr_base.sh
```

**Features:** base, character_arcs, emotional, all GC features, ngram, plot_shifts, rst, structure

**Use case:** Fast training without GPU-intensive surprisal computation

#### 3. **BERT Variant** (`run_lr_bert.sh`)
Includes BERT surprisal but excludes GPT surprisal:
```bash
sbatch run_lr_bert.sh
```

**Features:** base, bert_surprisal, character_arcs, emotional, all GC features, ngram, ngram_surprisal, plot_shifts, rst, structure

**Use case:** BERT-based surprisal analysis

#### 4. **GPT Variant** (`run_lr_gpt.sh`)
Includes GPT surprisal and n-gram surprisal but excludes BERT:
```bash
sbatch run_lr_gpt.sh
```

**Features:** base, character_arcs, emotional, all GC features, ngram, ngram_surprisal, plot_shifts, rst, structure, surprisal

**Use case:** GPT-based surprisal analysis

#### 5. **RFECV Feature Selection** (`run_lr_rfecv.sh`)
Full feature set with RFECV for optimal feature selection:
```bash
sbatch run_lr_rfecv.sh
```

**Features:** Same as full feature set

**Use case:** Automatic feature selection and model simplification

### Common Configuration

All scripts use these default settings:
- Oversampling: SMOTE (enabled)
- Undersampling: Cluster-based with 20 clusters (enabled)
- PCA variance threshold: 0.95
- Ensemble method: Voting with Dirichlet weight optimization
- Prefer GC overlap: Enabled
- Output: `/scratch/ishaan.karan/pca_ensemble.pkl` (or `lr_rfecv.pkl` for RFECV)

### Local Execution

For local execution or custom configurations:

```bash
# PCA Ensemble with custom groups
python train_pca_ensemble.py \
    --groups base gc_polarity emotional ngram \
    --method voting \
    --search dirichlet \
    --pca_n_components 0.95 \
    --oversample \
    --undersample \
    --undersample_method cluster_random \
    --undersample_clusters 20 \
    --output models/my_model.pkl

# RFECV with specific groups
python train_lr_rfecv.py \
    --groups base gc_polarity emotional \
    --oversample \
    --undersample \
    --output models/rfecv_model.pkl
```

## Project Structure

```
lr_saliency/
├── data.py                    # MENSA dataset loading
├── train_pca_ensemble.py      # PCA ensemble training
├── train_lr_rfecv.py          # RFECV feature selection training
├── pca_ensemble.py            # Ensemble weight optimization
├── pca_utils.py               # PCA utilities and threshold finding
├── requirements.txt           # Python dependencies
│
├── feature_cache/             # Feature loading infrastructure
│   ├── __init__.py
│   ├── load_hf.py            # HuggingFace feature loading
│   └── extract.py            # Feature extraction registry
│
├── features/                  # Feature extractors (30+ modules)
│   ├── base.py               # Basic features
│   ├── character_arcs.py     # Character features
│   ├── emotional.py          # Emotional features
│   ├── structure.py          # Structural features
│   ├── plot_shifts.py        # Plot shift detection
│   ├── bert_surprisal.py     # BERT surprisal
│   ├── surprisal.py          # GPT-2 surprisal
│   ├── ngram*.py             # N-gram features
│   ├── rst.py                # RST features
│   ├── gc_*.py               # 14 GC feature modules
│   └── ...
│
├── hf_feature_cache/          # Local HuggingFace cache
│
└── run_*.sh                   # SLURM job scripts (5 configs)
```

## Methodology

### Data Preprocessing

1. **Feature Loading**: Features loaded from HuggingFace Hub cache
2. **Feature Alignment**: Ensures consistent features across train/val/test splits
3. **Imputation**: Missing values filled with median strategy
4. **Class Balancing**:
   - SMOTE oversampling of minority class
   - Cluster-based undersampling of majority class (20 clusters)
5. **Standardization**: Features scaled with StandardScaler

### PCA Ensemble Method

1. **Group-wise PCA**: Each feature group transformed independently
2. **Variance Preservation**: Retains components explaining 95% variance
3. **Ensemble Training**: Individual classifiers per PCA-transformed group
4. **Weight Optimization**: 
   - Dirichlet sampling for voting weights
   - Bayesian optimization over weight simplex
5. **Threshold Tuning**: Optimal decision threshold found on validation set

### RFECV Method

1. **Recursive Elimination**: Iteratively removes least important features
2. **Cross-Validation**: 5-fold CV for robust feature ranking
3. **Scoring**: F1-macro score optimization
4. **Automatic Selection**: Determines optimal feature count

## Output

Both training scripts produce pickle files containing:
- Trained model(s)
- Feature importance/weights
- Selected features (RFECV)
- PCA transformers (PCA ensemble)
- Preprocessing pipeline
- Performance metrics

## Performance Metrics

The pipeline reports:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro-averaged F1 (primary metric)
- **Precision/Recall**: Per-class performance
- **ROC AUC**: Area under ROC curve
- **Confusion Matrix**: Classification breakdown


