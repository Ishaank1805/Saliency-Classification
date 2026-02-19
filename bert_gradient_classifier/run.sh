#!/bin/bash

# BERT Gradient Classifier - Quick Run Script

set -e

echo "=========================================="
echo "BERT Gradient Classifier - Training"
echo "=========================================="

# Configuration
FEATURE_DIR="../LLM's/features"
DATA_DIR="data"
MODELS_DIR="models"
BERT_MODEL="bert-base-uncased"

# Step 1: Prepare data (if needed)
echo ""
echo "Step 1: Preparing data..."
python quick_start.py \
    --feature-dir "$FEATURE_DIR" \
    --output-dir "$DATA_DIR" \
    --text-col "scene_text" \
    --label-col "label"

# Step 2: Train with all features
echo ""
echo "Step 2: Training with all features..."
python train.py \
    --train-path "$DATA_DIR/train.parquet" \
    --val-path "$DATA_DIR/validation.parquet" \
    --test-path "$DATA_DIR/test.parquet" \
    --use-gradients \
    --use-activations \
    --use-linguistic \
    --use-svm \
    --bert-model "$BERT_MODEL" \
    --load-texts-from-mensa \
    --output-dir "$MODELS_DIR"

# Step 3: Compare all configurations
echo ""
echo "Step 3: Comparing all configurations..."
python evaluate.py \
    --train-path "$DATA_DIR/train.parquet" \
    --val-path "$DATA_DIR/validation.parquet" \
    --test-path "$DATA_DIR/test.parquet" \
    --compare-all \
    --output-dir "comparison_results"

# Step 4: Generate report
echo ""
echo "Step 4: Generating analysis report..."
python report_generator.py \
    --results-dir "$MODELS_DIR" \
    --shap-dir "$MODELS_DIR/shap_svm" \
    --comparison-dir "comparison_results" \
    --output "analysis_report.md"

echo ""
echo "=========================================="
echo "Training complete! Check:"
echo "  - Models: $MODELS_DIR/"
echo "  - Comparison: comparison_results/"
echo "  - Report: analysis_report.md"
echo "=========================================="

