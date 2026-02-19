#!/bin/bash

# Script to run experiments by priority (recommended approach)
# Runs most important experiments first, then expands

set -e

# Configuration
TRAIN_PATH="${1:-data/train.parquet}"
VAL_PATH="${2:-data/validation.parquet}"
TEST_PATH="${3:-data/test.parquet}"
OUTPUT_DIR="${4:-./experiments}"
EPOCHS="${5:-25}"

echo "=================================================================================="
echo "RUNNING EXPERIMENTS BY PRIORITY"
echo "=================================================================================="

# Phase 1: Fair Comparison (MOST IMPORTANT - 3 experiments)
echo ""
echo "PHASE 1: Fair Comparison (3 experiments)"
echo "Comparing RoBERTa-large with/without linguistic features"
echo "=================================================================================="
python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group core_fair_comparison \
    --epochs "$EPOCHS" \
    --fp16 \
    --small-batch-large-models \
    --output-dir "$OUTPUT_DIR"

# Phase 2: Model Size Comparison (4 experiments)
echo ""
echo "PHASE 2: Model Size Comparison (4 experiments)"
echo "Comparing BERT-base, BERT-large, RoBERTa-base, RoBERTa-large"
echo "=================================================================================="
python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group model_size_comparison \
    --epochs "$EPOCHS" \
    --fp16 \
    --small-batch-large-models \
    --output-dir "$OUTPUT_DIR"

# Phase 3: Finetuning Impact (6 experiments)
echo ""
echo "PHASE 3: Finetuning Impact (6 experiments)"
echo "Comparing frozen vs finetuned encoders"
echo "=================================================================================="
python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group finetuning_impact \
    --epochs "$EPOCHS" \
    --fp16 \
    --small-batch-large-models \
    --output-dir "$OUTPUT_DIR"

# Phase 4: Fusion Methods (6 experiments)
echo ""
echo "PHASE 4: Fusion Methods (6 experiments)"
echo "Comparing concatenation vs attention fusion"
echo "=================================================================================="
python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group fusion_methods \
    --epochs "$EPOCHS" \
    --fp16 \
    --small-batch-large-models \
    --output-dir "$OUTPUT_DIR"

# Phase 5: All RoBERTa-large Experiments (9 experiments)
echo ""
echo "PHASE 5: All RoBERTa-large Experiments (9 experiments)"
echo "Comprehensive analysis with fair comparison"
echo "=================================================================================="
python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group roberta_large_all \
    --epochs "$EPOCHS" \
    --fp16 \
    --small-batch-large-models \
    --output-dir "$OUTPUT_DIR"

# Phase 6: Architecture Ablations (optional - can skip if time limited)
echo ""
echo "PHASE 6: Architecture Ablations (9 experiments)"
echo "Transformer layers, attention heads, positional encoding, classifiers"
echo "=================================================================================="
read -p "Run architecture ablations? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_sequence.py \
        --train-path "$TRAIN_PATH" \
        --val-path "$VAL_PATH" \
        --test-path "$TEST_PATH" \
        --experiment-group transformer_layers \
        --epochs "$EPOCHS" \
        --output-dir "$OUTPUT_DIR"
    
    python train_sequence.py \
        --train-path "$TRAIN_PATH" \
        --val-path "$VAL_PATH" \
        --test-path "$TEST_PATH" \
        --experiment-group attention_heads \
        --epochs "$EPOCHS" \
        --output-dir "$OUTPUT_DIR"
    
    python train_sequence.py \
        --train-path "$TRAIN_PATH" \
        --val-path "$VAL_PATH" \
        --test-path "$TEST_PATH" \
        --experiment-group positional_encoding \
        --epochs "$EPOCHS" \
        --output-dir "$OUTPUT_DIR"
    
    python train_sequence.py \
        --train-path "$TRAIN_PATH" \
        --val-path "$VAL_PATH" \
        --test-path "$TEST_PATH" \
        --experiment-group classifier \
        --epochs "$EPOCHS" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=================================================================================="
echo "EXPERIMENTS COMPLETED!"
echo "=================================================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generate reports:"
echo "  python detailed_report.py --experiments-dir $OUTPUT_DIR --output-dir ./analysis --generate-plots"
echo "=================================================================================="

