#!/bin/bash

# Script to run all ablation experiments
# This will run all 40+ experiments sequentially

set -e  # Exit on error

# Configuration
TRAIN_PATH="${1:-data/train.parquet}"
VAL_PATH="${2:-data/validation.parquet}"
TEST_PATH="${3:-data/test.parquet}"
OUTPUT_DIR="${4:-./experiments}"
EPOCHS="${5:-25}"
BATCH_SIZE="${6:-4}"
LR="${7:-8e-5}"

# Memory optimizations (recommended for large models)
FP16_FLAG="--fp16"
SMALL_BATCH_FLAG="--small-batch-large-models"

echo "=================================================================================="
echo "RUNNING ALL ABLATION EXPERIMENTS"
echo "=================================================================================="
echo "Train path: $TRAIN_PATH"
echo "Val path: $VAL_PATH"
echo "Test path: $TEST_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "=================================================================================="
echo ""

# Check if data files exist
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: Training file not found: $TRAIN_PATH"
    exit 1
fi

if [ ! -f "$VAL_PATH" ]; then
    echo "ERROR: Validation file not found: $VAL_PATH"
    exit 1
fi

if [ ! -f "$TEST_PATH" ]; then
    echo "ERROR: Test file not found: $TEST_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for pre-computed embeddings
EMBEDDINGS_CACHE="${OUTPUT_DIR}/../embeddings_cache"
if [ -d "$EMBEDDINGS_CACHE" ]; then
    echo "Found embeddings cache directory: $EMBEDDINGS_CACHE"
    EMBEDDINGS_FLAG="--embeddings-cache-dir $EMBEDDINGS_CACHE"
else
    echo "No embeddings cache found. Embeddings will be computed on-the-fly."
    echo "To speed up training, pre-compute embeddings first:"
    echo "  python precompute_embeddings.py --model-name roberta-large ..."
    EMBEDDINGS_FLAG=""
fi
echo ""

# Start time
START_TIME=$(date +%s)

# Run all experiments
echo "Running all experiments from experiment_config.py..."
echo "This will run 40+ experiments sequentially."
echo ""

python train_sequence.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    --experiment-group all \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --output-dir "$OUTPUT_DIR" \
    $FP16_FLAG \
    $SMALL_BATCH_FLAG \
    $EMBEDDINGS_FLAG

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=================================================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=================================================================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Compare results:"
echo "   python compare_experiments.py --experiments-dir $OUTPUT_DIR"
echo ""
echo "2. Generate detailed analysis:"
echo "   python detailed_report.py --experiments-dir $OUTPUT_DIR --output-dir ./analysis --generate-plots"
echo "=================================================================================="

