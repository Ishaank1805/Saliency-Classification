#!/bin/bash

# Pre-compute embeddings for all models that need it
# This will pre-compute for: roberta-large, bert-base-uncased, bert-large-uncased, roberta-base

set -e

TRAIN_PATH="${1:-data/train.parquet}"
VAL_PATH="${2:-data/validation.parquet}"
TEST_PATH="${3:-data/test.parquet}"
OUTPUT_DIR="${4:-./embeddings_cache}"

MODELS=(
    "roberta-large"
    "bert-base-uncased"
    "bert-large-uncased"
    "roberta-base"
)

echo "=================================================================================="
echo "PRE-COMPUTING EMBEDDINGS FOR ALL MODELS"
echo "=================================================================================="
echo "Train path: $TRAIN_PATH"
echo "Val path: $VAL_PATH"
echo "Test path: $TEST_PATH"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Models to pre-compute:"
for MODEL in "${MODELS[@]}"; do
    echo "  - $MODEL"
done
echo ""
echo "This will take ~30-50 minutes total."
echo "=================================================================================="
echo ""

START_TIME=$(date +%s)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))
    TOTAL=${#MODELS[@]}
    
    echo ""
    echo "=================================================================================="
    echo "[$NUM/$TOTAL] Pre-computing embeddings for: $MODEL"
    echo "=================================================================================="
    
    python precompute_embeddings.py \
        --model-name "$MODEL" \
        --train-path "$TRAIN_PATH" \
        --val-path "$VAL_PATH" \
        --test-path "$TEST_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 16
    
    echo ""
    echo "✓ Completed: $MODEL"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=================================================================================="
echo "ALL EMBEDDINGS PRE-COMPUTED SUCCESSFULLY!"
echo "=================================================================================="
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.pkl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "You can now run experiments with:"
echo "  python train_sequence.py --embeddings-cache-dir $OUTPUT_DIR ..."
echo "=================================================================================="

