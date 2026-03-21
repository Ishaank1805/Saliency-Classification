#!/bin/bash
set -e

BASE="/scratch/ishaan.karan/hnsd"

echo "================================================"
echo "  NADN: Narrative Arc Decomposition Network"
echo "  Base dir: $BASE"
echo "================================================"

pip install -q -r requirements.txt

mkdir -p "$BASE/logs" "$BASE/cache/scene_embeddings" "$BASE/figures" "$BASE/hf_cache"
export HF_HOME="$BASE/hf_cache"
export HF_DATASETS_CACHE="$BASE/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

# Encode scenes + summaries (skips if already cached)
echo "[1/4] Encoding scenes + summaries..."
python encode_all.py

# Train
echo "[2/4] Training NADN..."
python train.py --mode train 2>&1 | tee "$BASE/logs/training.log"

# Test (already done in train, but explicit)
echo "[3/4] Test results in $BASE/logs/test_results.json"

# Figures
echo "[4/4] Generating figures..."
python visualize.py 2>/dev/null || echo "Visualize skipped (update for NADN if needed)"

echo ""
echo "Done. Results: $BASE/logs/test_results.json"
cat "$BASE/logs/test_results.json"
