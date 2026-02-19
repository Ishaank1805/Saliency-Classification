# Hugging Face Features Integration Update

## Summary

Updated `train_sequence.py` to support loading features directly from Hugging Face datasets (`Ishaank18/screenplay-features`) instead of requiring local parquet files.

## Changes Made

### 1. New Function: `load_features_from_huggingface()`
- Loads and merges multiple feature groups from Hugging Face
- Handles merging on `movie_id` and `scene_index`
- Automatically removes duplicate columns
- Provides detailed logging of loaded features

### 2. Updated `MovieDataset` Class
- Added new parameters:
  - `use_huggingface`: Boolean to enable Hugging Face loading
  - `hf_repo`: Hugging Face repository name (default: "Ishaank18/screenplay-features")
  - `feature_groups`: List of feature groups to load (e.g., ["base", "gc_polarity", "rst"])
- Maintains backward compatibility with local file loading

### 3. Updated Command-Line Arguments
- `--use-huggingface`: Enable loading from Hugging Face
- `--hf-repo`: Specify Hugging Face repository (default: "Ishaank18/screenplay-features")
- `--feature-groups`: List of feature groups to load
- `--train-path`, `--val-path`, `--test-path`: Now optional when using Hugging Face

## Usage Examples

### Example 1: Load from Hugging Face with specific feature groups

```bash
python train_sequence.py \
    --use-huggingface \
    --feature-groups base gc_polarity rst emotional gc_discourse \
    --experiment bert_base_gated \
    --epochs 25 \
    --batch-size 4
```

### Example 2: Load from local files (original behavior)

```bash
python train_sequence.py \
    --train-path /path/to/train.parquet \
    --val-path /path/to/val.parquet \
    --test-path /path/to/test.parquet \
    --experiment bert_base_gated \
    --epochs 25
```

### Example 3: Load all available feature groups

```bash
python train_sequence.py \
    --use-huggingface \
    --feature-groups base bert_surprisal character_arcs emotional \
                     gc_academic gc_basic gc_char_diversity gc_concreteness \
                     gc_dialogue gc_discourse gc_narrative gc_polarity \
                     gc_pos gc_pronouns gc_punctuation gc_readability \
                     gc_syntax gc_temporal ngram ngram_surprisal \
                     plot_shifts rst structure surprisal \
    --experiment bert_base_gated \
    --epochs 25
```

## Available Feature Groups

The following 24 feature groups are available in the Hugging Face repository:

1. `base` - Basic features
2. `bert_surprisal` - BERT surprisal features
3. `character_arcs` - Character arc features
4. `emotional` - Emotional features
5. `gc_academic` - Academic writing features
6. `gc_basic` - Basic GC features
7. `gc_char_diversity` - Character diversity features
8. `gc_concreteness` - Concreteness features
9. `gc_dialogue` - Dialogue features
10. `gc_discourse` - Discourse features
11. `gc_narrative` - Narrative features
12. `gc_polarity` - Polarity features
13. `gc_pos` - Part-of-speech features
14. `gc_pronouns` - Pronoun features
15. `gc_punctuation` - Punctuation features
16. `gc_readability` - Readability features
17. `gc_syntax` - Syntax features
18. `gc_temporal` - Temporal features
19. `ngram` - N-gram features
20. `ngram_surprisal` - N-gram surprisal features
21. `plot_shifts` - Plot shift features
22. `rst` - Rhetorical Structure Theory features
23. `structure` - Structure features
24. `surprisal` - Surprisal features

## Requirements

- `datasets` library: `pip install datasets`
- Hugging Face account (for private repositories, if applicable)

## Notes

- When using Hugging Face, `--train-path`, `--val-path`, and `--test-path` are not required
- Feature groups are automatically merged on `movie_id` and `scene_index`
- The `base` group should typically be included as it contains essential metadata
- All other functionality (TextRank, top features, SHAP analysis) works the same way

## Backward Compatibility

The update maintains full backward compatibility. Existing scripts using local parquet files will continue to work without modification.

