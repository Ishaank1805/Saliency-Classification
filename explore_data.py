"""
Step 0: Explore the MENSA dataset structure.
Run this FIRST to understand the schema before training.
"""
from datasets import load_dataset


def explore():
    print("=" * 60)
    print("Loading MENSA dataset...")
    print("=" * 60)

    from config import HNSDConfig
    config = HNSDConfig()
    ds = load_dataset("rohitsaxena/MENSA", cache_dir=config.hf_cache_dir)

    # ── Splits ──
    print(f"\nSplits: {list(ds.keys())}")
    for split_name, split_data in ds.items():
        print(f"\n{'─' * 40}")
        print(f"Split: {split_name}")
        print(f"  Rows:    {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")
        print(f"  Features:{split_data.features}")

    # ── Inspect a few rows ──
    first_split = list(ds.keys())[0]
    data = ds[first_split]

    print(f"\n{'=' * 60}")
    print(f"First 3 rows from '{first_split}':")
    print("=" * 60)
    for i in range(min(3, len(data))):
        row = data[i]
        print(f"\n--- Row {i} ---")
        for col, val in row.items():
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {col}: {val_str}")

    # ── Check if data is per-scene or per-movie ──
    print(f"\n{'=' * 60}")
    print("Checking data granularity...")
    print("=" * 60)

    # Look for movie-id-like columns
    cols = data.column_names
    potential_movie_cols = [c for c in cols if any(k in c.lower() for k in ["movie", "film", "id", "title", "name"])]
    print(f"  Potential movie ID columns: {potential_movie_cols}")

    potential_scene_cols = [c for c in cols if any(k in c.lower() for k in ["scene", "text", "content", "script", "dialogue"])]
    print(f"  Potential scene/text columns: {potential_scene_cols}")

    potential_label_cols = [c for c in cols if any(k in c.lower() for k in ["label", "salient", "saliency", "target", "gold"])]
    print(f"  Potential label columns: {potential_label_cols}")

    # ── If per-movie, check nested structure ──
    print(f"\n{'=' * 60}")
    print("Checking for nested/list columns...")
    print("=" * 60)
    row0 = data[0]
    for col, val in row0.items():
        if isinstance(val, list):
            print(f"  {col}: list of length {len(val)}, first element type: {type(val[0]).__name__}")
            if len(val) > 0:
                elem = val[0]
                if isinstance(elem, dict):
                    print(f"    Keys: {list(elem.keys())}")
                    for k, v in elem.items():
                        v_str = str(v)[:100]
                        print(f"      {k}: {v_str}")
                elif isinstance(elem, str):
                    print(f"    First element: {str(elem)[:150]}...")
                else:
                    print(f"    First element: {elem}")
        elif isinstance(val, dict):
            print(f"  {col}: dict with keys {list(val.keys())}")

    # ── Label distribution ──
    print(f"\n{'=' * 60}")
    print("Label distribution check...")
    print("=" * 60)
    for split_name, split_data in ds.items():
        for col in split_data.column_names:
            vals = split_data[col]
            if isinstance(vals[0], list) and len(vals[0]) > 0 and isinstance(vals[0][0], (int, float, bool)):
                # Nested labels (per-movie, with scene-level labels)
                flat = [v for movie_vals in vals for v in movie_vals]
                if all(v in (0, 1, True, False) for v in flat[:1000]):
                    n_pos = sum(1 for v in flat if v)
                    print(f"  {split_name}/{col}: {n_pos}/{len(flat)} positive ({100*n_pos/len(flat):.1f}%)")
            elif isinstance(vals[0], (int, float, bool)):
                unique = set(vals[:1000])
                if unique.issubset({0, 1, True, False}):
                    n_pos = sum(1 for v in vals if v)
                    print(f"  {split_name}/{col}: {n_pos}/{len(vals)} positive ({100*n_pos/len(vals):.1f}%)")


if __name__ == "__main__":
    explore()
