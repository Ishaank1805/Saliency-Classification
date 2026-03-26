"""
Download NADN cached embeddings + checkpoints from HuggingFace Hub.

Usage:
  # Download everything to default location
  python download_from_hf.py

  # Download to custom location
  python download_from_hf.py --base-dir /my/path/hnsd

  # Download only embeddings (skip checkpoints)
  python download_from_hf.py --only-embeddings
"""
import os
import argparse
import shutil
from huggingface_hub import snapshot_download

# ══════════════════════════════════════════
#  CONFIGURE THESE
# ══════════════════════════════════════════
HF_REPO_ID = "Ishaank18/nadn-cache"  # change to your username
BASE_DIR = "/scratch/ishaan.karan/hnsd"
# ══════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-embeddings", action="store_true")
    parser.add_argument("--repo", type=str, default=HF_REPO_ID)
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    args = parser.parse_args()

    cache_dir = os.path.join(args.base_dir, "cache", "scene_embeddings")
    logs_dir = os.path.join(args.base_dir, "logs")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Check if already populated
    existing = [f for f in os.listdir(cache_dir) if f.endswith(".pt")] if os.path.exists(cache_dir) else []
    if len(existing) > 1800:  # 924 scenes + 924 summaries
        print(f"Cache already has {len(existing)} files. Skipping download.")
        print(f"Delete {cache_dir} to force re-download.")
        return

    print(f"Downloading from: https://huggingface.co/datasets/{args.repo}")
    print(f"Destination: {args.base_dir}")

    if args.only_embeddings:
        allow = ["cache/scene_embeddings/*"]
    else:
        allow = ["cache/scene_embeddings/*", "logs/*.pt"]

    # Download snapshot — HF handles caching/resumption
    downloaded_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=args.base_dir,
    )

    # Verify
    n_files = len([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
    print(f"\n✓ Downloaded {n_files} files to {cache_dir}")

    if not args.only_embeddings and os.path.exists(logs_dir):
        ckpts = [f for f in os.listdir(logs_dir) if f.endswith(".pt")]
        print(f"✓ Checkpoints: {ckpts}")

    print("\nReady to train: python train.py --mode train")


if __name__ == "__main__":
    main()
