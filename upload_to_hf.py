"""
Upload NADN cached embeddings + checkpoints to HuggingFace Hub.

Usage:
  # First time: login
  huggingface-cli login

  # Upload everything
  python upload_to_hf.py

  # Upload only embeddings (skip checkpoints)
  python upload_to_hf.py --only-embeddings
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo

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

    api = HfApi()

    # Create repo (dataset type — no model card needed)
    try:
        create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"Repo creation: {e} (may already exist)")

    cache_dir = os.path.join(args.base_dir, "cache", "scene_embeddings")
    logs_dir = os.path.join(args.base_dir, "logs")

    # ── Upload scene embeddings ──
    scene_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt") and "_summary" not in f]
    summary_files = [f for f in os.listdir(cache_dir) if f.endswith("_summary.pt")]
    print(f"\nScene embeddings: {len(scene_files)}")
    print(f"Summary embeddings: {len(summary_files)}")
    print(f"Total files: {len(scene_files) + len(summary_files)}")

    # Upload all embeddings in ONE commit
    print(f"\nUploading embeddings from {cache_dir}...")
    print("  (Single commit — may take a few minutes, be patient)")
    api.upload_folder(
        folder_path=cache_dir,
        path_in_repo="cache/scene_embeddings",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Upload all scene + summary embeddings",
        ignore_patterns=["*.tmp"],
    )
    print(f"  ✓ Uploaded {len(scene_files) + len(summary_files)} embedding files")

    # ── Upload checkpoints (optional) ──
    if not args.only_embeddings and os.path.exists(logs_dir):
        ckpt_files = [f for f in os.listdir(logs_dir) if f.endswith(".pt")]
        if ckpt_files:
            print(f"\nUploading checkpoints: {ckpt_files}")
            for f in ckpt_files:
                fpath = os.path.join(logs_dir, f)
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"logs/{f}",
                    repo_id=args.repo,
                    repo_type="dataset",
                )
            print(f"  ✓ Uploaded {len(ckpt_files)} checkpoints")

    print(f"\nDone! Access at: https://huggingface.co/datasets/{args.repo}")
    print(f"Pull with:  python download_from_hf.py")


if __name__ == "__main__":
    main()
