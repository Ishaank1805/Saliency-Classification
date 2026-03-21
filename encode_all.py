"""
Pre-encode all MENSA scenes AND summaries using all available GPUs.
Summaries are encoded once per movie and cached separately.

Usage: python encode_all.py
"""
import os
import time
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from config import HNSDConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def encode_worker(gpu_id, movie_shards, config):
    device = torch.device(f"cuda:{gpu_id}")
    cache_dir = config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[GPU {gpu_id}] Loading {config.encoder_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_name, use_fast=True, trust_remote_code=True,
        cache_dir=config.hf_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModel.from_pretrained(
        config.encoder_name,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=config.hf_cache_dir,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def encode_text(text):
        """Encode a single text → (1, d) tensor."""
        inputs = tokenizer(
            text, truncation=True,
            max_length=config.max_scene_tokens,
            return_tensors="pt",
        ).to(device)
        if inputs["input_ids"].shape[1] == 0:
            return torch.zeros(1, config.scene_embed_dim)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
            hidden = outputs.last_hidden_state.float()
            return hidden.mean(dim=1).cpu()

    print(f"[GPU {gpu_id}] Encoding {len(movie_shards)} movies...")
    t0 = time.time()

    for idx, (movie_id, scenes, summary) in enumerate(movie_shards):
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)
        scene_path = os.path.join(cache_dir, f"{safe_id}.pt")
        summary_path = os.path.join(cache_dir, f"{safe_id}_summary.pt")

        # Encode scenes (skip if cached)
        if not os.path.exists(scene_path):
            embeddings = []
            for text in scenes:
                if not isinstance(text, str) or len(text.strip()) == 0:
                    embeddings.append(torch.zeros(1, config.scene_embed_dim))
                    continue
                embeddings.append(encode_text(text))
            result = torch.cat(embeddings, dim=0)
            torch.save(result, scene_path)

        # Encode summary (skip if cached)
        if not os.path.exists(summary_path):
            if summary and isinstance(summary, str) and len(summary.strip()) > 0:
                # Summary can be long — encode in chunks and mean pool
                words = summary.split()
                chunk_size = 800  # words per chunk
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    chunks.append(encode_text(chunk_text))
                summary_emb = torch.cat(chunks, dim=0).mean(dim=0, keepdim=True)  # (1, d)
            else:
                summary_emb = torch.zeros(1, config.scene_embed_dim)
            torch.save(summary_emb, summary_path)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60
            print(f"[GPU {gpu_id}] {idx+1}/{len(movie_shards)} ({rate:.0f} movies/min)", flush=True)

    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] Done. {len(movie_shards)} movies in {elapsed/60:.1f} min")


def main():
    config = HNSDConfig()
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.hf_cache_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    print("\nLoading MENSA dataset...")
    all_movies = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset(config.dataset_name, split=split, cache_dir=config.hf_cache_dir)
        for row in ds:
            movie_id = row["name"]
            scenes = row["scenes"][:config.max_scenes_per_movie]
            summary = row.get("summary", "")
            all_movies.append((movie_id, scenes, summary))
    print(f"Total movies: {len(all_movies)}")

    # Count cached
    cached_scenes = 0
    cached_summaries = 0
    for movie_id, _, _ in all_movies:
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)
        if os.path.exists(os.path.join(config.cache_dir, f"{safe_id}.pt")):
            cached_scenes += 1
        if os.path.exists(os.path.join(config.cache_dir, f"{safe_id}_summary.pt")):
            cached_summaries += 1
    print(f"Cached: {cached_scenes} scenes, {cached_summaries} summaries out of {len(all_movies)}")

    if cached_scenes == len(all_movies) and cached_summaries == len(all_movies):
        print("All movies already encoded. Skipping.")
        return

    # Shard across GPUs
    shards = [[] for _ in range(n_gpus)]
    for i, movie in enumerate(all_movies):
        shards[i % n_gpus].append(movie)

    print(f"\nLaunching {n_gpus} encoding workers...")
    t0 = time.time()

    if n_gpus == 1:
        encode_worker(0, shards[0], config)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for gpu_id in range(n_gpus):
            p = mp.Process(target=encode_worker, args=(gpu_id, shards[gpu_id], config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                raise RuntimeError(f"Encoding failed on GPU {i}")

    elapsed = time.time() - t0
    print(f"\nAll encoding done in {elapsed/60:.1f} min")

    # Verify
    missing = []
    for movie_id, _, _ in all_movies:
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)
        if not os.path.exists(os.path.join(config.cache_dir, f"{safe_id}.pt")):
            missing.append(f"{movie_id} (scenes)")
        if not os.path.exists(os.path.join(config.cache_dir, f"{safe_id}_summary.pt")):
            missing.append(f"{movie_id} (summary)")
    if missing:
        print(f"WARNING: {len(missing)} missing: {missing[:5]}...")
    else:
        print(f"Verified: all {len(all_movies)} movies + summaries cached")


if __name__ == "__main__":
    main()
