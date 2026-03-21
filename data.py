"""
HNSD Data Module
Loads MENSA, encodes scenes, builds movie-level batches.

IMPORTANT: Run explore_data.py first and adjust COLUMN MAPPINGS
below based on the actual dataset schema.
"""
import os
import re
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import HNSDConfig


# ═══════════════════════════════════════════════════════════════
#  COLUMN MAPPINGS — verified from explore_data.py output
# ═══════════════════════════════════════════════════════════════
MOVIE_ID_COL = "name"        # e.g. "The_Ides_of_March_(film)"
SCENE_TEXT_COL = "scenes"    # list of scene text strings per movie
SCENE_LABEL_COL = "labels"   # list of 0.0/1.0 floats per movie
IS_NESTED = True             # one row = one movie
# ═══════════════════════════════════════════════════════════════


def extract_characters_from_scene(text: str) -> List[str]:
    """
    Extract character names from screenplay formatting.
    Screenplay convention: character names appear in ALL CAPS on their own line
    before dialogue.
    """
    lines = text.strip().split("\n")
    characters = set()
    for line in lines:
        stripped = line.strip()
        # Character cue: all caps, possibly with (V.O.) or (O.S.) or (CONT'D)
        if stripped and stripped == stripped.upper() and len(stripped.split()) <= 4:
            # Remove parentheticals
            name = re.sub(r"\(.*?\)", "", stripped).strip()
            if name and len(name) > 1 and not name.startswith("INT") and not name.startswith("EXT"):
                characters.add(name)
    return list(characters)


class MENSAMovieDataset(Dataset):
    """
    Each item is one movie: a dict of scene texts, labels,
    character lists, and precomputed pseudo-labels.
    """

    def __init__(
        self,
        split: str,
        config: HNSDConfig,
        scene_encoder: Optional[object] = None,
        tokenizer: Optional[object] = None,
    ):
        self.config = config
        self.split = split

        # ── Load raw data ──
        ds = load_dataset(config.dataset_name, split=split, cache_dir=config.hf_cache_dir)
        self.movies = self._parse_movies(ds)

        # ── Precompute per-movie metadata ──
        for movie in self.movies:
            movie["characters_per_scene"] = [
                extract_characters_from_scene(s) for s in movie["scenes"]
            ]
            movie["char_presence"] = self._build_character_presence(movie)
            movie["causal_pseudo_labels"] = self._build_causal_pseudolabels(movie)

    def _parse_movies(self, ds) -> List[Dict]:
        """Parse HF dataset into list of movie dicts."""
        if IS_NESTED:
            # One row per movie, lists of scenes
            movies = []
            for row in ds:
                scenes = row[SCENE_TEXT_COL]
                labels = [int(l) for l in row[SCENE_LABEL_COL]]  # float64 → int
                movie_id = row.get(MOVIE_ID_COL, f"movie_{len(movies)}")
                # Truncate to max scenes
                n = min(len(scenes), self.config.max_scenes_per_movie)
                movies.append({
                    "movie_id": movie_id,
                    "scenes": scenes[:n],
                    "labels": labels[:n],
                    "n_scenes": n,
                })
            return movies
        else:
            # One row per scene — group by movie
            from collections import defaultdict
            grouped = defaultdict(lambda: {"scenes": [], "labels": []})
            for row in ds:
                mid = row[MOVIE_ID_COL]
                grouped[mid]["scenes"].append(row[SCENE_TEXT_COL])
                grouped[mid]["labels"].append(row[SCENE_LABEL_COL])
            movies = []
            for mid, data in grouped.items():
                n = min(len(data["scenes"]), self.config.max_scenes_per_movie)
                movies.append({
                    "movie_id": mid,
                    "scenes": data["scenes"][:n],
                    "labels": data["labels"][:n],
                    "n_scenes": n,
                })
            return movies

    def _build_character_presence(self, movie: Dict) -> torch.Tensor:
        """
        Build K x N binary matrix: char_presence[k][t] = 1 if character k
        appears in scene t.
        Returns shape (K, N) where K = n_character_slots, N = n_scenes.
        """
        K = self.config.n_character_slots
        N = movie["n_scenes"]

        # Collect all characters, rank by frequency
        from collections import Counter
        char_counter = Counter()
        for chars in movie["characters_per_scene"]:
            char_counter.update(chars)

        # Top-K characters
        top_chars = [c for c, _ in char_counter.most_common(K)]
        char_to_idx = {c: i for i, c in enumerate(top_chars)}

        presence = torch.zeros(K, N)
        for t, chars in enumerate(movie["characters_per_scene"]):
            for c in chars:
                if c in char_to_idx:
                    presence[char_to_idx[c], t] = 1.0

        return presence

    def _build_causal_pseudolabels(self, movie: Dict) -> torch.Tensor:
        """
        Build causal pseudo-label matrix using entity overlap + TF-IDF.
        Returns shape (N, N) where entry [t, i] is the pseudo-probability
        that scene t depends on scene i (i < t only).
        """
        N = movie["n_scenes"]
        cfg = self.config

        # ── Entity overlap ──
        entity_sim = np.zeros((N, N))
        for t in range(N):
            chars_t = set(movie["characters_per_scene"][t])
            if not chars_t:
                continue
            for i in range(t):
                chars_i = set(movie["characters_per_scene"][i])
                if chars_i:
                    overlap = len(chars_t & chars_i) / len(chars_t | chars_i)
                    entity_sim[t, i] = overlap

        # ── TF-IDF similarity ──
        if N > 1:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            try:
                tfidf_matrix = vectorizer.fit_transform(movie["scenes"])
                tfidf_sim = cosine_similarity(tfidf_matrix).astype(np.float32)
            except ValueError:
                tfidf_sim = np.zeros((N, N))
        else:
            tfidf_sim = np.zeros((N, N))

        # ── Combine ──
        combined = cfg.causal_entity_weight * entity_sim + cfg.causal_tfidf_weight * tfidf_sim

        # Mask upper triangle (only past scenes)
        mask = np.tril(np.ones((N, N), dtype=bool), k=-1)
        combined = combined * mask

        # Binarize at percentile threshold
        nonzero_vals = combined[mask]
        if len(nonzero_vals) > 0 and nonzero_vals.max() > 0:
            threshold = np.percentile(nonzero_vals[nonzero_vals > 0], cfg.causal_threshold_percentile)
            binary = (combined >= threshold).astype(np.float32)
        else:
            binary = np.zeros((N, N), dtype=np.float32)

        return torch.from_numpy(binary)

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        return self.movies[idx]


class SceneEncoder:
    """
    Encodes scene text into fixed-dimensional vectors using frozen Qwen2.5-7B.
    Quantized to 4-bit NF4 via bitsandbytes (~4GB VRAM).
    Mean-pools last hidden states. Caches embeddings to disk.
    """

    def __init__(self, config: HNSDConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"  Loading {config.encoder_name} (4-bit quantized)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.encoder_name, use_fast=True, trust_remote_code=True,
            cache_dir=config.hf_cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── 4-bit NF4 quantization via bitsandbytes ──
        # Pin to single GPU with device_map={"": 0} to avoid device confusion
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModel.from_pretrained(
                config.encoder_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=config.hf_cache_dir,
            )
        else:
            self.model = AutoModel.from_pretrained(
                config.encoder_name,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=config.hf_cache_dir,
            )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._cache = {}
        self._cache_dir = config.cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    @torch.no_grad()
    def encode_scenes(self, scenes: List[str], movie_id: str = "") -> torch.Tensor:
        """
        Encode list of scene texts → (N, d) tensor in float32.
        Caches to disk; subsequent runs skip encoding entirely.
        """
        if movie_id:
            safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)
            cache_path = os.path.join(self._cache_dir, f"{safe_id}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")
            if movie_id in self._cache:
                return self._cache[movie_id]

        embeddings = []
        d = self.config.scene_embed_dim

        # Encode one scene at a time to avoid padding issues across
        # scenes of wildly different lengths. Quantized model is the
        # bottleneck anyway, not batching overhead.
        for idx, text in enumerate(scenes):
            # Guard: ensure text is a non-empty string
            if not isinstance(text, str) or len(text.strip()) == 0:
                embeddings.append(torch.zeros(1, d))
                continue

            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_scene_tokens,
                return_tensors="pt",
            ).to(self.device)

            # Guard: skip if tokenizer produced 0 tokens
            if inputs["input_ids"].shape[1] == 0:
                print(f"    WARNING: scene {idx} in {movie_id} tokenized to 0 tokens, using zero vector")
                embeddings.append(torch.zeros(1, d))
                continue

            outputs = self.model(**inputs, output_hidden_states=False)

            # Mean pooling over all tokens (no padding in single-scene mode)
            hidden = outputs.last_hidden_state.float()  # (1, T, d)
            pooled = hidden.mean(dim=1).cpu()            # (1, d)
            embeddings.append(pooled)

        result = torch.cat(embeddings, dim=0)  # (N, d)

        if movie_id:
            safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)
            torch.save(result, os.path.join(self._cache_dir, f"{safe_id}.pt"))
            self._cache[movie_id] = result

        return result

    def clear_cache(self):
        self._cache.clear()


def collate_movies(batch: List[Dict], config: HNSDConfig) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of movies into padded tensors.
    Each movie can have a different number of scenes.

    Returns dict with:
        scene_texts: list of list of strings (B, varying N)
        labels:      (B, max_N) long tensor, -1 for padding
        char_presence: (B, K, max_N) float tensor
        causal_pseudo: (B, max_N, max_N) float tensor
        mask:        (B, max_N) bool tensor (True = valid scene)
        n_scenes:    list of int
        movie_ids:   list of str
    """
    B = len(batch)
    max_N = max(m["n_scenes"] for m in batch)
    K = config.n_character_slots

    labels = torch.full((B, max_N), -1, dtype=torch.long)
    char_presence = torch.zeros(B, K, max_N)
    causal_pseudo = torch.zeros(B, max_N, max_N)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    scene_texts = []
    n_scenes = []
    movie_ids = []
    scene_lengths = torch.zeros(B, max_N)  # word count per scene

    for b, movie in enumerate(batch):
        n = movie["n_scenes"]
        labels[b, :n] = torch.tensor(movie["labels"][:n], dtype=torch.long)
        char_presence[b, :, :n] = movie["char_presence"][:, :n]
        causal_pseudo[b, :n, :n] = movie["causal_pseudo_labels"][:n, :n]
        mask[b, :n] = True
        scene_texts.append(movie["scenes"])
        n_scenes.append(n)
        movie_ids.append(movie["movie_id"])
        # Scene word counts (log-scaled to prevent dominating)
        for t, text in enumerate(movie["scenes"][:n]):
            wc = len(text.split()) if isinstance(text, str) else 0
            scene_lengths[b, t] = math.log1p(wc)  # log(1 + word_count)

    return {
        "scene_texts": scene_texts,
        "labels": labels,
        "char_presence": char_presence,
        "causal_pseudo": causal_pseudo,
        "mask": mask,
        "n_scenes": n_scenes,
        "movie_ids": movie_ids,
        "scene_lengths": scene_lengths,
    }


def get_dataloaders(config: HNSDConfig):
    """Build train/val/test dataloaders."""
    train_ds = MENSAMovieDataset("train", config)
    val_ds = MENSAMovieDataset("validation", config)
    test_ds = MENSAMovieDataset("test", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.phase1_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_movies(b, config),
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.phase2_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_movies(b, config),
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.phase2_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_movies(b, config),
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader
