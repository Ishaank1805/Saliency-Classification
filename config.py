"""
NADN Configuration — Narrative Arc Decomposition Network
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class HNSDConfig:
    # ─── Paths ───
    base_dir: str = "/scratch/ishaan.karan/hnsd"
    log_dir: str = ""
    cache_dir: str = ""
    figure_dir: str = ""
    hf_cache_dir: str = ""

    def __post_init__(self):
        self.log_dir = f"{self.base_dir}/logs"
        self.cache_dir = f"{self.base_dir}/cache/scene_embeddings"
        self.figure_dir = f"{self.base_dir}/figures"
        self.hf_cache_dir = f"{self.base_dir}/hf_cache"

    # ─── Dataset ───
    dataset_name: str = "rohitsaxena/MENSA"
    max_scenes_per_movie: int = 150

    # ─── Scene Encoder (frozen) ───
    encoder_name: str = "Qwen/Qwen3-8B"
    scene_embed_dim: int = 4096
    max_scene_tokens: int = 1024
    freeze_encoder: bool = True
    load_in_4bit: bool = True

    # ─── Bidirectional Scene Contextualizer ───
    d_model: int = 768
    n_heads: int = 8
    n_layers: int = 4
    dim_ff: int = 3072
    dropout: float = 0.15

    # ─── Slot Attention (Arc Discovery) ───
    n_arc_slots: int = 6       # K narrative arcs (main plot + subplots)
    slot_attn_rounds: int = 3  # iterative refinement rounds

    # ─── Classifier ───
    classifier_hidden: List[int] = field(default_factory=lambda: [384, 96])

    # ─── Losses ───
    focal_gamma: float = 2.0
    focal_alpha: float = 0.4
    f1_loss_weight: float = 0.35

    # ─── Loss Weights ───
    lambda_recon: float = 1.0
    lambda_coherence: float = 0.3
    lambda_diversity: float = 10.0   # 33x stronger — prevent arc collapse
    lambda_entropy: float = 2.0      # NEW: maximize scene-arc entropy
    lambda_balance: float = 5.0      # NEW: equal arc usage
    lambda_order: float = 0.3
    phase2_recon_weight: float = 0.3
    phase2_struct_weight: float = 0.1

    # ─── Training ───
    phase1_epochs: int = 30
    phase1_lr: float = 2e-4
    phase1_batch_size: int = 2
    grad_accum_steps: int = 2
    use_amp: bool = True

    phase2_epochs: int = 20
    phase2_lr: float = 3e-5
    phase2_batch_size: int = 2
    phase2_patience: int = 5

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"

    # ─── Augmentation ───
    order_permute_ratio: float = 0.10

    # ─── Compat (unused but data.py references) ───
    n_character_slots: int = 15
    causal_entity_weight: float = 0.7
    causal_tfidf_weight: float = 0.3
    causal_threshold_percentile: float = 90.0
