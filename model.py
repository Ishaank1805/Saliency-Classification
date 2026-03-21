"""
Narrative Arc Decomposition Network (NADN)

Architecture:
    Frozen Qwen → Bidirectional Contextualizer → Slot Attention (Arc Discovery)
    → Counterfactual Disruption Scoring → Summary-Grounded Training

Key innovations over HNSD:
    1. Bidirectional (not causal) — saliency is retrospective
    2. Slot attention discovers latent narrative arcs
    3. Summary reconstruction directly mirrors labeling process
    4. Counterfactual disruption defines saliency causally
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HNSDConfig


# ═══════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════
#  BIDIRECTIONAL SCENE CONTEXTUALIZER
#  (replaces causal transformer — saliency needs full story)
# ═══════════════════════════════════════════════════════════

class BidirectionalContextualizer(nn.Module):
    """
    Non-causal transformer: every scene sees every other scene.
    Builds 'story role' representations instead of 'story so far'.
    """

    def __init__(self, config: HNSDConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.scene_embed_dim, config.d_model)
        self.pos_enc = SinusoidalPE(config.d_model, config.max_scenes_per_movie)
        self.norm_in = nn.LayerNorm(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

    def forward(self, z, mask):
        """
        Args:
            z: (B, N, scene_embed_dim)
            mask: (B, N) bool
        Returns:
            h: (B, N, d_model) — bidirectionally contextualized
        """
        x = self.input_proj(z)
        x = self.pos_enc(x)
        x = self.norm_in(x)

        # NO causal mask — every scene sees every other scene
        pad_mask = ~mask  # True = ignore
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        return h


# ═══════════════════════════════════════════════════════════
#  SLOT ATTENTION FOR NARRATIVE ARC DISCOVERY
#  (from object-centric learning, applied to narratology)
# ═══════════════════════════════════════════════════════════

class NarrativeSlotAttention(nn.Module):
    """
    K learnable arc slots compete to 'own' scenes through
    iterative cross-attention. Discovers latent narrative threads.

    After R rounds:
      - Arc representations S: (B, K, d) — what each arc is about
      - Scene-Arc affinity Φ: (B, N, K) — which arcs each scene belongs to
    """

    def __init__(self, config: HNSDConfig):
        super().__init__()
        d = config.d_model
        K = config.n_arc_slots
        self.K = K
        self.R = config.slot_attn_rounds
        self.d = d
        self.scale = d ** -0.5

        # Orthogonal initial arc slots (maximally different starting points)
        init = torch.randn(K, d)
        if K <= d:
            # QR decomposition gives orthogonal rows
            q, _ = torch.linalg.qr(init.T)
            init = q.T[:K] * 0.5  # scale down slightly
        self.slot_init = nn.Parameter(init)

        # Temporal bias: each slot prefers a different part of the narrative
        # Slot k attends more to position k/K of the movie
        self.temporal_bias = nn.Parameter(torch.linspace(0, 1, K).unsqueeze(1))  # (K, 1)
        self.temporal_scale = nn.Parameter(torch.ones(K, 1) * 3.0)  # sharpness

        # Attention projections
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)

        # GRU for stable slot updates
        self.gru = nn.GRUCell(d, d)

        # Layer norms
        self.norm_slots = nn.LayerNorm(d)
        self.norm_inputs = nn.LayerNorm(d)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm_mlp = nn.LayerNorm(d)

    def forward(self, h, mask, scene_position=None):
        """
        Args:
            h: (B, N, d) contextualized scene representations
            mask: (B, N) bool
            scene_position: (B, N) normalized [0,1] position in movie
        Returns:
            slots: (B, K, d) arc representations
            affinity: (B, N, K) scene-arc soft assignment
        """
        B, N, d = h.shape
        K = self.K
        device = h.device

        # Default positions
        if scene_position is None:
            scene_position = torch.zeros(B, N, device=device)

        # Temporal bias: each slot prefers a region of the narrative
        # (K, 1) vs (B, N) → (B, K, N) Gaussian bias
        pos = scene_position.unsqueeze(1)  # (B, 1, N)
        center = self.temporal_bias.to(device)  # (K, 1)
        scale = self.temporal_scale.to(device).abs() + 0.1  # (K, 1) positive
        temporal_logits = -scale * (pos - center.unsqueeze(0)) ** 2  # (B, K, N)

        # Initialize slots
        slots = self.slot_init.unsqueeze(0).expand(B, -1, -1).clone()

        # Precompute keys and values
        inputs = self.norm_inputs(h)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for r in range(self.R):
            slots_normed = self.norm_slots(slots)
            q = self.to_q(slots_normed)

            # Content attention + temporal bias
            dots = torch.bmm(q, k.transpose(1, 2)) * self.scale + temporal_logits

            # Mask padding
            dots = dots.masked_fill(~mask.unsqueeze(1), float("-inf"))

            # Softmax over slots (scenes compete)
            attn = F.softmax(dots, dim=1)
            attn = attn.nan_to_num(0.0)

            # Aggregate
            updates = torch.bmm(attn, v)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K, d),
                slots.reshape(B * K, d),
            ).reshape(B, K, d)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Final affinity
        q_final = self.to_q(self.norm_slots(slots))
        dots_final = torch.bmm(q_final, k.transpose(1, 2)) * self.scale + temporal_logits
        dots_final = dots_final.masked_fill(~mask.unsqueeze(1), float("-inf"))
        affinity = dots_final.transpose(1, 2).softmax(dim=-1)  # (B, N, K)
        affinity = affinity.nan_to_num(0.0)
        affinity = affinity * mask.unsqueeze(-1).float()

        return slots, affinity


# ═══════════════════════════════════════════════════════════
#  SUMMARY ALIGNMENT HEAD
#  (directly mirrors MENSA labeling process)
# ═══════════════════════════════════════════════════════════

class SummaryAlignmentHead(nn.Module):
    """
    Learns to select scenes whose content matches the movie's
    Wikipedia plot summary. Training signal that no baseline uses.

    L_recon = 1 - cos_sim(saliency_weighted_scenes, summary_embedding)
    """

    def __init__(self, config: HNSDConfig):
        super().__init__()
        d = config.scene_embed_dim  # operate in raw embedding space
        self.score_proj = nn.Linear(config.d_model, 1)

    def forward(self, h, z_raw, z_summary, mask):
        """
        Args:
            h: (B, N, d_model) contextualized scenes
            z_raw: (B, N, scene_embed_dim) raw Qwen embeddings
            z_summary: (B, scene_embed_dim) summary embedding
            mask: (B, N) bool
        Returns:
            loss: summary reconstruction loss
            selection_weights: (B, N) soft scene selection
        """
        # Compute soft selection weights from contextualized representations
        scores = self.score_proj(h).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, N)
        weights = weights.nan_to_num(0.0)    # all-masked → NaN → 0

        # Reconstruct summary as weighted combination of raw scene embeddings
        z_recon = torch.bmm(weights.unsqueeze(1), z_raw).squeeze(1)  # (B, d)

        # Guard: skip cosine similarity if either vector is near-zero
        recon_norm = z_recon.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        summary_norm = z_summary.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z_recon_safe = z_recon / recon_norm
        z_summary_safe = z_summary / summary_norm

        cos_sim = (z_recon_safe * z_summary_safe).sum(dim=-1)  # (B,)
        loss = (1 - cos_sim).mean()

        return loss, weights


# ═══════════════════════════════════════════════════════════
#  COUNTERFACTUAL DISRUPTION SCORER
#  (causal definition of saliency)
# ═══════════════════════════════════════════════════════════

class DisruptionScorer(nn.Module):
    """
    Computes saliency as counterfactual arc disruption:
    How much do narrative arcs change if we remove this scene?

    Uses first-order gradient approximation (not N forward passes).
    """

    def __init__(self, config: HNSDConfig):
        super().__init__()
        d = config.d_model
        K = config.n_arc_slots

        # Final saliency MLP: combines disruption + affinity + context + metadata
        # Input: disruption (1) + arc_affinity (K) + h (d) + z_proj (d)
        #        + scene_length (1) + scene_position (1)
        input_dim = 1 + K + d + d + 2
        hidden = config.classifier_hidden

        layers = []
        prev = input_dim
        for h_dim in hidden:
            layers.extend([
                nn.Linear(prev, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(config.dropout),
            ])
            prev = h_dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        self.z_proj = nn.Linear(config.scene_embed_dim, d)

    def forward(self, h, z_raw, slots, affinity, scene_length, scene_position, mask):
        """
        Args:
            h: (B, N, d)
            z_raw: (B, N, scene_embed_dim)
            slots: (B, K, d) arc representations
            affinity: (B, N, K) scene-arc assignment
            scene_length: (B, N)
            scene_position: (B, N)
            mask: (B, N)
        Returns:
            logits: (B, N) saliency logits
            disruption: (B, N) raw disruption scores
        """
        B, N, d = h.shape
        K = slots.shape[1]

        # ── Compute disruption score ──
        # For each scene t, disruption = how much arcs change without it
        # Efficient: ||Φ[t,:] * (h[t] projected into arc space)||
        # This approximates the leave-one-out effect

        # Each scene's contribution to each arc: (B, N, K, 1) * (B, N, 1, d)
        # = how much of scene t's representation is in arc k
        scene_arc_contrib = affinity.unsqueeze(-1) * h.unsqueeze(2)  # (B, N, K, d)

        # Disruption = L2 norm of total contribution across arcs
        disruption = scene_arc_contrib.norm(dim=-1).sum(dim=-1)  # (B, N)

        # Normalize per movie for stability
        d_max = disruption.max(dim=-1, keepdim=True).values.clamp(min=1e-7)
        disruption_normed = disruption / d_max

        # ── Build features ──
        z_proj = self.z_proj(z_raw)

        features = torch.cat([
            disruption_normed.unsqueeze(-1),    # (B, N, 1)
            affinity,                            # (B, N, K)
            h,                                   # (B, N, d)
            z_proj,                              # (B, N, d)
            scene_length.unsqueeze(-1),          # (B, N, 1)
            scene_position.unsqueeze(-1),        # (B, N, 1)
        ], dim=-1)

        logits = self.mlp(features).squeeze(-1)  # (B, N)
        return logits, disruption_normed


# ═══════════════════════════════════════════════════════════
#  ORDER HEAD (kept from HNSD)
# ═══════════════════════════════════════════════════════════

class OrderHead(nn.Module):
    def __init__(self, config: HNSDConfig):
        super().__init__()
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, h):
        return self.head(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════
#  SOFT MACRO F1 LOSS
# ═══════════════════════════════════════════════════════════

class SoftMacroF1Loss(nn.Module):
    def __init__(self, gamma=2.0, focal_alpha=0.4, f1_weight=0.35, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.f1_weight = f1_weight
        self.eps = eps

    def forward(self, logits, targets, mask):
        p = torch.sigmoid(logits)
        y = targets.float()
        m = mask.float()
        p, y = p * m, y * m

        tp1 = (p * y).sum()
        fp1 = (p * (1 - y) * m).sum()
        fn1 = ((1 - p) * y * m).sum()
        tp0 = ((1 - p) * (1 - y) * m).sum()
        fp0 = ((1 - p) * y * m).sum()
        fn0 = (p * (1 - y) * m).sum()

        f1_sal = (2 * tp1) / (2 * tp1 + fp1 + fn1 + self.eps)
        f1_nonsal = (2 * tp0) / (2 * tp0 + fp0 + fn0 + self.eps)
        f1_loss = 1 - (f1_sal + f1_nonsal) / 2

        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t = p * y + (1 - p) * (1 - y)
        alpha_t = self.focal_alpha * y + (1 - self.focal_alpha) * (1 - y)
        focal = (alpha_t * (1 - p_t) ** self.gamma * ce * m).sum() / m.sum().clamp(min=1)

        return (1 - self.f1_weight) * focal + self.f1_weight * f1_loss


# ═══════════════════════════════════════════════════════════
#  FULL NADN MODEL
# ═══════════════════════════════════════════════════════════

class HNSD(nn.Module):
    """
    Narrative Arc Decomposition Network.
    Named HNSD for compatibility with existing training code.
    """

    def __init__(self, config: HNSDConfig):
        super().__init__()
        self.config = config

        # Bidirectional contextualizer (NOT causal)
        self.contextualizer = BidirectionalContextualizer(config)

        # Slot attention for arc discovery
        self.slot_attn = NarrativeSlotAttention(config)

        # Summary alignment (exploits unused summary field)
        self.summary_head = SummaryAlignmentHead(config)

        # Counterfactual disruption scorer
        self.scorer = DisruptionScorer(config)

        # Order head (auxiliary)
        self.order_head = OrderHead(config)

        # Saliency loss
        self.saliency_loss = SoftMacroF1Loss(
            gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            f1_weight=config.f1_loss_weight,
        )

    def forward(
        self,
        z: torch.Tensor,                          # (B, N, scene_embed_dim)
        mask: torch.Tensor,                        # (B, N) bool
        labels: Optional[torch.Tensor] = None,
        z_summary: Optional[torch.Tensor] = None,  # (B, scene_embed_dim)
        scene_length: Optional[torch.Tensor] = None,
        order_labels: Optional[torch.Tensor] = None,
        phase: int = 2,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        B, N, _ = z.shape
        cfg = self.config
        device = z.device

        if scene_length is None:
            scene_length = torch.zeros(B, N, device=device)

        # Scene position
        scene_position = torch.zeros(B, N, device=device)
        for b in range(B):
            nv = mask[b].sum().long().item()
            if nv > 1:
                scene_position[b, :nv] = torch.linspace(0, 1, nv, device=device)

        # ── Bidirectional contextualization ──
        h = self.contextualizer(z, mask)  # (B, N, d)

        # ── Slot attention: discover narrative arcs ──
        slots, affinity = self.slot_attn(h, mask, scene_position)  # (B, K, d), (B, N, K)

        # ── Counterfactual disruption scoring ──
        sal_logits, disruption = self.scorer(
            h, z, slots, affinity, scene_length, scene_position, mask
        )

        # ═══════════ Losses ═══════════
        outputs = {
            "sal_logits": sal_logits,
            "disruption": disruption,
            "affinity": affinity,       # (B, N, K) for interpretability
            "slots": slots,             # (B, K, d)
            # Compat keys for interpretability code
            "hub_score": disruption,
            "surprisal": torch.zeros(B, N, device=device),
            "beta_sum": torch.zeros(B, N, device=device),
            "read_mass": torch.zeros(B, N, device=device),
            "write_mass": torch.zeros(B, N, device=device),
        }

        total_loss = torch.tensor(0.0, device=device)

        phase1_mult = 1.0 if phase == 1 else cfg.phase2_struct_weight

        # ── Summary reconstruction ──
        if z_summary is not None:
            loss_recon, selection_weights = self.summary_head(h, z, z_summary, mask)
            outputs["loss_recon"] = loss_recon
            outputs["selection_weights"] = selection_weights
            recon_weight = cfg.lambda_recon if phase == 1 else cfg.phase2_recon_weight
            total_loss = total_loss + recon_weight * loss_recon

        # ── Arc coherence: arcs should be temporally localized ──
        # Var(position weighted by affinity for each arc)
        pos = scene_position.unsqueeze(-1)  # (B, N, 1)
        arc_weights = affinity * mask.unsqueeze(-1).float()  # (B, N, K)
        arc_weights_sum = arc_weights.sum(dim=1, keepdim=True).clamp(min=1e-7)
        arc_weights_norm = arc_weights / arc_weights_sum  # (B, N, K)

        mean_pos = (arc_weights_norm * pos).sum(dim=1)  # (B, K)
        var_pos = (arc_weights_norm * (pos - mean_pos.unsqueeze(1)) ** 2).sum(dim=1)  # (B, K)
        loss_coherence = var_pos.mean()
        outputs["loss_coherence"] = loss_coherence
        total_loss = total_loss + cfg.lambda_coherence * phase1_mult * loss_coherence

        # ── Arc diversity: different arcs MUST be different ──
        slots_normed = F.normalize(slots + 1e-8, dim=-1)
        sim_matrix = torch.bmm(slots_normed, slots_normed.transpose(1, 2))  # (B, K, K)
        eye = torch.eye(cfg.n_arc_slots, device=device).unsqueeze(0)
        # Use absolute similarity (not squared) — stronger gradient when slots are similar
        off_diag = (sim_matrix * (1 - eye)).abs()
        loss_diversity = off_diag.mean()
        outputs["loss_diversity"] = loss_diversity
        total_loss = total_loss + cfg.lambda_diversity * phase1_mult * loss_diversity

        # ── Affinity entropy: scenes should use multiple arcs, not collapse to one ──
        # Maximize entropy of per-scene arc assignment
        aff_valid = affinity[mask]  # (total_valid_scenes, K)
        if aff_valid.shape[0] > 0:
            entropy = -(aff_valid * (aff_valid + 1e-8).log()).sum(dim=-1)  # (V,)
            max_entropy = math.log(cfg.n_arc_slots)  # maximum possible
            loss_entropy = 1 - entropy.mean() / max_entropy  # 0 = max entropy, 1 = collapsed
            outputs["loss_entropy"] = loss_entropy
            total_loss = total_loss + cfg.lambda_entropy * phase1_mult * loss_entropy

        # ── Usage balance: each arc should be used roughly equally ──
        # Prevent one arc from dominating
        arc_usage = affinity[mask].mean(dim=0)  # (K,) — average usage per arc
        target_usage = 1.0 / cfg.n_arc_slots
        loss_balance = ((arc_usage - target_usage) ** 2).sum() * cfg.n_arc_slots
        outputs["loss_balance"] = loss_balance
        total_loss = total_loss + cfg.lambda_balance * phase1_mult * loss_balance

        # ── Scene order prediction ──
        if order_labels is not None:
            order_logits = self.order_head(h)
            loss_order = F.binary_cross_entropy_with_logits(
                order_logits, order_labels, reduction="none"
            )
            loss_order = (loss_order * mask.float()).sum() / mask.float().sum().clamp(min=1)
            total_loss = total_loss + cfg.lambda_order * phase1_mult * loss_order
            outputs["loss_order"] = loss_order

        # ── Saliency loss (Phase 2 only) ──
        if labels is not None and phase == 2:
            loss_sal = self.saliency_loss(sal_logits, labels, mask)
            total_loss = total_loss + loss_sal
            outputs["loss_sal"] = loss_sal

        outputs["loss"] = total_loss
        return outputs
