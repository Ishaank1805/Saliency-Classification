"""
NADN Debug — test every component individually.
Run: python debug_nadn.py
"""
import os
import sys
import torch
import numpy as np

from config import HNSDConfig

config = HNSDConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
fails = 0


def check(name, condition, detail=""):
    global fails
    if condition:
        print(f"  {PASS} {name}")
    else:
        fails += 1
        print(f"  {FAIL} {name} — {detail}")


def has_nan(t):
    return torch.isnan(t).any().item()


def has_inf(t):
    return torch.isinf(t).any().item()


# ═══════════════════════════════════════════════════════
print("=" * 50)
print("1. SUMMARY EMBEDDINGS CHECK")
print("=" * 50)

from data import MENSAMovieDataset
ds = MENSAMovieDataset("validation", config)
movie = ds[0]
safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in movie["movie_id"])

scene_path = os.path.join(config.cache_dir, f"{safe_id}.pt")
summary_path = os.path.join(config.cache_dir, f"{safe_id}_summary.pt")

check("Scene cache exists", os.path.exists(scene_path), scene_path)
check("Summary cache exists", os.path.exists(summary_path), summary_path)

if os.path.exists(scene_path):
    z_scene = torch.load(scene_path, map_location="cpu")
    check(f"Scene shape: {z_scene.shape}", z_scene.shape[1] == config.scene_embed_dim)
    check("Scene no NaN", not has_nan(z_scene))
    check("Scene non-zero", z_scene.abs().sum() > 0)

if os.path.exists(summary_path):
    z_sum = torch.load(summary_path, map_location="cpu")
    print(f"  Summary raw shape: {z_sum.shape}")
    check(f"Summary shape ok", z_sum.shape[-1] == config.scene_embed_dim)
    check("Summary no NaN", not has_nan(z_sum))
    check("Summary non-zero", z_sum.abs().sum() > 0, f"norm={z_sum.norm().item():.4f}")

    # Flatten to (d,)
    if z_sum.dim() == 2:
        z_sum_flat = z_sum[0]
    else:
        z_sum_flat = z_sum
    check(f"Summary flat shape: ({z_sum_flat.shape[0]},)", z_sum_flat.shape[0] == config.scene_embed_dim)


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. BIDIRECTIONAL CONTEXTUALIZER")
print("=" * 50)

from model import BidirectionalContextualizer

ctx = BidirectionalContextualizer(config).to(device)

B, N, d_in = 2, 20, config.scene_embed_dim
z_fake = torch.randn(B, N, d_in, device=device)
mask_fake = torch.ones(B, N, dtype=torch.bool, device=device)
mask_fake[1, 15:] = False  # movie 2 has only 15 scenes

with torch.no_grad():
    h = ctx(z_fake, mask_fake)

check(f"Output shape: {h.shape}", h.shape == (B, N, config.d_model))
check("No NaN", not has_nan(h))
check("No Inf", not has_inf(h))


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. SLOT ATTENTION")
print("=" * 50)

from model import NarrativeSlotAttention

slot_attn = NarrativeSlotAttention(config).to(device)

with torch.no_grad():
    slots, affinity = slot_attn(h, mask_fake)

check(f"Slots shape: {slots.shape}", slots.shape == (B, config.n_arc_slots, config.d_model))
check(f"Affinity shape: {affinity.shape}", affinity.shape == (B, N, config.n_arc_slots))
check("Slots no NaN", not has_nan(slots))
check("Affinity no NaN", not has_nan(affinity))
check("Affinity in [0,1]", (affinity >= 0).all() and (affinity <= 1).all())
check("Affinity sums ~1 for valid scenes",
      (affinity[0, 0].sum() - 1.0).abs() < 0.01,
      f"sum={affinity[0, 0].sum().item():.4f}")
check("Affinity 0 for padded scenes",
      affinity[1, 15:].abs().sum() < 1e-5,
      f"padded sum={affinity[1, 15:].abs().sum().item():.6f}")


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("4. SUMMARY ALIGNMENT HEAD")
print("=" * 50)

from model import SummaryAlignmentHead

summary_head = SummaryAlignmentHead(config).to(device)

z_summary_fake = torch.randn(B, config.scene_embed_dim, device=device)

with torch.no_grad():
    loss_recon, weights = summary_head(h, z_fake, z_summary_fake, mask_fake)

check(f"Recon loss: {loss_recon.item():.4f}", not has_nan(loss_recon))
check("Recon loss finite", torch.isfinite(loss_recon).item())
check(f"Weights shape: {weights.shape}", weights.shape == (B, N))
check("Weights no NaN", not has_nan(weights))
check("Weights sum ~1", (weights[0].sum() - 1.0).abs() < 0.01,
      f"sum={weights[0].sum().item():.4f}")

# Test with ZERO summary (edge case)
z_summary_zero = torch.zeros(B, config.scene_embed_dim, device=device)
with torch.no_grad():
    loss_zero, weights_zero = summary_head(h, z_fake, z_summary_zero, mask_fake)
check("Zero summary → no NaN", not has_nan(loss_zero), f"loss={loss_zero.item()}")
check("Zero summary → finite", torch.isfinite(loss_zero).item())


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("5. DISRUPTION SCORER")
print("=" * 50)

from model import DisruptionScorer

scorer = DisruptionScorer(config).to(device)
scene_length = torch.rand(B, N, device=device) * 5
scene_position = torch.zeros(B, N, device=device)
scene_position[0] = torch.linspace(0, 1, N)
scene_position[1, :15] = torch.linspace(0, 1, 15)

with torch.no_grad():
    logits, disruption = scorer(h, z_fake, slots, affinity, scene_length, scene_position, mask_fake)

check(f"Logits shape: {logits.shape}", logits.shape == (B, N))
check(f"Disruption shape: {disruption.shape}", disruption.shape == (B, N))
check("Logits no NaN", not has_nan(logits))
check("Disruption no NaN", not has_nan(disruption))
check("Disruption in [0,1]", disruption.min() >= 0 and disruption.max() <= 1.01,
      f"range=[{disruption.min():.4f}, {disruption.max():.4f}]")


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("6. FULL NADN FORWARD (Phase 1)")
print("=" * 50)

from model import HNSD as NADN

model = NADN(config).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

order_labels = torch.zeros(B, N, device=device)
order_labels[:, 3] = 1.0

with torch.amp.autocast("cuda", enabled=config.use_amp):
    outputs = model(
        z=z_fake, mask=mask_fake,
        z_summary=z_summary_fake,
        scene_length=scene_length,
        order_labels=order_labels,
        phase=1,
    )

check("Phase 1 loss exists", "loss" in outputs)
check(f"Phase 1 loss: {outputs['loss'].item():.4f}", not has_nan(outputs["loss"]))
check("Phase 1 loss finite", torch.isfinite(outputs["loss"]).item())
check("Has loss_recon", "loss_recon" in outputs)
check(f"loss_recon: {outputs['loss_recon'].item():.4f}", not has_nan(outputs["loss_recon"]))
check("Has loss_coherence", "loss_coherence" in outputs)
check(f"loss_coherence: {outputs['loss_coherence'].item():.4f}", not has_nan(outputs["loss_coherence"]))
check("Has loss_diversity", "loss_diversity" in outputs)
check(f"loss_diversity: {outputs['loss_diversity'].item():.4f}", not has_nan(outputs["loss_diversity"]))
check("Has loss_order", "loss_order" in outputs)
check(f"loss_order: {outputs['loss_order'].item():.4f}", not has_nan(outputs["loss_order"]))
check("sal_logits no NaN", not has_nan(outputs["sal_logits"]))
check("disruption no NaN", not has_nan(outputs["disruption"]))
check("affinity no NaN", not has_nan(outputs["affinity"]))


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("7. FULL NADN FORWARD (Phase 2)")
print("=" * 50)

labels_fake = torch.randint(0, 2, (B, N), device=device)

with torch.amp.autocast("cuda", enabled=config.use_amp):
    outputs2 = model(
        z=z_fake, mask=mask_fake,
        labels=labels_fake,
        z_summary=z_summary_fake,
        scene_length=scene_length,
        phase=2,
    )

check(f"Phase 2 loss: {outputs2['loss'].item():.4f}", not has_nan(outputs2["loss"]))
check("Phase 2 has loss_sal", "loss_sal" in outputs2)
check(f"loss_sal: {outputs2['loss_sal'].item():.4f}", not has_nan(outputs2["loss_sal"]))


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("8. BACKWARD PASS")
print("=" * 50)

model.zero_grad()
with torch.amp.autocast("cuda", enabled=config.use_amp):
    out = model(
        z=z_fake, mask=mask_fake,
        labels=labels_fake,
        z_summary=z_summary_fake,
        scene_length=scene_length,
        order_labels=order_labels,
        phase=2,
    )

scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
scaler.scale(out["loss"]).backward()

has_grad = False
grad_nan = False
for name, p in model.named_parameters():
    if p.grad is not None:
        has_grad = True
        if has_nan(p.grad):
            grad_nan = True
            print(f"    NaN grad in: {name}")
            break

check("Gradients computed", has_grad)
check("No NaN in gradients", not grad_nan)


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("9. REAL DATA FORWARD")
print("=" * 50)

del model
torch.cuda.empty_cache()

from data import collate_movies
batch = collate_movies([ds[0], ds[1]], config)

# Load real embeddings
from train import Trainer
trainer = Trainer(config)

z_real, z_summary_real = trainer._encode_batch(batch)
mask_real = batch["mask"].to(device)
labels_real = batch["labels"].to(device)
scene_length_real = batch["scene_lengths"].to(device)

check(f"Real z shape: {z_real.shape}", z_real.shape[2] == config.scene_embed_dim)
check(f"Real z_summary shape: {z_summary_real.shape}", z_summary_real.shape[1] == config.scene_embed_dim)
check("Real z no NaN", not has_nan(z_real))
check("Real z_summary no NaN", not has_nan(z_summary_real))
check("Real z_summary non-zero", z_summary_real.abs().sum() > 0,
      f"norm={z_summary_real.norm().item():.4f}")

with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=config.use_amp):
        out_real = trainer.model(
            z=z_real, mask=mask_real,
            labels=labels_real,
            z_summary=z_summary_real,
            scene_length=scene_length_real,
            phase=2,
        )

check(f"Real forward loss: {out_real['loss'].item():.4f}", not has_nan(out_real["loss"]))
check("Real forward finite", torch.isfinite(out_real["loss"]).item())
check("Real sal_logits no NaN", not has_nan(out_real["sal_logits"]))
check("Real disruption no NaN", not has_nan(out_real["disruption"]))

# Try backward
trainer.model.zero_grad()
with torch.amp.autocast("cuda", enabled=config.use_amp):
    out_bwd = trainer.model(
        z=z_real, mask=mask_real,
        labels=labels_real,
        z_summary=z_summary_real,
        scene_length=scene_length_real,
        phase=2,
    )
scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
scaler.scale(out_bwd["loss"]).backward()

grad_ok = True
for name, p in trainer.model.named_parameters():
    if p.grad is not None and has_nan(p.grad):
        grad_ok = False
        print(f"    NaN grad: {name}")
        break
check("Real backward no NaN grads", grad_ok)


# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print(f"RESULTS: {fails} failures")
print("=" * 50)

if fails > 0:
    print("Fix failures before training.")
    sys.exit(1)
else:
    print("All clear! Run: python train.py --mode train")
    sys.exit(0)
