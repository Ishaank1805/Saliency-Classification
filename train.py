"""
NADN Training Pipeline
Phase 1: Arc discovery (summary reconstruction + coherence + diversity + order)
Phase 2: Saliency fine-tuning (soft macro F1 + summary regularizer)
"""
import os
import json
import random
import time
import math
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from config import HNSDConfig
from data import MENSAMovieDataset, collate_movies, get_dataloaders
from model import HNSD
from evaluate import compute_metrics, evaluate_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def permute_scenes(z, mask, ratio=0.10):
    B, N, d = z.shape
    z_perm = z.clone()
    order_labels = torch.zeros(B, N, device=z.device)
    for b in range(B):
        n_valid = mask[b].sum().item()
        n_displace = max(1, int(n_valid * ratio))
        indices = list(range(n_valid))
        displaced = random.sample(indices, min(n_displace, len(indices)))
        targets = displaced.copy()
        random.shuffle(targets)
        for src, tgt in zip(displaced, targets):
            if src != tgt:
                z_perm[b, tgt] = z[b, src]
                order_labels[b, tgt] = 1.0
    return z_perm, order_labels


class Trainer:
    def __init__(self, config: HNSDConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {self.device}")
        set_seed(config.seed)

        # Embedding cache (scene + summary)
        self._embedding_cache_dir = config.cache_dir
        self._scene_cache = {}
        self._summary_cache = {}
        print(f"Loading embeddings from cache: {self._embedding_cache_dir}")

        # Data
        print("Loading MENSA dataset...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config)
        print(f"  Train: {len(self.train_loader.dataset)} movies")
        print(f"  Val:   {len(self.val_loader.dataset)} movies")
        print(f"  Test:  {len(self.test_loader.dataset)} movies")

        # Model
        self.model = HNSD(config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"NADN trainable parameters: {n_params:,}")

        # AMP
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_accum_steps = config.grad_accum_steps
        if self.use_amp:
            print("Mixed precision: ON")
        print(f"Gradient accumulation: {self.grad_accum_steps} steps")

        # Logging
        self.log_dir = config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = {"phase1": [], "phase2": []}

    def _safe_id(self, movie_id):
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in movie_id)

    def _encode_batch(self, batch):
        """Load scene embeddings + summary embeddings from disk cache."""
        all_z = []
        all_z_summary = []

        for b in range(len(batch["scene_texts"])):
            movie_id = batch["movie_ids"][b]
            safe_id = self._safe_id(movie_id)

            # Scene embeddings
            if movie_id in self._scene_cache:
                z = self._scene_cache[movie_id]
            else:
                path = os.path.join(self._embedding_cache_dir, f"{safe_id}.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Scene cache not found: {path}")
                z = torch.load(path, map_location="cpu")
                self._scene_cache[movie_id] = z
            all_z.append(z)

            # Summary embedding
            if movie_id in self._summary_cache:
                z_sum = self._summary_cache[movie_id]
            else:
                sum_path = os.path.join(self._embedding_cache_dir, f"{safe_id}_summary.pt")
                if os.path.exists(sum_path):
                    z_sum = torch.load(sum_path, map_location="cpu")
                else:
                    z_sum = torch.zeros(1, self.config.scene_embed_dim)
                self._summary_cache[movie_id] = z_sum
            all_z_summary.append(z_sum)

        # Pad scenes
        max_n = batch["mask"].shape[1]
        B = len(all_z)
        d = all_z[0].shape[1]
        z_padded = torch.zeros(B, max_n, d)
        for b, z in enumerate(all_z):
            n = min(z.shape[0], max_n)
            z_padded[b, :n] = z[:n]

        # Stack summaries: (B, d)
        z_summary_list = []
        for s in all_z_summary:
            if s.dim() == 2:
                z_summary_list.append(s[0])  # (d,)
            elif s.dim() == 1:
                z_summary_list.append(s)      # (d,)
            else:
                z_summary_list.append(s.reshape(-1)[:self.config.scene_embed_dim])
        z_summary = torch.stack(z_summary_list, dim=0)  # (B, d)

        return z_padded.to(self.device), z_summary.to(self.device)

    def train_phase1(self):
        print("\n" + "=" * 60)
        print("PHASE 1: Narrative Arc Discovery")
        print("=" * 60)

        config = self.config
        optimizer = AdamW(
            self.model.parameters(),
            lr=config.phase1_lr,
            weight_decay=config.weight_decay,
        )
        total_steps = len(self.train_loader) * config.phase1_epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=config.phase1_lr,
            total_steps=total_steps,
            pct_start=config.warmup_ratio, anneal_strategy="cos",
        )

        for epoch in range(config.phase1_epochs):
            self.model.train()
            epoch_losses = {"loss": 0, "loss_recon": 0, "loss_coherence": 0,
                            "loss_diversity": 0, "loss_entropy": 0, "loss_balance": 0,
                            "loss_order": 0}
            n_batches = 0
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                z, z_summary = self._encode_batch(batch)
                mask = batch["mask"].to(self.device)
                scene_length = batch["scene_lengths"].to(self.device)

                z_perm, order_labels = permute_scenes(z, mask, config.order_permute_ratio)
                order_labels = order_labels.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        z=z_perm, mask=mask,
                        z_summary=z_summary,
                        scene_length=scene_length,
                        order_labels=order_labels,
                        phase=1,
                    )
                    loss = outputs["loss"] / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                for key in epoch_losses:
                    if key in outputs:
                        epoch_losses[key] += outputs[key].item()
                n_batches += 1

            avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            self.history["phase1"].append(avg)

            print(
                f"  Epoch {epoch+1}/{config.phase1_epochs} | "
                f"Loss: {avg['loss']:.4f} | "
                f"Recon: {avg.get('loss_recon', 0):.4f} | "
                f"Divers: {avg.get('loss_diversity', 0):.4f} | "
                f"Entropy: {avg.get('loss_entropy', 0):.4f} | "
                f"Balance: {avg.get('loss_balance', 0):.4f} | "
                f"Order: {avg.get('loss_order', 0):.4f}"
            )

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "phase1.pt"))
        print("Phase 1 complete. Checkpoint saved.")

    def train_phase2(self):
        print("\n" + "=" * 60)
        print("PHASE 2: Saliency Fine-tuning")
        print("=" * 60)

        config = self.config
        optimizer = AdamW(
            self.model.parameters(),
            lr=config.phase2_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=config.phase2_patience, min_lr=1e-6
        )

        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(config.phase2_epochs):
            self.model.train()
            epoch_losses = {"loss": 0, "loss_sal": 0, "loss_recon": 0}
            n_batches = 0
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                z, z_summary = self._encode_batch(batch)
                mask = batch["mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                scene_length = batch["scene_lengths"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        z=z, mask=mask, labels=labels,
                        z_summary=z_summary,
                        scene_length=scene_length,
                        phase=2,
                    )
                    loss = outputs["loss"] / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()

                for key in epoch_losses:
                    if key in outputs:
                        epoch_losses[key] += outputs[key].item()
                n_batches += 1

            # Validation
            val_metrics = self._evaluate(self.val_loader)
            macro_f1 = val_metrics["macro_f1"]
            scheduler.step(macro_f1)

            avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            self.history["phase2"].append({**avg, **val_metrics})

            print(
                f"  Epoch {epoch+1}/{config.phase2_epochs} | "
                f"Loss: {avg['loss']:.4f} | "
                f"Sal: {avg.get('loss_sal', 0):.4f} | "
                f"Recon: {avg.get('loss_recon', 0):.4f} | "
                f"Val P: {val_metrics['precision_salient']:.2f} | "
                f"Val R: {val_metrics['recall_salient']:.2f} | "
                f"Val F1: {val_metrics['f1_salient']:.2f} | "
                f"Val MacroF1: {macro_f1:.2f}"
            )

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                patience_counter = 0
                best_state = deepcopy(self.model.state_dict())
                torch.save(best_state, os.path.join(self.log_dir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= config.phase2_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        print(f"Phase 2 complete. Best Val Macro F1: {best_f1:.2f}")

    @torch.no_grad()
    def _evaluate(self, dataloader):
        """Evaluate with summary embeddings."""
        self.model.eval()
        all_logits = []
        all_labels = []

        for batch in dataloader:
            z, z_summary = self._encode_batch(batch)
            mask = batch["mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            scene_length = batch["scene_lengths"].to(self.device)

            outputs = self.model(
                z=z, mask=mask, z_summary=z_summary,
                scene_length=scene_length, phase=2,
            )
            sal_logits = outputs["sal_logits"]

            for b in range(z.shape[0]):
                n = batch["n_scenes"][b]
                probs = torch.sigmoid(sal_logits[b, :n]).cpu().numpy()
                lbls = labels[b, :n].cpu().numpy()
                valid = lbls >= 0
                all_logits.append(probs[valid])
                all_labels.append(lbls[valid])

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        from evaluate import find_optimal_threshold
        threshold = find_optimal_threshold(all_logits, all_labels)
        preds = (all_logits >= threshold).astype(int)
        metrics = compute_metrics(preds, all_labels)
        metrics["threshold"] = threshold
        return metrics

    def test(self):
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)

        test_metrics = self._evaluate(self.test_loader)
        print(f"  Precision (salient): {test_metrics['precision_salient']:.2f}%")
        print(f"  Recall    (salient): {test_metrics['recall_salient']:.2f}%")
        print(f"  F1        (salient): {test_metrics['f1_salient']:.2f}%")
        print(f"  Macro F1:            {test_metrics['macro_f1']:.2f}%")

        with open(os.path.join(self.log_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        return test_metrics

    def run_interpretability(self):
        print("\n" + "=" * 60)
        print("INTERPRETABILITY ANALYSIS")
        print("=" * 60)

        self.model.eval()
        all_disruption = []
        all_affinity = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                z, z_summary = self._encode_batch(batch)
                mask = batch["mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                scene_length = batch["scene_lengths"].to(self.device)

                outputs = self.model(
                    z=z, mask=mask, z_summary=z_summary,
                    scene_length=scene_length, phase=2,
                )

                for b in range(z.shape[0]):
                    n = batch["n_scenes"][b]
                    all_disruption.append(outputs["disruption"][b, :n].cpu())
                    all_affinity.append(outputs["affinity"][b, :n].cpu())
                    all_labels.append(labels[b, :n].cpu())

        disruption = torch.cat(all_disruption).numpy()
        labels = torch.cat(all_labels).numpy()
        affinity = torch.cat(all_affinity).numpy()  # (total_scenes, K)

        salient = labels == 1
        non_salient = labels == 0

        print("\n  Feature means (Salient vs Non-Salient):")
        s_d = disruption[salient].mean()
        ns_d = disruption[non_salient].mean()
        print(f"    {'Disruption Score':25s}: Sal={s_d:.4f}  NonSal={ns_d:.4f}  Δ={s_d-ns_d:+.4f}")

        # Per-arc affinity
        K = affinity.shape[1]
        for k in range(K):
            s_a = affinity[salient, k].mean()
            ns_a = affinity[non_salient, k].mean()
            print(f"    {'Arc ' + str(k+1) + ' affinity':25s}: Sal={s_a:.4f}  NonSal={ns_a:.4f}  Δ={s_a-ns_a:+.4f}")

        # Arc entropy (salient scenes should have lower entropy — more focused on specific arcs)
        aff_sal = affinity[salient]
        aff_nonsal = affinity[non_salient]
        eps = 1e-7
        ent_sal = -(aff_sal * np.log(aff_sal + eps)).sum(axis=1).mean()
        ent_nonsal = -(aff_nonsal * np.log(aff_nonsal + eps)).sum(axis=1).mean()
        print(f"    {'Arc Entropy':25s}: Sal={ent_sal:.4f}  NonSal={ent_nonsal:.4f}  Δ={ent_sal-ent_nonsal:+.4f}")

        np.savez(
            os.path.join(self.log_dir, "interpretability.npz"),
            disruption=disruption,
            affinity=affinity,
            labels=labels,
            # Compat keys
            surprisal=np.zeros_like(disruption),
            beta_sum=np.zeros_like(disruption),
            read_mass=np.zeros_like(disruption),
            write_mass=np.zeros_like(disruption),
            hub_score=disruption,
        )
        print("  Saved interpretability data.")

    def train(self):
        start = time.time()
        self.train_phase1()
        self.train_phase2()
        self.test()
        self.run_interpretability()
        elapsed = time.time() - start
        print(f"\nTotal training time: {elapsed/3600:.1f} hours")
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NADN Training")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-phase1", action="store_true")
    args = parser.parse_args()

    config = HNSDConfig()

    if args.mode == "train":
        trainer = Trainer(config)
        if args.skip_phase1:
            p1_path = os.path.join(config.log_dir, "phase1.pt")
            print(f"Loading Phase 1 checkpoint: {p1_path}")
            trainer.model.load_state_dict(torch.load(p1_path, map_location=trainer.device))
            trainer.train_phase2()
            trainer.test()
            trainer.run_interpretability()
        else:
            if args.checkpoint:
                trainer.model.load_state_dict(torch.load(args.checkpoint))
            trainer.train()

    elif args.mode == "test":
        trainer = Trainer(config)
        if args.checkpoint:
            trainer.model.load_state_dict(torch.load(args.checkpoint))
        trainer.test()
        trainer.run_interpretability()
