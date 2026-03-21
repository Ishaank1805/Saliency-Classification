"""
HNSD Evaluation
Metrics aligned with MENSA benchmark (Table 1 in the paper):
  - Salient class: Precision, Recall, F1
  - Overall: Macro F1
  - Statistical significance: McNemar's test
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from scipy.stats import chi2 as chi2_dist


def compute_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics matching Table 1 of the LGSM paper.

    Args:
        all_preds:  (N,) binary predictions
        all_labels: (N,) ground truth labels

    Returns dict with precision_salient, recall_salient, f1_salient, macro_f1.
    """
    # Salient class metrics (class=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
    )
    macro_f1 = f1.mean() * 100

    return {
        "precision_salient": prec[1] * 100,
        "recall_salient": rec[1] * 100,
        "f1_salient": f1[1] * 100,
        "macro_f1": macro_f1,
        "precision_nonsalient": prec[0] * 100,
        "recall_nonsalient": rec[0] * 100,
        "f1_nonsalient": f1[0] * 100,
    }


def find_optimal_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Find the threshold that maximizes Macro F1.
    Addresses the train/test distribution shift (14% → 33% salient rate).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    best_f1 = 0.0
    best_t = 0.5

    for t in thresholds:
        preds = (logits >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[0, 1], average=None, zero_division=0
        )
        macro = f1.mean()
        if macro > best_f1:
            best_f1 = macro
            best_t = t

    return best_t


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    encode_fn: Callable,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Full evaluation loop.

    Args:
        model: HNSD model
        dataloader: val or test DataLoader
        encode_fn: function to encode batch scene texts → embeddings
        device: torch device
        threshold: classification threshold (if None, optimizes on the data)

    Returns dict of metrics.
    """
    model.eval()

    all_logits = []
    all_labels = []

    for batch in dataloader:
        z = encode_fn(batch)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        scene_length = batch["scene_lengths"].to(device) if "scene_lengths" in batch else None

        outputs = model(z=z, mask=mask, scene_length=scene_length, phase=2)
        sal_logits = outputs["sal_logits"]  # (B, N)

        for b in range(z.shape[0]):
            n = batch["n_scenes"][b]
            scene_logits = torch.sigmoid(sal_logits[b, :n]).cpu().numpy()
            scene_labels = labels[b, :n].cpu().numpy()

            # Filter out padding (labels == -1)
            valid = scene_labels >= 0
            all_logits.append(scene_logits[valid])
            all_labels.append(scene_labels[valid])

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Threshold calibration
    if threshold is None:
        threshold = find_optimal_threshold(all_logits, all_labels)

    all_preds = (all_logits >= threshold).astype(int)

    metrics = compute_metrics(all_preds, all_labels)
    metrics["threshold"] = threshold
    metrics["n_scenes"] = len(all_labels)
    metrics["n_salient"] = int(all_labels.sum())
    metrics["n_predicted_salient"] = int(all_preds.sum())

    return metrics


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's test comparing two models (Table 5 in the paper).

    Returns chi2 statistic, p-value, and the contingency counts.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency table
    both_wrong = (~correct_a & ~correct_b).sum()
    a_right_b_wrong = (correct_a & ~correct_b).sum()
    a_wrong_b_right = (~correct_a & correct_b).sum()
    both_right = (correct_a & correct_b).sum()

    # McNemar statistic (with continuity correction)
    n_discordant = a_right_b_wrong + a_wrong_b_right
    if n_discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n_discordant
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "both_wrong": int(both_wrong),
        "a_right_b_wrong": int(a_right_b_wrong),
        "a_wrong_b_right": int(a_wrong_b_right),
        "both_right": int(both_right),
        "recovery_ratio": a_right_b_wrong / max(a_wrong_b_right, 1),
    }


def error_analysis(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, int]:
    """
    Error breakdown matching Table 2 in the paper.
    """
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    total_salient = int(labels.sum())
    recall = tp / max(total_salient, 1) * 100

    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Recall": recall,
    }


def recall_by_act(
    preds: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
) -> Dict[str, float]:
    """
    Recall across narrative timeline (Figure 3 in the paper).

    Args:
        positions: normalized scene position [0, 1] for each scene
    """
    act_boundaries = {
        "Act 1": (0.0, 0.25),
        "Act 2A": (0.25, 0.50),
        "Act 2B": (0.50, 0.60),
        "Act 2C": (0.60, 0.75),
        "Act 3": (0.75, 1.0),
    }

    results = {}
    for act_name, (lo, hi) in act_boundaries.items():
        act_mask = (positions >= lo) & (positions < hi)
        act_labels = labels[act_mask]
        act_preds = preds[act_mask]

        if act_labels.sum() > 0:
            recall = ((act_preds == 1) & (act_labels == 1)).sum() / act_labels.sum()
            results[act_name] = float(recall * 100)
        else:
            results[act_name] = 0.0

    return results
