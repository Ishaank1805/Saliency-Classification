"""
NADN Visualization — arc decomposition figures
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

from config import HNSDConfig
config = HNSDConfig()
SAVE_DIR = config.figure_dir
LOG_DIR = config.log_dir
os.makedirs(SAVE_DIR, exist_ok=True)


def load_data():
    data = np.load(os.path.join(LOG_DIR, "interpretability.npz"))
    return {k: data[k] for k in data.files}


def fig1_disruption_vs_saliency(data):
    disruption = data["disruption"]
    labels = data["labels"]
    n_bins = 20
    bins = np.linspace(0, np.percentile(disruption, 98), n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    probs = []
    for i in range(n_bins):
        m = (disruption >= bins[i]) & (disruption < bins[i+1])
        probs.append(labels[m].mean() if m.sum() > 10 else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(centers, probs, s=60, c="steelblue", zorder=3)
    valid = ~np.isnan(probs)
    if valid.sum() > 3:
        c = np.polyfit(centers[valid], np.array(probs)[valid], 2)
        xs = np.linspace(bins[0], bins[-2], 100)
        ax.plot(xs, np.polyval(c, xs), "r-", lw=2)
    ax.set_xlabel("Counterfactual Disruption Score")
    ax.set_ylabel("Salience Probability")
    ax.set_title("Salience vs. Arc Disruption")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "fig1_disruption.pdf"))
    plt.close()
    print("  Saved fig1_disruption.pdf")


def fig2_arc_affinity(data):
    affinity = data["affinity"]
    labels = data["labels"]
    K = affinity.shape[1]

    sal_means = affinity[labels == 1].mean(axis=0)
    nonsal_means = affinity[labels == 0].mean(axis=0)

    x = np.arange(K)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, sal_means, w, label="Salient", color="steelblue")
    ax.bar(x + w/2, nonsal_means, w, label="Non-Salient", color="lightcoral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Arc {i+1}" for i in range(K)])
    ax.set_ylabel("Mean Affinity")
    ax.set_title("Scene-Arc Affinity: Salient vs Non-Salient")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(SAVE_DIR, "fig2_arc_affinity.pdf"))
    plt.close()
    print("  Saved fig2_arc_affinity.pdf")


def fig3_arc_entropy(data):
    affinity = data["affinity"]
    labels = data["labels"]
    eps = 1e-7
    entropy = -(affinity * np.log(affinity + eps)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, entropy.max(), 30)
    ax.hist(entropy[labels == 0], bins=bins, alpha=0.5, density=True, color="gray", label="Non-salient")
    ax.hist(entropy[labels == 1], bins=bins, alpha=0.6, density=True, color="steelblue", label="Salient")
    ax.set_xlabel("Arc Assignment Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Salient Scenes Have More Focused Arc Assignments")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "fig3_arc_entropy.pdf"))
    plt.close()
    print("  Saved fig3_arc_entropy.pdf")


if __name__ == "__main__":
    print("Generating NADN figures...")
    data = load_data()
    fig1_disruption_vs_saliency(data)
    fig2_arc_affinity(data)
    fig3_arc_entropy(data)
    print(f"Figures saved to {SAVE_DIR}/")
