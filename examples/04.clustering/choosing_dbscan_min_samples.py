"""
Choosing DBSCAN's min_samples
==============================
min_samples controls how many points must be within eps of a point
for it to be considered a CORE point (i.e., inside a dense region).

Rules of thumb:
  - min_samples >= 2 × n_features  (most common starting point)
  - Larger min_samples → stricter density → more noise, fewer/smaller clusters
  - Smaller min_samples → looser density → less noise, bigger clusters

Strategy: fix a reasonable eps, then try several min_samples values
and observe how the clustering changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ── 1. Create data: 3 clusters + random noise ─────────────────────────────
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=[(-3,-3),(0,0),(4,2)],
                  cluster_std=0.8, random_state=42)
noise = np.random.uniform(-7, 7, size=(30, 2))
X = StandardScaler().fit_transform(np.vstack([X, noise]))

# ── 2. Fix eps (chosen from the k-distance elbow plot) ───────────────────
eps = 0.35   # determined separately via the elbow method

# ── 3. Try different min_samples values ──────────────────────────────────
min_samples_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(min_samples_values), figsize=(16, 4), sharey=True)
fig.suptitle(
    f"Effect of min_samples on DBSCAN  (eps = {eps})\n"
    "Grey = noise   |   Colours = clusters",
    fontweight="bold"
)

cmap = plt.colormaps["tab10"]

for ax, ms in zip(axes, min_samples_values):
    labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    n_noise    = (labels == -1).sum()

    # Plot noise first (grey), then each cluster
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(X[mask,0], X[mask,1], c="lightgrey",
                       edgecolors="grey", s=20, lw=0.4, label="Noise")
        else:
            ax.scatter(X[mask,0], X[mask,1], color=cmap(lbl),
                       edgecolors="k", s=25, lw=0.3, label=f"Cluster {lbl}")

    ax.set_title(f"min_samples = {ms}\n"
                 f"{n_clusters} cluster(s), {n_noise} noise pts")
    ax.set_xlabel("Feature 1"); ax.grid(alpha=0.3)
    print(f"min_samples={ms:2d}  ->  clusters: {n_clusters},  noise: {n_noise}")

axes[0].set_ylabel("Feature 2")
plt.tight_layout()
plt.savefig("choosing_dbscan_min_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Key insight ───────────────────────────────────────────────────────────
print("\nKey insight:")
print("  Low  min_samples -> more points qualify as core points -> less noise")
print("  High min_samples -> fewer core points -> more points labelled as noise")
print(f"\n  Rule of thumb: min_samples = 2 * n_features = {2*X.shape[1]}")