"""
Choosing DBSCAN's Epsilon (ε) Using the k-Distance / Elbow Method
==================================================================

DBSCAN requires two hyper-parameters:
  • min_samples  – minimum number of points that must be within ε of a core point
  • eps (ε)      – the neighbourhood radius that defines "closeness"

The most principled way to pick a good ε is the **k-distance plot** (also
called the elbow or knee curve).  Here is the idea step by step:

  Step 1 – Fix min_samples (a common rule of thumb: 2 × n_features, minimum 4).
  Step 2 – For every point, compute the distance to its k-th nearest neighbour
            where k = min_samples.
  Step 3 – Sort those distances in ascending order and plot them.
  Step 4 – Look for the "elbow" (a sharp bend in the curve).
            • Points to the LEFT of the elbow have a small k-distance  → they
              live in dense regions and will be classified as core points.
            • Points to the RIGHT of the elbow have a large k-distance → they
              live in sparse regions and will become noise (label = -1).
            The ε value at the elbow is a natural separation between the two
            regimes.

Why does this work?
  If we set ε to the knee value, we are saying: "a core point must have at
  least min_samples neighbours within distance ε."  At the knee the density
  transitions from cluster-like to noise-like, so the knee ε cleanly
  separates clusters from noise.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Step 0 – Create realistic synthetic data
# ---------------------------------------------------------------------------
# We use make_blobs so there are genuine clusters for DBSCAN to discover.
# Using random noise alone (np.random.randn) would produce a single blob with
# no clear elbow, making the method hard to illustrate.

np.random.seed(42)

X_blobs, _ = make_blobs(
    n_samples=300,
    centers=[(-3, -3), (0, 0), (4, 2)],   # three distinct cluster centres
    cluster_std=0.8,
    random_state=42,
)

# Add a sprinkle of uniform noise to simulate real-world outliers
noise = np.random.uniform(-7, 7, size=(30, 2))
X = np.vstack([X_blobs, noise])

# Always scale features so that ε is measured in a meaningful unit
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Dataset shape:", X.shape)
print("(300 clustered points + 30 uniform noise points, then standardised)\n")

# ---------------------------------------------------------------------------
# Step 1 – Choose min_samples
# ---------------------------------------------------------------------------
# Rule of thumb: min_samples ≥ ln(n)  or  min_samples = 2 × n_features.
# For 2-D data with ~330 points both heuristics give roughly 4-6.
# We use 5 here.

min_samples = 5
print(f"min_samples = {min_samples}")
print(
    "  (Rule of thumb: 2 × number of features = 2 × 2 = 4; "
    "we round up to 5 for robustness)\n"
)

# ---------------------------------------------------------------------------
# Step 2 – Compute the k-th nearest-neighbour distance for every point
# ---------------------------------------------------------------------------
# NearestNeighbors with n_neighbors=min_samples returns, for each point, the
# distances to its min_samples closest neighbours (including itself at index 0,
# which always has distance 0).  We take the LAST column (index -1) which is
# the distance to the FURTHEST of the k neighbours – the k-th NN distance.

nbrs = NearestNeighbors(n_neighbors=min_samples)
nbrs.fit(X)
distances, _ = nbrs.kneighbors(X)

# k-th nearest-neighbour distance for each point
k_distances = distances[:, -1]   # shape: (n_samples,)

# Sort in ascending order so the curve goes smoothly from small to large
k_distances_sorted = np.sort(k_distances)

print("k-distance statistics (before sorting):")
print(f"  min  = {k_distances.min():.4f}")
print(f"  mean = {k_distances.mean():.4f}")
print(f"  max  = {k_distances.max():.4f}\n")

# ---------------------------------------------------------------------------
# Step 3 – Detect the knee automatically
# ---------------------------------------------------------------------------
# The knee is where the curve bends most sharply.  A simple robust approach:
# fit a straight line from the first to the last point, then find the point
# with the greatest perpendicular distance from that line.

def find_knee(sorted_distances):
    """
    Return the index of the 'knee' point using the maximum-distance-to-line
    heuristic (the Kneedle concept, implemented without extra dependencies).

    How it works:
      1. Draw a straight line between the first and last points of the curve.
      2. For every point on the curve, compute its perpendicular distance to
         that line.
      3. The point with the LARGEST perpendicular distance is the knee.
    """
    n = len(sorted_distances)
    # Index array (x-axis of the plot)
    x = np.arange(n)
    y = sorted_distances

    # Vector from first to last point
    x_start, y_start = x[0],  y[0]
    x_end,   y_end   = x[-1], y[-1]

    # Perpendicular distance from each point (xi, yi) to the line
    # defined by (x_start, y_start) – (x_end, y_end)
    numerator   = abs((y_end - y_start) * x - (x_end - x_start) * y
                      + x_end * y_start - y_end * x_start)
    denominator = np.sqrt((y_end - y_start)**2 + (x_end - x_start)**2)
    distances_to_line = numerator / (denominator + 1e-12)

    knee_idx = int(np.argmax(distances_to_line))
    return knee_idx

knee_idx = find_knee(k_distances_sorted)
eps_auto = k_distances_sorted[knee_idx]

print(f"Automatically detected knee index : {knee_idx}")
print(f"Suggested eps from elbow          : {eps_auto:.4f}\n")

# ---------------------------------------------------------------------------
# Step 4 – Fit DBSCAN using the detected ε
# ---------------------------------------------------------------------------

eps = eps_auto   # ← this is the value chosen from the elbow plot

db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Labels: -1 means noise; 0, 1, 2, … are cluster ids
n_clusters = len(set(labels) - {-1})
n_noise    = int(np.sum(labels == -1))

print("DBSCAN results:")
print(f"  eps               = {eps:.4f}")
print(f"  min_samples       = {min_samples}")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points      : {n_noise} ({100*n_noise/len(labels):.1f} %)\n")

# ---------------------------------------------------------------------------
# Step 5 – Visualise: elbow plot  +  final clusters side by side
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Choosing DBSCAN's ε with the k-Distance Elbow Method",
             fontsize=14, fontweight="bold")

# ── Left panel: k-distance / elbow plot ─────────────────────────────────────
ax = axes[0]
ax.plot(k_distances_sorted, color="steelblue", linewidth=1.8,
        label=f"{min_samples}-NN distance (sorted)")

# Mark the knee with a vertical dashed line and an annotation
ax.axvline(x=knee_idx, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Elbow at index {knee_idx}")
ax.axhline(y=eps_auto, color="darkorange", linestyle=":", linewidth=1.5,
           label=f"Suggested ε = {eps_auto:.3f}")
ax.scatter([knee_idx], [eps_auto], color="crimson", zorder=5, s=80)

ax.annotate(
    f"  ε ≈ {eps_auto:.3f}\n  (elbow / knee)",
    xy=(knee_idx, eps_auto),
    xytext=(knee_idx + len(k_distances_sorted) * 0.08, eps_auto * 1.3),
    fontsize=9,
    color="crimson",
    arrowprops=dict(arrowstyle="->", color="crimson"),
)

ax.set_title(f"k-Distance Plot  (k = min_samples = {min_samples})")
ax.set_xlabel("Points sorted by distance (index)")
ax.set_ylabel(f"Distance to {min_samples}-th nearest neighbour")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Add a text box explaining how to read the plot
ax.text(
    0.02, 0.97,
    "← dense region\n(cluster points)\n\n"
    "→ sparse region\n(noise / outliers)",
    transform=ax.transAxes,
    fontsize=8, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
)

# ── Right panel: DBSCAN clustering result ───────────────────────────────────
ax = axes[1]

unique_labels = sorted(set(labels))
# Build a colour map: noise → grey, clusters → tab10 colours
colours = plt.colormaps["tab10"].resampled(max(n_clusters, 1))

for label in unique_labels:
    mask = labels == label
    if label == -1:
        ax.scatter(X[mask, 0], X[mask, 1],
                   c="lightgrey", edgecolors="grey", s=30,
                   linewidths=0.5, label="Noise", zorder=2)
    else:
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=colours(label), edgecolors="black", s=40,
                   linewidths=0.4, label=f"Cluster {label}", zorder=3)

ax.set_title(
    f"DBSCAN Result  (ε={eps:.3f}, min_samples={min_samples})\n"
    f"{n_clusters} cluster(s) found, {n_noise} noise point(s)"
)
ax.set_xlabel("Feature 1 (standardised)")
ax.set_ylabel("Feature 2 (standardised)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("choosing_dbscan_epsilon.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Takeaway summary (printed to console)
# ---------------------------------------------------------------------------
print("=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. SORT k-distances and PLOT them.
   The curve rises slowly in dense regions then sharply for
   sparse (noise) regions.

2. The ELBOW (knee) marks the natural density threshold.
   Points to the LEFT are inside clusters;
   points to the RIGHT will be labelled as noise.

3. Set ε = the k-distance value at the elbow.
   This gives DBSCAN the right scale to separate clusters
   from noise.

4. If DBSCAN finds too many noise points → increase ε (move
   the elbow to the right / raise min_samples).
   If DBSCAN merges clusters → decrease ε.

5. Always STANDARDISE your features first so that ε has a
   consistent meaning across all dimensions.
""")