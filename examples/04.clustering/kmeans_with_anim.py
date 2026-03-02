import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import make_blobs

# ---------------------------------------------------------------------------
# Core K-Means Logic
# ---------------------------------------------------------------------------

def initialize_centroids(dataset, k, random_state=42):
    """
    Places k centroids at uniformly random positions inside the data's
    bounding box — NOT sampled from the data points themselves.

    Why this matters for education
    --------------------------------
    When centroids are seeded *from* the data they accidentally start
    near real clusters, so convergence is fast and uninteresting.
    Starting them anywhere inside the bounding box forces every centroid
    to travel visibly across the plot before it settles, making the
    assign → update rhythm much easier to follow frame by frame.

    Parameters
    ----------
    dataset      : ndarray of shape (N, 2)  — the data points
    k            : int                      — number of clusters
    random_state : int                      — seed for reproducibility

    Returns
    -------
    centroids : ndarray of shape (k, 2)
    """
    rng = np.random.default_rng(random_state)
    # Compute the axis-wise min and max of the data
    x_min, y_min = dataset.min(axis=0)
    x_max, y_max = dataset.max(axis=0)
    # Sample k (x, y) positions uniformly within [min, max] on each axis
    xs = rng.uniform(x_min, x_max, size=k)
    ys = rng.uniform(y_min, y_max, size=k)
    return np.column_stack([xs, ys])


def assign_clusters(dataset, centroids):
    """
    Assigns every data point to the nearest centroid.

    For each point we compute the Euclidean distance to every centroid
    and pick the index of the smallest one.

    Parameters
    ----------
    dataset   : ndarray of shape (N, 2)
    centroids : ndarray of shape (k, 2)

    Returns
    -------
    labels : ndarray of shape (N,) with integer cluster IDs in [0, k-1]
    """
    # dataset[:, np.newaxis, :] has shape (N, 1, 2)
    # centroids[np.newaxis, :, :] has shape (1, k, 2)
    # Broadcasting gives differences of shape (N, k, 2)
    diff = dataset[:, np.newaxis, :] - centroids[np.newaxis, :, :]

    # Sum of squared differences along the coordinate axis → shape (N, k)
    squared_distances = np.sum(diff ** 2, axis=2)

    # Index of the closest centroid for each point → shape (N,)
    return np.argmin(squared_distances, axis=1)


def update_centroids(dataset, labels, k):
    """
    Moves each centroid to the mean position of the points assigned to it.

    Parameters
    ----------
    dataset : ndarray of shape (N, 2)
    labels  : ndarray of shape (N,)  — cluster assignments
    k       : int

    Returns
    -------
    new_centroids : ndarray of shape (k, 2)
    """
    new_centroids = np.zeros((k, 2))
    for cluster_id in range(k):
        # Boolean mask selecting all points in this cluster
        mask = labels == cluster_id
        if mask.any():
            # Mean of all points in the cluster = new centroid position
            new_centroids[cluster_id] = dataset[mask].mean(axis=0)
        else:
            # Edge case: empty cluster — keep centroid where it was
            # (This rarely happens with a good initialisation.)
            new_centroids[cluster_id] = 0.0
    return new_centroids


def centroids_converged(old_centroids, new_centroids, tolerance=1e-4):
    """
    Checks whether the centroids have stopped moving.

    Convergence is declared when the maximum shift of any centroid
    is smaller than 'tolerance'.

    Parameters
    ----------
    old_centroids : ndarray of shape (k, 2)
    new_centroids : ndarray of shape (k, 2)
    tolerance     : float

    Returns
    -------
    bool — True if converged, False otherwise
    """
    # Maximum Euclidean shift across all centroids
    max_shift = np.max(np.linalg.norm(new_centroids - old_centroids, axis=1))
    return max_shift < tolerance


def kmeans_step_by_step(dataset, k, max_iterations=20, random_state=42):
    """
    Runs K-Means and yields a snapshot after every meaningful change
    so the animation can display the algorithm state frame by frame.

    The algorithm alternates between two steps:
        1. ASSIGN  — label each point with its nearest centroid
        2. UPDATE  — move each centroid to the mean of its assigned points

    Yields
    ------
    dict with keys:
        'labels'     — current cluster assignment for every point (or None)
        'centroids'  — current centroid positions
        'old_centroids' — centroid positions before the latest move (or None)
        'phase'      — 'init' | 'assign' | 'update' | 'converged'
        'iteration'  — current iteration number (1-based)
    """
    centroids = initialize_centroids(dataset, k, random_state)
    labels = np.full(len(dataset), -1, dtype=int)  # -1 = unassigned

    # --- Frame 0: show the initial random centroids ---
    yield {
        'labels': labels.copy(),
        'centroids': centroids.copy(),
        'old_centroids': None,
        'phase': 'init',
        'iteration': 0,
    }

    for iteration in range(1, max_iterations + 1):

        # ---- ASSIGN STEP ------------------------------------------------
        labels = assign_clusters(dataset, centroids)
        yield {
            'labels': labels.copy(),
            'centroids': centroids.copy(),
            'old_centroids': None,
            'phase': 'assign',
            'iteration': iteration,
        }

        # ---- UPDATE STEP ------------------------------------------------
        old_centroids = centroids.copy()
        centroids = update_centroids(dataset, labels, k)
        yield {
            'labels': labels.copy(),
            'centroids': centroids.copy(),
            'old_centroids': old_centroids.copy(),
            'phase': 'update',
            'iteration': iteration,
        }

        # ---- CONVERGENCE CHECK ------------------------------------------
        if centroids_converged(old_centroids, centroids):
            yield {
                'labels': labels.copy(),
                'centroids': centroids.copy(),
                'old_centroids': None,
                'phase': 'converged',
                'iteration': iteration,
            }
            return  # stop the generator — we are done


# ---------------------------------------------------------------------------
# Visualization Logic
# ---------------------------------------------------------------------------

# A visually distinct palette for up to 8 clusters
CLUSTER_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]

def create_kmeans_animation(dataset, k, max_iterations=20, random_state=42,
                            interval=700):
    """
    Creates and displays an animated scatter plot of the K-Means algorithm.

    Each frame shows one of three phases:
      • init      — initial random centroid placement
      • assign    — points re-coloured to their nearest centroid
      • update    — arrows showing centroid movement to cluster means
      • converged — final result with a congratulatory title

    Parameters
    ----------
    dataset        : ndarray of shape (N, 2)
    k              : int   — number of clusters
    max_iterations : int   — safety cap on iterations
    random_state   : int
    interval       : int   — milliseconds between frames
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#1a1a2e')   # dark background for visual appeal
    ax.set_facecolor('#16213e')

    generator = kmeans_step_by_step(dataset, k, max_iterations, random_state)

    def update(frame_data):
        """Draws a single animation frame."""
        ax.clear()
        ax.set_facecolor('#16213e')

        labels     = frame_data['labels']
        centroids  = frame_data['centroids']
        old_cents  = frame_data['old_centroids']
        phase      = frame_data['phase']
        iteration  = frame_data['iteration']

        # ---- Plot data points -------------------------------------------
        for point_idx, (x, y) in enumerate(dataset):
            cluster_id = labels[point_idx]

            if cluster_id == -1:
                # Unassigned (only during 'init' phase)
                color = '#888888'
            else:
                color = CLUSTER_PALETTE[cluster_id % len(CLUSTER_PALETTE)]

            ax.plot(x, y, 'o', color=color, markersize=5,
                    alpha=0.75, zorder=2)

        # ---- Plot centroids ---------------------------------------------
        for c_idx, (cx, cy) in enumerate(centroids):
            color = CLUSTER_PALETTE[c_idx % len(CLUSTER_PALETTE)]
            ax.plot(cx, cy, '*', color=color, markersize=18,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)

        # ---- Draw movement arrows (UPDATE phase only) -------------------
        if phase == 'update' and old_cents is not None:
            for c_idx in range(k):
                ox, oy = old_cents[c_idx]
                nx, ny = centroids[c_idx]
                # Only draw when the centroid actually moved
                if np.linalg.norm([nx - ox, ny - oy]) > 1e-6:
                    ax.annotate(
                        '',
                        xy=(nx, ny), xytext=(ox, oy),
                        arrowprops=dict(
                            arrowstyle='->', color='white',
                            lw=1.8, connectionstyle='arc3,rad=0.0'
                        ),
                        zorder=6
                    )

        # ---- Title / subtitle -------------------------------------------
        phase_labels = {
            'init':      'Step 0 — Random initialisation of centroids (★)',
            'assign':    f'Step {iteration}a — Assign each point to its nearest centroid',
            'update':    f'Step {iteration}b — Move centroids to cluster means  →',
            'converged': f'✓ Converged after {iteration} iteration(s)!',
        }
        ax.set_title(phase_labels[phase],
                     color='white', fontsize=12, pad=12)

        # ---- Axes styling -----------------------------------------------
        ax.set_xlabel('Feature 1', color='#aaaaaa')
        ax.set_ylabel('Feature 2', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

        # ---- Legend — one entry per cluster -----------------------------
        legend_handles = [
            plt.Line2D([0], [0], marker='*', color='w', linestyle='None',
                       markerfacecolor=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
                       markeredgecolor='white', markersize=12,
                       label=f'Cluster {i + 1} centroid')
            for i in range(k)
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', linestyle='None',
                       markerfacecolor='#888888', markersize=7,
                       label='Unassigned / data points')
        )
        ax.legend(handles=legend_handles, loc='upper right',
                  facecolor='#1a1a2e', edgecolor='#444466',
                  labelcolor='white', fontsize=9)

    # cache_frame_data=False avoids the save_count warning with generators
    ani = FuncAnimation(fig, update, frames=generator,
                        repeat=False, interval=interval,
                        cache_frame_data=False)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Generate synthetic data: 4 overlapping Gaussian blobs
    #
    # cluster_std=1.6 makes the blobs overlap significantly so that
    # boundary points switch cluster assignments across several
    # iterations — giving students more frames to watch the algorithm
    # "think" before it settles on a final partition.
    # ------------------------------------------------------------------
    X, _ = make_blobs(n_samples=240, centers=4, cluster_std=1.6,
                      random_state=7)

    # ------------------------------------------------------------------
    # K-Means parameters
    # ------------------------------------------------------------------
    K = 4            # number of clusters to find
    MAX_ITER = 30    # safety cap — more iterations needed for hard data

    # ------------------------------------------------------------------
    # Run the animated visualisation
    # interval=1200 ms gives viewers time to read each step's title
    # and observe which points changed colour before the next frame.
    # ------------------------------------------------------------------
    create_kmeans_animation(X, k=K, max_iterations=MAX_ITER,
                            random_state=7, interval=1200)
