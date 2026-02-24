import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons

# --- Core DBSCAN Logic ---

def get_neighbors(dataset, point_index, eps):
    """
    Finds the indices of all points within distance 'eps' of a given point.
    
    This uses efficient NumPy broadcasting to calculate all distances at once.
    """
    # Select the specific point from the dataset
    point = dataset[point_index]
    # Calculate the Euclidean distance from the point to every other point.
    # Step 1: subtract the point's coordinates from every row  → shape (N, 2)
    diff = dataset - point
    # Step 2: square each difference and sum across the two coordinates (axis=1)
    squared_distances = np.sum(diff ** 2, axis=1)
    # Step 3: take the square root  →  distance = sqrt((x2-x1)² + (y2-y1)²)
    distances = np.sqrt(squared_distances)
    # Return the indices of points that are within the eps radius
    return np.where(distances < eps)[0]

def dbscan_step_by_step(dataset, eps, min_pts):
    """
    Performs the DBSCAN clustering algorithm step by step.
    This is a generator — it yields a snapshot of the algorithm state
    after each meaningful change so the animation can display it.

    Label conventions:
        0  = unvisited
       -1  = noise
       1+  = cluster ID

    Yields:
        dict with keys:
            'labels'         - current label for every point
            'processing_idx' - index of the point being examined right now
            'neighbors_idx'  - neighbors found for the current point
    """
    labels = np.zeros(len(dataset), dtype=int)
    cluster_id = 0

    # Show the initial blank state
    yield {'labels': labels.copy(), 'processing_idx': None, 'neighbors_idx': None}

    for point_index in range(len(dataset)):
        # Skip points already classified
        if labels[point_index] != 0:
            continue

        # --- Check whether this point is a core point ---
        neighbors = get_neighbors(dataset, point_index, eps)
        yield {'labels': labels.copy(), 'processing_idx': point_index, 'neighbors_idx': neighbors}

        if len(neighbors) < min_pts:
            # Not enough neighbors → noise
            labels[point_index] = -1
            yield {'labels': labels.copy(), 'processing_idx': point_index, 'neighbors_idx': neighbors}
            continue

        # --- Core point: grow a new cluster ---
        cluster_id += 1
        labels[point_index] = cluster_id
        yield {'labels': labels.copy(), 'processing_idx': point_index, 'neighbors_idx': neighbors}

        # 'seeds' holds every point that still needs to be examined for this cluster.
        # We iterate with a plain for-loop; Python allows appending to the list
        # mid-loop, so newly discovered neighbors are processed automatically.
        seeds = list(neighbors)

        for seed_index in seeds:
            if labels[seed_index] == -1:
                # Former noise point → now a border point of this cluster
                labels[seed_index] = cluster_id
                yield {'labels': labels.copy(), 'processing_idx': seed_index, 'neighbors_idx': []}

            elif labels[seed_index] == 0:
                # Unvisited point → join this cluster
                labels[seed_index] = cluster_id
                seed_neighbors = get_neighbors(dataset, seed_index, eps)
                yield {'labels': labels.copy(), 'processing_idx': seed_index, 'neighbors_idx': seed_neighbors}

                # If it is also a core point, add its unvisited neighbors to seeds
                if len(seed_neighbors) >= min_pts:
                    for n in seed_neighbors:
                        if n not in seeds:
                            seeds.append(n)

# --- Visualization Logic ---

def create_dbscan_animation(dataset, eps, min_pts):
    """
    Creates and displays an animated scatter plot of the DBSCAN algorithm.
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    dbscan_generator = dbscan_step_by_step(dataset, eps, min_pts)
    
    # Define a color map for clusters.
    # np.max(labels) cannot be used before labels are known, so we estimate
    # a reasonable upper bound: one color per potential cluster + noise/unvisited.
    # int() is required because the lut argument must be an integer.
    cluster_colors = plt.colormaps.get_cmap('tab10')

    def get_point_types(labels, dataset, eps, min_pts):
        """Classify each point as core, border, noise, or unvisited."""
        # Use a set for O(1) membership checks
        core_set = {
            i for i in range(len(dataset))
            if labels[i] > 0 and len(get_neighbors(dataset, i, eps)) >= min_pts
        }

        point_types = {}
        for i in range(len(dataset)):
            if labels[i] > 0:       # assigned to a cluster
                point_types[i] = 'core' if i in core_set else 'border'
            elif labels[i] == -1:   # explicitly marked as noise
                point_types[i] = 'noise'
            else:                   # label == 0 → not yet visited
                point_types[i] = 'unvisited'
        return point_types

    def update(frame_data):
        """The function that draws each frame of the animation."""
        ax.clear()
        
        labels = frame_data['labels']
        processing_idx = frame_data['processing_idx']
        neighbors_idx = frame_data['neighbors_idx']
        
        point_types = get_point_types(labels, dataset, eps, min_pts)

        # Plot each point based on its determined type
        for i, (x, y) in enumerate(dataset):
            if point_types[i] == 'core':
                # Normalize cluster id to [0, 1] for the colormap
                ax.plot(x, y, 'o', c=cluster_colors(labels[i] / 10), markersize=10, zorder=3)
            elif point_types[i] == 'border':
                ax.plot(x, y, 'x', c=cluster_colors(labels[i] / 10), markersize=8, zorder=2)
            elif point_types[i] == 'noise':
                ax.plot(x, y, '.', c='gray', markersize=5, zorder=1)
            else: # unvisited
                ax.plot(x, y, '.', c='black', markersize=5, zorder=1)

        # Highlight the point currently being processed
        if processing_idx is not None:
            ax.scatter(dataset[processing_idx, 0], dataset[processing_idx, 1],
                       facecolors='none', edgecolors='red', s=200, linewidths=2, zorder=4,
                       label='Processing Point')
            
            # Draw a circle representing the eps radius
            circle = plt.Circle((dataset[processing_idx, 0], dataset[processing_idx, 1]), eps,
                                color='red', fill=False, linestyle='--', alpha=0.5, zorder=4)
            ax.add_artist(circle)

        ax.set_title(f"DBSCAN Clustering (eps={eps}, min_pts={min_pts})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend([plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='green', markersize=10),
                   plt.Line2D([0], [0], color='w', marker='x', markerfacecolor='blue', markeredgecolor='blue', markersize=8),
                   plt.Line2D([0], [0], color='w', marker='.', markerfacecolor='gray', markersize=5),
                   plt.Line2D([0], [0], color='w', marker='.', markerfacecolor='black', markersize=5)],
                  ['Core', 'Border', 'Noise', 'Unvisited'], loc='upper right')
        ax.set_aspect('equal', adjustable='box')


    # Create the animation.
    # cache_frame_data=False avoids the save_count warning when using a generator.
    ani = FuncAnimation(fig, update, frames=dbscan_generator, repeat=False,
                        interval=100, cache_frame_data=False)
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Generate sample data for demonstration
    X, y = make_moons(n_samples=200, noise=0.07, random_state=42)

    # Set DBSCAN parameters
    EPSILON = 0.15
    MIN_POINTS = 5

    # Run the animation
    create_dbscan_animation(X, EPSILON, MIN_POINTS)
