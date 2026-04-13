"""
Live gradient descent animation for f(x) = x^2.

Concise idea:
- Function: f(x) = x^2
- Gradient: f'(x) = 2x
- Update rule: x <- x - alpha * 2x
Each frame performs one update and shows the new point on the curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main() -> None:
    # Hyperparameters and starting point
    alpha = 0.15
    x = 4.5
    max_iters = 45

    # Curve for visualization
    xs = np.linspace(-5, 5, 500)
    ys = xs**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="steelblue", linewidth=2, label="f(x) = x^2")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 26)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Live Gradient Descent on f(x) = x^2")
    ax.grid(alpha=0.25)

    # Animated point and path
    point, = ax.plot([], [], "o", color="crimson", markersize=8, label="current x")
    path_line, = ax.plot([], [], "-", color="crimson", alpha=0.6, linewidth=1.5, label="descent path")
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="upper right")

    x_hist = [x]
    y_hist = [x**2]

    def init():
        point.set_data([x_hist[0]], [y_hist[0]])
        path_line.set_data(x_hist, y_hist)
        text.set_text(f"iter = 0\nx = {x_hist[0]:.4f}\nf(x) = {y_hist[0]:.4f}")
        return point, path_line, text

    def update(frame_idx: int):
        nonlocal x
        grad = 2 * x
        x = x - alpha * grad
        y = x**2

        x_hist.append(x)
        y_hist.append(y)

        point.set_data([x], [y])
        path_line.set_data(x_hist, y_hist)
        text.set_text(f"iter = {frame_idx + 1}\nx = {x:.4f}\nf(x) = {y:.6f}")
        return point, path_line, text

    FuncAnimation(
        fig,
        update,
        frames=max_iters,
        init_func=init,
        interval=180,   # milliseconds between steps (live feel)
        blit=True,
        repeat=False,
    )

    print("Live explanation: each frame applies x <- x - alpha*(2x), so x moves toward 0.")
    plt.show()


if __name__ == "__main__":
    main()
