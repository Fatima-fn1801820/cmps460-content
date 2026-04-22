"""
Interactive gradient descent animation for linear regression.

Controls:
- Start/Stop animation
- Reset run
- Set initial w and b
- Set learning rate alpha
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


@dataclass
class Config:
    initial_w: float = 0.0
    initial_b: float = 0.0
    alpha: float = 0.05
    frame_interval_ms: int = 220
    max_iters: int = 250
    convergence_tolerance: float = 1e-5


class LinearRegressionGDAnimator:
    def __init__(self, config: Config) -> None:
        self.config = config

        # Small synthetic dataset (roughly y = 2x + 1 with noise).
        self.x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        self.y = np.array([1.0, 3.0, 4.8, 7.2, 8.9, 10.8], dtype=float)

        self.w = config.initial_w
        self.b = config.initial_b
        self.iteration = 0
        self.running = False
        self.converged = False
        self.converged_iteration: int | None = None

        self.w_history: list[float] = [self.w]
        self.b_history: list[float] = [self.b]
        self.cost_history: list[float] = [self._compute_mse()]

        self.fig: plt.Figure
        self.ax_data: plt.Axes
        self.ax_cost: plt.Axes
        self.data_scatter = None
        self.fit_line = None
        self.path3d = None
        self.current_point3d = None
        self.status_text = None
        self.start_button: Button
        self.stop_button: Button
        self.reset_button: Button
        self.w_slider: Slider
        self.b_slider: Slider
        self.alpha_slider: Slider
        self.animation: FuncAnimation

        self._build_figure()
        self._build_controls()
        self._build_animation()
        self._redraw_state()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(13, 7))
        self.fig.subplots_adjust(left=0.06, right=0.98, bottom=0.33, top=0.90, wspace=0.22)
        self.ax_data = self.fig.add_subplot(1, 2, 1)
        self.ax_cost = self.fig.add_subplot(1, 2, 2, projection="3d")

        # Left: data + model line
        self.data_scatter = self.ax_data.scatter(
            self.x,
            self.y,
            color="royalblue",
            edgecolors="white",
            linewidths=1.0,
            s=65,
            zorder=3,
            label="Data",
        )
        x_line = np.linspace(self.x.min() - 0.5, self.x.max() + 0.5, 200)
        y_line = self.w * x_line + self.b
        self.fit_line, = self.ax_data.plot(
            x_line, y_line, color="crimson", linewidth=2, alpha=0.85, zorder=2, label="Model line"
        )
        self.ax_data.set_title("Linear Regression Fit")
        self.ax_data.set_xlabel("x")
        self.ax_data.set_ylabel("y")
        self.ax_data.grid(alpha=0.25)
        self.ax_data.legend(loc="upper right")
        self.status_text = self.ax_data.text(
            0.03,
            0.97,
            "",
            transform=self.ax_data.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # Right: 3D cost surface J(w,b) and GD path
        w_grid = np.linspace(-2.0, 4.0, 70)
        b_grid = np.linspace(-2.0, 4.0, 70)
        W, B = np.meshgrid(w_grid, b_grid)
        Z = self._compute_mse_grid(W, B)

        self.ax_cost.plot_surface(W, B, Z, cmap="viridis", alpha=0.65, linewidth=0)
        self.path3d, = self.ax_cost.plot(
            [], [], [], color="crimson", marker="o", markersize=3, linewidth=2, label="GD path"
        )
        self.current_point3d, = self.ax_cost.plot(
            [], [], [], "o", color="black", markersize=6, label="Current point"
        )
        self.ax_cost.set_title("MSE Surface J(w, b)")
        self.ax_cost.set_xlabel("w")
        self.ax_cost.set_ylabel("b")
        self.ax_cost.set_zlabel("MSE")
        self.ax_cost.set_xlim(-2.0, 4.0)
        self.ax_cost.set_ylim(-2.0, 4.0)
        self.ax_cost.view_init(elev=30, azim=-55)
        self.ax_cost.legend(loc="upper left")

    def _build_controls(self) -> None:
        start_ax = self.fig.add_axes([0.08, 0.20, 0.14, 0.07])
        stop_ax = self.fig.add_axes([0.25, 0.20, 0.14, 0.07])
        reset_ax = self.fig.add_axes([0.42, 0.20, 0.14, 0.07])

        w_slider_ax = self.fig.add_axes([0.08, 0.10, 0.22, 0.04])
        b_slider_ax = self.fig.add_axes([0.38, 0.10, 0.22, 0.04])
        alpha_slider_ax = self.fig.add_axes([0.68, 0.10, 0.24, 0.04])

        self.start_button = Button(start_ax, "Start", color="#d6f5d6", hovercolor="#b9ecb9")
        self.stop_button = Button(stop_ax, "Stop", color="#f5d6d6", hovercolor="#ecb9b9")
        self.reset_button = Button(reset_ax, "Reset", color="#d9e5ff", hovercolor="#c4d6ff")

        self.w_slider = Slider(
            ax=w_slider_ax,
            label="Initial w",
            valmin=-2.0,
            valmax=4.0,
            valinit=self.config.initial_w,
            valstep=0.1,
        )
        self.b_slider = Slider(
            ax=b_slider_ax,
            label="Initial b",
            valmin=-2.0,
            valmax=4.0,
            valinit=self.config.initial_b,
            valstep=0.1,
        )
        self.alpha_slider = Slider(
            ax=alpha_slider_ax,
            label="alpha",
            valmin=0.001,
            valmax=0.2,
            valinit=self.config.alpha,
            valstep=0.001,
        )

        self.start_button.on_clicked(self._on_start)
        self.stop_button.on_clicked(self._on_stop)
        self.reset_button.on_clicked(self._on_reset)

    def _build_animation(self) -> None:
        self.animation = FuncAnimation(
            self.fig,
            self._animate_frame,
            interval=self.config.frame_interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def _compute_mse(self) -> float:
        y_hat = self.w * self.x + self.b
        return float(np.mean((y_hat - self.y) ** 2))

    def _compute_mse_grid(self, W: np.ndarray, B: np.ndarray) -> np.ndarray:
        predictions = W[..., np.newaxis] * self.x + B[..., np.newaxis]
        return np.mean((predictions - self.y) ** 2, axis=2)

    def _gradient_step(self) -> None:
        m = self.x.shape[0]
        error = (self.w * self.x + self.b) - self.y

        dw = (1 / m) * np.sum(error * self.x)
        db = (1 / m) * np.sum(error)

        alpha = float(self.alpha_slider.val)
        self.w = self.w - alpha * dw
        self.b = self.b - alpha * db
        self.iteration += 1

        mse = self._compute_mse()
        self.w_history.append(self.w)
        self.b_history.append(self.b)
        self.cost_history.append(mse)

        if mse <= self.config.convergence_tolerance:
            self.converged = True
            self.converged_iteration = self.iteration
            self.running = False

    def _animate_frame(self, _frame_idx: int):
        if self.running and self.iteration < self.config.max_iters:
            self._gradient_step()
            self._redraw_state()
        return self.fit_line, self.path3d, self.current_point3d, self.status_text

    def _redraw_state(self) -> None:
        x_line = np.linspace(self.x.min() - 0.5, self.x.max() + 0.5, 200)
        y_line = self.w * x_line + self.b
        self.fit_line.set_data(x_line, y_line)

        mse = self._compute_mse()
        self.path3d.set_data(self.w_history, self.b_history)
        self.path3d.set_3d_properties(self.cost_history)
        self.current_point3d.set_data([self.w], [self.b])
        self.current_point3d.set_3d_properties([mse])

        state = "Running" if self.running else "Stopped"
        converge_msg = (
            f"Converged at iter = {self.converged_iteration}"
            if self.converged and self.converged_iteration is not None
            else "Converged at iter = -"
        )

        self.status_text.set_text(
            f"state = {state}\n"
            f"iter = {self.iteration}\n"
            f"w = {self.w:.4f}\n"
            f"b = {self.b:.4f}\n"
            f"mse = {mse:.6f}\n"
            f"alpha = {float(self.alpha_slider.val):.3f}\n"
            f"{converge_msg}"
        )
        self.fig.canvas.draw_idle()

    def _on_start(self, _event) -> None:
        if self.converged:
            self._on_reset(_event)
        self.running = True
        self._redraw_state()

    def _on_stop(self, _event) -> None:
        self.running = False
        self._redraw_state()

    def _on_reset(self, _event) -> None:
        self.running = False
        self.converged = False
        self.converged_iteration = None
        self.w = float(self.w_slider.val)
        self.b = float(self.b_slider.val)
        self.iteration = 0
        self.w_history = [self.w]
        self.b_history = [self.b]
        self.cost_history = [self._compute_mse()]
        self._redraw_state()

    def show(self) -> None:
        plt.show()


def main() -> None:
    app = LinearRegressionGDAnimator(Config())
    print("Use Start/Stop/Reset and sliders to explore gradient descent.")
    app.show()


if __name__ == "__main__":
    main()
