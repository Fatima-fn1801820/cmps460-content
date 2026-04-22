"""
Interactive gradient descent animation for f(x) = x^2.

Controls:
- Start/Stop animation
- Reset trajectory
- Set initial x value
- Set learning rate alpha
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


@dataclass
class Config:
    initial_x: float = 4.5
    alpha: float = 0.15
    frame_interval_ms: int = 180
    max_iters: int = 250
    convergence_tolerance: float = 1e-8


class GradientDescentAnimator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.current_x = config.initial_x
        self.iteration = 0
        self.running = False
        self.converged = False
        self.converged_iteration: int | None = None

        self.x_hist = [self.current_x]
        self.y_hist = [self.current_x**2]

        self.fig: plt.Figure
        self.ax_main: plt.Axes
        self.point = None
        self.path_line = None
        self.status_text = None
        self.start_button: Button
        self.stop_button: Button
        self.reset_button: Button
        self.x_slider: Slider
        self.alpha_slider: Slider
        self.animation: FuncAnimation

        self._build_figure()
        self._build_controls()
        self._build_animation()
        self._redraw_state()

    def _build_figure(self) -> None:
        self.fig, self.ax_main = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.95, bottom=0.30, top=0.92)

        xs = np.linspace(-6, 6, 600)
        ys = xs**2

        self.ax_main.plot(xs, ys, color="steelblue", linewidth=2, label="f(x) = x^2")
        self.ax_main.axhline(0, color="gray", linewidth=0.8)
        self.ax_main.axvline(0, color="gray", linewidth=0.8)
        self.ax_main.set_xlim(-6, 6)
        self.ax_main.set_ylim(0, 36)
        self.ax_main.set_xlabel("x")
        self.ax_main.set_ylabel("f(x)")
        self.ax_main.set_title("Gradient Descent on f(x) = x^2")
        self.ax_main.grid(alpha=0.25)

        self.point, = self.ax_main.plot(
            [], [], "o", color="crimson", markersize=8, label="Current point"
        )
        self.path_line, = self.ax_main.plot(
            [], [], "-", color="crimson", alpha=0.6, linewidth=1.8, label="Descent path"
        )
        self.status_text = self.ax_main.text(
            0.02,
            0.98,
            "",
            transform=self.ax_main.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        self.ax_main.legend(loc="upper right")

    def _build_controls(self) -> None:
        start_ax = self.fig.add_axes([0.08, 0.18, 0.16, 0.07])
        stop_ax = self.fig.add_axes([0.27, 0.18, 0.16, 0.07])
        reset_ax = self.fig.add_axes([0.46, 0.18, 0.16, 0.07])
        # Keep a larger center gap so slider labels and value texts do not overlap.
        x_slider_ax = self.fig.add_axes([0.08, 0.08, 0.33, 0.05])
        alpha_slider_ax = self.fig.add_axes([0.60, 0.08, 0.33, 0.05])

        self.start_button = Button(start_ax, "Start", color="#d6f5d6", hovercolor="#b9ecb9")
        self.stop_button = Button(stop_ax, "Stop", color="#f5d6d6", hovercolor="#ecb9b9")
        self.reset_button = Button(reset_ax, "Reset", color="#d9e5ff", hovercolor="#c4d6ff")

        self.x_slider = Slider(
            ax=x_slider_ax,
            label="Initial x",
            valmin=-5.0,
            valmax=5.0,
            valinit=self.config.initial_x,
            valstep=0.1,
        )
        self.alpha_slider = Slider(
            ax=alpha_slider_ax,
            label="Learning rate (alpha)",
            valmin=0.01,
            valmax=1.0,
            valinit=self.config.alpha,
            valstep=0.01,
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

    def _step_gradient_descent(self) -> None:
        alpha = float(self.alpha_slider.val)
        gradient = 2 * self.current_x
        self.current_x = self.current_x - alpha * gradient
        self.iteration += 1
        self.x_hist.append(self.current_x)
        current_y = self.current_x**2
        self.y_hist.append(current_y)

        if current_y <= self.config.convergence_tolerance:
            self.converged = True
            self.converged_iteration = self.iteration
            self.running = False

    def _animate_frame(self, _frame_idx: int):
        if self.running and self.iteration < self.config.max_iters:
            self._step_gradient_descent()
            self._redraw_state()
        return self.point, self.path_line, self.status_text

    def _redraw_state(self) -> None:
        current_y = self.current_x**2
        self.point.set_data([self.current_x], [current_y])
        self.path_line.set_data(self.x_hist, self.y_hist)

        state = "Running" if self.running else "Stopped"
        converge_msg = (
            f"Converged at iter = {self.converged_iteration}"
            if self.converged and self.converged_iteration is not None
            else "Converged at iter = -"
        )
        self.status_text.set_text(
            f"state = {state}\n"
            f"iter = {self.iteration}\n"
            f"x = {self.current_x:.5f}\n"
            f"f(x) = {current_y:.6f}\n"
            f"alpha = {float(self.alpha_slider.val):.2f}\n"
            f"{converge_msg}"
        )
        self.fig.canvas.draw_idle()

    def _on_start(self, _event) -> None:
        if self.converged:
            # Resume from a fresh run if user starts after convergence.
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
        self.current_x = float(self.x_slider.val)
        self.iteration = 0
        self.x_hist = [self.current_x]
        self.y_hist = [self.current_x**2]
        self._redraw_state()

    def show(self) -> None:
        plt.show()


def main() -> None:
    app = GradientDescentAnimator(Config())
    print("Use Start/Stop/Reset and sliders to control the gradient descent animation.")
    app.show()


if __name__ == "__main__":
    main()
