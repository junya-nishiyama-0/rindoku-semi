from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "images" / "single_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([2.0, 5.0, 4.0, 7.0])

BEST = (1.4, 1.0)
ALTERNATIVE = (-0.5, 5.75)

COLORS = {
    "points": "#202124",
    "best": "#2563eb",
    "alternative": "#dc2626",
    "candidate": "#7c3aed",
    "residual": "#f59e0b",
    "grid": "#d1d5db",
}


def setup_axis(ax, *, title=None):
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_xticks(X)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0, labelpad=12)
    if title:
        ax.set_title(title, pad=12)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter(
        X,
        Y,
        s=90,
        color=COLORS["points"],
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    for index, (x, y) in enumerate(zip(X, Y), start=1):
        ax.annotate(
            f"$({int(x)}, {int(y)})$",
            (x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=10,
        )


def draw_line(ax, a, b, *, color, label=None, linewidth=2.5):
    x_line = np.linspace(0.5, 4.5, 200)
    ax.plot(x_line, a * x_line + b, color=color, linewidth=linewidth, label=label)


def draw_residuals(ax, a, b):
    predictions = a * X + b
    for x, y, prediction in zip(X, Y, predictions):
        ax.plot(
            [x, x],
            [prediction, y],
            color=COLORS["residual"],
            linestyle=(0, (4, 3)),
            linewidth=2.2,
            zorder=3,
        )
    return Y - predictions


def draw_residual_squares(ax, a, b):
    """Draw screen-space squares whose vertical sides are the residuals."""
    predictions = a * X + b
    residuals = Y - predictions

    # Convert a residual's on-screen height to the equivalent x-axis length,
    # so the patches remain visually square even when x/y scales differ.
    origin = ax.transData.transform((0, 0))
    x_pixels_per_unit = abs(ax.transData.transform((1, 0))[0] - origin[0])
    y_pixels_per_unit = abs(ax.transData.transform((0, 1))[1] - origin[1])
    x_min, x_max = ax.get_xlim()

    for x, y, prediction, residual in zip(X, Y, predictions, residuals):
        side_y = abs(residual)
        side_x = side_y * y_pixels_per_unit / x_pixels_per_unit

        # Extend inward at the right edge and outward elsewhere.
        square_x = x - side_x if x + side_x > x_max else x
        square_y = min(y, prediction)

        square = Rectangle(
            (square_x, square_y),
            side_x,
            side_y,
            facecolor=COLORS["residual"],
            edgecolor=COLORS["residual"],
            linewidth=1.6,
            linestyle=(0, (3, 2)),
            alpha=0.22,
            zorder=2,
        )
        ax.add_patch(square)


def save_figure(fig, filename):
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def candidate_lines_figure():
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    setup_axis(ax)
    draw_line(
        ax,
        *BEST,
        color=COLORS["best"],
        label=r"$\hat{y}=1.4x+1.0$",
        linewidth=3,
    )
    draw_line(
        ax,
        0.3,
        3.75,
        color=COLORS["candidate"],
        label=r"$\hat{y}=0.3x+3.75$",
    )
    draw_line(
        ax,
        *ALTERNATIVE,
        color=COLORS["alternative"],
        label=r"$\hat{y}=-0.5x+5.75$",
    )
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=10)
    save_figure(fig, "candidate_lines.png")


def residuals_figure():
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    setup_axis(ax)
    draw_line(ax, *BEST, color=COLORS["best"], label=r"$\hat{y}=1.4x+1.0$")
    residuals = draw_residuals(ax, *BEST)
    predictions = BEST[0] * X + BEST[1]
    label_offsets = {
        1: (-64, 0),
        2: (10, 0),
        3: (10, 0),
        4: (-64, 0),
    }
    for index, (x, y, prediction, residual) in enumerate(
        zip(X, Y, predictions, residuals), start=1
    ):
        x_offset, y_offset = label_offsets[index]
        ax.annotate(
            rf"$\epsilon_{index}={residual:.1f}$",
            (x, (y + prediction) / 2),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color="#92400e",
            fontsize=10,
            va="center",
        )
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, "residuals.png")


def zero_sum_residuals_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    configurations = [
        (*BEST, COLORS["best"]),
        (*ALTERNATIVE, COLORS["alternative"]),
    ]
    for ax, (a, b, color) in zip(axes, configurations):
        setup_axis(ax, title=rf"$\hat{{y}}={a:g}x{b:+g}$")
        draw_line(ax, a, b, color=color)
        residuals = draw_residuals(ax, a, b)
        ax.text(
            0.5,
            0.94,
            rf"$\sum_i \epsilon_i={residuals.sum():.1f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": color},
        )
    save_figure(fig, "zero_sum_residuals.png")


def mse_comparison_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    configurations = [
        (*BEST, COLORS["best"]),
        (*ALTERNATIVE, COLORS["alternative"]),
    ]
    for ax, (a, b, color) in zip(axes, configurations):
        setup_axis(ax, title=rf"$\hat{{y}}={a:g}x{b:+g}$")
        draw_line(ax, a, b, color=color)
        residuals = draw_residuals(ax, a, b)
        mse = np.mean(residuals**2)
        ax.text(
            0.5,
            0.94,
            rf"$\hat{{L}}={mse:.2f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": color},
        )

    # Layout must be finalized before converting y-lengths to screen-space
    # x-lengths for visually square patches.
    fig.canvas.draw()
    for ax, (a, b, _) in zip(axes, configurations):
        draw_residual_squares(ax, a, b)

    save_figure(fig, "mse_comparison.png")


def optimal_line_figure():
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    setup_axis(ax)
    draw_line(
        ax,
        *BEST,
        color=COLORS["best"],
        label=r"$\hat{y}=1.4x+1.0$",
        linewidth=3,
    )
    draw_residuals(ax, *BEST)
    ax.text(
        0.03,
        0.90,
        r"$a^*=1.4,\quad b^*=1.0,\quad \hat{L}=0.80$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": COLORS["best"],
        },
    )
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    save_figure(fig, "optimal_line.png")


def main():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "mathtext.fontset": "stix",
        }
    )
    candidate_lines_figure()
    residuals_figure()
    zero_sum_residuals_figure()
    mse_comparison_figure()
    optimal_line_figure()


if __name__ == "__main__":
    main()
