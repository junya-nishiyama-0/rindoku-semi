from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "images" / "neural_network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pos": "#dc2626",
    "neg": "#2563eb",
    "line": "#111827",
    "accent": "#7c3aed",
    "forward": "#2563eb",
    "backward": "#dc2626",
    "node": "#f3f4f6",
    "grid": "#d1d5db",
}

plt.rcParams.update({"font.size": 12, "mathtext.fontset": "cm"})


def save(fig, filename):
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))


# ---------------------------------------------------------------- networks
def draw_layer(ax, x, ys, labels, color="white"):
    for y, label in zip(ys, labels):
        ax.add_patch(Circle((x, y), 0.22, facecolor=color, edgecolor=COLORS["line"], lw=1.5, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=15, zorder=4)


def connect(ax, x0, ys0, x1, ys1):
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0 + 0.22, x1 - 0.22], [y0, y1], color="#6b7280", lw=1, zorder=1)


def network_figures():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    xin, xout = [0.5, -0.5], [0.9, 0.0, -0.9]
    connect(ax, 0, xin, 2, xout)
    draw_layer(ax, 0, xin, ["$x_1$", "$x_2$"])
    draw_layer(ax, 2, xout, ["$y_1$", "$y_2$", "$y_3$"], "#dbeafe")
    ax.text(1, -1.55, r"$\boldsymbol{y} = g(\boldsymbol{x}\boldsymbol{W} + \boldsymbol{b})$", ha="center", fontsize=15)
    ax.set_title("Single-layer", pad=8)

    ax = axes[1]
    hs = [0.9, 0.0, -0.9]
    connect(ax, 0, xin, 2, hs)
    connect(ax, 2, hs, 4, [0.0])
    draw_layer(ax, 0, xin, ["$x_1$", "$x_2$"])
    draw_layer(ax, 2, hs, ["$h_1$", "$h_2$", "$h_3$"], "#ede9fe")
    draw_layer(ax, 4, [0.0], ["$y$"], "#dbeafe")
    ax.text(2, -1.55, r"$\boldsymbol{h} = g^{(1)}(\boldsymbol{x}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)}),\ \ y = g^{(2)}(\boldsymbol{h}\boldsymbol{W}^{(2)} + b^{(2)})$", ha="center", fontsize=13)
    for x, name in [(0, "input"), (2, "hidden"), (4, "output")]:
        ax.text(x, 1.4, name, ha="center", fontsize=11, color="#4b5563")
    ax.set_title("Two-layer", pad=8)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_ylim(-1.8, 1.6)
    axes[0].set_xlim(-0.5, 2.5)
    axes[1].set_xlim(-0.5, 4.5)
    save(fig, "networks.png")


# ---------------------------------------------------------------- logic gates
POINTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
GATES = {
    "AND": ((1, 1, -1.5), [0, 0, 0, 1]),
    "OR": ((1, 1, -0.5), [0, 1, 1, 1]),
    "NAND": ((-1, -1, 1.5), [1, 1, 1, 0]),
    "XOR": (None, [0, 1, 1, 0]),
}


def plot_points(ax, points, outputs, labels=None):
    for (x1, x2), y in zip(points, outputs):
        ax.scatter(
            x1, x2, s=170, zorder=5,
            color=COLORS["pos"] if y else "white",
            edgecolor=COLORS["pos"] if y else COLORS["neg"], linewidth=2,
        )
    if labels:
        for (x1, x2), label in zip(points, labels):
            ax.annotate(label, (x1, x2), xytext=(9, -16), textcoords="offset points", fontsize=11)


def setup_unit_square(ax, xlabel="$x_1$", ylabel="$x_2$"):
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, labelpad=10)
    ax.grid(True, color=COLORS["grid"], alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def draw_boundary(ax, w1, w2, b, color=COLORS["line"], ls="-"):
    xs = np.linspace(-0.4, 1.4, 50)
    ax.plot(xs, -(w1 * xs + b) / w2, color=color, lw=2, ls=ls)


def logic_gate_figure():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (name, (params, outputs)) in zip(axes, GATES.items()):
        setup_unit_square(ax)
        if params:
            draw_boundary(ax, *params)
            w1, w2, b = params
            ax.set_title(f"{name}\n$(w_1, w_2, b)=({w1:g}, {w2:g}, {b:g})$", fontsize=12)
        else:
            ax.set_title("XOR\nno single line separates", fontsize=12)
        plot_points(ax, POINTS, outputs)
    save(fig, "logic_gates.png")


def xor_hidden_figure():
    or_out = [0, 1, 1, 1]
    nand_out = [1, 1, 1, 0]
    xor_out = [0, 1, 1, 0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    setup_unit_square(ax)
    draw_boundary(ax, 1, 1, -0.5, COLORS["accent"])
    draw_boundary(ax, -1, -1, 1.5, "#059669", ls="--")
    ax.text(-0.35, 0.65, "OR", color=COLORS["accent"], fontsize=12)
    ax.text(0.85, 1.25, "NAND", color="#059669", fontsize=12)
    plot_points(ax, POINTS, xor_out, ["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    ax.set_title(r"input space $(x_1, x_2)$")

    ax = axes[1]
    setup_unit_square(ax, "$h_1$ (OR)", "$h_2$\n(NAND)")
    hidden = np.array(list(zip(or_out, nand_out)))
    draw_boundary(ax, 1, 1, -1.5)
    plot_points(ax, hidden, xor_out)
    ax.annotate("(0,1),(1,0)", (1, 1), xytext=(-20, 14), textcoords="offset points", fontsize=11)
    ax.annotate("(0,0)", (0, 1), xytext=(9, -16), textcoords="offset points", fontsize=11)
    ax.annotate("(1,1)", (1, 0), xytext=(9, -16), textcoords="offset points", fontsize=11)
    ax.set_title(r"hidden space $(h_1, h_2)$: AND separates")
    fig.tight_layout(w_pad=3)
    save(fig, "xor_hidden_space.png")


# ---------------------------------------------------------------- universal approximation
def universal_figures():
    xs = np.linspace(-1, 4, 2000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for w, color in [(1, "#93c5fd"), (4, "#3b82f6"), (1000, "#1e3a8a")]:
        ax.plot(xs - 1.5, sigmoid(w * (xs - 1.5)), color=color, lw=2.2, label=f"$w={w}$")
    ax.set_title(r"$\sigma(wx)$: large $w$ $\to$ step")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-2.5, 2.5)

    ax = axes[1]
    s1 = sigmoid(1000 * xs - 1000)
    s2 = -sigmoid(1000 * xs - 2000)
    ax.plot(xs, s1, color="#3b82f6", lw=1.6, ls="--", label=r"$\sigma(1000x-1000)$")
    ax.plot(xs, s2, color="#059669", lw=1.6, ls="--", label=r"$-\sigma(1000x-2000)$")
    ax.plot(xs, s1 + s2, color=COLORS["pos"], lw=2.8, label="sum")
    ax.set_title("two sigmoids make a rectangle")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.set_ylim(-1.25, 1.25)
    for ax in axes:
        ax.grid(True, color=COLORS["grid"], alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("$x$")
    save(fig, "sigmoid_rectangle.png")

    target = lambda x: np.sin(1.5 * x) + 0.3 * x
    xs = np.linspace(0, 4, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, m in zip(axes, [5, 25]):
        edges = np.linspace(0, 4, m + 1)
        approx = np.zeros_like(xs)
        for left, right in zip(edges[:-1], edges[1:]):
            height = target((left + right) / 2)
            approx += height * (sigmoid(1000 * (xs - left)) - sigmoid(1000 * (xs - right)))
        ax.plot(xs, target(xs), color=COLORS["line"], lw=1.6, ls="--", label="target")
        ax.plot(xs, approx, color=COLORS["pos"], lw=2.2, label=f"{m} rectangles ({2 * m} hidden units)")
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        ax.set_ylim(-0.3, 1.9)
        ax.grid(True, color=COLORS["grid"], alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("$x$")
    save(fig, "universal_approximation.png")


# ---------------------------------------------------------------- computation graphs
def draw_graph(nodes, edges, filename, *, mode, figsize, xlim, ylim):
    """nodes: name -> (x, y, label, value, grad, kind); kind is 'leaf' or 'op'."""
    fig, ax = plt.subplots(figsize=figsize)
    for src, dst, local in edges:
        x0, y0 = nodes[src][:2]
        x1, y1 = nodes[dst][:2]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
            shrinkA=26, shrinkB=26, color="#6b7280", lw=1.4, zorder=1,
        ))
        if mode == "local" and local:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.28, local, ha="center", va="top",
                    fontsize=12, color=COLORS["accent"])
    for name, (x, y, label, value, grad, kind) in nodes.items():
        if kind == "leaf":
            ax.text(x, y, label, ha="center", va="center", fontsize=16, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["line"], lw=1.3))
        else:
            ax.add_patch(Circle((x, y), 0.4, facecolor=COLORS["node"], edgecolor=COLORS["line"], lw=1.3, zorder=2))
            ax.text(x, y, label, ha="center", va="center", fontsize=13, zorder=3)
        if mode in ("forward", "backward") and value is not None:
            ax.text(x, y + 0.5, value, ha="center", va="bottom", fontsize=12, color=COLORS["forward"])
        if mode == "backward" and grad is not None:
            ax.text(x, y - 0.5, grad, ha="center", va="top", fontsize=12, color=COLORS["backward"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, filename)


def linear_graphs():
    nodes = {
        "a": (0, 1.4, "$a$", "$-2$", r"$\frac{\partial y}{\partial a}=3$", "leaf"),
        "x": (0, -1.0, "$x$", "$3$", r"$\frac{\partial y}{\partial x}=-2$", "leaf"),
        "alpha": (2, 0.2, r"$\times$", r"$\alpha=-6$", r"$\frac{\partial y}{\partial \alpha}=1$", "op"),
        "b": (2, -1.6, "$b$", "$4$", r"$\frac{\partial y}{\partial b}=1$", "leaf"),
        "y": (4, -0.4, "$+$", "$y=-2$", r"$\frac{\partial y}{\partial y}=1$", "op"),
    }
    edges = [
        ("a", "alpha", r"$\times x$"),
        ("x", "alpha", r"$\times a$"),
        ("alpha", "y", r"$\times 1$"),
        ("b", "y", r"$\times 1$"),
    ]
    kwargs = dict(figsize=(7, 4.6), xlim=(-0.8, 4.8), ylim=(-2.6, 2.3))
    draw_graph(nodes, edges, "graph_linear_forward.png", mode="forward", **kwargs)
    draw_graph(nodes, edges, "graph_linear_backward.png", mode="backward", **kwargs)


def bce_graphs():
    x1, x2, w1, w2 = 1.0, -1.0, 1.0, 0.5
    alpha, beta = w1 * x1, w2 * x2
    mu = alpha + beta
    p = sigmoid(mu)
    loss = -np.log(p)
    g_p = -1 / p
    g_mu = g_p * p * (1 - p)
    f = lambda v: f"{v:.4f}".rstrip("0").rstrip(".")
    print(f"mu={mu}, p={p:.4f}, loss={loss:.4f}, dl/dp={g_p:.4f}, dl/dmu={g_mu:.4f}, "
          f"dl/dw1={g_mu * x1:.4f}, dl/dw2={g_mu * x2:.4f}")

    nodes = {
        "w1": (0, 2.7, "$w_1$", "$1$", rf"${f(g_mu * x1)}$", "leaf"),
        "x1": (0, 1.0, "$x_1$", "$1$", None, "leaf"),
        "w2": (0, -1.0, "$w_2$", "$0.5$", rf"${f(g_mu * x2)}$", "leaf"),
        "x2": (0, -2.7, "$x_2$", "$-1$", None, "leaf"),
        "alpha": (2.2, 1.85, r"$\times$", rf"$\alpha={f(alpha)}$", rf"${f(g_mu)}$", "op"),
        "beta": (2.2, -1.85, r"$\times$", rf"$\beta={f(beta)}$", rf"${f(g_mu)}$", "op"),
        "mu": (4.4, 0, "$+$", rf"$\mu={f(mu)}$", rf"${f(g_mu)}$", "op"),
        "p": (6.4, 0, r"$\sigma$", rf"$p={f(p)}$", rf"${f(g_p)}$", "op"),
        "l": (8.4, 0, r"$-\log$", rf"$\hat{{l}}={f(loss)}$", "$1$", "op"),
    }
    edges = [
        ("w1", "alpha", r"$\times x_1$"),
        ("x1", "alpha", None),
        ("w2", "beta", r"$\times x_2$"),
        ("x2", "beta", None),
        ("alpha", "mu", r"$\times 1$"),
        ("beta", "mu", r"$\times 1$"),
        ("mu", "p", r"$\times p(1-p)$"),
        ("p", "l", r"$\times(-\frac{1}{p})$"),
    ]
    kwargs = dict(figsize=(11, 5.6), xlim=(-0.8, 9.3), ylim=(-3.5, 3.5))
    draw_graph(nodes, edges, "graph_bce_forward.png", mode="forward", **kwargs)
    draw_graph(nodes, edges, "graph_bce_local.png", mode="local", **kwargs)
    draw_graph(nodes, edges, "graph_bce_backward.png", mode="backward", **kwargs)


if __name__ == "__main__":
    network_figures()
    logic_gate_figure()
    xor_hidden_figure()
    universal_figures()
    linear_graphs()
    bce_graphs()
