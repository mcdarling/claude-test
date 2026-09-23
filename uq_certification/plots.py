"""Figures for the demo (static PNGs via matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import measures  # noqa: E402
from .maturity import LEVEL_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "font.size": 9,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def uncertainty_vs_error(outputs: dict, y, path):
    """outputs: {label: predict() dict}. Error by uncertainty decile + risk-coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.38
    for i, (label, out) in enumerate(outputs.items()):
        err = measures.error_by_uncertainty_quantile(out["total"], out["p"], y)
        x = np.arange(1, len(err) + 1) + (i - 0.5) * width
        axes[0].bar(x, err, width=width - 0.04, color=SERIES[i], label=label)
        cov, risk = measures.risk_coverage(out["total"], out["p"], y)
        axes[1].plot(cov, risk, color=SERIES[i], label=label)
    axes[0].set(
        title="Higher uncertainty → more errors",
        xlabel="Uncertainty decile (1 = most confident)",
        ylabel="Error rate",
        xticks=range(1, 11),
    )
    axes[0].grid(axis="x", visible=False)
    axes[1].set(
        title="Deferring uncertain cases lowers error",
        xlabel="Fraction of cases decided automatically",
        ylabel="Error rate on automatic decisions",
    )
    for ax in axes:
        ax.legend()
    _save(fig, path)


def calibration(outputs: dict, y, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0.5, 1], [0.5, 1], color=TEXT_2, lw=1, ls="--", label="perfect calibration")
    for i, (label, out) in enumerate(outputs.items()):
        curve = measures.reliability_curve(out["p"], y)
        ece = measures.expected_calibration_error(out["p"], y)
        ax.plot(curve[:, 0], curve[:, 1], color=SERIES[i], marker="o", ms=4, label=f"{label} (ECE {ece:.3f})")
    ax.set(
        title="Reliability diagram",
        xlabel="Predicted confidence",
        ylabel="Observed accuracy",
        xlim=(0.5, 1.0),
        ylim=(0.3, 1.02),
    )
    ax.legend(loc="lower right")
    _save(fig, path)


def _grid_mean(values, lighting, weather, n=5):
    li = np.minimum((lighting * n).astype(int), n - 1)
    wi = np.minimum((weather * n).astype(int), n - 1)
    grid = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            m = (wi == a) & (li == b)
            if m.any():
                grid[a, b] = values[m].mean()
    return grid


def uncertainty_drivers(panels: list[tuple[str, np.ndarray]], scenes, importance, path):
    """Heatmaps of uncertainty over lighting x weather + scene-parameter importance."""
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.0 * (len(panels) + 1), 3.8))
    vmax = max(np.nanmax(_grid_mean(v, scenes.lighting, scenes.weather)) for _, v in panels)
    for i, (ax, (title, values)) in enumerate(zip(axes, panels)):
        grid = _grid_mean(values, scenes.lighting, scenes.weather)
        im = ax.imshow(grid, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=vmax, extent=(0, 1, 0, 1))
        ax.set(title=title, xlabel="Lighting (0 = night, 1 = day)")
        if i == 0:
            ax.set_ylabel("Weather (0 = clear, 1 = severe)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(False)
        for a in range(grid.shape[0]):
            for b in range(grid.shape[1]):
                val = grid[a, b]
                colour = "white" if val > 0.55 * vmax else TEXT
                ax.text((b + 0.5) / 5, (a + 0.5) / 5, f"{val:.2f}", ha="center", va="center", fontsize=7, color=colour)
    fig.colorbar(im, ax=list(axes[:-1]), shrink=0.8, label="Mean uncertainty (bits)")
    names, vals = zip(*sorted(importance.items(), key=lambda kv: kv[1]))
    axes[-1].barh(names, vals, color=SERIES[0], height=0.6)
    axes[-1].set(title="What drives uncertainty?", xlabel="Permutation importance")
    axes[-1].grid(axis="y", visible=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def closed_loop(histories: dict, added: dict, path):
    """histories: {strategy: [history per seed]}; added: {strategy: Scenes added by loop}."""
    metrics = [
        ("accuracy", "Test accuracy"),
        ("worst_cell_accuracy", "Worst-cell accuracy"),
        ("mean_epistemic", "Mean epistemic uncertainty (bits)"),
        ("ece", "Calibration error (ECE)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    flat = axes.ravel()
    for i, (strategy, runs) in enumerate(histories.items()):
        n_train = [h["n_train"] for h in runs[0]]
        for ax, (key, title) in zip(flat, metrics):
            vals = np.array([[h[key] for h in run] for run in runs])
            mean, sd = vals.mean(axis=0), vals.std(axis=0)
            ax.plot(n_train, mean, color=SERIES[i], marker="o", ms=4, label=strategy)
            ax.fill_between(n_train, mean - sd, mean + sd, color=SERIES[i], alpha=0.15, lw=0)
            ax.set(title=title, xlabel="Training samples")
    for ax in flat[:4]:
        ax.legend()
    for ax, (strategy, scenes) in zip(flat[4:], added.items()):
        i = list(histories).index(strategy)
        ax.scatter(scenes.lighting, scenes.weather, s=6, color=SERIES[i], alpha=0.35, lw=0)
        ax.set(
            title=f"Where the {strategy} strategy added data",
            xlabel="Lighting (0 = night, 1 = day)",
            ylabel="Weather (0 = clear, 1 = severe)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    _save(fig, path)


def tradeoffs(rates, pareto, chosen_idx, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        rates[~pareto, 1], rates[~pareto, 0], c=rates[~pareto, 2], cmap=SEQUENTIAL, s=10, alpha=0.35, lw=0,
        vmin=0, vmax=rates[:, 2].max(),
    )
    sc = ax.scatter(
        rates[pareto, 1], rates[pareto, 0], c=rates[pareto, 2], cmap=SEQUENTIAL, s=26,
        edgecolors=SURFACE, linewidths=1, vmin=0, vmax=rates[:, 2].max(), label="Pareto-optimal policies",
    )
    no_defer = rates[:, 2] == 0
    order = np.argsort(rates[no_defer, 1])
    ax.plot(rates[no_defer, 1][order], rates[no_defer, 0][order], color=SERIES[1], lw=1.5, label="no deferral")
    ax.scatter(
        rates[chosen_idx, 1], rates[chosen_idx, 0], s=140, marker="*", color=SERIES[1],
        edgecolors=TEXT, linewidths=0.8, zorder=5, label="chosen (min expected cost)",
    )
    fig.colorbar(sc, ax=ax, label="Deferral rate (operator workload)")
    ax.set(
        title="Missed drones vs false alarms vs operator load",
        xlabel="False alarm rate",
        ylabel="Miss rate",
        xlim=(0, 0.6),
        ylim=(0, 0.6),
    )
    ax.legend(loc="upper right")
    _save(fig, path)


def maturity(scorecards: dict, path):
    """scorecards: {label: [Assessment, ...]}."""
    labels = list(scorecards)
    chars = [a.characteristic for a in scorecards[labels[0]]]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    height = 0.8 / len(labels)
    y = np.arange(len(chars))
    for i, label in enumerate(labels):
        levels = [a.level for a in scorecards[label]]
        offset = (i - (len(labels) - 1) / 2) * height
        ax.barh(y + offset, levels, height=height - 0.04, color=SERIES[i], label=label)
        for yy, lv in zip(y + offset, levels):
            ax.text(lv + 0.05, yy, f"L{lv} {LEVEL_NAMES[lv]}", va="center", fontsize=8, color=TEXT)
    ax.set(yticks=y, yticklabels=chars, xlim=(0, 5.9), xlabel="Maturity level", title="Toy maturity scorecard")
    ax.set_xticks(range(0, 6))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, path)
