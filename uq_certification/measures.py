"""Quantifiable measurement mechanisms: calibration, error detection, per-cell accuracy."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE (Guo et al., 2017) for a binary classifier.

    Uses the confidence of the predicted class, so a model that is 80% sure
    should be right 80% of the time.
    """
    conf, correct = _confidence_and_correct(p, y)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Mean confidence, accuracy and count per confidence bin (for plotting)."""
    conf, correct = _confidence_and_correct(p, y)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = [
        (conf[bins == b].mean(), correct[bins == b].mean(), (bins == b).sum())
        for b in range(n_bins)
        if (bins == b).any()
    ]
    return np.array(rows)


def _confidence_and_correct(p, y):
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    return conf, (pred == y).astype(float)


def error_detection_auroc(uncertainty: np.ndarray, p: np.ndarray, y: np.ndarray) -> float:
    """How well uncertainty ranks mistakes above correct predictions (0.5 = chance)."""
    errors = ((p >= 0.5).astype(int) != y).astype(int)
    if errors.min() == errors.max():
        return float("nan")
    return float(roc_auc_score(errors, uncertainty))


def error_by_uncertainty_quantile(uncertainty, p, y, n_bins: int = 10):
    """Error rate within each uncertainty decile (the paper's preliminary finding)."""
    errors = ((p >= 0.5).astype(int) != y).astype(float)
    edges = np.quantile(uncertainty, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, uncertainty, side="right") - 1, 0, n_bins - 1)
    return np.array([errors[bins == b].mean() if (bins == b).any() else np.nan for b in range(n_bins)])


def risk_coverage(uncertainty, p, y):
    """Error rate on the retained samples when the most uncertain are deferred."""
    errors = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(uncertainty)
    cum_err = np.cumsum(errors[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return coverage, cum_err


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided lower confidence bound on a binomial proportion."""
    if n == 0:
        return 0.0
    z = stats.norm.ppf(confidence)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return float((centre - margin) / denom)


def per_cell_accuracy(p, y, cells, n_cells: int):
    """Accuracy, count and Wilson lower bound for each operational-domain cell."""
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    rows = []
    for c in range(n_cells):
        mask = cells == c
        n = int(mask.sum())
        k = int(correct[mask].sum())
        rows.append((c, n, k / n if n else np.nan, wilson_lower_bound(k, n)))
    return rows
