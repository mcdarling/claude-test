"""Split conformal prediction for the binary detector (Shafer & Vovk, 2008).

A prediction set contains every label whose nonconformity score is below a
threshold learned on a held-out calibration set. With probability at least
1 - alpha the set contains the true label. That guarantee is the formal
property the paper asks measurement mechanisms to provide.

"Mondrian" (group-conditional) conformal learns a separate threshold per
group, so the guarantee holds *within* every group. Here a group is an
operational-domain cell combined with the candidate label. Coverage then
holds per condition and per class, and coverage of the drone class is a
bound on the missed-detection rate.

Sets of size 2 ({no drone, drone}) mean the model can't decide at the
requested confidence. They are the natural trigger for operator review.
"""

from __future__ import annotations

import numpy as np


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return 1.0  # too few calibration points: include every label
    return float(np.quantile(scores, level, method="higher"))


class ConformalDetector:
    def __init__(self, alpha: float = 0.1, mondrian: bool = False):
        self.alpha = alpha
        self.mondrian = mondrian

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray, groups_cal: np.ndarray | None = None):
        scores = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
        if not self.mondrian:
            self.q_global = _quantile(scores, self.alpha)
            return self
        self.q = {}
        for g in np.unique(groups_cal):
            for label in (0, 1):
                mask = (groups_cal == g) & (y_cal == label)
                self.q[(g, label)] = _quantile(scores[mask], self.alpha)
        return self

    def predict_sets(self, p: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Boolean array (n, 2): column k is True if label k is in the set."""
        score = np.column_stack([p, 1 - p])  # score of label 0 is p, of label 1 is 1 - p
        if not self.mondrian:
            return score <= self.q_global
        q = np.array([[self.q.get((g, 0), 1.0), self.q.get((g, 1), 1.0)] for g in groups])
        return score <= q


def coverage(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample indicator that the true label is in the prediction set."""
    return sets[np.arange(len(y)), y]
