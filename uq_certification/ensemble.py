"""Deep-ensemble detector with uncertainty decomposition.

Each member is a small neural network trained on a bootstrap resample with its
own random initialisation (Lakshminarayanan et al., 2017). Disagreement
between members is the uncertainty signal used throughout the demo.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def _fit_member(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), len(X))  # bootstrap resample
    model = make_pipeline(
        StandardScaler(),
        # Early stopping on a held-out 10% guards against overfitting, which
        # otherwise makes members overconfident (poorly calibrated).
        MLPClassifier(
            hidden_layer_sizes=(64, 64), alpha=1e-3, early_stopping=True, max_iter=400, random_state=seed
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X[idx], y[idx])
    return model


class DeepEnsemble:
    def __init__(self, n_members: int = 5, seed: int = 0, n_jobs: int = 1):
        self.n_members = n_members
        self.seed = seed
        self.n_jobs = n_jobs
        self.members = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        seeds = [self.seed * 1000 + i for i in range(self.n_members)]
        self.members = Parallel(n_jobs=self.n_jobs)(delayed(_fit_member)(X, y, s) for s in seeds)
        return self

    def member_probs(self, X: np.ndarray) -> np.ndarray:
        """P(drone) from each member, shape (n_members, n_samples)."""
        return np.stack([m.predict_proba(X)[:, 1] for m in self.members])

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Mean probability plus total / aleatoric / epistemic uncertainty.

        total      entropy of the mean prediction
        aleatoric  mean entropy of the members (noise inherent in the data)
        epistemic  total - aleatoric = mutual information (member disagreement,
                   reducible with more data)
        """
        probs = self.member_probs(X)
        p_mean = probs.mean(axis=0)
        total = binary_entropy(p_mean)
        aleatoric = binary_entropy(probs).mean(axis=0)
        return {
            "p": p_mean,
            "total": total,
            "aleatoric": aleatoric,
            "epistemic": np.maximum(total - aleatoric, 0.0),
            "variance": probs.var(axis=0),
        }


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy in bits of a Bernoulli(p) prediction."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
