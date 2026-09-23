"""Closed-loop, uncertainty-guided synthetic data generation.

Each round:
  1. train the deep ensemble on the current data,
  2. score a pool of candidate scenes by epistemic uncertainty,
  3. generate new samples *similar* to the most uncertain candidates
     (perturbing their scene parameters),
  4. add them to the training set and repeat.

The "random" strategy spends the same data budget on uniformly sampled
scenes. It is the baseline showing whether uncertainty guidance helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import simulator as sim
from .ensemble import DeepEnsemble
from .measures import expected_calibration_error


@dataclass
class LoopResult:
    strategy: str
    seed: int
    history: list[dict] = field(default_factory=list)
    initial_model: DeepEnsemble | None = None
    final_model: DeepEnsemble | None = None
    train_scenes: sim.Scenes | None = None


def run_closed_loop(
    strategy: str,
    test_scenes: sim.Scenes,
    X_test: np.ndarray,
    seed: int = 0,
    n_rounds: int = 5,
    n_initial: int = 1500,
    budget: int = 400,
    pool_size: int = 4000,
    n_seed_scenes: int = 50,
) -> LoopResult:
    if strategy not in ("uncertainty", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(seed)
    scenes, X = sim.generate(n_initial, rng, biased=True)
    cells = sim.condition_cell(test_scenes)
    result = LoopResult(strategy, seed)

    for round_ in range(n_rounds + 1):
        model = DeepEnsemble(seed=seed * 100 + round_).fit(X, scenes.drone)
        if round_ == 0:
            result.initial_model = model
        result.history.append(_evaluate(model, X_test, test_scenes.drone, cells, round_, len(X)))
        if round_ == n_rounds:
            break
        if strategy == "uncertainty":
            pool_scenes, X_pool = sim.generate(pool_size, rng)
            epistemic = model.predict(X_pool)["epistemic"]
            top = np.argsort(epistemic)[-n_seed_scenes:]
            new_scenes = sim.perturb_scenes(pool_scenes.subset(top), budget // n_seed_scenes, rng)
        else:
            new_scenes = sim.sample_scenes(budget, rng)
        scenes = sim.Scenes.concat([scenes, new_scenes])
        X = np.vstack([X, sim.render_features(new_scenes, rng)])

    result.final_model = model
    result.train_scenes = scenes
    return result


def _evaluate(model, X_test, y_test, cells, round_, n_train):
    out = model.predict(X_test)
    correct = (out["p"] >= 0.5).astype(int) == y_test
    cell_acc = [correct[cells == c].mean() for c in range(sim.N_CELLS)]
    return {
        "round": round_,
        "n_train": n_train,
        "accuracy": float(correct.mean()),
        "worst_cell_accuracy": float(np.min(cell_acc)),
        "ece": expected_calibration_error(out["p"], y_test),
        "mean_epistemic": float(out["epistemic"].mean()),
        "mean_aleatoric": float(out["aleatoric"].mean()),
    }
