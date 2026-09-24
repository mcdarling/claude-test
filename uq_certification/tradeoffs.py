"""Multi-objective trade-off between missed detections, false alarms and operator load.

The runtime policy has two knobs:
  t    alarm threshold on P(drone)
  tau  uncertainty threshold above which the case is deferred to a human
       operator (the actionable output of the UQ measurement)

Total uncertainty is the right runtime signal: it flags errors whether they
come from model ignorance or from noisy sensors. Epistemic uncertainty alone
is the right signal for deciding what data to collect (see closed_loop.py).

For simplicity deferred cases are assumed to be resolved correctly by the
operator, at a workload cost. That makes three objectives to minimise:
  miss rate         drones the system auto-dismissed / all drones
  false alarm rate  empty scenes the system auto-alarmed on / all empty scenes
  deferral rate     fraction of cases sent to the operator
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    t: float
    tau: float


def policy_rates(p, uncertainty, y, policy: Policy) -> dict[str, float]:
    deferred = uncertainty > policy.tau
    alarm = (p >= policy.t) & ~deferred
    dismiss = (p < policy.t) & ~deferred
    drones = y == 1
    return {
        "miss": float((dismiss & drones).sum() / max(drones.sum(), 1)),
        "false_alarm": float((alarm & ~drones).sum() / max((~drones).sum(), 1)),
        "deferral": float(deferred.mean()),
    }


def sweep(p, uncertainty, y, n_t: int = 41, taus=None):
    """Evaluate a grid of policies; returns (policies, rates array [n, 3])."""
    if taus is None:
        # "never defer" plus deferring the top 50%, 40%, ... 2% most uncertain
        qs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
        taus = [np.inf] + [float(np.quantile(uncertainty, q)) for q in qs]
    policies, rates = [], []
    for tau in taus:
        for t in np.linspace(0.02, 0.98, n_t):
            policy = Policy(float(t), float(tau))
            r = policy_rates(p, uncertainty, y, policy)
            policies.append(policy)
            rates.append([r["miss"], r["false_alarm"], r["deferral"]])
    return policies, np.array(rates)


def pareto_mask(costs: np.ndarray) -> np.ndarray:
    """True for points not dominated by any other (all objectives minimised)."""
    n = len(costs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        mask[i] = not dominated.any()
    return mask


def choose_policy(policies, rates, weights=(5.0, 1.0, 0.3), base_rate: float = 0.5, max_deferral: float = 0.2):
    """Pick the policy with the lowest expected cost per scene, subject to a
    cap on operator workload (the epsilon-constraint method).

    ``weights`` are the notional costs of one missed drone, one false alarm
    and one operator deferral. Making them explicit is the point: the trade-off
    becomes a documented, auditable decision instead of an ad-hoc threshold.
    A weighted sum alone tends to land on extremes (alarm on everything, or
    defer everything), so operator capacity enters as a hard constraint.
    """
    w_miss, w_fa, w_defer = weights
    cost = (
        w_miss * base_rate * rates[:, 0]
        + w_fa * (1 - base_rate) * rates[:, 1]
        + w_defer * rates[:, 2]
    )
    feasible = rates[:, 2] <= max_deferral
    best = int(np.argmin(np.where(feasible, cost, np.inf)))
    return policies[best], best, cost
