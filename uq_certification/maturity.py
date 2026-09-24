"""A toy maturity scorecard: measurements to levels with explicit evidence.

Follows the paper's structure: each trustworthiness characteristic has levels
1-5, each level lists the evidence required, and levels are cumulative (level
3 needs everything from levels 1-3). The numeric thresholds are notional and
meant for illustration only. They were fixed before the first run, except the
false-alarm limit in Safety L3, which was added after that run exposed a
loophole (see results/report.md).

Level 5 needs formal verification of system components. Nothing in this demo
attempts that, so it is always reported as not met.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import simulator as sim
from .conformal import coverage
from .measures import error_detection_auroc, expected_calibration_error, wilson_lower_bound

LEVEL_NAMES = {
    0: "Not assessed",
    1: "Ad-hoc",
    2: "Structured",
    3: "Measurement-driven",
    4: "Statistical guarantees",
    5: "Formal verification",
}

# Thresholds (all fixed before the first run except FALSE_ALARM_L3).
MIN_CELL_SAMPLES = 50
WORST_CELL_ACC = 0.70
ECE_L2, ECE_L3 = 0.10, 0.05
AUROC_L3 = 0.75
TARGET_COVERAGE = 0.90
COVERAGE_TOL = 0.02
MAX_SET_DEFERRAL = 0.30
MISS_L3, FALSE_ALARM_L3, DEFER_L3 = 0.05, 0.10, 0.25
SIGNIFICANCE = 0.05


@dataclass
class Criterion:
    level: int
    requirement: str
    passed: bool
    evidence: str


@dataclass
class Assessment:
    characteristic: str
    criteria: list[Criterion]

    @property
    def level(self) -> int:
        achieved = 0
        for level in range(1, 6):
            at_level = [c for c in self.criteria if c.level == level]
            if not at_level or not all(c.passed for c in at_level):
                break
            achieved = level
        return achieved


def _cells_significantly_below(hit: np.ndarray, cells: np.ndarray, target: float):
    """Cells whose coverage is significantly below target (Bonferroni-corrected)."""
    present = [c for c in np.unique(cells)]
    failing = []
    for c in present:
        h = hit[cells == c]
        test = stats.binomtest(int(h.sum()), len(h), target, alternative="less")
        if test.pvalue < SIGNIFICANCE / len(present):
            failing.append(int(c))
    return failing


def assess(ev: dict) -> list[Assessment]:
    """Score one model from its evidence bundle (see run_demo.collect_evidence)."""
    y, p, cells = ev["y"], ev["p"], ev["cells"]
    correct = (p >= 0.5).astype(int) == y
    acc = correct.mean()

    # Reliability / validity -------------------------------------------------
    ece = expected_calibration_error(p, y)
    auroc = error_detection_auroc(ev["uncertainty"], p, y)
    hit = coverage(ev["sets_marginal"], y)
    drone_hit = hit[y == 1]
    reliability = Assessment(
        "Reliability / validity (calibration)",
        [
            Criterion(1, "Held-out accuracy reported", True, f"accuracy = {acc:.3f}"),
            Criterion(2, f"ECE <= {ECE_L2}", ece <= ECE_L2, f"ECE = {ece:.3f}"),
            Criterion(3, f"ECE <= {ECE_L3}", ece <= ECE_L3, f"ECE = {ece:.3f}"),
            Criterion(
                3,
                f"Uncertainty flags errors: AUROC >= {AUROC_L3}",
                auroc >= AUROC_L3,
                f"AUROC = {auroc:.3f}",
            ),
            Criterion(
                4,
                f"Conformal coverage within {TARGET_COVERAGE:.0%} +/- {COVERAGE_TOL:.0%}",
                abs(hit.mean() - TARGET_COVERAGE) <= COVERAGE_TOL,
                f"coverage = {hit.mean():.3f}",
            ),
            Criterion(
                4,
                f"Drone-class coverage >= {TARGET_COVERAGE - COVERAGE_TOL:.0%}",
                drone_hit.mean() >= TARGET_COVERAGE - COVERAGE_TOL,
                f"drone coverage = {drone_hit.mean():.3f}",
            ),
            Criterion(5, "Formal verification of components", False, "not attempted"),
        ],
    )

    # Robustness -------------------------------------------------------------
    counts = np.array([(cells == c).sum() for c in range(sim.N_CELLS)])
    declared = ev["declared_cells"]
    in_od = np.isin(cells, declared)
    cell_acc = {c: correct[cells == c].mean() for c in declared}
    cell_lb = {
        c: wilson_lower_bound(int(correct[cells == c].sum()), int((cells == c).sum())) for c in declared
    }
    worst = min(cell_acc, key=cell_acc.get)
    worst_lb = min(cell_lb, key=cell_lb.get)
    mondrian_hit = coverage(ev["sets_mondrian"], y)
    under = _cells_significantly_below(mondrian_hit[in_od], cells[in_od], TARGET_COVERAGE)
    set_defer = (ev["sets_mondrian"][in_od].sum(axis=1) == 2).mean()
    robustness = Assessment(
        "Robustness",
        [
            Criterion(1, "Held-out accuracy reported", True, f"accuracy = {acc:.3f}"),
            Criterion(
                2,
                f"Every operational cell tested with >= {MIN_CELL_SAMPLES} samples",
                bool(counts[declared].min() >= MIN_CELL_SAMPLES),
                f"{len(declared)} declared cells, min n = {counts[declared].min()}",
            ),
            Criterion(
                2,
                f"Worst-cell accuracy >= {WORST_CELL_ACC}",
                cell_acc[worst] >= WORST_CELL_ACC,
                f"{sim.cell_label(worst)}: {cell_acc[worst]:.3f}",
            ),
            Criterion(
                3,
                "Uncertainty-guided scenario generation performed",
                ev["guided_rounds"] > 0,
                f"{ev['guided_rounds']} guided rounds",
            ),
            Criterion(
                3,
                f"Worst-cell accuracy 95% lower bound >= {WORST_CELL_ACC}",
                cell_lb[worst_lb] >= WORST_CELL_ACC,
                f"{sim.cell_label(worst_lb)}: {cell_lb[worst_lb]:.3f}",
            ),
            Criterion(
                4,
                "Per-cell conformal coverage not significantly below target",
                not under,
                "all cells OK" if not under else "failing: " + ", ".join(sim.cell_label(c) for c in under),
            ),
            Criterion(
                4,
                f"Runtime monitor defers <= {MAX_SET_DEFERRAL:.0%} (ambiguous sets)",
                set_defer <= MAX_SET_DEFERRAL,
                f"ambiguous-set rate = {set_defer:.3f}",
            ),
            Criterion(5, "Formal verification of components", False, "not attempted"),
        ],
    )

    # Safety (runtime guardrails) -------------------------------------------
    rates = ev["policy_rates"]
    drones_in_od = (y == 1) & in_od
    drone_under = _cells_significantly_below(
        mondrian_hit[drones_in_od], cells[drones_in_od], TARGET_COVERAGE
    )
    safety = Assessment(
        "Safety (runtime guardrails)",
        [
            Criterion(1, "Fixed decision threshold", True, "P(drone) >= 0.5"),
            Criterion(
                2,
                "Operating point chosen by explicit cost trade-off",
                ev["policy"] is not None,
                f"t = {ev['policy'].t:.2f}, weights (miss, false alarm, defer) = {ev['weights']}",
            ),
            Criterion(
                3,
                f"Uncertainty-triggered deferral: miss <= {MISS_L3:.0%}, false alarm <= {FALSE_ALARM_L3:.0%} "
                f"at deferral <= {DEFER_L3:.0%}",
                rates["miss"] <= MISS_L3
                and rates["false_alarm"] <= FALSE_ALARM_L3
                and rates["deferral"] <= DEFER_L3
                and np.isfinite(ev["policy"].tau),
                f"miss = {rates['miss']:.3f}, false alarm = {rates['false_alarm']:.3f}, "
                f"deferral = {rates['deferral']:.3f}",
            ),
            Criterion(
                4,
                "Drone-class conformal guarantee holds in every cell (miss-rate bound)",
                not drone_under,
                "all cells OK"
                if not drone_under
                else "failing: " + ", ".join(sim.cell_label(c) for c in drone_under),
            ),
            Criterion(5, "Formally verified safety monitor", False, "not attempted"),
        ],
    )
    return [reliability, robustness, safety]
