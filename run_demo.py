"""End-to-end demo: UQ as a measurement mechanism for maturity-based certification.

Run with:  python run_demo.py            (about a minute)
Outputs:   results/*.png, results/report.md, results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from uq_certification import plots
from uq_certification import simulator as sim
from uq_certification.closed_loop import run_closed_loop
from uq_certification.conformal import ConformalDetector, coverage
from uq_certification.maturity import LEVEL_NAMES, TARGET_COVERAGE, assess
from uq_certification.measures import error_detection_auroc, expected_calibration_error
from uq_certification.tradeoffs import choose_policy, pareto_mask, policy_rates, sweep

# Notional costs of a missed drone, a false alarm and an operator review. A miss
# is 5x worse than a false alarm; an operator review costs less than a nuisance
# alarm because it is a deliberate look rather than an interruption.
COST_WEIGHTS = (5.0, 1.0, 0.3)
MAX_DEFERRAL = 0.20  # operator capacity: at most 20% of cases can be reviewed
DECLARED_OD_MIN_ACC = 0.75  # restricted operational domain: cells meeting this on calibration data


def collect_evidence(model, X_test, s_test, X_cal, s_cal, guided_rounds, restrict_od=False):
    """Run every measurement mechanism for one model and bundle the evidence.

    With ``restrict_od`` the operational domain is first narrowed to cells
    where the model meets DECLARED_OD_MIN_ACC on calibration data (a
    requirements-phase decision), and every measurement is taken inside it.
    """
    declared = list(range(sim.N_CELLS))
    if restrict_od:
        p_c, cells_c = model.predict(X_cal)["p"], sim.condition_cell(s_cal)
        correct_c = (p_c >= 0.5).astype(int) == s_cal.drone
        declared = [c for c in declared if correct_c[cells_c == c].mean() >= DECLARED_OD_MIN_ACC]
        keep_t = np.isin(sim.condition_cell(s_test), declared)
        keep_c = np.isin(cells_c, declared)
        s_test, X_test = s_test.subset(keep_t), X_test[keep_t]
        s_cal, X_cal = s_cal.subset(keep_c), X_cal[keep_c]

    out_t, out_c = model.predict(X_test), model.predict(X_cal)
    y_t, y_c = s_test.drone, s_cal.drone
    cells_t, cells_c = sim.condition_cell(s_test), sim.condition_cell(s_cal)

    alpha = 1 - TARGET_COVERAGE
    marginal = ConformalDetector(alpha).fit(out_c["p"], y_c)
    mondrian = ConformalDetector(alpha, mondrian=True).fit(out_c["p"], y_c, cells_c)

    # Trade-off analysis and threshold selection use calibration data only.
    policies, rates = sweep(out_c["p"], out_c["total"], y_c)
    policy, idx, _ = choose_policy(policies, rates, COST_WEIGHTS, max_deferral=MAX_DEFERRAL)

    return {
        "y": y_t,
        "p": out_t["p"],
        "uncertainty": out_t["total"],
        "outputs": out_t,
        "cells": cells_t,
        "sets_marginal": marginal.predict_sets(out_t["p"]),
        "sets_mondrian": mondrian.predict_sets(out_t["p"], cells_t),
        "policy": policy,
        "policy_rates": policy_rates(out_t["p"], out_t["total"], y_t, policy),
        "sweep": (policies, rates, idx),
        "weights": COST_WEIGHTS,
        "declared_cells": declared,
        "guided_rounds": guided_rounds,
    }


def uncertainty_importance(out, scenes, seed=0):
    """Which scene parameters explain the model's uncertainty (feature attribution)."""
    Z = scenes.param_matrix()
    half = len(Z) // 2
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=seed, n_jobs=-1)
    rf.fit(Z[:half], out["total"][:half])
    imp = permutation_importance(rf, Z[half:], out["total"][half:], n_repeats=5, random_state=seed)
    return dict(zip(sim.PARAM_MATRIX_NAMES, imp.importances_mean))


def summarise(ev):
    y, p = ev["y"], ev["p"]
    hit_m = coverage(ev["sets_marginal"], y)
    hit_g = coverage(ev["sets_mondrian"], y)
    return {
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "ece": expected_calibration_error(p, y),
        "error_auroc_total": error_detection_auroc(ev["outputs"]["total"], p, y),
        "error_auroc_epistemic": error_detection_auroc(ev["outputs"]["epistemic"], p, y),
        "mean_epistemic": float(ev["outputs"]["epistemic"].mean()),
        "mean_aleatoric": float(ev["outputs"]["aleatoric"].mean()),
        "conformal_coverage_marginal": float(hit_m.mean()),
        "conformal_coverage_mondrian": float(hit_g.mean()),
        "conformal_drone_coverage_mondrian": float(hit_g[y == 1].mean()),
        "ambiguous_set_rate_mondrian": float((ev["sets_mondrian"].sum(axis=1) == 2).mean()),
        "policy": {"t": ev["policy"].t, "tau": ev["policy"].tau},
        "policy_rates": ev["policy_rates"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="repetitions of the closed-loop comparison")
    parser.add_argument("--rounds", type=int, default=5, help="closed-loop rounds")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    start = time.time()

    # Fixed evaluation data spanning the whole operational domain.
    s_test, X_test = sim.generate(8000, np.random.default_rng(12345))
    s_cal, X_cal = sim.generate(8000, np.random.default_rng(54321))

    # 1. Closed loop: uncertainty-guided vs random data generation.
    print(f"Closed loop: {args.seeds} seeds x {args.rounds} rounds x 2 strategies")
    runs = {"uncertainty": [], "random": []}
    for seed in range(args.seeds):
        for strategy in runs:
            runs[strategy].append(run_closed_loop(strategy, s_test, X_test, seed=seed, n_rounds=args.rounds))
        print(f"  seed {seed} done ({time.time() - start:.0f}s)")
    histories = {k: [r.history for r in v] for k, v in runs.items()}
    n_init = runs["uncertainty"][0].history[0]["n_train"]
    added = {k: v[0].train_scenes.subset(np.arange(n_init, len(v[0].train_scenes))) for k, v in runs.items()}
    plots.closed_loop(histories, added, args.out / "fig4_closed_loop.png")

    # 2. Measurement mechanisms on the initial model and the UQ-guided final model.
    primary = runs["uncertainty"][0]
    models = {"initial": (primary.initial_model, 0), "after closed loop": (primary.final_model, args.rounds)}
    evidence = {k: collect_evidence(m, X_test, s_test, X_cal, s_cal, g) for k, (m, g) in models.items()}
    outputs = {k: ev["outputs"] for k, ev in evidence.items()}
    plots.uncertainty_vs_error(outputs, s_test.drone, args.out / "fig1_uncertainty_vs_error.png")
    plots.calibration(outputs, s_test.drone, args.out / "fig2_calibration.png")

    # 3. Feature attribution: what drives uncertainty?
    importance = uncertainty_importance(outputs["initial"], s_test)
    plots.uncertainty_drivers(
        [
            ("Epistemic, initial model", outputs["initial"]["epistemic"]),
            ("Epistemic, after closed loop", outputs["after closed loop"]["epistemic"]),
            ("Aleatoric, after closed loop", outputs["after closed loop"]["aleatoric"]),
        ],
        s_test,
        importance,
        args.out / "fig3_uncertainty_drivers.png",
    )

    # 4. Multi-objective trade-off (final model, calibration data).
    policies, rates, idx = evidence["after closed loop"]["sweep"]
    plots.tradeoffs(rates, pareto_mask(rates), idx, args.out / "fig5_tradeoffs.png")

    # 5. Maturity scorecards: full operational domain and a restricted, declared one.
    scorecards = {k: assess(ev) for k, ev in evidence.items()}
    restricted_ev = collect_evidence(primary.final_model, X_test, s_test, X_cal, s_cal, args.rounds, restrict_od=True)
    scorecards_restricted = assess(restricted_ev)
    plots.maturity(
        {**scorecards, "after closed loop, restricted domain": scorecards_restricted},
        args.out / "fig6_maturity.png",
    )

    metrics = {
        "closed_loop_final_round": {
            k: {m: _mean_sd([h[-1][m] for h in hs]) for m in hs[0][-1] if m != "round"}
            for k, hs in histories.items()
        },
        "closed_loop_initial_round": {
            m: _mean_sd([h[0][m] for h in histories["uncertainty"]]) for m in histories["uncertainty"][0][0] if m != "round"
        },
        "models": {k: summarise(ev) for k, ev in evidence.items()},
        "uncertainty_importance": importance,
        "maturity_levels": {k: {a.characteristic: a.level for a in sc} for k, sc in scorecards.items()},
        "maturity_levels_restricted_od": {a.characteristic: a.level for a in scorecards_restricted},
        "model_restricted_od": summarise(restricted_ev),
        "restricted_od_excluded_cells": [
            sim.cell_label(c) for c in range(sim.N_CELLS) if c not in restricted_ev["declared_cells"]
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(args, metrics, scorecards, scorecards_restricted, restricted_ev, evidence)
    print(f"Done in {time.time() - start:.0f}s. See {args.out}/report.md")


def _mean_sd(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def _scorecard_table(assessments):
    lines = ["| Characteristic | Level | Requirement | Evidence | Met |", "|---|---|---|---|---|"]
    for a in assessments:
        for c in a.criteria:
            lines.append(
                f"| {a.characteristic} | L{c.level} | {c.requirement} | {c.evidence} | {'✅' if c.passed else '❌'} |"
            )
    return "\n".join(lines)


def write_report(args, metrics, scorecards, scorecards_restricted, restricted_ev, evidence):
    cl = metrics["closed_loop_final_round"]
    init = metrics["closed_loop_initial_round"]
    m_init, m_final = metrics["models"]["initial"], metrics["models"]["after closed loop"]

    def fmt(d):
        return f"{d['mean']:.3f} ± {d['sd']:.3f}"

    loop_rows = "\n".join(
        f"| {label} | {src['n_train']['mean']:.0f} | {fmt(src['accuracy'])} | {fmt(src['worst_cell_accuracy'])} "
        f"| {fmt(src['mean_epistemic'])} | {fmt(src['ece'])} |"
        for label, src in [("initial (skewed data)", init), ("+ random data", cl["random"]), ("+ uncertainty-guided data", cl["uncertainty"])]
    )
    measure_rows = "\n".join(
        f"| {name} | {m_init[key]:.3f} | {m_final[key]:.3f} |"
        for name, key in [
            ("Accuracy", "accuracy"),
            ("Expected calibration error", "ece"),
            ("Error-detection AUROC (total uncertainty)", "error_auroc_total"),
            ("Error-detection AUROC (epistemic only)", "error_auroc_epistemic"),
            ("Mean epistemic uncertainty (bits)", "mean_epistemic"),
            ("Mean aleatoric uncertainty (bits)", "mean_aleatoric"),
            ("Conformal coverage, marginal (target 0.90)", "conformal_coverage_marginal"),
            ("Conformal coverage, per-cell Mondrian (target 0.90)", "conformal_coverage_mondrian"),
            ("Drone-class coverage, Mondrian", "conformal_drone_coverage_mondrian"),
            ("Ambiguous prediction sets → operator", "ambiguous_set_rate_mondrian"),
        ]
    )
    pol = m_final["policy"]
    pr = m_final["policy_rates"]
    imp = sorted(metrics["uncertainty_importance"].items(), key=lambda kv: -kv[1])
    levels = metrics["maturity_levels"]
    level_rows = "\n".join(
        f"| {c} | L{levels['initial'][c]} {LEVEL_NAMES[levels['initial'][c]]} "
        f"| L{levels['after closed loop'][c]} {LEVEL_NAMES[levels['after closed loop'][c]]} "
        f"| L{metrics['maturity_levels_restricted_od'][c]} {LEVEL_NAMES[metrics['maturity_levels_restricted_od'][c]]} |"
        for c in levels["initial"]
    )
    excluded = metrics["restricted_od_excluded_cells"]
    rr = metrics["model_restricted_od"]

    report = f"""# Results: UQ as a measurement mechanism for maturity-based certification

Generated by `python run_demo.py --seeds {args.seeds} --rounds {args.rounds}`.
All data is synthetic (see `uq_certification/simulator.py`). The numbers show
how the mechanisms behave. They are not claims about any real detector.

## 1. Uncertainty tracks error
The paper's preliminary finding: the model is less likely to be correct when
its uncertainty is high.

![uncertainty vs error](fig1_uncertainty_vs_error.png)

## 2. Calibration
![calibration](fig2_calibration.png)

| Measurement | Initial model | After closed loop |
|---|---|---|
{measure_rows}

## 3. What drives uncertainty
The heatmaps split uncertainty into **epistemic** (model ignorance, which more
data can fix) and **aleatoric** (noise in the sensors themselves, which it
can't). A random forest attributes the initial model's uncertainty to scene
parameters. Ranked by permutation importance:
{", ".join(f"`{k}` ({v:.3f})" for k, v in imp)}.

![drivers](fig3_uncertainty_drivers.png)

## 4. Closed-loop, uncertainty-guided data generation
Each round adds 400 samples. The guided strategy perturbs the scene
parameters of the most uncertain candidates. The random strategy samples the
whole domain uniformly. Mean ± sd over {args.seeds} seeds on the same test set:

| Training data | n | Accuracy | Worst-cell accuracy | Mean epistemic | ECE |
|---|---|---|---|---|---|
{loop_rows}

![closed loop](fig4_closed_loop.png)

## 5. Multi-objective trade-off and runtime guardrail
Policies trade missed drones against false alarms and operator workload
(deferring uncertain cases). The chosen policy minimises expected cost with
explicit weights (miss, false alarm, deferral) = {COST_WEIGHTS}, subject to an
operator-capacity constraint of at most {MAX_DEFERRAL:.0%} deferrals (the
epsilon-constraint method: a weighted sum alone lands on extremes such as
alarming on everything or deferring half of all cases). Selected on
calibration data: alarm threshold t = {pol['t']:.2f}, deferral when total
uncertainty > {pol['tau']:.3f} bits. On the test set: miss rate {pr['miss']:.3f},
false-alarm rate {pr['false_alarm']:.3f}, deferral rate {pr['deferral']:.3f}.

![trade-offs](fig5_tradeoffs.png)

## 6. Toy maturity scorecard
Levels are cumulative (see `uq_certification/maturity.py`). The thresholds
were fixed before the first run, with one exception: that run showed a
heavily miss-averse cost weighting could pass Safety L3 and L4 while raising
false alarms on 69% of empty scenes. That is the operator alarm fatigue the
paper warns about, and a textbook case of checkbox compliance. So a
false-alarm limit was added to Safety L3. The third column restricts the declared
operational domain to cells where the final model reaches
{DECLARED_OD_MIN_ACC:.0%} accuracy on calibration data, which excludes
{len(excluded)} of {sim.N_CELLS} cells: {", ".join(excluded) if excluded else "none"}.
Every measurement in that column (calibration, conformal, operating point) is
redone inside the restricted domain. There the chosen policy gets accuracy
{rr['accuracy']:.3f}, miss rate {rr['policy_rates']['miss']:.3f}, false-alarm
rate {rr['policy_rates']['false_alarm']:.3f} and deferral rate
{rr['policy_rates']['deferral']:.3f}.

| Characteristic | Initial | After closed loop | After closed loop, restricted domain |
|---|---|---|---|
{level_rows}

![maturity](fig6_maturity.png)

### Evidence: initial model
{_scorecard_table(scorecards['initial'])}

### Evidence: after closed loop (full operational domain)
{_scorecard_table(scorecards['after closed loop'])}

### Evidence: after closed loop (restricted operational domain)
{_scorecard_table(scorecards_restricted)}
"""
    (args.out / "report.md").write_text(report)


if __name__ == "__main__":
    main()
