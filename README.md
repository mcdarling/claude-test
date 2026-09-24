# UQ as a measurement mechanism for maturity-based certification

A small, runnable toy version of the ideas in
[*Toward Maturity-Based Certification of Embodied AI: Quantifying
Trustworthiness Through Measurement Mechanisms*](https://arxiv.org/abs/2601.03470)
(Darling, Hesu, Mardikes, McGuigan & Milewicz, 2026).

The paper argues that certifying embodied AI needs **maturity models** fed by
**measurement mechanisms** that are quantifiable, actionable and, where
possible, formally grounded. It uses uncertainty quantification (UQ) as the
worked example, with a UAS (drone) detection case study. This repo builds a
miniature version of that pipeline end to end with synthetic data, so every
idea in the paper becomes a number you can see.

**Everything here is synthetic and notional.** The sensor model, thresholds
and cost weights are illustrative. They say nothing about any real system.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py          # about a minute; writes results/
pytest -q                   # unit tests for the core math
```

Then open [`results/report.md`](results/report.md).

## Certification report site

`docs/index.html` is a self-contained report for a general technical audience.
It explains maturity models from scratch, shows the rubric, and walks through
how the system scored, with the evidence behind every level. `run_demo.py`
regenerates it automatically (or run `python build_site.py` on its own).

To publish it with GitHub Pages: **Settings → Pages → Build and deployment →
Deploy from a branch → `main` / `docs`**.

## How the paper maps to the code

| Paper concept | Where it lives | What it does here |
|---|---|---|
| Closed-loop synthetic data pipeline with control over UAS, environment and confounders | `simulator.py` | Scene parameters (size, range, lighting, weather, terrain, birds) are rendered into 8 noisy sensor features. The detector only sees the features. |
| Deep ensembles for UQ (Sahay et al., 2022) | `ensemble.py` | 5 bootstrapped MLPs. Uncertainty is split into **epistemic** (member disagreement, reducible with data) and **aleatoric** (sensor noise, irreducible). |
| Quantifiable metrics: calibration, error detection | `measures.py` | ECE, reliability curves, error-by-uncertainty deciles, risk-coverage, Wilson confidence bounds per operational cell. |
| Formal properties: conformal prediction | `conformal.py` | Split conformal with a 90% coverage guarantee, including a per-condition, per-class (Mondrian) variant whose drone-class coverage bounds the miss rate. |
| Closed loop: generate data similar to high-uncertainty samples, retrain, reassess | `closed_loop.py` | Epistemic uncertainty on a candidate pool picks seeds; their scene parameters are perturbed to make new data. Compared against a random-data baseline. |
| Feature attribution (open question in the paper) | `run_demo.py` | A random forest plus permutation importance attributes uncertainty back to scene parameters. |
| Multi-objective trade-offs: misses vs false alarms vs operator trust | `tradeoffs.py` | Sweep of alarm threshold and deferral threshold, a miss vs false-alarm Pareto front among policies within operator capacity, and an explicit cost-weighted choice. |
| Actionable outputs and runtime guardrails | `tradeoffs.py`, `conformal.py` | High uncertainty or an ambiguous conformal set sends the case to a human operator. |
| Maturity levels with required evidence | `maturity.py` | Cumulative levels 1–5 for reliability, robustness and safety. Every criterion records the evidence behind it. |
| Operational domain specification (requirements phase) | `run_demo.py` | Scores the full domain and a restricted one that excludes cells the system can't handle. |

## What the demo shows

These numbers come from the latest `results/report.md`. Rerun to regenerate them.

1. **Uncertainty tracks error**, reproducing the paper's preliminary finding.
   Error rate climbs from under 1% in the most confident decile to over 40% in
   the least. Uncertainty ranks errors with AUROC of about 0.8.
2. **The two kinds of uncertainty do different jobs.** Epistemic uncertainty
   starts high at night and in severe weather, where the initial data is
   thin, and the closed loop drives it down. What remains is mostly aleatoric,
   meaning the sensors are simply noisy there. So epistemic uncertainty should
   decide *what data to collect*, and total uncertainty should decide *what to
   send to an operator*. Once the loop has run, epistemic uncertainty alone
   barely predicts errors (AUROC around 0.5).
3. **Uncertainty-guided data generation helps, but modestly.** Over 5 seeds it
   edges out random data on accuracy and worst-cell accuracy, by about one
   standard deviation. It's a small effect in this toy, so don't read it as a
   general result.
4. **Weighted-sum trade-offs land on extremes.** A heavily miss-averse
   weighting alarmed on 69% of empty scenes. Cheap operator reviews made it
   defer half of all cases. Adding an operator-capacity constraint (the
   ε-constraint method) gave a sensible operating point. Pareto filtering
   over all three objectives ruled out almost nothing, because more deferral
   always buys fewer errors. It only became informative once the workload
   budget was fixed and the front was taken over misses and false alarms.
5. **The scorecard produces useful results:**
   - **Reliability drops from L4 to L3 after the loop.** Conformal prediction's
     *marginal* 90% guarantee held, but coverage on the drone class alone fell
     below 88%. A marginal guarantee says nothing per class; group-conditional
     (Mondrian) conformal fixes it.
   - **Robustness only reaches L4 once the operational domain is restricted.**
     A few night or dusk, severe-weather, urban cells are too noisy for these
     sensors no matter how much data is added.
   - **Safety stays at L2.** Within the operator budget, false alarms stay above
     the 10% limit, so the detector itself would need to improve.
   - **The rubric was revised once, openly.** The false-alarm criterion was
     added after the first run exposed a checkbox-compliance loophole, the risk
     the paper's research agenda warns about. The report says so.

## Layout

```
uq_certification/
  simulator.py    scene parameters -> sensor features; operational-domain cells
  ensemble.py     deep ensemble + epistemic/aleatoric decomposition
  measures.py     calibration, error detection, confidence bounds
  conformal.py    split + Mondrian conformal prediction
  closed_loop.py  uncertainty-guided vs random data generation
  tradeoffs.py    policy sweep, Pareto filtering, constrained cost-based choice
  maturity.py     toy maturity rubric and scoring
  plots.py        figures
run_demo.py       runs everything, writes results/ and docs/
build_site.py     builds the report site in docs/ from results/metrics.json
tests/            unit tests
```

## Ideas for extending it

- Replace the toy features with images and a CNN, and measure similarity in a
  UMAP latent space, as the paper does.
- Add out-of-distribution scores, such as a new drone type never seen in
  training, as another robustness mechanism.
- Model sensor degradation over time and use longitudinal uncertainty
  tracking for the operations and maintenance phase.
- Extend the rubric to more NIST characteristics, and tackle the paper's
  question of how to make levels comparable across characteristics.
