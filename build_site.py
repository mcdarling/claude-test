"""Build the GitHub Pages certification report from results/metrics.json.

Run after run_demo.py (which calls this automatically):  python build_site.py
Output: docs/index.html plus docs/figures/*.png
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from uq_certification import simulator as sim
from uq_certification.maturity import LEVEL_NAMES

ROOT = Path(__file__).parent
FIGURES = [
    "fig1_uncertainty_vs_error.png",
    "fig2_calibration.png",
    "fig3_uncertainty_drivers.png",
    "fig4_closed_loop.png",
    "fig5_tradeoffs.png",
    "fig6_maturity.png",
]
CONFIGS = [
    ("initial", "Initial model", "Trained on a skewed first data collection"),
    ("after closed loop", "After closed loop", "Five rounds of uncertainty-guided data added"),
    (
        "after closed loop, restricted domain",
        "Restricted domain",
        "Same model, certified only where it can perform",
    ),
]
LEVEL_MEANING = {
    1: "Limited, informal testing. Claims rest on a few demonstrations.",
    2: "Documented tests that cover the conditions the system is meant to operate in.",
    3: "Measurements steer what gets tested and improved, and performance comes with quantified bounds.",
    4: "Formal statistical guarantees, validated across the operating domain, plus runtime monitoring that acts on measurements.",
    5: "Mathematical proofs about components that combine into whole-system guarantees, with formally verified runtime safety mechanisms.",
}
SHORT_NAMES = {
    "Reliability / validity (calibration)": "Reliability",
    "Robustness": "Robustness",
    "Safety (runtime guardrails)": "Safety",
}
CHAR_QUESTION = {
    "Reliability / validity (calibration)": "Can you trust how confident the system says it is?",
    "Robustness": "Does it keep working across every condition it is meant for?",
    "Safety (runtime guardrails)": "When it is unsure, does it do something safe about it?",
}


def esc(text) -> str:
    return html.escape(str(text)).replace("&lt;=", "≤").replace("&gt;=", "≥").replace("+/-", "±")


def pct(x, digits=0) -> str:
    return f"{100 * x:.{digits}f}%"


def ladder(level: int) -> str:
    cells = "".join(
        f'<span class="rung{" on" if i <= level else ""}" aria-hidden="true"></span>' for i in range(1, 6)
    )
    return f'<span class="ladder" role="img" aria-label="Level {level} of 5">{cells}</span>'


def verdict_table(cards) -> str:
    head = "".join(f"<th scope=\"col\">{esc(name)}<small>{esc(note)}</small></th>" for _, name, note in CONFIGS)
    rows = []
    for i, char in enumerate(c["characteristic"] for c in cards[CONFIGS[0][0]]):
        cells = []
        for key, _, _ in CONFIGS:
            lv = cards[key][i]["level"]
            cells.append(
                f'<td>{ladder(lv)}<span class="lv">L{lv}</span> <span class="lvname">{esc(LEVEL_NAMES[lv])}</span></td>'
            )
        rows.append(
            f'<tr><th scope="row">{esc(SHORT_NAMES[char])}<small>{esc(CHAR_QUESTION[char])}</small></th>{"".join(cells)}</tr>'
        )
    return f'<div class="scroll"><table class="verdict"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def rubric_blocks(cards) -> str:
    blocks = []
    for assessment in cards[CONFIGS[0][0]]:
        char = assessment["characteristic"]
        items = []
        for level in range(1, 6):
            reqs = [c["requirement"] for c in assessment["criteria"] if c["level"] == level]
            lis = "".join(f"<li>{esc(r)}</li>" for r in reqs)
            items.append(f'<li><span class="tag">L{level}</span><ul>{lis}</ul></li>')
        blocks.append(
            f'<section class="rubric"><h4>{esc(SHORT_NAMES[char])}</h4>'
            f'<p class="q">{esc(CHAR_QUESTION[char])}</p><ol class="reqs">{"".join(items)}</ol></section>'
        )
    return f'<div class="rubrics">{"".join(blocks)}</div>'


def evidence_tables(cards) -> str:
    out = []
    for i, assessment in enumerate(cards[CONFIGS[0][0]]):
        char = assessment["characteristic"]
        head = "".join(f'<th scope="col">{esc(name)}</th>' for _, name, _ in CONFIGS)
        levels = "".join(
            f'<td class="achieved">Level {cards[k][i]["level"]}</td>' for k, _, _ in CONFIGS
        )
        rows = []
        for j, crit in enumerate(assessment["criteria"]):
            cells = []
            for key, _, _ in CONFIGS:
                c = cards[key][i]["criteria"][j]
                mark = '<span class="pass">Met</span>' if c["passed"] else '<span class="fail">Not met</span>'
                cells.append(f"<td>{mark}<span class=\"ev\">{esc(c['evidence'])}</span></td>")
            rows.append(
                f'<tr><td class="lvcell">L{crit["level"]}</td><th scope="row">{esc(crit["requirement"])}</th>{"".join(cells)}</tr>'
            )
        out.append(
            f'<h4 id="evidence-{SHORT_NAMES[char].lower()}">{esc(SHORT_NAMES[char])}</h4>'
            f'<div class="scroll"><table class="evidence"><thead><tr><th>Level</th><th>Requirement</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody><tfoot><tr><td></td><th scope="row">Level achieved</th>{levels}</tr></tfoot></table></div>'
        )
    return "\n".join(out)


def figure(name, caption, alt) -> str:
    return (
        f'<figure><div class="plate"><img src="figures/{name}" alt="{esc(alt)}" loading="lazy"></div>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def build(results: Path = ROOT / "results", out: Path = ROOT / "docs") -> Path:
    m = json.loads((results / "metrics.json").read_text())
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy(results / name, out / "figures" / name)
    (out / ".nojekyll").write_text("")

    cards = m["scorecards"]
    init, final = m["models"]["initial"], m["models"]["after closed loop"]
    restricted = m["model_restricted_od"]
    loop_u, loop_r = m["closed_loop_final_round"]["uncertainty"], m["closed_loop_final_round"]["random"]
    loop_0 = m["closed_loop_initial_round"]
    par = m["pareto"]
    w_miss, w_fa, w_defer = m["cost_weights"]
    excluded = m["restricted_od_excluded_cells"]
    imp = sorted(m["uncertainty_importance"].items(), key=lambda kv: -kv[1])
    dec = init["error_by_uncertainty_decile"]
    rates = final["policy_rates"]
    rrates = restricted["policy_rates"]
    lv = m["maturity_levels"]
    lv_r = m["maturity_levels_restricted_od"]
    rel, rob, saf = (
        "Reliability / validity (calibration)",
        "Robustness",
        "Safety (runtime guardrails)",
    )

    def ms(d, digits=3):
        return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"

    loop_rows = "".join(
        f"<tr><th scope=\"row\">{label}</th><td>{d['n_train']['mean']:.0f}</td><td>{ms(d['accuracy'])}</td>"
        f"<td>{ms(d['worst_cell_accuracy'])}</td><td>{ms(d['mean_epistemic'])}</td></tr>"
        for label, d in [
            ("Initial (skewed data)", loop_0),
            ("+ random data", loop_r),
            ("+ uncertainty-guided data", loop_u),
        ]
    )
    measure_rows = "".join(
        f"<tr><th scope=\"row\">{name}</th><td>{init[key]:.3f}</td><td>{final[key]:.3f}</td><td>{note}</td></tr>"
        for name, key, note in [
            ("Accuracy", "accuracy", "Share of test scenes classified correctly"),
            ("Calibration error (ECE)", "ece", "Gap between stated confidence and actual accuracy; lower is better"),
            ("Error detection, total uncertainty (AUROC)", "error_auroc_total", "0.5 = no better than chance, 1.0 = perfect"),
            ("Error detection, epistemic only (AUROC)", "error_auroc_epistemic", "Falls once the data gaps are filled"),
            ("Conformal coverage (target 0.90)", "conformal_coverage_marginal", "How often the prediction set contains the truth"),
            ("Drone-class coverage, per-condition sets", "conformal_drone_coverage_mondrian", "Coverage for drones only, guaranteed per condition"),
        ]
    )
    n_cells = sim.N_CELLS
    lighting = ", ".join(b[0] for b in sim.LIGHTING_BINS)
    weather = ", ".join(b[0] for b in sim.WEATHER_BINS)
    terrains = ", ".join(sim.TERRAINS)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Drone Detector Maturity Assessment</title>
<meta name="description" content="A worked example of maturity-based certification for an AI drone detector, using uncertainty quantification as the measurement mechanism.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
:root {{
  color-scheme: light;
  --paper: #f3f5f7;
  --sheet: #ffffff;
  --ink: #14202b;
  --muted: #56626e;
  --rule: #d6dce2;
  --accent: #1c5cab;
  --accent-soft: #dce8f7;
  --rung: #d3dbe4;
  --pass: #1d7446;
  --pass-bg: #e2f2e8;
  --fail: #a8261c;
  --fail-bg: #f9e3e0;
  --note-bg: #fff6df;
  --note-rule: #c98500;
  --display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --paper: #0e141a;
    --sheet: #151d25;
    --ink: #e6ebf0;
    --muted: #9ba7b3;
    --rule: #2a3642;
    --accent: #6da7ec;
    --accent-soft: #1b2e45;
    --rung: #2c3947;
    --pass: #5fcb91;
    --pass-bg: #163424;
    --fail: #f08a80;
    --fail-bg: #3a1c1a;
    --note-bg: #2e2612;
    --note-rule: #c98500;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper: #0e141a;
  --sheet: #151d25;
  --ink: #e6ebf0;
  --muted: #9ba7b3;
  --rule: #2a3642;
  --accent: #6da7ec;
  --accent-soft: #1b2e45;
  --rung: #2c3947;
  --pass: #5fcb91;
  --pass-bg: #163424;
  --fail: #f08a80;
  --fail-bg: #3a1c1a;
  --note-bg: #2e2612;
  --note-rule: #c98500;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; scroll-padding-top: 1rem; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 400 1.0625rem/1.65 var(--body);
  padding-inline: 16px;
  padding-block: 0 4rem;
}}
a {{ color: var(--accent); text-underline-offset: 2px; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }}
img {{ max-width: 100%; height: auto; display: block; }}
h1, h2, h3, h4 {{ font-family: var(--display); font-weight: 600; line-height: 1.1; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(2.4rem, 6vw, 3.8rem); letter-spacing: -0.005em; }}
h2 {{ font-size: clamp(1.8rem, 3.6vw, 2.3rem); }}
h3 {{ font-size: 1.45rem; }}
h4 {{ font-size: 1.2rem; letter-spacing: 0.01em; }}
p {{ margin: 0; }}
.eyebrow {{ font: 500 0.78rem/1.3 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
.layout {{ max-width: 1180px; margin-inline: auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 3rem; }}
@media (min-width: 1100px) {{ .layout {{ grid-template-columns: 200px minmax(0, 1fr); }} }}
nav.toc {{ display: none; }}
@media (min-width: 1100px) {{
  nav.toc {{ display: block; position: sticky; top: 1.5rem; align-self: start; padding-top: 2.5rem; }}
  nav.toc ol {{ list-style: none; margin: 0.75rem 0 0; padding: 0; display: grid; gap: 0.45rem; }}
  nav.toc a {{ color: var(--muted); text-decoration: none; font: 500 0.95rem/1.3 var(--display); letter-spacing: 0.02em; }}
  nav.toc a:hover {{ color: var(--accent); }}
}}
main {{ min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr); gap: 4.5rem; }}
.prose {{ max-width: 68ch; display: grid; gap: 1.1rem; }}
.prose ul, .prose ol {{ margin: 0; padding-left: 1.3rem; display: grid; gap: 0.4rem; }}
header.cover {{ padding-top: 2.5rem; display: grid; grid-template-columns: minmax(0, 1fr); gap: 1.4rem; padding-bottom: 0.5rem; }}
.cover .meta {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.6rem; font: 400 0.82rem/1.4 var(--mono); color: var(--muted); }}
.cover .meta b {{ color: var(--ink); font-weight: 500; }}
.lede {{ font-size: 1.22rem; line-height: 1.55; max-width: 60ch; }}
.note {{ background: var(--note-bg); border-left: 3px solid var(--note-rule); padding: 0.85rem 1rem; font-size: 0.98rem; max-width: 68ch; }}
section.part {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 1.5rem; }}
section.part > header {{ display: grid; gap: 0.35rem; border-top: 1px solid var(--rule); padding-top: 1.4rem; }}
.scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
th, td {{ text-align: left; vertical-align: top; padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--rule); }}
thead th {{ font: 600 0.95rem/1.2 var(--display); letter-spacing: 0.03em; border-bottom: 2px solid var(--ink); }}
th small, td small {{ display: block; font: 400 0.82rem/1.35 var(--body); color: var(--muted); margin-top: 0.2rem; letter-spacing: 0; }}
td {{ font-variant-numeric: tabular-nums; }}
table.verdict {{ background: var(--sheet); min-width: 640px; }}
table.verdict tbody th {{ font: 600 1.15rem/1.2 var(--display); width: 26%; }}
table.verdict td {{ white-space: nowrap; vertical-align: middle; }}
.ladder {{ display: inline-flex; gap: 3px; vertical-align: middle; margin-right: 0.55rem; }}
.rung {{ width: 14px; height: 18px; background: var(--rung); border-radius: 2px; }}
.rung.on {{ background: var(--accent); }}
.lv {{ font: 600 1.1rem/1 var(--display); vertical-align: middle; }}
.lvname {{ color: var(--muted); font-size: 0.92rem; vertical-align: middle; }}
.levels {{ display: grid; gap: 0; border-top: 2px solid var(--ink); max-width: 820px; }}
.levels > div {{ display: grid; grid-template-columns: 4.2rem 1fr; gap: 1rem; padding: 0.8rem 0; border-bottom: 1px solid var(--rule); align-items: baseline; }}
.levels .n {{ font: 700 1.6rem/1 var(--display); color: var(--accent); }}
.levels .n span {{ display: block; font: 500 0.7rem/1.3 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-top: 0.25rem; }}
.levels h4 {{ margin-bottom: 0.2rem; }}
.props {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); max-width: 980px; }}
.props > div {{ background: var(--sheet); border: 1px solid var(--rule); padding: 1rem; display: grid; gap: 0.35rem; align-content: start; }}
.props p {{ font-size: 0.95rem; color: var(--muted); }}
.rubrics {{ display: grid; gap: 1.2rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
.rubric {{ background: var(--sheet); border: 1px solid var(--rule); padding: 1.1rem 1.1rem 0.6rem; display: grid; gap: 0.5rem; align-content: start; }}
.rubric .q {{ color: var(--muted); font-size: 0.93rem; font-style: italic; }}
.reqs {{ list-style: none; margin: 0.3rem 0 0; padding: 0; display: grid; }}
.reqs > li {{ display: grid; grid-template-columns: 2.4rem 1fr; gap: 0.5rem; padding: 0.55rem 0; border-top: 1px solid var(--rule); font-size: 0.93rem; line-height: 1.45; }}
.reqs ul {{ margin: 0; padding-left: 1rem; display: grid; gap: 0.25rem; }}
.tag {{ font: 600 0.95rem/1.4 var(--display); color: var(--accent); }}
table.evidence {{ min-width: 860px; background: var(--sheet); }}
table.evidence tbody th {{ font: 400 0.93rem/1.4 var(--body); width: 30%; }}
table.evidence td.lvcell {{ font: 600 1rem/1.4 var(--display); color: var(--accent); width: 3.2rem; }}
table.evidence tfoot th, table.evidence tfoot td {{ border-bottom: none; border-top: 2px solid var(--ink); font: 600 1.05rem/1.3 var(--display); }}
.pass, .fail {{ display: inline-block; font: 500 0.72rem/1 var(--mono); letter-spacing: 0.05em; text-transform: uppercase; padding: 0.3rem 0.45rem; border-radius: 3px; }}
.pass {{ color: var(--pass); background: var(--pass-bg); }}
.fail {{ color: var(--fail); background: var(--fail-bg); }}
.ev {{ display: block; font: 400 0.8rem/1.4 var(--mono); color: var(--muted); margin-top: 0.35rem; word-break: break-word; }}
figure {{ margin: 0; display: grid; gap: 0.6rem; max-width: 980px; }}
.plate {{ background: #fcfcfb; border: 1px solid var(--rule); padding: 0.6rem; }}
figcaption {{ font-size: 0.92rem; color: var(--muted); max-width: 72ch; }}
.pipeline {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: stretch; max-width: 980px; }}
.pipeline > div {{ flex: 1 1 150px; background: var(--sheet); border: 1px solid var(--rule); padding: 0.8rem; display: grid; gap: 0.25rem; align-content: start; }}
.pipeline .step {{ font: 500 0.72rem/1.2 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent); }}
.pipeline p {{ font-size: 0.9rem; color: var(--muted); line-height: 1.45; }}
.takeaways {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
.takeaways > div {{ border-top: 3px solid var(--accent); padding-top: 0.7rem; display: grid; gap: 0.35rem; align-content: start; }}
.takeaways p {{ font-size: 0.97rem; }}
dl.gloss {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 0; max-width: 820px; margin: 0; }}
dl.gloss > div {{ display: grid; gap: 0.2rem; padding: 0.7rem 0; border-bottom: 1px solid var(--rule); }}
@media (min-width: 700px) {{ dl.gloss > div {{ grid-template-columns: 13rem 1fr; gap: 1.2rem; }} }}
dl.gloss dt {{ font: 600 1.05rem/1.4 var(--display); }}
dl.gloss dd {{ margin: 0; font-size: 0.96rem; }}
code {{ font: 400 0.88em var(--mono); background: var(--accent-soft); padding: 0.05em 0.3em; border-radius: 3px; }}
footer {{ font-size: 0.9rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 1.2rem; display: grid; gap: 0.4rem; }}
</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents">
  <p class="eyebrow">Contents</p>
  <ol>
    <li><a href="#verdict">Verdict</a></li>
    <li><a href="#why">Why certify</a></li>
    <li><a href="#maturity">Maturity models</a></li>
    <li><a href="#system">System assessed</a></li>
    <li><a href="#measure">How we measured</a></li>
    <li><a href="#scores">Scores and evidence</a></li>
    <li><a href="#lessons">What we learned</a></li>
    <li><a href="#glossary">Glossary</a></li>
  </ol>
</nav>
<main>

<header class="cover">
  <p class="eyebrow">Maturity assessment · worked example</p>
  <h1>Can we certify an AI drone detector?</h1>
  <p class="lede">A step-by-step demonstration of maturity-based certification, the approach proposed in
  <a href="https://arxiv.org/abs/2601.03470"><em>Toward Maturity-Based Certification of Embodied AI</em></a>.
  We define what each maturity level requires, measure a drone detector against those requirements, and show how it scored and why.</p>
  <div class="meta">
    <span>System <b>Toy UAS detector (synthetic)</b></span>
    <span>Mechanism <b>Uncertainty quantification</b></span>
    <span>Characteristics <b>Reliability · Robustness · Safety</b></span>
    <span>Test scenes <b>8,000</b></span>
  </div>
  <p class="note"><strong>This is a teaching example, not a real certification.</strong> The detector, its sensors and its data are simulated,
  and every threshold was chosen for illustration. The goal is to show how the process works, including where it gets difficult.</p>
</header>

<section class="part" id="verdict">
  <header><p class="eyebrow">Verdict</p><h2>How the system scored</h2></header>
  <div class="prose"><p>Each characteristic gets a maturity level from 1 to 5. We scored the same detector three times: as first trained,
  after improving it with targeted data, and after limiting the conditions it is certified for.</p></div>
  {verdict_table(cards)}
  <div class="takeaways">
    <div><h4>Reliability: {_change(lv['initial'][rel], lv['after closed loop'][rel])}</h4>
      <p>The model's confidence is well calibrated throughout. It slipped a level after improving, because a 90% guarantee that held on average
      fell to {pct(_drone_cov(cards), 1)} for drones specifically.</p></div>
    <div><h4>Robustness: {_change(lv['initial'][rob], lv_r[rob])}</h4>
      <p>Targeted data fixed most weak conditions. Level 4 needed the certified domain to exclude {len(excluded)} of {n_cells} conditions
      where the sensors are too noisy for any amount of data to help.</p></div>
    <div><h4>Safety: {_change(lv['initial'][saf], lv_r[saf])}</h4>
      <p>Within the operator's review budget, the detector still raises false alarms on {pct(rates['false_alarm'])} of empty scenes,
      above the 10% limit. Better sensors or models are needed, not better paperwork.</p></div>
  </div>
</section>

<section class="part" id="why">
  <header><p class="eyebrow">Section 1</p><h2>Why certification for embodied AI is hard</h2></header>
  <div class="prose">
    <p><strong>Embodied AI</strong> is AI with a physical presence: autonomous vehicles, drones, medical devices. Before such a system is trusted in a
    safety-critical role, someone has to decide it is trustworthy enough. That decision is <strong>certification</strong>.</p>
    <p>A counter-drone detection system is a good example. It combines physical sensors (radar, radio receivers, cameras, microphones) with AI that
    decides whether a drone is present. Its mistakes cost something in both directions:</p>
    <ul>
      <li>A <strong>missed drone</strong> lets a threat into restricted airspace.</li>
      <li>A <strong>false alarm</strong> wastes an operator's attention. Too many of them and operators start ignoring alarms, including real ones.</li>
    </ul>
    <p>The U.S. National Institute of Standards and Technology (NIST) lists the characteristics of trustworthy AI: valid and reliable, safe, secure and
    resilient, accountable and transparent, explainable, privacy-enhanced, and fair. These are good goals, but they are not tests. Certification needs a
    way to turn each goal into <strong>measurable evidence</strong> with a clear pass or fail.</p>
  </div>
</section>

<section class="part" id="maturity">
  <header><p class="eyebrow">Section 2</p><h2>How a maturity model works</h2></header>
  <div class="prose">
    <p>A <strong>maturity model</strong> grades a capability on a ladder of levels instead of a single yes or no. The idea comes from software engineering,
    where models such as CMMI grade how disciplined an organization's processes are. The paper proposes the same structure for AI trustworthiness,
    with five levels. Using robustness as the example:</p>
  </div>
  <div class="levels">
    {"".join(f'<div><p class="n">{i}<span>Level</span></p><div><h4>{esc(LEVEL_NAMES[i])}</h4><p>{esc(LEVEL_MEANING[i])}</p></div></div>' for i in range(1, 6))}
  </div>
  <div class="prose">
    <p>Two rules make the ladder meaningful:</p>
    <ul>
      <li><strong>Every level names the evidence that proves it.</strong> A claim without a measurement behind it doesn't count.</li>
      <li><strong>Levels are cumulative.</strong> To hold level 3 a system must also meet everything at levels 1 and 2. One failed requirement caps the level,
      even if the system passes requirements higher up.</li>
    </ul>
    <p>The evidence comes from <strong>measurement mechanisms</strong>: techniques that turn a trustworthiness goal into numbers. The paper says a good
    mechanism should be:</p>
  </div>
  <div class="props">
    <div><h4>Quantifiable</h4><p>Produces numbers with clear thresholds that can mark the boundary between levels.</p></div>
    <div><h4>Actionable</h4><p>Drives decisions, such as alerting an operator or switching to a safe fallback.</p></div>
    <div><h4>Formally grounded</h4><p>Offers mathematical guarantees where possible, not just heuristics.</p></div>
    <div><h4>Lifecycle-wide</h4><p>Applies from requirements through training, testing, deployment and maintenance.</p></div>
  </div>
  <div class="prose">
    <p>This assessment uses <strong>uncertainty quantification</strong> as its mechanism: measuring how sure the AI is about each decision, and whether that
    sureness can be trusted. Below is the rubric we scored against: three characteristics, each with requirements at every level. The numeric thresholds
    are ours, chosen for illustration.</p>
  </div>
  {rubric_blocks(cards)}
  <p class="note">One rubric change was made after seeing results. The first run passed Safety levels 3 and 4 with a system that raised false alarms on 69%
  of empty scenes. That is a loophole, not a safe system, so we added the false-alarm limit to Safety level 3. Every other threshold was set before
  the first run.</p>
</section>

<section class="part" id="system">
  <header><p class="eyebrow">Section 3</p><h2>The system under assessment</h2></header>
  <div class="prose">
    <p>The detector looks at a scene and answers one question: <em>is there a drone?</em> Everything is simulated, which lets us control every condition
    and know the right answer for every scene.</p>
  </div>
  <div class="pipeline">
    <div><p class="step">Scene</p><h4>What is out there</h4><p>A drone or not, maybe a bird, its size and distance, lighting, weather and terrain.</p></div>
    <div><p class="step">Sensors</p><h4>What gets measured</h4><p>Eight noisy readings: optical and thermal blobs, rotor signature, motion, shape, clutter, light, visibility.</p></div>
    <div><p class="step">Model</p><h4>An ensemble of five</h4><p>Five small neural networks, each trained on a different resample of the data.</p></div>
    <div><p class="step">Decision</p><h4>Alarm, dismiss or defer</h4><p>Raise an alarm, dismiss the scene, or hand it to a human operator when too uncertain.</p></div>
  </div>
  <div class="prose">
    <p>The <strong>operating domain</strong> is the set of conditions the system is meant to handle. We split it into {n_cells} cells:
    3 lighting levels ({lighting}) × 3 weather levels ({weather}) × 4 terrains ({terrains}). Robustness is judged cell by cell,
    so a system can't hide a weak spot behind a good average.</p>
    <p>The first training set was deliberately skewed, like a real first data-collection campaign: mostly clear daylight over cities and desert,
    with little night, fog or forest data.</p>
  </div>
</section>

<section class="part" id="measure">
  <header><p class="eyebrow">Section 4</p><h2>How we measured</h2></header>

  <div class="prose">
    <h3>4.1 Does uncertainty predict mistakes?</h3>
    <p>The five networks vote. When they agree, the ensemble is confident; when they disagree, it is uncertain. The first thing to check is whether
    uncertainty means anything. It does: in the most confident tenth of scenes the initial model was wrong {pct(dec[0], 1)} of the time, and in the least
    confident tenth, {pct(dec[-1])}.</p>
  </div>
  {figure("fig1_uncertainty_vs_error.png", "Left: error rate rises steadily with uncertainty. Right: handing the most uncertain scenes to an operator lowers the error rate on the scenes decided automatically.", "Bar chart of error rate by uncertainty decile and line chart of error versus fraction decided automatically, for the initial and improved models.")}

  <div class="prose">
    <h3>4.2 Is its confidence honest?</h3>
    <p><strong>Calibration</strong> asks whether a model that says it is 80% sure is right 80% of the time. The calibration error measures the gap.</p>
  </div>
  {figure("fig2_calibration.png", f"Reliability diagram: points on the dashed line mean stated confidence matches actual accuracy. Calibration error was {init['ece']:.3f} initially and {final['ece']:.3f} after improvement.", "Reliability diagram showing both models close to the perfect-calibration diagonal.")}
  <div class="prose">
    <p><strong>Conformal prediction</strong> adds a formal guarantee. Instead of one answer, the model gives a set of possible answers: "drone", "no drone",
    or both when it can't tell. The method guarantees the set contains the right answer at least 90% of the time. A set with both answers is a natural
    trigger for operator review. A variant that learns separate thresholds per condition and per class keeps the guarantee inside every cell and for
    drones specifically, which bounds the miss rate.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Measurement</th><th>Initial</th><th>After closed loop</th><th>Meaning</th></tr></thead>
    <tbody>{measure_rows}</tbody>
  </table></div>

  <div class="prose">
    <h3>4.3 What makes it uncertain?</h3>
    <p>Uncertainty comes in two kinds, and they call for different responses:</p>
    <ul>
      <li><strong>Epistemic</strong> uncertainty means the model hasn't seen enough examples like this. More data fixes it.</li>
      <li><strong>Aleatoric</strong> uncertainty means the sensors themselves are too noisy to tell. More data doesn't help.</li>
    </ul>
    <p>Tracing uncertainty back to scene conditions, the biggest drivers were {", ".join(f"<code>{k}</code>" for k, _ in imp[:3])}.</p>
  </div>
  {figure("fig3_uncertainty_drivers.png", "Mean uncertainty over lighting and weather. Epistemic uncertainty starts high at night and in bad weather, where training data was thin, and targeted data removes most of it. Aleatoric uncertainty stays high in the dark and in fog, because those sensor readings are noisy.", "Three heatmaps of uncertainty over lighting and weather, and a bar chart ranking scene parameters by influence on uncertainty.")}

  <div class="prose">
    <h3>4.4 Using measurements to improve the system</h3>
    <p>The paper proposes a <strong>closed loop</strong>: find where the model is most uncertain, generate more synthetic data like those scenes, retrain,
    and measure again. We ran five rounds of 400 scenes and compared against adding the same amount of random data, repeating everything with five
    different random seeds.</p>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Training data</th><th>Scenes</th><th>Accuracy</th><th>Worst-cell accuracy</th><th>Mean epistemic uncertainty</th></tr></thead>
    <tbody>{loop_rows}</tbody>
  </table></div>
  {figure("fig4_closed_loop.png", "Top and bottom left: both strategies improve the model, with uncertainty guidance slightly ahead on accuracy. Bottom right: guided data concentrates at night and in bad weather, while random data spreads evenly. Shaded bands show variation across the five seeds.", "Line charts of accuracy, worst-cell accuracy, epistemic uncertainty and calibration against training size, and scatter plots of where each strategy added data.")}
  <div class="prose"><p>Guidance helped, but modestly: about one standard deviation better than random data. Some of the most uncertain conditions are uncertain
  because of sensor noise, and extra data there cannot fix that.</p></div>

  <div class="prose">
    <h3>4.5 Choosing an operating point</h3>
    <p>The system has two settings: how likely a drone must be before it raises an alarm, and how uncertain it must be before it hands a scene to an
    operator. Every combination trades three things against each other: missed drones, false alarms and operator workload. We tested
    {par['n_policies']} combinations.</p>
    <p>A combination is <strong>Pareto-optimal</strong> if no other combination beats it on every measure at once. Across all three measures,
    {par['n_pareto_3_objectives']} of {par['n_policies']} qualified. More deferral always means fewer errors, so almost nothing is ruled out.
    The trade-off only sharpened once we fixed the operator's capacity: at most {pct(m['max_deferral'])} of scenes can be reviewed. Of the
    {par['n_within_capacity']} combinations within that budget, {par['n_constrained_front']} form the front between misses and false alarms.</p>
    <p>From that front we chose the combination with the lowest expected cost, counting a missed drone as {w_miss:g}× as costly as a false alarm
    and an operator review as {w_defer:g}×. These weights are value judgments that stakeholders should own. Writing them down makes the choice auditable.
    On the test scenes the chosen setting missed {pct(rates['miss'], 1)} of drones, raised false alarms on {pct(rates['false_alarm'], 1)} of empty scenes
    and sent {pct(rates['deferral'])} of scenes to the operator.</p>
  </div>
  {figure("fig5_tradeoffs.png", "Each dot is one combination of settings, shaded by how many scenes it sends to the operator. The orange line is the best available trade-off within the review budget; the star is the chosen setting.", "Scatter plot of miss rate against false alarm rate for all policies, with the constrained Pareto front and the chosen policy highlighted.")}
</section>

<section class="part" id="scores">
  <header><p class="eyebrow">Section 5</p><h2>Scores and the evidence behind them</h2></header>
  <div class="prose">
    <p>Every requirement is checked against the test scenes, and every result is recorded with its evidence. Reading down a column shows where each
    configuration stopped climbing. The cell names read lighting / weather / terrain.</p>
    <p>The <strong>restricted domain</strong> column certifies the system only for the conditions where it reaches 75% accuracy on separate calibration data,
    excluding {len(excluded)} of {n_cells}: {", ".join(f"<code>{esc(c)}</code>" for c in excluded)}. All measurements, including the operating point,
    are redone inside that domain. There the chosen setting missed {pct(rrates['miss'], 1)} of drones with {pct(rrates['false_alarm'], 1)} false alarms.
    Restricting the domain is legitimate only if it is decided as a requirement before testing, not after seeing which cells fail.</p>
  </div>
  {figure("fig6_maturity.png", "Maturity level per characteristic for each configuration.", "Horizontal bar chart of maturity levels for reliability, robustness and safety across the three configurations.")}
  {evidence_tables(cards)}
</section>

<section class="part" id="lessons">
  <header><p class="eyebrow">Section 6</p><h2>What we learned</h2></header>
  <div class="prose">
    <p><strong>The approach works in principle.</strong> Uncertainty quantification met every property the paper asks of a measurement mechanism. Each
    level traced to recorded evidence, and the scorecard pointed at real weaknesses rather than paperwork gaps.</p>
    <p><strong>It also exposed problems a real certification scheme must solve:</strong></p>
    <ul>
      <li><strong>Levels can move the wrong way.</strong> Reliability dropped after the model improved, because a noisy measurement crossed a hard
      threshold. Requirements should use statistical tests or confidence bounds, and scores should report how stable they are.</li>
      <li><strong>Loopholes appear fast.</strong> The first rubric could be satisfied by alarming on almost everything. Requirements for different
      characteristics have to be checked together.</li>
      <li><strong>The operating domain drives the score.</strong> Excluding {len(excluded)} conditions lifted robustness from level {lv['after closed loop'][rob]} to level {lv_r[rob]}. The domain must be
      fixed before testing.</li>
      <li><strong>Averages hide weak spots.</strong> A 90% guarantee on average fell short for drones specifically. Guarantees should hold within the
      groups that matter for safety.</li>
      <li><strong>Different uncertainty for different jobs.</strong> Epistemic uncertainty tells you what data to collect. Total uncertainty tells you what
      to hand to an operator. Once data gaps were filled, epistemic uncertainty alone barely predicted errors (AUROC {final['error_auroc_epistemic']:.2f}).</li>
      <li><strong>Trade-offs need constraints as well as weights.</strong> Weights alone produced extreme settings. A hard cap on operator workload,
      the kind of limit requirements already state, produced a sensible one.</li>
    </ul>
    <p>Uncertainty is one mechanism among many. A full scheme would add others for the remaining NIST characteristics, such as test-coverage metrics,
    adversarial robustness checks, drift detection in the field and formal verification for level 5.</p>
  </div>
</section>

<section class="part" id="glossary">
  <header><p class="eyebrow">Reference</p><h2>Glossary</h2></header>
  <dl class="gloss">
    <div><dt>Aleatoric uncertainty</dt><dd>Uncertainty from noise in the data itself. More data doesn't reduce it.</dd></div>
    <div><dt>AUROC</dt><dd>How well a score separates two groups, here mistakes from correct answers. 0.5 is chance, 1.0 is perfect.</dd></div>
    <div><dt>Calibration error (ECE)</dt><dd>The average gap between how confident a model says it is and how often it is right.</dd></div>
    <div><dt>Closed loop</dt><dd>Measure uncertainty, generate data where it is high, retrain, and measure again.</dd></div>
    <div><dt>Conformal prediction</dt><dd>A method that outputs a set of possible answers guaranteed to contain the right one at a chosen rate, such as 90%.</dd></div>
    <div><dt>Coverage</dt><dd>How often a conformal prediction set actually contains the right answer.</dd></div>
    <div><dt>Deferral</dt><dd>Handing a decision to a human operator instead of deciding automatically.</dd></div>
    <div><dt>Ensemble</dt><dd>Several models trained separately whose predictions are combined. Their disagreement measures uncertainty.</dd></div>
    <div><dt>Epistemic uncertainty</dt><dd>Uncertainty from lack of relevant training data. More data reduces it.</dd></div>
    <div><dt>Operating domain</dt><dd>The set of conditions a system is designed and certified to operate in.</dd></div>
    <div><dt>Pareto-optimal</dt><dd>A choice that no other choice beats on every objective at once.</dd></div>
    <div><dt>Wilson lower bound</dt><dd>A conservative estimate of a success rate that accounts for how many samples it rests on.</dd></div>
  </dl>
</section>

<footer>
  <p>Based on Darling, Hesu, Mardikes, McGuigan &amp; Milewicz, <a href="https://arxiv.org/abs/2601.03470"><em>Toward Maturity-Based Certification of Embodied AI:
  Quantifying Trustworthiness Through Measurement Mechanisms</em></a> (2026).</p>
  <p>Generated by <code>build_site.py</code> from the results of <code>run_demo.py</code>. All data is synthetic.</p>
</footer>

</main>
</div>
</body>
</html>
"""
    (out / "index.html").write_text(page)
    return out / "index.html"


def _change(before: int, after: int) -> str:
    return f"L{before} throughout" if before == after else f"L{before} to L{after}"


def _drone_cov(cards) -> float:
    """Drone-class coverage evidence for the final model (parsed from the recorded evidence)."""
    for crit in cards["after closed loop"][0]["criteria"]:
        if crit["requirement"].startswith("Drone-class coverage"):
            return float(crit["evidence"].split("=")[1])
    return float("nan")


if __name__ == "__main__":
    print(f"Wrote {build()}")
