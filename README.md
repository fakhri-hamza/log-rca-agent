# log-rca-agent

![CI](https://github.com/fakhri-hamza/log-rca-agent/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A tool-using **root-cause analysis agent** for log incidents, with an **offline evaluation on 457 real incidents** from the labelled BlueGene/L (BGL) supercomputer log.

The on-call question is: *an alert fired at time T, which log message represents the root cause, and on which node did it start?*
The agent only gets what an on-call engineer would get: the alert time, an unlabelled window of logs around it (30 min before, 10 min after) and the template counts of the previous 24 h. It investigates through six small tools and then submits a diagnosis.

## How it works

| Piece | File |
|---|---|
| Incidents from labelled BGL: alert lines grouped on 5-minute gaps; ground truth = main alert category (its templates) + node of the first alert line; labels never reach the agent | `src/rca/data.py` |
| Tools: `window_summary`, `top_changes` (rate vs previous 24 h), `alert_context`, `first_occurrence`, `nodes_for_template`, `search_templates`, `template_text`, each returning a small JSON dict | `src/rca/tools.py` |
| Agents: two baselines, a scripted tool-using agent, and a Claude agent (manual tool-use loop, `submit_diagnosis` as a strict tool, refusal handling, server-side fallbacks) | `src/rca/agents.py` |
| Scoring: template accuracy, node accuracy, both, tool calls, time | `src/rca/evaluate.py` |

## Results (all 457 BGL incidents with ≥ 3 alert lines)

| Agent | Template acc. | Node acc. | Both | Tool calls |
|---|---|---|---|---|
| **scripted agent** (alert context + 24 h rate change) | **0.781** | 0.978 | **0.779** | 2.0 |
| first severe line at the alert (the paging line) | 0.751 | 0.978 | 0.751 | 0 |
| most frequent template in the window | 0.643 | 0.363 | 0.302 | 0 |

What this shows, honestly:

- **The node is easy, the cause is not.** The line that paged you already names the right node 97.8 % of the time. Picking the right *message* is where the work is.
- **Comparing against the last 24 h helps, a bit.** Re-ranking the templates seen at the alert by their rate rise adds 3 points over the paging line. I designed the agent on a 200-incident sample (where the gain was 7 points), so the full-set number is the one to trust.
- **Failure case: KERNMC (machine checks), 0/23.** A machine-check dump writes several lines at once; the agent picks a detail line (`capture first ... error address`) that is part of the same dump but is not labelled as the alert. A per-category view is in `results/by_category.md`.
- **The Claude agent is wired and unit-tested (with a fake client) but not yet scored** on the real incidents: I have not run it, so there is no number for it here. Run it with `--claude` and an `ANTHROPIC_API_KEY`.

## Run it

```bash
pip install pandas numpy pytest            # + anthropic for the Claude agent
pytest -q                                  # synthetic incident, no data needed
python scripts/run_eval.py --bgl-dir ../log-anomaly-bench/data            # all incidents
python scripts/run_eval.py --bgl-dir ../log-anomaly-bench/data --sample 50 --claude
```

The BGL data (Loghub) is prepared by [log-anomaly-bench](https://github.com/fakhri-hamza/log-anomaly-bench): `processed/BGL_events.csv`, `processed/BGL_templates.csv` and `raw/BGL.log`.

## Licence

MIT
