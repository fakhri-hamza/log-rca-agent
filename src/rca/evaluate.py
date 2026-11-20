"""Score agents on labelled incidents.

  template accuracy : the predicted template carries the incident's alert category
  node accuracy     : the predicted node is the node of the first alert line
  both              : both right
"""
from __future__ import annotations

import time

import pandas as pd

from rca.agents import Diagnosis
from rca.data import Incident
from rca.tools import Toolbox


def score(inc: Incident, d: Diagnosis) -> dict:
    t_ok = d.root_template_id is not None and d.root_template_id in inc.root_templates
    n_ok = d.root_node is not None and d.root_node == inc.root_node
    return {"incident": inc.id, "category": inc.category, "template_ok": t_ok, "node_ok": n_ok,
            "both_ok": t_ok and n_ok, "tool_calls": d.tool_calls,
            "input_tokens": d.input_tokens, "output_tokens": d.output_tokens}


def run(agents: dict, incidents: list[Incident], templates: dict[int, str]) -> pd.DataFrame:
    rows = []
    for inc in incidents:
        for name, agent in agents.items():
            tb = Toolbox(inc.window, inc.baseline, templates, inc.alert_time)
            t0 = time.perf_counter()
            d = agent(tb, inc.alert_time) if getattr(agent, "needs_alert_time", False) else agent(tb)
            if not d.tool_calls:
                d.tool_calls = tb.calls
            rows.append({"agent": name, **score(inc, d), "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("agent")
    return pd.DataFrame({
        "incidents": g.size(),
        "template_acc": g["template_ok"].mean().round(3),
        "node_acc": g["node_ok"].mean().round(3),
        "both_acc": g["both_ok"].mean().round(3),
        "avg_tool_calls": g["tool_calls"].mean().round(1),
        "avg_seconds": g["seconds"].mean().round(3),
    }).sort_values("both_acc", ascending=False)
