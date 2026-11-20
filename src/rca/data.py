"""Turn the labelled BGL log into root-cause-analysis incidents.

BGL (Loghub) labels every line: the first field is "-" for normal lines or an alert category
such as KERNDTLB, APPREAD... We group alert lines into incidents (a new incident starts after
`gap_s` seconds without any alert) and keep, for each incident:

  * the time the first alert fired (what an on-call engineer would be paged with),
  * the ground truth: the incident's main alert category, the template ids that carry it,
    and the node of the first alert line,
  * a label-free window of log lines around that time for the agent to investigate,
  * template counts over the previous 24 h, so tools can tell a burst from business as usual.

Labels never reach the agent: the window and baseline only contain timestamp, node, component,
level and template id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Incident:
    id: int
    alert_time: int                       # unix seconds of the first alert line
    category: str                         # main alert category (ground truth)
    root_templates: set[int]              # templates labelled with that category in the incident
    root_node: str                        # node of the first alert line
    window: pd.DataFrame                  # label-free lines in [alert_time - before, alert_time + after]
    baseline: pd.Series                   # template_id -> count over the previous 24 h
    n_alert_lines: int = 0
    nodes_affected: set[str] = field(default_factory=set)


def read_bgl_labels(raw_log: Path) -> np.ndarray:
    """First whitespace-separated field of every line ('-' = normal)."""
    out = []
    with open(raw_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sp = line.find(" ")
            out.append(line[:sp] if sp > 0 else line.strip())
    return np.array(out, dtype=object)


def load_bgl(events_csv: Path, raw_log: Path) -> pd.DataFrame:
    ev = pd.read_csv(events_csv, usecols=["timestamp", "node", "component", "level", "template_id"],
                     dtype={"node": "category", "component": "category", "level": "category"})
    labels = read_bgl_labels(raw_log)
    if len(labels) != len(ev):
        raise ValueError(f"raw log has {len(labels)} lines but events file has {len(ev)}")
    ev["label"] = pd.Categorical(labels)
    return ev


def build_incidents(ev: pd.DataFrame, gap_s: int = 300, before_s: int = 1800, after_s: int = 600,
                    min_alert_lines: int = 3, baseline_s: int = 86_400, max_window_lines: int = 50_000,
                    sample: int | None = None, seed: int = 0) -> list[Incident]:
    ts = ev["timestamp"].to_numpy()
    is_alert = (ev["label"] != "-").to_numpy()
    alert_idx = np.flatnonzero(is_alert)
    if len(alert_idx) == 0:
        return []
    # split alert lines into incidents on time gaps
    breaks = np.flatnonzero(np.diff(ts[alert_idx]) > gap_s) + 1
    groups = np.split(alert_idx, breaks)
    groups = [g for g in groups if len(g) >= min_alert_lines]
    rng = np.random.default_rng(seed)
    if sample is not None and len(groups) > sample:
        groups = [groups[i] for i in sorted(rng.choice(len(groups), sample, replace=False))]

    keep_cols = ["timestamp", "node", "component", "level", "template_id"]
    incidents = []
    for k, g in enumerate(groups):
        t0 = int(ts[g[0]])
        lab = ev["label"].iloc[g].astype(str)
        category = lab.value_counts().index[0]
        root_templates = set(ev["template_id"].iloc[g][lab.to_numpy() == category].unique().tolist())
        lo, hi = np.searchsorted(ts, t0 - before_s), np.searchsorted(ts, t0 + after_s, side="right")
        if hi - lo > max_window_lines:                       # keep the lines closest to the alert
            hi = min(hi, lo + max_window_lines)
        window = ev.iloc[lo:hi][keep_cols].reset_index(drop=True)
        b_lo = np.searchsorted(ts, t0 - before_s - baseline_s)
        baseline = ev["template_id"].iloc[b_lo:lo].value_counts()
        incidents.append(Incident(
            id=k, alert_time=t0, category=category, root_templates=root_templates,
            root_node=str(ev["node"].iloc[g[0]]), window=window, baseline=baseline,
            n_alert_lines=len(g), nodes_affected=set(ev["node"].iloc[g].astype(str).unique().tolist())))
    return incidents


def load_templates(templates_csv: Path) -> dict[int, str]:
    t = pd.read_csv(templates_csv, usecols=["template_id", "template"])
    return dict(zip(t["template_id"].astype(int), t["template"].astype(str)))
