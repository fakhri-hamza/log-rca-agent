"""Investigation tools over one incident window. Every tool returns a small JSON-able dict.

The same functions back both agents: the scripted baseline calls them directly, and the Claude
agent calls them through tool use (their JSON schemas are in TOOL_SPECS).
"""
from __future__ import annotations

import math

import pandas as pd

SEVERE = {"FATAL", "FAILURE", "SEVERE", "ERROR"}


class Toolbox:
    def __init__(self, window: pd.DataFrame, baseline: pd.Series, templates: dict[int, str],
                 alert_time: int, baseline_hours: float = 24.0):
        self.w = window
        self.baseline = baseline
        self.templates = templates
        self.alert_time = alert_time
        self.window_hours = max((window["timestamp"].max() - window["timestamp"].min()) / 3600, 1 / 60) \
            if len(window) else 1 / 60
        self.baseline_hours = baseline_hours
        self.calls = 0

    # -- tools -------------------------------------------------------------------------------
    def window_summary(self) -> dict:
        self.calls += 1
        w = self.w
        return {"lines": int(len(w)), "nodes": int(w["node"].nunique()),
                "templates": int(w["template_id"].nunique()),
                "by_level": {str(k): int(v) for k, v in w["level"].value_counts().items() if v},
                "by_component": {str(k): int(v) for k, v in w["component"].value_counts().head(8).items() if v}}

    def top_changes(self, k: int = 10, min_level: str | None = None) -> dict:
        """Templates ranked by how unusual their rate is in the window vs the previous 24 h.
        score = log((window rate + 1) / (baseline rate + 1)); new templates get is_new=True."""
        self.calls += 1
        w = self.w if min_level is None else self.w[self.w["level"].astype(str).isin(_levels_at_least(min_level))]
        rows = []
        for tid, cnt in w["template_id"].value_counts().items():
            base = int(self.baseline.get(tid, 0))
            r_w, r_b = cnt / self.window_hours, base / self.baseline_hours
            rows.append({"template_id": int(tid), "count": int(cnt), "baseline_24h": base,
                         "is_new": base == 0, "score": round(math.log((r_w + 1) / (r_b + 1)), 3),
                         "level": str(w.loc[w["template_id"] == tid, "level"].mode().iloc[0]),
                         "text": self.templates.get(int(tid), "?")[:120]})
        rows.sort(key=lambda r: (-r["score"], -r["count"]))
        return {"templates": rows[:k]}

    def first_occurrence(self, template_id: int) -> dict:
        self.calls += 1
        rows = self.w[self.w["template_id"] == template_id]
        if rows.empty:
            return {"error": f"template {template_id} not in window"}
        r = rows.iloc[0]
        return {"template_id": int(template_id), "first_time": int(r["timestamp"]),
                "seconds_before_alert": int(self.alert_time - r["timestamp"]),
                "node": str(r["node"]), "component": str(r["component"]), "level": str(r["level"])}

    def nodes_for_template(self, template_id: int, k: int = 5) -> dict:
        self.calls += 1
        rows = self.w[self.w["template_id"] == template_id]
        firsts = rows.groupby("node", observed=True)["timestamp"].min().sort_values()
        counts = rows["node"].value_counts()
        return {"template_id": int(template_id), "n_nodes": int(rows["node"].nunique()),
                "nodes_by_first_seen": [{"node": str(n), "first_time": int(t), "count": int(counts[n])}
                                        for n, t in firsts.head(k).items()]}

    def alert_context(self, seconds: int = 60, k: int = 20) -> dict:
        """Severe lines logged from the alert time to `seconds` after it, in time order."""
        self.calls += 1
        w = self.w
        m = (w["timestamp"] >= self.alert_time) & (w["timestamp"] <= self.alert_time + seconds)             & w["level"].astype(str).isin(SEVERE)
        rows = w[m].head(k)
        return {"lines": [{"time": int(r.timestamp), "node": str(r.node), "component": str(r.component),
                           "level": str(r.level), "template_id": int(r.template_id),
                           "text": self.templates.get(int(r.template_id), "?")[:100]}
                          for r in rows.itertuples(index=False)]}

    def template_text(self, template_id: int) -> dict:
        self.calls += 1
        return {"template_id": int(template_id), "text": self.templates.get(int(template_id), "?")}

    def search_templates(self, keyword: str, k: int = 10) -> dict:
        self.calls += 1
        kw = keyword.lower()
        present = self.w["template_id"].value_counts()
        hits = [{"template_id": int(t), "count": int(c), "text": self.templates.get(int(t), "?")[:120]}
                for t, c in present.items() if kw in self.templates.get(int(t), "").lower()]
        return {"keyword": keyword, "matches": hits[:k]}

    # -- dispatch for tool use ----------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as exc:                          # noqa: BLE001 - report to the agent
            return {"error": f"{type(exc).__name__}: {exc}"}


def _levels_at_least(level: str) -> set[str]:
    order = ["INFO", "WARNING", "ERROR", "SEVERE", "FAILURE", "FATAL"]
    level = level.upper()
    return set(order[order.index(level):]) if level in order else set(order)


TOOL_SPECS = [
    {"name": "window_summary", "description": "Overview of the log window around the alert: line, node and template counts, lines per level and per component.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "top_changes", "description": "Log templates ranked by how unusual their rate is in the window compared with the previous 24 hours (score = log rate ratio; is_new = never seen in the previous 24 h). Optionally keep only lines at or above a severity level.",
     "input_schema": {"type": "object", "properties": {
         "k": {"type": "integer", "description": "how many templates to return (default 10)"},
         "min_level": {"type": "string", "enum": ["INFO", "WARNING", "ERROR", "SEVERE", "FAILURE", "FATAL"]}},
         "required": [], "additionalProperties": False}},
    {"name": "first_occurrence", "description": "When and where a template first appears in the window, and how many seconds before the alert.",
     "input_schema": {"type": "object", "properties": {"template_id": {"type": "integer"}},
                      "required": ["template_id"], "additionalProperties": False}},
    {"name": "nodes_for_template", "description": "Nodes that logged a template, ordered by when they first logged it.",
     "input_schema": {"type": "object", "properties": {"template_id": {"type": "integer"}, "k": {"type": "integer"}},
                      "required": ["template_id"], "additionalProperties": False}},
    {"name": "alert_context", "description": "Severe (ERROR or worse) lines logged from the alert time to `seconds` after it, in time order: what the page was about.",
     "input_schema": {"type": "object", "properties": {"seconds": {"type": "integer"}, "k": {"type": "integer"}},
                      "required": [], "additionalProperties": False}},
    {"name": "template_text", "description": "Full text of a log template.",
     "input_schema": {"type": "object", "properties": {"template_id": {"type": "integer"}},
                      "required": ["template_id"], "additionalProperties": False}},
    {"name": "search_templates", "description": "Templates present in the window whose text contains a keyword (case-insensitive).",
     "input_schema": {"type": "object", "properties": {"keyword": {"type": "string"}, "k": {"type": "integer"}},
                      "required": ["keyword"], "additionalProperties": False}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SPECS}

SUBMIT_SPEC = {
    "name": "submit_diagnosis",
    "description": "Submit the final root-cause diagnosis. Call this exactly once, when you are done investigating.",
    "strict": True,
    "input_schema": {"type": "object", "properties": {
        "root_template_id": {"type": "integer", "description": "the log template that best represents the root cause"},
        "root_node": {"type": "string", "description": "the node where the problem started"},
        "explanation": {"type": "string", "description": "two or three sentences of reasoning"}},
        "required": ["root_template_id", "root_node", "explanation"], "additionalProperties": False},
}
