"""Tests on a small synthetic log: a fault starts on node B with a new FATAL template, then spreads."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from rca.agents import ClaudeAgent, first_severe, most_frequent, scripted_agent
from rca.data import build_incidents
from rca.evaluate import run, summarize
from rca.tools import Toolbox

TEMPLATES = {1: "heartbeat ok", 2: "ciod: failed to read message", 3: "machine check interrupt",
             4: "noisy error that is always there"}


def synthetic() -> pd.DataFrame:
    rows = []
    t = 1_000_000
    for i in range(3000):                                   # 25 h of normal traffic, one line / 30 s
        rows.append((t + 30 * i, f"N{i % 5}", "KERNEL", "INFO", 1, "-"))
        if i % 10 == 0:
            rows.append((t + 30 * i + 1, "N9", "APP", "ERROR", 4, "-"))
    t_fault = t + 30 * 3000
    rows.append((t_fault, "B", "KERNEL", "FATAL", 3, "KERNMC"))        # root cause on B
    for j in range(1, 40):                                              # symptoms on other nodes
        rows.append((t_fault + 5 * j, f"N{j % 5}", "APP", "FATAL", 2, "APPREAD"))
    for j in range(1, 4):
        rows.append((t_fault + 7 * j, "B", "KERNEL", "FATAL", 3, "KERNMC"))
    for k in range(60):
        rows.append((t_fault + 400 + 30 * k, f"N{k % 5}", "KERNEL", "INFO", 1, "-"))
    ev = pd.DataFrame(rows, columns=["timestamp", "node", "component", "level", "template_id", "label"])
    ev = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for c in ["node", "component", "level", "label"]:
        ev[c] = ev[c].astype("category")
    return ev


@pytest.fixture(scope="module")
def incident():
    incs = build_incidents(synthetic())
    assert len(incs) == 1
    return incs[0]


def test_incident_ground_truth_and_no_label_leak(incident):
    assert incident.root_node == "B"
    assert incident.category == "APPREAD"                  # majority category
    assert incident.root_templates == {2}
    assert "label" not in incident.window.columns
    assert incident.baseline.get(1, 0) > 0 and incident.baseline.get(2, 0) == 0


def test_tools(incident):
    tb = Toolbox(incident.window, incident.baseline, TEMPLATES, incident.alert_time)
    top = tb.top_changes(k=3, min_level="ERROR")["templates"]
    assert {r["template_id"] for r in top[:2]} == {2, 3} and all(r["is_new"] for r in top[:2])
    assert tb.first_occurrence(3)["node"] == "B"
    assert tb.call("nope", {})["error"].startswith("unknown")
    assert "error" in tb.call("first_occurrence", {"template_id": 999})
    assert tb.alert_context(seconds=10)["lines"][0]["node"] == "B"
    assert tb.search_templates("ciod")["matches"][0]["template_id"] == 2


def test_baselines_and_scripted_agent(incident):
    tb = lambda: Toolbox(incident.window, incident.baseline, TEMPLATES, incident.alert_time)  # noqa: E731
    assert most_frequent(tb()).root_template_id == 1        # fooled by the heartbeat
    d = scripted_agent(tb())
    assert d.root_node == "B" and d.root_template_id == 2 and d.tool_calls == 2
    fs = first_severe(tb())                                 # the paging line itself: right node, wrong template
    assert fs.root_node == "B" and fs.root_template_id == 3


class FakeClient:
    """Plays back two turns: one tool call, then submit_diagnosis."""

    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        usage = NS(input_tokens=100, output_tokens=20)
        if len(self.calls) == 1:
            return NS(stop_reason="tool_use", usage=usage, content=[
                NS(type="tool_use", id="t1", name="top_changes", input={"k": 3, "min_level": "ERROR"})])
        last = kw["messages"][-1]["content"][0]
        assert last["type"] == "tool_result" and last["tool_use_id"] == "t1"
        return NS(stop_reason="tool_use", usage=usage, content=[
            NS(type="tool_use", id="t2", name="submit_diagnosis",
               input={"root_template_id": 2, "root_node": "B", "explanation": "new FATAL burst"})])


def test_claude_agent_loop_with_fake_client(incident):
    client = FakeClient()
    agent = ClaudeAgent(client=client)
    df = run({"claude": agent}, [incident], TEMPLATES)
    assert df.iloc[0]["template_ok"] and df.iloc[0]["node_ok"]
    assert df.iloc[0]["input_tokens"] == 200
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert any(t["name"] == "submit_diagnosis" for t in kw["tools"])


def test_refusal_is_handled(incident):
    client = NS(beta=NS(messages=NS(create=lambda **kw: NS(
        stop_reason="refusal", usage=NS(input_tokens=1, output_tokens=0), content=[]))))
    tb = Toolbox(incident.window, incident.baseline, TEMPLATES, incident.alert_time)
    d = ClaudeAgent(client=client)(tb, incident.alert_time)
    assert d.root_template_id is None and d.explanation == "model declined"


def test_summarize(incident):
    df = run({"mf": most_frequent, "sa": scripted_agent}, [incident], TEMPLATES)
    s = summarize(df)
    assert set(s.index) == {"mf", "sa"} and np.all(s["incidents"] == 1)
