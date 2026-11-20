"""Root-cause agents and baselines. Every agent maps (incident, toolbox) -> Diagnosis."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from rca.tools import SEVERE, SUBMIT_SPEC, TOOL_SPECS, Toolbox


@dataclass
class Diagnosis:
    root_template_id: int | None
    root_node: str | None
    explanation: str = ""
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    trace: list = field(default_factory=list)


# ---------------------------------------------------------------- baselines (no tools, no reasoning)
def most_frequent(tb: Toolbox) -> Diagnosis:
    """The template with the most lines in the window, and the node that logged it first."""
    w = tb.w
    if w.empty:
        return Diagnosis(None, None, "empty window")
    tid = int(w["template_id"].value_counts().index[0])
    node = str(w.loc[w["template_id"] == tid, "node"].iloc[0])
    return Diagnosis(tid, node, "most frequent template")


def first_severe(tb: Toolbox) -> Diagnosis:
    """The first ERROR-or-worse line at or after the alert time (the line that paged us)."""
    w = tb.w
    sev = w[w["level"].astype(str).isin(SEVERE) & (w["timestamp"] >= tb.alert_time)]
    if sev.empty:
        return most_frequent(tb)
    r = sev.iloc[0]
    return Diagnosis(int(r["template_id"]), str(r["node"]), "first severe line at the alert")


# ---------------------------------------------------------------- scripted tool-using agent
def scripted_agent(tb: Toolbox, seconds: int = 60) -> Diagnosis:
    """A fixed investigation plan with the same tools the LLM agent gets:
    1. read the severe lines around the alert (alert_context);
    2. the root node is the node of the first of them;
    3. the root template is, among the templates in that context, the one whose rate rose the most
       against the previous 24 h (top_changes), ties broken by first appearance."""
    trace = []
    ctx = tb.alert_context(seconds=seconds, k=200)["lines"]
    trace.append(("alert_context", {"seconds": seconds}, ctx[:5]))
    if not ctx:
        d = most_frequent(tb)
        d.tool_calls, d.trace = tb.calls, trace
        return d
    node = ctx[0]["node"]
    in_ctx = {line["template_id"]: i for i, line in reversed(list(enumerate(ctx)))}
    ranked = tb.top_changes(k=1000, min_level="ERROR")["templates"]
    trace.append(("top_changes", {"min_level": "ERROR"}, ranked[:5]))
    score = {r["template_id"]: r["score"] for r in ranked}
    tid = max(in_ctx, key=lambda t: (score.get(t, 0.0), -in_ctx[t]))
    return Diagnosis(tid, node, f"largest rate rise among templates at the alert: {tb.templates.get(tid, '?')[:80]}",
                     tb.calls, trace=trace)


# ---------------------------------------------------------------- Claude tool-use agent
SYSTEM = """You are an SRE investigating an alert on a BlueGene/L supercomputer. You get a window of
log lines around the time the alert fired (from 30 minutes before to 10 minutes after) and the
template counts of the previous 24 hours. Use the tools to find the root cause: the log template
that best represents what went wrong first, and the node where it started. Compare against the
24-hour baseline before deciding: a template that is always there is rarely the cause. Keep the
investigation short (a handful of tool calls), then call submit_diagnosis exactly once."""


class ClaudeAgent:
    """Manual tool-use loop on the Anthropic Messages API (client injected so tests can fake it)."""

    needs_alert_time = True

    def __init__(self, client=None, model: str = "claude-opus-5-5", effort: str = "medium",
                 max_turns: int = 12):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort, self.max_turns = client, model, effort, max_turns

    def __call__(self, tb: Toolbox, alert_time: int) -> Diagnosis:
        messages = [{"role": "user", "content":
                     f"An alert fired at unix time {alert_time}. Investigate and submit your diagnosis."}]
        diag = Diagnosis(None, None)
        for _ in range(self.max_turns):
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM,
                tools=TOOL_SPECS + [SUBMIT_SPEC], messages=messages,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            diag.input_tokens += resp.usage.input_tokens
            diag.output_tokens += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                diag.explanation = "model declined"
                return diag
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:                                   # ended without submitting
                diag.explanation = next((b.text for b in resp.content if b.type == "text"), "")
                return diag
            results = []
            for u in uses:
                args = u.input if isinstance(u.input, dict) else json.loads(u.input)
                if u.name == "submit_diagnosis":
                    diag.root_template_id = int(args["root_template_id"])
                    diag.root_node = str(args["root_node"])
                    diag.explanation = args.get("explanation", "")
                    diag.tool_calls = tb.calls
                    return diag
                out = tb.call(u.name, args)
                diag.trace.append((u.name, args, out))
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out),
                                **({"is_error": True} if "error" in out else {})})
            messages.append({"role": "user", "content": results})
        diag.explanation = "turn limit reached"
        diag.tool_calls = tb.calls
        return diag
