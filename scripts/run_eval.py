"""Build incidents from labelled BGL and score the agents.

    python scripts/run_eval.py --bgl-dir ../log-anomaly-bench/data --sample 200
    python scripts/run_eval.py ... --claude          # also run the Claude agent (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rca.agents import ClaudeAgent, first_severe, most_frequent, scripted_agent  # noqa: E402
from rca.data import build_incidents, load_bgl, load_templates  # noqa: E402
from rca.evaluate import run, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bgl-dir", type=Path, default=ROOT.parent / "log-anomaly-bench" / "data")
    ap.add_argument("--sample", type=int, default=0, help="0 = all incidents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    t0 = time.time()
    ev = load_bgl(a.bgl_dir / "processed" / "BGL_events.csv", a.bgl_dir / "raw" / "BGL.log")
    templates = load_templates(a.bgl_dir / "processed" / "BGL_templates.csv")
    all_inc = build_incidents(ev, sample=None)
    incidents = build_incidents(ev, sample=a.sample or None, seed=a.seed)
    print(f"loaded {len(ev):,} lines, {len(all_inc)} incidents, scoring {len(incidents)} "
          f"({time.time() - t0:.0f}s)")

    agents = {"most_frequent": most_frequent, "first_severe": first_severe, "scripted_agent": scripted_agent}
    if a.claude:
        agents["claude_agent"] = ClaudeAgent()
    df = run(agents, incidents, templates)
    a.out.mkdir(exist_ok=True)
    df.to_csv(a.out / "per_incident.csv", index=False)
    s = summarize(df)
    print(s.to_string())
    (a.out / "summary.md").write_text(s.to_string() + "\n", encoding="utf-8")
    cats = df[df.agent == "scripted_agent"].groupby("category")["template_ok"].agg(["size", "mean"]).round(3)
    (a.out / "by_category.md").write_text(cats.sort_values("size", ascending=False).to_string() + "\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
