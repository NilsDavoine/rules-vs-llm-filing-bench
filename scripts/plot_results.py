"""Draw docs/results.png from results/predictions.jsonl (written by filing_bench.bench).

One bar per approach: the share of the 51 test pieces (never used to write the rules)
whose file name AND folder are exactly right. Needs requirements-plot.txt.

    python scripts/plot_results.py
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {"Rules": "#2a78d6", "Local model": "#eb6834", "Hybrid (rules first)": "#1baf7a",
          "Claude": "#6250d6"}
LABELS = {"rules-v0": "Rules v0  naive", "rules-v1": "Rules v1  + dates",
          "rules-v2": "Rules v2  + vocabulary", "rules-v3": "Rules v3  + parties",
          "qwen2.5-0.5b": "Qwen2.5 0.5B", "qwen2.5-1.5b": "Qwen2.5 1.5B",
          "hybrid v3+qwen2.5-1.5b": "Rules v3 + Qwen2.5 1.5B"}


def group(name):
    if name.startswith("rules"):
        return "Rules"
    if name.startswith("hybrid"):
        return "Hybrid (rules first)"
    if name.startswith("claude"):
        return "Claude"
    return "Local model"


def label(name):
    if name in LABELS:
        return LABELS[name]
    m = re.match(r"claude-(\w+)-(\d+)-(\d+)$", name)
    return f"Claude {m.group(1).title()} {m.group(2)}.{m.group(3)}" if m else name


def main():
    ok, n = {}, {}
    with open(os.path.join(ROOT, "results", "predictions.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                ok[r["approach"]] = ok.get(r["approach"], 0) + r["ok"]
                n[r["approach"]] = n.get(r["approach"], 0) + 1
    # One hybrid is enough: both pick "never ask" on dev and score the same.
    names = [a for a in ok if a != "hybrid v3+qwen2.5-0.5b"]
    order = list(GROUPS)
    names.sort(key=lambda a: (order.index(group(a)), list(ok).index(a)))
    rates = [100 * ok[a] / n[a] for a in names]
    pieces = n[names[0]]

    h = 0.42 * len(names) + 1.9
    fig, ax = plt.subplots(figsize=(8, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(names)))[::-1]
    ax.barh(y, rates, height=0.62, color=[GROUPS[group(a)] for a in names])
    for yi, v, a in zip(y, rates, names):
        ax.text(v + 1.2, yi, f"{v:.0f} %  ({ok[a]}/{n[a]})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [label(a) for a in names], fontsize=9.5, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=8.5, color=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)
    fig.text(0.02, 1 - 0.28 / h, "Documents filed exactly right (name and folder)",
             fontsize=12, color=INK, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.62 / h, f"Test split: {pieces} invented pieces never used to write the rules. "
             "Source: results/predictions.jsonl", fontsize=8.5, color=INK2, va="top")
    present = [g for g in GROUPS if any(group(a) == g for a in names)]
    fig.legend(handles=[Patch(color=GROUPS[g], label=g) for g in present], loc="upper left",
               bbox_to_anchor=(0.012, 1 - 0.86 / h), ncol=len(present), frameon=False,
               fontsize=8.5, labelcolor=INK2, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.25 / h))
    out = os.path.join(ROOT, "docs", "results.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
