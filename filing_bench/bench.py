"""Run every approach on both splits and write results.md.

    python -m filing_bench.bench              # rules + local models (+ Claude if its answers are cached)
    python -m filing_bench.bench --no-llm     # rules only, a few seconds, no download
    python -m filing_bench.bench --claude     # also asks Claude (ANTHROPIC_API_KEY), cost estimate first

Every number in results.md comes from this script. Model answers are cached
in results/cache/, so a second run re-scores without re-generating.
"""
import argparse
import collections
import datetime as dt
import json
import os
import statistics
import sys
import time

from . import corpus, guard
from .llm import LOCAL_MODELS, ClaudeModel, LocalModel
from .rules import V3, VERSIONS
from .schema import NATURE_FOLDER, NO_DATE, NATURES, place
from .score import METRICS, score, summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SPLITS = ("dev", "test")
THRESHOLDS = [0.0, 0.5, 0.6, 0.75, 0.9, 1.01]


def run_one(piece, propose):
    """propose(piece) -> (Proposal, seconds, extra row or None)."""
    t0 = time.perf_counter()
    out = propose(piece)
    elapsed = time.perf_counter() - t0
    prop, row = out if isinstance(out, tuple) else (out, None)
    seconds = row["seconds"] if row else elapsed
    raw_period = prop.period
    prop, invented = guard.enforce(prop, piece["text"])
    r = score(piece, prop, invented)
    return {
        "id": piece["id"], "kind": piece["kind"], "lang": piece["lang"], "traps": piece["traps"],
        "expected": r["expected_name"], "expected_folder": piece["expected"]["folder"],
        "got": prop.name, "got_folder": prop.folder, "raw_period": raw_period,
        "confidence": prop.confidence, "seconds": seconds,
        "tokens_in": row["tokens_in"] if row else 0, "tokens_out": row["tokens_out"] if row else 0,
        "model_raw": row["raw"] if row else "", "device": row.get("device", "") if row else "cpu",
        **{k: r[k] for k in METRICS},
    }


def hybrid(rules, model, threshold):
    """Rules first; the model is asked only when the rules doubt, and only
    for what they doubt about. Its date still has to pass the guard."""
    def propose(piece):
        p = rules.propose(piece)
        if p.confidence >= threshold:
            return p, None
        q, row = model.propose(piece)
        scores = sorted(rules.nature_scores(piece).values(), reverse=True)
        doubt_nature = not scores or (len(scores) > 1 and scores[0] - scores[1] < 3)
        if doubt_nature and q.nature:
            p.nature = q.nature
            p.emitter = rules.emitter(piece, p.nature) or q.emitter
        if p.period == NO_DATE and q.period != NO_DATE and guard.supported(q.period, piece["text"]):
            p.period = q.period
        if NATURES.get(p.nature) and len(p.emitter) < 3 and q.emitter:
            p.emitter = q.emitter
        p.folder = rules.folder(piece, p)
        row = dict(row, asked=True)
        return p, row
    return propose


def approaches(args):
    out = [(v.name, v.propose) for v in VERSIONS]
    models = []
    if not args.no_llm:
        for name in args.models:
            m = LocalModel(name)
            out.append((name, m.propose))
            models.append(m)
        c = ClaudeModel()
        if args.claude or not c.missing(args.pieces):
            # A complete cache re-scores without a key; new calls need --claude.
            out.append((c.name, c.propose))
    return out, models


def confirm_claude(args, pieces):
    """Print the cost estimate before any call; returns False to stop."""
    c = ClaudeModel()
    e = c.estimate(pieces)
    print(f"Claude ({c.model}): {e['calls']} pieces to send, {e['cached']} already cached. "
          f"${e['price_in']:.2f} / ${e['price_out']:.2f} per 1M input / output tokens.\n"
          f"  expected  : {e['tokens_in']:,.0f} in + {e['expected_out']:,.0f} out  ~ ${e['expected_usd']:.3f}\n"
          f"  worst case: {e['tokens_in']:,.0f} in + {e['worst_out']:,.0f} out  ~ ${e['worst_usd']:.3f}"
          f"  (every answer at max_tokens)", flush=True)
    if e["calls"] == 0:
        return True
    if not ClaudeModel.available():
        print("set ANTHROPIC_API_KEY to run Claude", file=sys.stderr)
        return False
    if args.yes:
        return True
    return input("Run it, billed to your key? [y/N] ").strip().lower() in ("y", "yes")


def pct(x):
    return f"{100 * x:.0f} %"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def latency(records):
    s = [r["seconds"] for r in records]
    return statistics.mean(s), sorted(s)[int(0.95 * (len(s) - 1))]


def fmt_s(x):
    return f"{x * 1000:.2f} ms" if x < 1 else f"{x:.2f} s"


# --------------------------------------------------------- error analysis --

def category(r):
    wrong = []
    if r["invented"]:
        wrong.append("invented date (blocked)")
    if not r["period"]:
        wrong.append("period")
    if not r["nature"]:
        wrong.append("nature")
    elif not r["name"]:
        wrong.append("emitter")
    if not r["folder"]:
        wrong.append("folder")
    return " + ".join(wrong) or "?"


def typical_failures(records, n=10):
    """The n failures that best represent what goes wrong: one example per
    (kind of error, kind of piece), most frequent first."""
    fails = [r for r in records if not r["ok"]]
    groups = collections.OrderedDict()
    for r in sorted(fails, key=lambda r: r["id"]):
        groups.setdefault((category(r), r["kind"]), []).append(r)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:n]
    rows = []
    for (cat, kind), rs in ranked:
        r = rs[0]
        traps = ", ".join(t for t in r["traps"] if t not in ("body_keywords",))[:70]
        got = r["got"] + ("" if r["folder"] else f" -> {r['got_folder'] or '(none)'}")
        exp = r["expected"] + ("" if r["folder"] else f" -> {r['expected_folder']}")
        rows.append([len(rs), cat, kind, r["id"], f"`{exp}`", f"`{got}`", traps])
    return rows, len(fails)


# ------------------------------------------------------------------ main --

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(LOCAL_MODELS))
    ap.add_argument("--claude", action="store_true",
                    help="also run Claude (CLAUDE_MODEL, default claude-haiku-5-5) on every piece")
    ap.add_argument("--yes", action="store_true", help="with --claude: skip the confirmation")
    args = ap.parse_args(argv)
    data = {s: corpus.load(s) for s in SPLITS}
    args.pieces = data["dev"] + data["test"]
    if args.claude and not confirm_claude(args, args.pieces):
        print("Cancelled, nothing was sent to Claude.")
        return 1
    runs, models = approaches(args)
    rec = {}
    for name, propose in runs:
        for s in SPLITS:
            rec[(name, s)] = [run_one(p, propose) for p in data[s]]
            print(f"{name:>16} {s:>4}: ok {pct(summary(rec[(name, s)])['ok'])}", flush=True)
        for m in models:
            m.unload()

    # Hybrid: threshold chosen on dev only, then applied once to test.
    sweep = []
    hybrid_names = []
    for m in models:
        best = None
        for t in THRESHOLDS:
            rs = [run_one(p, hybrid(V3(), m, t)) for p in data["dev"]]
            asked = sum(1 for p in data["dev"] if V3().propose(p).confidence < t) / len(data["dev"])
            ok = summary(rs)["ok"]
            # Test is scored too, for the reader only: the choice below looks at dev alone.
            ts = [run_one(p, hybrid(V3(), m, t)) for p in data["test"]]
            t_asked = sum(1 for p in data["test"] if V3().propose(p).confidence < t) / len(data["test"])
            sweep.append([m.name, t, pct(ok), pct(asked), pct(summary(ts)["ok"]), pct(t_asked)])
            if best is None or ok > best[1] + 1e-9:
                best = (t, ok)
        hname = f"hybrid v3+{m.name}"
        hybrid_names.append((hname, m.name, best[0]))
        for s in SPLITS:
            rec[(hname, s)] = [run_one(p, hybrid(V3(), m, best[0])) for p in data[s]]
            print(f"{hname:>16} {s:>4}: ok {pct(summary(rec[(hname, s)])['ok'])} (threshold {best[0]})", flush=True)

    names = [n for n, _ in runs] + [h[0] for h in hybrid_names]
    write_report(data, rec, names, sweep, hybrid_names, models)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for (name, s), rs in rec.items():
            for r in rs:
                f.write(json.dumps({"approach": name, "split": s, **r}, ensure_ascii=False) + "\n")


def write_report(data, rec, names, sweep, hybrid_names, models):
    out = ["# Results", "",
           f"Generated by `python -m filing_bench.bench` on {dt.date.today().isoformat()}. "
           "Do not edit by hand.", "",
           f"Corpus: {len(data['dev'])} dev pieces (rules written while looking at them) and "
           f"{len(data['test'])} test pieces (other emitters, plus wording that never appears in dev). "
           "A line is **ok** when both the file name and the folder are exactly right.", ""]

    out += ["## The road of the rules", "",
            "Each version adds one idea (see `filing_bench/rules.py`).", ""]
    rows = []
    prev = {}
    for v in VERSIONS:
        d, t = summary(rec[(v.name, "dev")]), summary(rec[(v.name, "test")])
        gain = "" if not prev else f"+{100 * (t['ok'] - prev['ok']):.0f} pts"
        rows.append([v.name, pct(d["ok"]), pct(t["ok"]), gain, pct(t["period"]), pct(t["nature"]),
                     pct(t["name"]), pct(t["folder"]),
                     sum(r["invented"] for s in SPLITS for r in rec[(v.name, s)])])
        prev = t
    out += [table(["version", "ok (dev)", "ok (test)", "gain on test", "period", "nature", "name",
                   "folder", "invented dates blocked (all 150)"], rows), "",
            "Period, nature, name and folder columns are on the test split.", ""]

    out += ["## Rules versus language models", ""]
    rows = []
    for n in names:
        if n.startswith("rules-v") and n not in ("rules-v3",):
            continue
        d, t = summary(rec[(n, "dev")]), summary(rec[(n, "test")])
        allr = rec[(n, "dev")] + rec[(n, "test")]
        mean, p95 = latency(allr)
        asked = ""
        if n.startswith("hybrid"):
            k = sum(1 for r in allr if r["tokens_in"])
            asked = f"{k}/{len(allr)}"
        cost = "0 (CPU)" if n.startswith("rules") else ("0 (local GPU)" if not n.startswith("claude") else "")
        if n.startswith("claude"):
            c = ClaudeModel(n)
            cost = f"${sum(c.cost(r) for r in allr) / len(allr) * 1000:.2f} per 1000"
        rows.append([n, pct(d["ok"]), pct(t["ok"]), pct(t["period"]), pct(t["nature"]), pct(t["name"]),
                     pct(t["folder"]), sum(r["invented"] for r in allr), fmt_s(mean), fmt_s(p95),
                     asked or "-", cost])
    out += [table(["approach", "ok (dev)", "ok (test)", "period", "nature", "name", "folder",
                   "invented dates blocked", "mean latency", "p95 latency", "model calls", "API cost"],
                  rows), ""]
    if models:
        device = rec[(models[0].name, "dev")][0]["device"]
        out += [f"Local models ran with greedy decoding on `{device}` (GTX 1650, 4 GB, float16); "
                "latency is generation time per piece, model loading excluded. "
                "Rules latency is measured on one CPU core.", ""]
    if not any(n.startswith("claude") for n in names):
        out += ["Claude was **not** run for these results (no API key during development). "
                "`./run.sh --claude` adds its row; nothing else changes.", ""]

    if sweep:
        out += ["### Choosing when the hybrid asks the model (dev split only)", "",
                "The rules give a confidence between 0 and 1; below the threshold the model is asked "
                "about what the rules doubt. The threshold with the best dev score is then used once on test.",
                "The two test columns are shown for information only; they played no part in the choice.",
                "", table(["model", "threshold", "ok (dev)", "sent to the model (dev)",
                           "ok (test, not used to choose)", "sent to the model (test)"], sweep), ""]

    out += ["## Where each approach fails, by trap (dev + test)", "",
            "Share of pieces carrying the trap that end **ok**.", ""]
    traps = collections.Counter(t for s in SPLITS for p in data[s] for t in p["traps"])
    shown = ["rules-v0", "rules-v1", "rules-v2", "rules-v3"] + [n for n in names if not n.startswith("rules")]
    rows = []
    for trap, count in traps.most_common():
        row = [trap, count]
        for n in shown:
            rs = [r for s in SPLITS for r in rec[(n, s)] if trap in r["traps"]]
            row.append(pct(sum(r["ok"] for r in rs) / len(rs)))
        rows.append(row)
    out += [table(["trap", "pieces"] + shown, rows), ""]

    out += ["## Ten typical failures of each approach (dev + test)", "",
            "One example per kind of error and kind of piece, most frequent first. "
            "`n` is how many failures share that pattern.", ""]
    for n in ["rules-v3"] + [n for n in names if not n.startswith("rules")]:
        rs, total = typical_failures(rec[(n, "dev")] + rec[(n, "test")])
        out += [f"### {n} - {total} failures out of {len(data['dev']) + len(data['test'])}", ""]
        if rs:
            out += [table(["n", "what is wrong", "piece", "id", "expected", "proposed", "traps"], rs), ""]
        else:
            out += ["No failure.", ""]

    with open(os.path.join(ROOT, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("results.md written")


if __name__ == "__main__":
    sys.exit(main())
