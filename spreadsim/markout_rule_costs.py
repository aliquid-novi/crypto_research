"""Follow-up to markout_rule.py: the other half of the hypothesis.

"Negative markouts -> push out" says nothing about what to do when markouts are positive. This run
starts from a ladder that is too tight to cover its own hedging cost, in worlds with little or no
continuation after a fill. There the first clips' markouts are positive, and the question is whether
pushing out helps anyway, and whether judging markouts against the cost of a fill (instead of against
zero) fixes the rule.

    python experiments/markout_rule_costs.py            # about 3 minutes on 2 cores
    python experiments/markout_rule_costs.py --quick
"""
# %%
try:
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, HERE)
except NameError:
    HERE = "."
import argparse

import numpy as np
import pandas as pd

import markout_rule as mr
import spreadsim as ss
from spreadsim import report as rp

pd.set_option("display.width", 220, "display.max_columns", 40)
TIGHT = (1.0, 5.0)            # clips at 1 and 5 bps from fair value: inside the cost of hedging one fill
COST_TICKS = 1.65             # hedge crossing + fees per unit, in quoted-leg ticks (measured in markout_rule.py)
WORLDS = {k: mr.WORLDS[k] for k in ("no flow pressure", "mild pressure", "moderate, fades very fast", "slow quoted leg")}


def quoter(**kw):
    return ss.SpreadQuoter("SPOT", "PERP", levels_bps=TIGHT, clip=10, max_clips=4, **kw)


STATIC = {"shift 0 (baseline)": 0, "shift 1": 1, "shift 2": 2, "shift 3": 3}
VARIANTS = {name: (lambda k=k: quoter(shift_ticks=k)) for name, k in STATIC.items()}
VARIANTS["markout rule vs zero"] = lambda: quoter(controller=ss.MarkoutFeedback(horizon=5.0, by_vol=False))
VARIANTS["markout rule vs cost"] = lambda: quoter(controller=ss.MarkoutFeedback(horizon=5.0, by_vol=False, hurdle_ticks=COST_TICKS))


def main(seeds, minutes, workers, out):
    exp = ss.Experiment(WORLDS, VARIANTS, seeds=seeds, duration=minutes * 60, metrics=mr.metrics,
                        sim_kw=lambda: {"latency": ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003)})
    runs = exp.run(workers=workers)
    T, pair = runs.table, runs.paired("pnl", mr.BASE)
    base = T[T.variant == mr.BASE].groupby("world").mean(numeric_only=True).loc[runs.worlds]
    W = pd.DataFrame(index=runs.worlds)
    W["markout 5s"], W["cost per unit"], W["markout 5s net of cost"] = base.s5, base.cost_ticks, base.s5 - base.cost_ticks
    W["baseline pnl"] = runs.mean("pnl")[mr.BASE]
    for v in ("shift 1", "shift 2", "markout rule vs zero", "markout rule vs cost"):
        W[f"{v}: pnl change"], W[f"{v}: se"] = pair.xs(v, level="variant")["diff"], pair.xs(v, level="variant")["diff_se"]
    for v in ("markout rule vs zero", "markout rule vs cost"):
        W[f"{v}: avg shift"] = T[T.variant == v].groupby("world").avg_shift.mean()
    W["best static shift"] = [STATIC[v] for v in runs.best("pnl", list(STATIC))]
    print("\n", W.round(2).T.to_string())
    print("\n", pair.round(2).to_string())

    R = rp.Report("Markout rule, part 2: positive markouts on a ladder that is too tight",
                  f"{len(WORLDS)} worlds x {len(VARIANTS)} variants x {len(seeds)} seeds, {minutes:g} simulated minutes each. "
                  f"Baseline clips at {TIGHT[0]:g} and {TIGHT[1]:g} bps from fair value; markouts in ticks of 2 bps.")
    R.heading("Markouts are positive, and pushing out still helps")
    R.table(W, "Markouts: spread markout of the first two clips at the baseline, in ticks, before and after the per-unit cost of hedging and fees. "
               "PnL changes are paired differences against the baseline.")
    R.figure(rp.plot_value_curves(runs, STATIC, {"markout rule vs zero": "rule judging markouts against zero",
                                                 "markout rule vs cost": "rule judging markouts against cost"},
                                  xlabel="static shift, ticks", ylabel="PnL", ncols=4, title="PnL by how far the tight ladder is pushed out"),
             "Grey: static ladders, mean +/- one standard error. Blue dot: best static. Dashed: the two versions of the rule.")
    R.heading("Every variant against the baseline").table(pair)
    print("report:", R.save(out))
    W.to_csv(out.replace(".html", "_worlds.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "markout_rule_costs_report.html"))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.minutes = 2, 10.0
    main(list(range(a.seeds)), a.minutes, a.workers, a.out)
