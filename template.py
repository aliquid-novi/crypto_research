"""Template for testing one hypothesis about a spread trade. Copy, rename, fill in the four blocks.

    python experiments/template.py --quick

Before running anything, write three lines here:
    HYPOTHESIS   what you think is true, as a rule someone could trade on
    PREDICTION   which variant wins, in which worlds, and by roughly how much
    REFUTATION   the result that would make you drop the idea

The structure does the rest: every variant faces the same markets, differences come with standard
errors, and the worlds differ in one planted property each, so you can see where the idea holds.
"""
# %%
try:
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, ".."))
except NameError:
    HERE = "."
import argparse

import pandas as pd

import spreadsim as ss
from spreadsim import report as rp

pd.set_option("display.width", 220, "display.max_columns", 40)

# %% 1. WORLDS: the same market with one property changed each time. Include worlds where you expect
#    the idea to fail; a test that can only confirm is not a test.
WORLDS = {
    "base": ss.usd_spread(),
    "slow quoted leg": ss.tweak(ss.usd_spread(), "SPOT", lag=1.5),
    "heavy flow pressure": ss.tweak(ss.usd_spread(), "SPOT", flow__impact_ticks=0.5),
}


# %% 2. VARIANTS: the baseline first, then the idea. Each is a function returning a fresh strategy.
#    A new placement rule is a Controller (see spreadsim/controllers.py); anything else is a
#    Strategy subclass or a different argument to an existing one.
class MyRule(ss.Controller):
    """Example: stand one tick further out while the hedge leg's book is thin on the side we would hit."""

    def shift(self, sim, strat):
        b = sim.book(strat.hedge_leg)
        return 1 if min(b.bid_qty, b.ask_qty) < 5 else 0


VARIANTS = {
    "baseline": lambda: ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9)),
    "my rule": lambda: ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9), controller=MyRule()),
}


# %% 3. METRICS: anything beyond the standard summary you want per run. Keys starting with "_" may
#    hold whole objects (a fill table) and are kept in runs.extras instead of the results table.
def metrics(res, world, variant, seed):
    ft = ss.fill_table(res, "SPOT", hedge="PERP")
    return {"quote_fills": len(ft), "s5": ft.s5.mean(), "s120": ft.s120.mean(), "_fills": ft.assign(world=world, variant=variant)}


# %% 4. RUN AND READ
def main(seeds, minutes, workers, out):
    exp = ss.Experiment(WORLDS, VARIANTS, seeds=seeds, duration=minutes * 60, metrics=metrics,
                        sim_kw=lambda: {"latency": ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003)})
    runs = exp.run(workers=workers)
    pair = runs.paired("pnl", baseline="baseline")
    print("\n", pair.round(2).to_string())

    R = rp.Report("My hypothesis", f"{len(WORLDS)} worlds x {len(VARIANTS)} variants x {len(seeds)} seeds, {minutes:g} minutes each")
    R.heading("Result").table(pair, "PnL against the baseline on the same seeds. |t| under about 2 is noise.")
    d = pair.xs("my rule", level="variant")
    R.figure(rp.plot_bars(d, "diff", "diff_se", title="PnL change from the rule, by world", xlabel="paired PnL difference vs baseline"),
             "Bars: mean paired difference; whiskers: +/- one standard error.")
    R.heading("Why")
    fills = pd.concat([v["_fills"] for v in runs.extras.values()])
    R.table(ss.conditional(fills, ["world", "variant"], ["s5", "s120"]),
            "Spread markouts of passive fills in ticks: 5 seconds and 2 minutes after the fill.")
    R.table(runs.table.groupby(["world", "variant"])[["quote_fills", "spread_capture", "hedge_cost", "adverse_selection",
                                                      "inventory", "fees", "funding"]].mean(),
            "Where the PnL came from, averaged over seeds.")
    print("report:", R.save(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "template_report.html"))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.minutes = 2, 10.0
    main(list(range(a.seeds)), a.minutes, a.workers, a.out)
