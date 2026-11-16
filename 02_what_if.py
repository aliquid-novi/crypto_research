"""The A/B pattern: change one thing, hold the market fixed, repeat over seeds, read the standard errors.

    python examples/02_what_if.py
"""
# %%
try:                                   # lets the examples run without installing the package
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
except NameError:
    pass
import pandas as pd

import spreadsim as ss

pd.set_option("display.width", 200, "display.max_columns", 30)
SEEDS = range(4)
markets = {s: ss.simulate(ss.usd_spread(), duration=2400, seed=s) for s in SEEDS}     # generate once, reuse


def run(seed, lat=0.003, queue="prob", levels=(4, 9)):
    q = {"risk": ss.RiskAverseQueue(), "prob": ss.ProbQueue(1.0), "opt": ss.OptimisticQueue()}[queue]
    strat = ss.SpreadQuoter("SPOT", "PERP", levels_bps=levels, clip=10, max_clips=4)
    return markets[seed].run(strat, latency=ss.ConstantLatency(feed=lat, entry=lat, response=lat), queue=q)


show = ["pnl", "pnl_se", "maker_fills", "spread_capture", "hedge_cost", "adverse_selection", "fees",
        "maker_markout_5s_bps", "taker_slippage_bps", "cancel_rejects"]

# %% 1. latency, same strategy, same four markets
print("\n=== latency (seconds on every hop)")
print(ss.sweep(run, {"lat": [0.0, 0.003, 0.03, 0.3]}, seeds=SEEDS, verbose=False)[show].round(2))

# %% 2. the queue assumption: the two bounds and the middle
print("\n=== queue model")
print(ss.sweep(run, {"queue": ["risk", "prob", "opt"]}, seeds=SEEDS, verbose=False)[show].round(2))

# %% 3. where the clips sit
print("\n=== clip distances from fair value, bps")
print(ss.sweep(run, {"levels": [(2, 6), (4, 9), (8, 14), (12, 20)]}, seeds=SEEDS, verbose=False)[show].round(2))

# A difference smaller than about two standard errors (the _se columns) is noise: add seeds or lengthen
# the run before believing it. Absolute PnL in a toy market means nothing; differences between
# configurations facing the same markets are what to read.
