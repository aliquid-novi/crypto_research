"""Run a baseline spread quoter on a synthetic SPOT-vs-PERP market and read every report.

    python examples/01_quickstart.py
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

# %% one hour of market: PERP leads, SPOT has a large tick, PERP pays funding every 30 minutes
mk = ss.simulate(ss.usd_spread(), duration=3600, seed=1)
print(f"{len(mk.events):,} events on {mk.names}")

# %% quote SPOT passively around a PERP-implied fair value; hedge each fill by taking PERP
strategy = ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9), clip=10, max_clips=4)
latency = ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003)       # seconds
res = mk.run(strategy, latency=latency)

# %% headline numbers
print("\n--- summary");            print(pd.Series(ss.summary(res)).round(3).to_string())
print("\n--- where the PnL came from (base currency)"); print(ss.pnl_attribution(res).round(2))
print("\n--- markouts of passive fills, bps (positive = good)"); print(ss.markout_curve(res, maker=True).round(2))
print("\n--- hedge orders: cost in bps"); print(ss.slippage(res, tag="hedge")[["delay_ms", "delay_cost", "cross_cost", "total"]].describe().round(2))
print("\n--- each hedged fill as a pair"); print(ss.spread_pairs(res)[["quote_edge_bps", "hedge_cost_bps", "hedge_delay_ms", "net_bps"]].describe().round(2))
print("\n--- passive order stats"); print(ss.maker_stats(res).round(2).T)
print("\n--- funding settlements"); print(res.funding)
print("\n--- strategy fair value vs the latent truth"); print(ss.fair_value_error(res, "fair", "SPOT"))

# %% raw records are plain DataFrames: res.fills, res.orders, res.bbo["SPOT"], res.equity, res.logs["fair"]
print("\n", res.fills.head(3).T)

# %% pictures (needs matplotlib)
try:
    from spreadsim.plots import plot_markouts, plot_run
    plot_run(res, "SPOT", t0=600, t1=1500).savefig("quickstart_run.png", dpi=130)
    plot_markouts({"passive SPOT fills": res}, maker=True).savefig("quickstart_markouts.png", dpi=130)
    print("\nsaved quickstart_run.png and quickstart_markouts.png")
except ImportError:
    pass
