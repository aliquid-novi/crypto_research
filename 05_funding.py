"""Funding in the simulator: the running prediction, the settlement, and its effect on fair value.

    python examples/05_funding.py
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


class WatchFunding(ss.Strategy):
    """Logs what a strategy can know about funding as the interval goes on."""

    def on_start(self, sim):
        sim.timer(60.0)

    def on_timer(self, sim, tag):
        sim.timer(60.0)
        c = sim.funding("PERP")                           # a FundingCalc tracking the venue's computation
        locked, remaining = c.locked()
        sim.log("predicted_bps", c.predicted() * 1e4)     # the rate if the average premium stayed where it is
        sim.log("avg_premium_bps", c.avg() * 1e4)
        sim.log("weight_remaining", remaining)            # share of the final average not yet determined
        sim.log("secs_to_settle", c.time_to_settle(sim.now))

    def on_funding(self, sim, inst, rate, payment):
        sim.log("settled_bps", rate * 1e4)


# %% 1. watch the prediction converge (or not) on the settled rate.
# In this seed the second interval predicts about twice the baseline rate, then settles at baseline.
mk = ss.simulate(ss.usd_spread(funding_interval=1800, passthrough=0.5), duration=3600, seed=1)
res = mk.run(WatchFunding())
view = pd.DataFrame({k: v.set_index("t").value for k, v in res.logs.items() if k != "settled_bps"})
print(view.iloc[::3].round(3))
print("\nsettled:", res.logs["settled_bps"].round(3).to_dict("records"), " baseline rate, bps:", round(mk.funding()["PERP"].baseline() * 1e4, 3))
print("In this toy market the perp also dips before each settlement and rebounds after (passthrough=0.5):")
t, sh = mk.truth["t"], mk.truth["shift"]["PERP"] * 1e4
print("  latent dip in bps at 20, 25, 29 and 31 minutes:", [round(float(sh[int(m * 60 / mk.market.dt)]), 3) for m in (20, 25, 29, 31)])

# %% 2. does a funding adjustment to fair value help? no adjustment vs the naive one, over seeds
SEEDS = range(4)
markets = {s: ss.simulate(ss.usd_spread(), duration=3600, seed=s) for s in SEEDS}


def run(seed, adj="none"):
    f = None if adj == "none" else ss.naive_funding_adj("PERP", perp_is="hedge")
    strat = ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9), funding_adj=f)
    return markets[seed].run(strat, latency=ss.ConstantLatency(0.002, 0.003, 0.003))


cols = ["pnl", "pnl_se", "maker_fills", "spread_capture", "hedge_cost", "adverse_selection", "inventory", "fees", "funding", "funding_se"]
print("\n", ss.sweep(run, {"adj": ["none", "naive"]}, seeds=SEEDS, verbose=False)[cols].round(2))
# funding_adj is any function f(sim, strategy) -> log shift of fair value. Write a better one.
# ss.expected_rate(calc, mu, sigma) prices the kinked payoff if you can supply a mean and a standard
# deviation for the final average premium; calc.locked() gives the part that is already fixed.
