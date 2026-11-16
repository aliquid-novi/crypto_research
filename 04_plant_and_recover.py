"""Change the market itself, then check whether your measurement finds what you planted.

This is the main way to practise research here: the generator knows the truth, so an estimator can be
graded. Plant an effect, measure it from the data alone (res.bbo, res.fills), compare with mk.truth.

    python examples/04_plant_and_recover.py
"""
# %%
try:                                   # lets the examples run without installing the package
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
except NameError:
    pass
import numpy as np
import pandas as pd

import spreadsim as ss

pd.set_option("display.width", 200, "display.max_columns", 30)

# %% every knob of the market is a dataclass field: see Flow, Leg, Basis, Regime, Factor in synth.py
base = ss.usd_spread()
slow_spot = ss.tweak(base, "SPOT", lag=1.0)                                   # SPOT now trails by ~1 s
toxic_spot = ss.tweak(base, "SPOT", flow__snipe_prob=0.95, flow__snipe_delay=0.01, flow__pull_delay=0.5)
thin_spot = ss.tweak(base, "SPOT", flow__target_depth=300.0)

def stats(market, seeds=range(3)):
    """A plain market maker on SPOT (it ignores PERP, so it feels SPOT's microstructure directly)."""
    rows = []
    for seed in seeds:
        mk = ss.simulate(market, duration=1200, seed=seed)
        res = mk.run(ss.SimpleMM("SPOT", half_spread_ticks=0.5, size=10, max_pos=60), latency=ss.ConstantLatency(0.002, 0.003, 0.003))
        s = ss.summary(res)
        rows.append({"fills": s["maker_fills"], "edge_at_fill_bps": s["maker_edge_bps"], "markout_5s_bps": s["maker_markout_5s_bps"],
                     "queue_ahead_on_joining": ss.maker_stats(res).queue_ahead_at_join.iloc[0], "cancel_rejects": s["cancel_rejects"]})
    return pd.DataFrame(rows).mean()


print(pd.DataFrame({name: stats(m) for name, m in
                    [("base", base), ("slow SPOT", slow_spot), ("toxic SPOT", toxic_spot), ("thin SPOT", thin_spot)]}).round(2))

# %% the truth you are trying to recover
mk = ss.simulate(slow_spot, duration=1800, seed=7)
res = mk.run(ss.Strategy())                       # a do-nothing strategy: just replay and record
t = mk.truth["t"]
print("\ntruth keys:", list(mk.truth))
print("planted SPOT lag: 1.0 s.   latent premium of PERP, bps: mean %.1f, sd %.1f"
      % (mk.truth["basis"]["PERP"].mean() * 1e4, mk.truth["basis"]["PERP"].std() * 1e4))

# %% observable data for your estimator: mids on a regular grid
grid = np.arange(60, 1790, 0.1)
mid = pd.DataFrame({nm: res.mid_at(nm, grid) for nm in mk.names}, index=grid)
print(mid.describe().round(3))
# YOUR TURN: from `mid` alone, estimate how far SPOT trails PERP, then compare with the planted 1.0 s.
# Then re-run with lag=0.25 and lag=3.0. Does your estimate move the right way, and by the right amount?
