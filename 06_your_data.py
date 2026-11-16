"""Replaying your own data. The simulator needs four tables; this shows their exact shape by exporting
a synthetic market to DataFrames and replaying it from them.

    python examples/06_your_data.py
"""
# %%
try:                                   # lets the examples run without installing the package
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
except NameError:
    pass
import pandas as pd

import spreadsim as ss
from spreadsim import DEPTH, FUNDING, INDEX, TRADE

mk = ss.simulate(ss.usd_spread(), duration=600, seed=1)
names = mk.names

# %% the four tables (only `depth` and `trades` are required)
ev = pd.DataFrame(mk.events, columns=["t", "kind", "i", "a", "b", "c"])
ev["inst"] = ev.i.map(dict(enumerate(names)))
tick = ev.inst.map({x.name: x.tick for x in mk.instruments})
d, tr = ev[ev.kind == DEPTH], ev[ev.kind == TRADE]
depth = pd.DataFrame({"t": d.t, "inst": d.inst, "side": d.a.astype(int), "price": d.b * tick[d.index], "qty": d.c})
trades = pd.DataFrame({"t": tr.t, "inst": tr.inst, "side": tr.a.astype(int), "price": tr.b * tick[tr.index], "qty": tr.c})
index = ev[ev.kind == INDEX][["t", "inst", "a"]].rename(columns={"a": "price"})
funding = ev[ev.kind == FUNDING][["t", "inst", "a"]].rename(columns={"a": "rate"})
for name, df in [("depth", depth), ("trades", trades), ("index", index), ("funding", funding)]:
    print(f"\n{name}: {len(df):,} rows"); print(df.head(3).to_string(index=False))

# %% describe your instruments, build the event stream, run
instruments = [
    ss.Instrument("SPOT", tick=0.05, lot=1.0, maker_fee=0.0, taker_fee=1.0e-4),
    ss.Instrument("PERP", tick=0.01, lot=1.0, maker_fee=0.2e-4, taker_fee=2.0e-4, kind="perp"),
]
events = ss.events_from_frames(instruments, depth=depth, trades=trades, index=index, funding=funding)
sim = ss.Sim(
    instruments, events, ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9)),
    latency=ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003),
    funding={"PERP": ss.FundingCalc(interval=1800, sample=5, weights="equal", impact_notional=5_000, multiplier=8.0)},
    funding_from="events",          # funding prints are in the data; use "model" to let the calculator settle
    # index_fn={"PERP": ss.mid_index("SPOT")},   # if you have no index feed: use the other leg's mid
)
res = sim.run()
print("\nreplayed from frames: pnl %.4f, fills %d" % (res.pnl, len(res.fills)))
direct = mk.run(ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9)), latency=ss.ConstantLatency(0.002, 0.003, 0.003))
print("replayed directly:    pnl %.4f, fills %d" % (direct.pnl, len(direct.fills)))
# Same quantities and PnL. The fill count can differ by a few because the loader puts same-timestamp
# trades ahead of same-timestamp depth updates, which can split one fill into two.

# Notes for real data:
#  * `t` in seconds. Subtract the session start if you keep epoch timestamps, to preserve precision.
#  * Start `depth` with a full snapshot of each book (one row per visible level).
#  * Only have periodic top-N snapshots? ss.snapshots_to_frame(...) diffs them into this `depth` shape.
#  * With real data there is no latent truth: analytics that say vs="ref" are unavailable, the rest work.
