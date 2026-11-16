"""Hypothesis: "if the markouts of our first clips are consistently negative in the current volatility
state, push the limit orders a tick or two further out."

Tested the only way a rule like this can be tested: in worlds where the right answer is known and differs.

    python experiments/markout_rule.py                  # full run: 10 seeds x 60 minutes, about 10 minutes on 2 cores
    python experiments/markout_rule.py --quick          # 2 seeds x 15 minutes, to check it runs

Design
  worlds     the same two-leg market with one thing changed each time (how hard order flow pushes the
             quoted leg, how fast that decays, fair-value breaks, a slower quoted leg, ...)
  variants   static ladders pushed out 0, 1, 2, 3 ticks; the markout rule reading 5-second markouts (with
             and without a separate shift per volatility state) and 60-second markouts; and a rule that
             only looks at volatility
  seeds      every variant runs on exactly the same markets, so differences are paired
  question   does the SIGN of the first clips' markout at the baseline ladder tell you whether pushing
             out helps? And does the rule end up better than the ladder it started from?

Writes markout_rule_report.html next to this file and prints the main tables.
"""
# %%
try:                                   # lets the script run without installing the package
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, ".."))
except NameError:
    HERE = "."
import argparse

import numpy as np
import pandas as pd

import spreadsim as ss
from spreadsim import report as rp

pd.set_option("display.width", 220, "display.max_columns", 40)
TICK = 0.02                                   # quoted leg: 2 bps tick at a price of 100
LEVELS = (5.0, 9.0)                           # baseline clips, bps from fair value
HORIZONS = (1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0)
FIRST_CLIPS = (0, 1)


# %% worlds: one knob changed at a time
def world(impact=0.3, half_life=20.0, lag=0.25, jumps=False, **flow):
    """SPOT (quoted, 2 bps tick) against PERP (hedge). `impact`: ticks the SPOT price is pushed by one
    average market order; `half_life`: seconds for that push to fade."""
    m = ss.usd_spread()
    spot = ss.Instrument("SPOT", tick=TICK, lot=1.0, maker_fee=0.0, taker_fee=1.0e-4)
    m = ss.tweak(m, "SPOT", inst=spot, lag=lag, flow__impact_ticks=impact, flow__impact_half_life=half_life,
                 flow__depth_ticks=2.0, **flow)
    if jumps:                                 # SPOT-only jumps: the spread's fair value itself breaks
        from dataclasses import replace
        m = replace(ss.tweak(m, "SPOT", factors=("U", "J")),
                    factors=m.factors + [ss.Factor("J", start=1.0, vol=0.0, jump_rate=1 / 150.0, jump_sd=8e-4)])
    return m


WORLDS = {
    "no flow pressure": world(impact=0.0),
    "mild pressure": world(impact=0.3),
    "strong pressure": world(impact=0.8),
    "very strong pressure": world(impact=1.4),
    "strong, fades fast": world(impact=1.0, half_life=4.0),
    "moderate, fades very fast": world(impact=0.7, half_life=2.5),
    "strong, fades slowly": world(impact=0.8, half_life=90.0),
    "fair-value breaks": world(impact=0.3, jumps=True),
    "slow quoted leg": world(impact=0.3, lag=1.5),
}


# %% variants: all start from the same baseline ladder
def quoter(**kw):
    return ss.SpreadQuoter("SPOT", "PERP", levels_bps=LEVELS, clip=10, max_clips=4, **kw)


STATIC = {"shift 0 (baseline)": 0, "shift 1": 1, "shift 2": 2, "shift 3": 3}
VARIANTS = {name: (lambda k=k: quoter(shift_ticks=k)) for name, k in STATIC.items()}
VARIANTS["markout rule, 5s"] = lambda: quoter(controller=ss.MarkoutFeedback(horizon=5.0, clips=FIRST_CLIPS))
VARIANTS["markout rule, 5s, one state"] = lambda: quoter(controller=ss.MarkoutFeedback(horizon=5.0, clips=FIRST_CLIPS, by_vol=False))
VARIANTS["markout rule, 60s"] = lambda: quoter(controller=ss.MarkoutFeedback(horizon=60.0, clips=FIRST_CLIPS))
VARIANTS["volatility only"] = lambda: quoter(controller=ss.VolShift(high=1, low=0))
BASE = "shift 0 (baseline)"


# %% what to record from each run
def metrics(res, world_name, variant, seed):
    out = {"quote_fills": int((res.fills.tag == "quote").sum())}
    s = res.logs.get("shift")
    if s is not None and len(s) > 1:
        out["avg_shift"] = float(np.average(s.value.values[:-1], weights=np.diff(s.t.values)))
    if variant != BASE:
        return out
    ft = ss.fill_table(res, "SPOT", hedge="PERP", horizons=HORIZONS, window=20.0)
    first = ft[ft.k.isin(FIRST_CLIPS)]
    for h in HORIZONS:
        out[f"s{h:g}"] = first[f"s{h:g}"].mean()          # spread markout of the first clips, ticks
    out["m5_raw"] = first["m5"].mean()
    for state, name in ((0, "low"), (1, "high")):
        out[f"s5_{name}_vol"] = first[first.vol_state == state]["s5"].mean()
    att = ss.pnl_attribution(res).loc["TOTAL"]
    qty = ft.qty.sum()
    cost = -(att.hedge_cost + att.fees) / (qty * TICK) if qty else np.nan      # hedge + fees per unit, in ticks
    out["cost_ticks"] = cost
    for rule in ("through", "touch"):
        cf = ss.shift_counterfactual(ft, shifts=(1, 2), value="s120", short="s5", tick_value=TICK, rule=rule, cost_ticks=cost)
        for k in (1, 2):
            out[f"cf{k}_{rule}"] = cf.loc[k, "net_money"]
        if rule == "through":
            out["share_kept_1"] = cf.loc[1, "share_kept"]
    out["_fills"] = ft.assign(world=world_name, seed=seed)                      # kept aside, not in the table
    return out


def main(seeds, duration, workers, out_path):
    exp = ss.Experiment(
        WORLDS, VARIANTS, seeds=seeds, duration=duration, metrics=metrics,
        sim_kw=lambda: {"latency": ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003)},
        keep=lambda w, v, s: v == "markout rule, 5s" and s == seeds[0] and w == "strong pressure")
    runs = exp.run(workers=workers)
    T = runs.table
    base = T[T.variant == BASE].groupby("world").mean(numeric_only=True).loc[runs.worlds]
    pair = runs.paired("pnl", BASE)
    n = len(seeds)

    # ---- one row per world: the signal, and what acting on it did
    W = pd.DataFrame(index=runs.worlds)
    W["markout 1s"], W["markout 5s"], W["markout 60s"] = base.s1, base.s5, base.s60
    for k in (1, 2):
        W[f"push {k}: pnl change"] = pair.xs(f"shift {k}", level="variant")["diff"]
        W[f"push {k}: se"] = pair.xs(f"shift {k}", level="variant")["diff_se"]
    W["best static shift"] = [STATIC[v] for v in runs.best("pnl", list(STATIC))]
    W["predicted, push 1"] = base.cf1_through
    for v in ("markout rule, 5s", "markout rule, 5s, one state", "markout rule, 60s", "volatility only"):
        W[f"{v}: pnl change"] = pair.xs(v, level="variant")["diff"]
        W[f"{v}: se"] = pair.xs(v, level="variant")["diff_se"]
        W[f"{v}: avg shift"] = T[T.variant == v].groupby("world").avg_shift.mean()
    best_static_pnl = runs.mean("pnl")[list(STATIC)].max(axis=1)
    W["rule 5s: gap to best static"] = runs.mean("pnl")["markout rule, 5s"] - best_static_pnl

    # Does the sign of a signal call the right move? Scored only where the truth is clear:
    # worlds in which pushing out one tick changed PnL by at least two standard errors.
    t1 = W["push 1: pnl change"] / W["push 1: se"]
    clear, helped = (t1.abs() >= 2).values, np.sign(W["push 1: pnl change"]).values
    sig = {f"markout {h:g}s": -np.sign(base[f"s{h:g}"].values) for h in HORIZONS}        # negative markout -> push out
    sig["keep/lose counterfactual"] = np.sign(base.cf1_through.values)
    calls = pd.DataFrame({k: {"right": int((v[clear] == helped[clear]).sum()), "wrong": int((v[clear] != helped[clear]).sum())}
                          for k, v in sig.items()}).T
    calls["clear worlds"] = int(clear.sum())
    num = {w: i + 1 for i, w in enumerate(runs.worlds)}
    Wn = W.rename(index={w: f"{i}. {w}" for w, i in num.items()})

    print("\n=== per world\n", W.round(2).T.to_string())
    W.to_csv(out_path.replace(".html", "_worlds.csv"))
    runs.table.to_csv(out_path.replace(".html", "_runs.csv"), index=False)
    print("\n=== does the sign of a signal call the +1 tick move correctly? (worlds with a clear effect only)\n", calls)
    print("\n=== every variant against the baseline, same seeds\n", pair.round(2).to_string())

    # ---- report
    R = rp.Report("Markout rule for clip placement: does it work?",
                  f"{len(runs.worlds)} worlds x {len(VARIANTS)} variants x {n} seeds, {duration / 60:.0f} simulated minutes each. "
                  f"Quoted leg tick {TICK / 100 * 1e4:.0f} bps, baseline clips at {LEVELS[0]:g} and {LEVELS[1]:g} bps from fair value. "
                  "PnL in quote currency; markouts are spread markouts (hedge leg's move removed) of the first two clips, in ticks.")
    R.heading("1. Does a negative markout mean 'push out'?")
    R.text("Each point is one world. Left: the average 5-second markout of the first clips at the baseline ladder, against "
           "what pushing every clip one tick further out actually did to PnL on the same markets. If the hypothesis held, "
           "points would sit top-left (negative markout, pushing out helped) or bottom-right. Right: the same outcome "
           "against the keep/lose counterfactual computed from the baseline run's own fills.",
           "The counterfactual sees only the baseline's fills. It cannot see the fills a different ladder would take instead, "
           "or what freed-up position capacity is worth, so read it as a description and the re-run as the test.")
    plt = rp._plt()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    d = W.reset_index().rename(columns={"index": "world"})
    d["n"] = d.world.map(num)
    rp.plot_signal_vs_effect(d, "markout 5s", "push 1: pnl change", "push 1: se", "n",
                             xlabel="average 5s markout of first clips at baseline, ticks",
                             ylabel="PnL change from pushing out 1 tick", agree="opposite", ax=ax[0],
                             title="Markout sign vs the effect of pushing out")
    rp.plot_signal_vs_effect(d, "predicted, push 1", "push 1: pnl change", "push 1: se", "n",
                             xlabel="PnL change predicted from baseline fills (keep/lose)",
                             ylabel="PnL change from pushing out 1 tick", agree="same", ax=ax[1],
                             title="Keep/lose counterfactual vs the effect")
    lim = [min(ax[1].get_xlim()[0], ax[1].get_ylim()[0]), max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])]
    ax[1].plot(lim, lim, color="#bdbcb6", lw=1, ls=(0, (4, 2)), zorder=0)
    fig.tight_layout()
    R.figure(fig, "Numbers are the worlds in the table below. Error bars: +/- one standard error of the paired difference across seeds. "
                  "Grey points: the outcome is within two standard errors of zero. Dashed line on the right: prediction = outcome.")
    R.table(Wn[["markout 1s", "markout 5s", "markout 60s", "push 1: pnl change", "push 1: se", "push 2: pnl change", "push 2: se",
                "best static shift", "predicted, push 1"]],
            "Markouts: average spread markout of the first two clips at the baseline ladder, in ticks. PnL changes: paired differences against the baseline.")
    R.table(calls, "The call 'push out when the signal says so' (markout negative, or predicted change positive), scored against the realised "
                   "sign in the worlds where pushing out one tick moved PnL by at least two standard errors.", fmt="{:.0f}")

    R.heading("2. Where is the best ladder, and where does the rule end up?")
    fig = rp.plot_value_curves(runs, STATIC, {"markout rule, 5s": "markout rule, 5s", "markout rule, 60s": "markout rule, 60s",
                                              "volatility only": "volatility only"}, ncols=3,
                               xlabel="static shift, ticks", ylabel="PnL", title="PnL by how far the ladder is pushed out")
    R.figure(fig, "Grey: static ladders, mean +/- one standard error across seeds. Blue dot: best static. Dashed lines: the adaptive rules.")
    rules = ["markout rule, 5s", "markout rule, 5s, one state", "markout rule, 60s", "volatility only"]
    R.table(Wn[["best static shift"] + [f"{v}: {c}" for v in rules for c in ("pnl change", "se", "avg shift")] + ["rule 5s: gap to best static"]],
            "Each rule against the baseline ladder it starts from (paired PnL difference, its standard error, and the average ticks it "
            "pushed out). Last column: the 5-second rule's PnL minus the best static ladder's.")

    R.heading("3. Why: which fills does pushing out keep?")
    R.text("Pushing a clip out does not remove a random sample of fills. It keeps the fills the market went on to trade "
           "through and drops the ones where it turned around first. The panels split the baseline's first-clip fills that way.")
    fills = pd.concat([v["_fills"] for v in runs.extras.values()])
    show = ["no flow pressure", "strong pressure", "fair-value breaks"]
    fig, ax = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.4))
    for a, w in zip(ax, show):
        rp.plot_markout_split(fills[(fills.world == w) & fills.k.isin(FIRST_CLIPS)], k=1, prefix="s", ax=a, title=w)
    fig.tight_layout()
    R.figure(fig, "Spread markouts of first-clip fills at the baseline ladder, pooled over seeds; bars are +/- one standard error.")
    for w in show:
        ft = fills[fills.world == w]
        R.table(ss.shift_counterfactual(ft, value="s120", short="s5", tick_value=TICK, cost_ticks=base.loc[w, "cost_ticks"]),
                f"{w}: keep/lose arithmetic per baseline fill, in ticks: net = gain_kept - value_lost (net_money in quote currency, summed over all seeds). "
                f"Hedge and fees cost {base.loc[w, 'cost_ticks']:.2f} ticks per unit filled.")

    R.heading("4. The markouts by volatility state and clip")
    w = "strong pressure"
    R.table(ss.conditional(fills[fills.world == w].dropna(subset=["k"]).astype({"k": int}), ["vol_state", "k"], ["s5", "s120"]),
            f"{w}: spread markouts in ticks at the baseline ladder. vol_state 1 = spread volatility above its median; k = clip (0 nearest).")

    R.heading("5. The rule at work")
    key = next(iter(runs.results), None)
    if key is not None:
        R.figure(rp.plot_controller(runs.results[key], t0=0, t1=min(duration, 2400)),
                 f"{key[0]}, seed {key[2]}: the 5-second markout rule. The shift shown is the one for the volatility state at that moment.")
    R.heading("Every variant against the baseline")
    R.table(pair, "diff = mean paired PnL difference vs the baseline; t = diff / diff_se; wins = seeds in which the variant beat the baseline.")
    path = R.save(out_path)
    print("\nreport written to", path)
    return runs, W


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "markout_rule_report.html"))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.minutes = 2, 15.0
    main(list(range(a.seeds)), a.minutes * 60.0, a.workers, a.out)
