# %% [markdown]
# # 02 · Markouts
#
# **Purpose.** How did the price move after each fill, relative to the price we traded at? Several reference
# prices, wall-clock and event-time horizons, before and after fees, decomposed into spread captured and
# adverse move, and conditioned on whatever you want to bucket by.
#
# **Convention.** d = +1 our buy, −1 our sell; markout_h = d × (m_{t+h} − p), in bps of p; **positive = good
# for us**. For h > 0 the reference is the first update strictly after t+h. For h ≤ 0 it is the last update
# strictly before t+h, so h = 0 is the book our fill hit. Horizons that cross a session closure (e.g. CME's
# daily halt) are NaN.
#
# **Tables.** fills, book_l1 (+ market_trades for event time; a lead venue in `config.LEAD_MAP`).
#
# **What to look at first.** The mid-reference curve for passive fills: where it crosses zero, and whether
# it is still falling at the longest horizon.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posttrade import _util, config, hygiene, markouts, plots, reference, store

plots.style()
T, META = store.load_tables()
fills, _ = hygiene.dedupe_fills(T["fills"])
book, trades = T["book_l1"], T["market_trades"]
HORIZONS = config.HORIZONS_S  # edit in config.py: log-spaced, includes negative horizons
print("horizons (s):", HORIZONS)

# %% [markdown]
# ## Which fills
# Passive quoting fills and spread-trade legs behave differently, so analyse them separately. Here: maker fills,
# aggregated per aggressor event (one taker sweeping several of our levels at one timestamp counts **once**).

# %%
maker = fills[fills["liquidity"] == "maker"]
events = markouts.aggregate_events(maker)
print(f"{len(maker):,} maker fills -> {len(events):,} aggressor events "
      f"({(events['n_fills'] > 1).mean():.1%} events had several fills)")
clusters = markouts.time_clusters(events, cluster_s=60)  # bootstrap unit: 60 s buckets share price paths

# %% [markdown]
# ## References: mid, weighted mid, Stoikov microprice, lead venue

# %%
ref_mid = reference.mid_ref(book)
ref_wmid = reference.to_ref(book, reference.weighted_mid(book))
refs = {"mid": ref_mid, "weighted mid": ref_wmid}
our_v, our_s = events.groupby(["venue", "symbol"]).size().idxmax()
our_book = book[(book["venue"] == our_v) & (book["symbol"] == our_s)]
tick = reference.tick_sizes(our_book, T["instruments"])[(our_v, our_s)]
mp_model = reference.fit_microprice(our_book, tick=tick, n_imb=10, n_spread=2)
refs["microprice"] = reference.to_ref(our_book, mp_model.apply(our_book))
lead = reference.lead_mid(book)
if len(lead):
    refs["lead mid"] = lead
else:
    print("lead mid UNAVAILABLE: set config.LEAD_MAP")
plots.heatmap(mp_model.table / tick, title="Microprice adjustment (ticks) by spread × imbalance", fmt="{:.2f}", cbar_label="ticks")
plt.show()

# %% [markdown]
# **How to read this.** Each cell is how far the microprice sits from the mid (in ticks) given the spread
# and queue imbalance I = bid_sz/(bid_sz+ask_sz). A monotone left-to-right gradient means imbalance predicts the
# next mid move. Bigger values in the wider-spread row mean the adjustment matters most when the spread is
# wide. A near-zero table means imbalance carries no information on this venue; then the microprice adds
# nothing over the mid. The model is fitted on this sample: for out-of-sample use, fit on the prior day.

# %% [markdown]
# ## Markout curves by reference

# %%
mo = {name: markouts.compute_markouts(events, r, HORIZONS) for name, r in refs.items()}
curves = {name: markouts.markout_curve(m, clusters=clusters, n_boot=300) for name, m in mo.items()}
plots.markout_curves(curves, title=f"Markouts of passive events on {our_v}:{our_s} (per event, 95% cluster-bootstrap CI)")
plt.show()
pd.concat({k: v["mean"] for k, v in curves.items()}, axis=1).round(2)

# %% [markdown]
# **How to read this.** At h = 0 the curve is the spread you captured (half-spread for touch fills). Moving
# right, it drops by the adverse move. Good market making keeps the curve above zero at the horizon where you
# would hedge or unwind. A curve still falling at 30 min means persistent informed flow. A drop that happens
# only at long horizons is often just drift and inventory, not adverse selection. Against the **lead mid** the
# drop happens earlier: if the lead is far below our mid at t = 0, you were filled on stale quotes, so look at
# 06 latency. Negative horizons show what the price did before the fill (see pre-fill drift in 03).

# %% [markdown]
# ## Decomposition: spread captured + adverse move

# %%
dec = markouts.decompose(events, ref_mid, HORIZONS)
adv_curve = markouts.markout_curve(dec["adverse"], clusters=clusters, n_boot=300)
plots.decomposition(dec["spread_captured"].mean(), adv_curve)
plt.show()

# %% [markdown]
# **How to read this.** The green line (spread captured) is what you earn if the price never moves. The
# orange curve is what the market takes back. The blue curve is their sum (the identity is asserted exactly
# in the code). If the orange curve's long-horizon level is larger in magnitude than the green line, you pay
# out more to informed flow than you earn in spread. Then either quote wider, or find which fills are toxic
# (conditional curves below, the toxicity model in 05).

# %% [markdown]
# ## Before / after fees, per event vs notional-weighted, vol-scaled

# %%
net = markouts.net_of_fees(mo["mid"], events)
vol = reference.trailing_vol(ref_mid, window_s=300)
vs = markouts.vol_scaled(mo["mid"], events, vol)
c2 = {"gross (per event)": curves["mid"],
      "gross (notional-weighted)": markouts.markout_curve(mo["mid"], weights=events["notional"].to_numpy(), clusters=clusters, n_boot=300),
      "after fees": markouts.markout_curve(net, clusters=clusters, n_boot=300)}
plots.markout_curves(c2, title="Gross vs after fees; per event vs notional-weighted")
plt.show()
plots.markout_curves({"vol-scaled": markouts.markout_curve(vs, clusters=clusters, n_boot=300)},
                     title="Vol-scaled markout: markout / (σ_trailing · √h)", ylabel="σ units")
plt.show()

# %% [markdown]
# **How to read this.** The gap between gross and after-fees is your fee or rebate (it shifts the whole curve).
# Notional-weighted below per-event means your **large** fills are worse, which is typical of size-aware
# informed takers. The vol-scaled curve makes days and instruments comparable. A value of −0.3 at a horizon
# means the adverse move was 0.3 standard deviations of the normal price move over that horizon. A
# consistently negative vol-scaled value at short h is a sign of being picked off, not of volatility.

# %% [markdown]
# ## Event time: next n trades and next n mid changes

# %%
if trades is not None:
    et = markouts.event_time_markouts(events, ref_mid, trades, config.EVENT_HORIZONS["trades"])
    em = markouts.event_time_markouts(events, ref_mid, markouts.mid_change_events(ref_mid), config.EVENT_HORIZONS["mid_changes"])
    plots.markout_curves({"next n trades": markouts.markout_curve(et, clusters=clusters, n_boot=200),
                          "next n mid changes": markouts.markout_curve(em, clusters=clusters, n_boot=200)},
                         title="Event-time markouts", xlabel="n events after the fill")
    plt.show()
else:
    print("UNAVAILABLE: event time needs market_trades")

# %% [markdown]
# **How to read this.** Event time removes the effect of activity: a fill in a busy minute and one in a quiet
# minute are compared after the same amount of "market news". If the wall-clock curve looks bad but the
# event-time curve is flat, the adverse moves are simply what happens when the market is busy. If adverse
# selection shows up within the first one or two mid changes, the takers knew where the next tick was going.

# %% [markdown]
# ## Taker view: effective / realised spread and impact
# effective = realised + impact (asserted), in bps of the pre-trade mid, from the taker's side.

# %%
if trades is not None:
    rows = {}
    for h in [5.0, 30.0, 60.0, 300.0]:
        ts_ = markouts.taker_spreads(trades[(trades["venue"] == our_v) & (trades["symbol"] == our_s)], ref_mid, h)
        rows[h] = ts_.mean()
    display(pd.DataFrame(rows).T.rename_axis("horizon_s").round(3))
ours = markouts.taker_spreads(events, ref_mid, 60.0)
print("our passive events, taker view at 60 s:\n", ours.mean().round(3))

# %% [markdown]
# Realised spread is what liquidity providers kept after h. Impact is the information in the trade. Your own
# passive fills' realised spread, compared with the venue-wide figure, says whether you are filled by better
# or worse takers than the average liquidity provider.

# %% [markdown]
# ## Conditional curves: bucket by any column

# %%
imb = _util.asof_values(events, book.assign(imb=reference.imbalance(book)), ["imb"], direction="backward", allow_exact=False)["imb"]
cond = pd.DataFrame({
    "signed imbalance": events["side"].to_numpy() * (imb.to_numpy() - 0.5),  # + = book leaned our way
    "trailing vol": _util.asof_values(events, vol, ["vol_bps"], direction="backward", allow_exact=False)["vol_bps"].to_numpy(),
    "size (qty)": events["qty"].to_numpy(),
})
if "lead mid" in mo:
    m0 = markouts.ref_at_fill(events, ref_mid)
    lm = _util.asof_values(events, lead, ["value"], direction="backward", allow_exact=False)["value"].to_numpy()
    cond["lead gap (signed bps)"] = 1e4 * events["side"].to_numpy() * (lm - m0) / m0
BUCKET = "lead gap (signed bps)" if "lead gap (signed bps)" in cond else "signed imbalance"  # <- pick any column
cc = markouts.conditional_curves(mo["mid"], markouts.quantile_buckets(cond[BUCKET], 5), clusters=clusters, n_boot=200)
plots.markout_curves({b: g.set_index("horizon_s") for b, g in cc.groupby("bucket")}, title=f"Markout by quintile of {BUCKET}")
plt.show()
plots.heatmap(cc.pivot(index="bucket", columns="horizon_s", values="mean"), title=f"Mean markout (bps): {BUCKET} quintile × horizon")
plt.show()

# %% [markdown]
# **How to read this.** Quintiles that fan out with horizon mean the variable predicts adverse selection.
# The heatmap shows the same thing: one row turning orange (bad) while the others stay blue is a filter
# candidate. The **lead gap** (lead mid minus our mid, signed by our side) is the classic one. Fills where the
# lead had already moved against us are the ones takers picked off. Overlapping CIs between adjacent quintiles
# mean no difference was detected, not that the effect is zero. Feed the variable into the regressions in 05
# to control for the others.

# %% [markdown]
# ## Large data: one day at a time
# `daily_markout_means` processes each UTC day separately (memory is bounded by the largest day). The day-level
# series is also the right input for CUSUM and day-clustered confidence intervals.

# %%
daily = markouts.daily_markout_means(maker, ref_mid, [0.0, 5.0, 60.0, 300.0], weight_col="notional")
display(daily.round(3))
print("combined:", markouts.combine_daily(daily).round(3).to_dict())
