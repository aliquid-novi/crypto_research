# %% [markdown]
# # 03 · Adverse selection
#
# **Purpose.** Who is picking you off, and when? This notebook covers the adverse-selection ratio over all
# resting orders, pre-fill drift, late cancels, full vs partial fills, concentration of losses by counterparty,
# and venue-wide toxicity scores per taker address.
#
# **Tables.** orders + book_l1 (AS ratio, late cancels, full/partial), fills.counterparty (concentration),
# market_trades with buyer/seller (venue-wide toxicity; Hyperliquid publishes addresses, most venues don't).
#
# **What to look at first.** The AS ratio, then the counterparty Lorenz curve.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posttrade import adverse, hygiene, markouts, plots, reference, store

plots.style()
T, META = store.load_tables()
fills, _ = hygiene.dedupe_fills(T["fills"])
book, trades, orders = T["book_l1"], T["market_trades"], T["orders"]
ref = reference.mid_ref(book)
maker = fills[fills["liquidity"] == "maker"]
H = 60.0  # <- the horizon you care about for "toxic" (seconds)

# %% [markdown]
# ## Adverse-selection ratio
# P(fill | next move against) / P(fill | next move in favour), over **all** resting orders (filled and cancelled).

# %%
if orders is not None:
    passive_orders = orders[orders["order_id"].isin(maker["order_id"]) | (orders["final_status"] == "cancelled")]
    for h in [None, 5.0, 30.0]:
        r = adverse.adverse_selection_ratio(passive_orders, ref, horizon_s=h)
        print(f"move = {'next mid change' if h is None else f'{h:g}s after done'}: AS ratio {r['ratio']:.2f}")
    display(r["table"])
else:
    print("UNAVAILABLE: needs the orders table")

# %% [markdown]
# **How to read this.** A ratio of 1 means fills are unrelated to what happens next. A ratio above 1 means
# you are filled preferentially just before the price moves against you: the classic signature of being
# picked off. Read it next to the markout curve in 02. A ratio near 1 with a negative markout points to drift
# or inventory rather than to informed takers. Cancelled orders define the baseline, so if you cancel
# reactively (after the move) the ratio understates the problem.

# %% [markdown]
# ## Pre-fill drift

# %%
drift = adverse.pre_fill_drift(maker, ref)
clusters = markouts.time_clusters(maker, 60)
plots.markout_curves({"pre-fill drift": markouts.markout_curve(drift, clusters=clusters, n_boot=300)},
                     title="Pre-fill drift: d·(m_t − m_{t+h}) for h < 0", xlabel="seconds before the fill")
plt.show()

# %% [markdown]
# **How to read this.** Positive values mean the price was already moving in your direction before you were
# filled: you bought into a rise and were hit by momentum traders. Negative means the price had moved
# against you and the taker hit your stale quote (latency arbitrage). Either way a non-zero curve means
# fills are predictable from public price history, so a feature built from it belongs in the toxicity model (05).

# %% [markdown]
# ## Late cancels

# %%
if orders is not None:
    lc = adverse.late_cancels(maker, orders)
    mo = markouts.compute_markouts(lc, ref, [5.0, 60.0, 300.0])
    print(f"{lc['late_cancel'].sum()} of {len(lc):,} maker fills ({lc['late_cancel'].mean():.2%}) had a cancel in flight")
    display(mo.groupby(lc["late_cancel"].to_numpy()).mean().rename(index={True: "late cancel", False: "other"}).round(2))
    plots.hist(lc.loc[lc["late_cancel"], "cancel_lead_ms"], bins=40, title="How long before the fill the cancel was sent",
               xlabel="ms", vline=None)
    plt.show()

# %% [markdown]
# **How to read this.** Late cancels are fills you tried to avoid and lost the race on. If their markouts are
# much worse than other fills (as they should be), the race matters: the histogram shows how much faster you
# would need to be. Most mass within your round-trip latency says shaving latency (06) converts these into
# avoided fills.

# %% [markdown]
# ## Full vs partial fills

# %%
if orders is not None:
    comp = adverse.fill_completeness(maker, orders)
    mo_all = markouts.compute_markouts(maker, ref, [1.0, 5.0, 30.0, 60.0, 300.0])
    plots.markout_curves({k: markouts.markout_curve(mo_all[comp.to_numpy() == k], clusters=clusters[comp.to_numpy() == k], n_boot=200)
                          for k in ["full", "partial"] if (comp == k).any()}, title="Markouts: fully vs partially filled orders")
    plt.show()

# %% [markdown]
# **How to read this.** Takers who take **all** of your size are often the informed ones: they wanted
# more than you showed. A full-fill curve well below the partial-fill curve suggests showing less size, or
# size-dependent quoting. If partial fills are worse instead, look at queue position: partial fills at the back of a
# level that just traded through are the dangerous ones.

# %% [markdown]
# ## Counterparty concentration

# %%
if "counterparty" in maker.columns and maker["counterparty"].notna().any():
    mo_h = markouts.compute_markouts(maker, ref, [H])[H].to_numpy()
    cc = adverse.counterparty_concentration(maker, mo_h)
    display(cc.head(15).round(3))
    plots.lorenz(np.clip(-cc["pnl"].to_numpy(), 0, None), title=f"Concentration of {H:g}s markout losses by counterparty",
                 ylabel="share of net losses")
    plt.show()
else:
    print("UNAVAILABLE: fills have no counterparty column")

# %% [markdown]
# **How to read this.** The table ranks counterparties by total PnL against you at the chosen horizon. A
# Lorenz curve that rises steeply means a handful of counterparties account for most of your losses. That
# finding is actionable: score them (below), avoid them, or skew against them. A curve near the diagonal
# means losses are spread evenly, so the problem is your quotes, not a particular counterparty. Rank by
# **total** loss, but check `n`: an address with three big losses is probably luck.

# %% [markdown]
# ## Venue-wide taker toxicity (needs buyer/seller in the public tape)

# %%
if trades is not None and {"buyer", "seller"} <= set(trades.columns) and trades["buyer"].notna().any():
    v, s = maker.groupby(["venue", "symbol"]).size().idxmax()
    tape = trades[(trades["venue"] == v) & (trades["symbol"] == s)]
    tox = adverse.taker_toxicity(tape, ref, horizon_s=30.0)
    print(f"shrinkage prior n = {tox.attrs['prior_n']:.1f}, grand mean = {tox.attrs['grand_mean_bps']:.2f} bps")
    top = tox.head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top))[::-1]
    ax.errorbar(top["mean_bps"], y, xerr=1.96 * top["se_bps"], fmt="o", color=plots.MUTED, ms=3, label="raw mean ± 1.96 se")
    ax.scatter(top["score_bps"], y, color=plots.SERIES[0], zorder=3, label="shrunk score")
    ax.set_yticks(y, [f"{a} (n={n})" for a, n in zip(top.index, top["n"])])
    ax.axvline(tox.attrs["grand_mean_bps"], color=plots.BASE, lw=1)
    ax.set(title="Most toxic takers on the venue (30 s taker markout)", xlabel="bps (positive = taker gained)")
    ax.legend()
    plt.show()
    if META.get("truth"):
        print("synthetic truth, toxic addresses:", META["truth"]["toxic_addresses"])
else:
    print("UNAVAILABLE: market_trades has no buyer/seller addresses on this venue")

# %% [markdown]
# **How to read this.** Each address's score is its mean taker markout shrunk toward the venue average.
# Addresses with few trades are pulled in, so a lucky small trader cannot top the list. A score far right
# of the grand mean with a tight raw CI is a reliably informed taker. Cross-check with the counterparty table
# above: the same addresses appearing in both is a strong finding. The scores use the whole sample, so to
# **act** on them, score on past days and test on later days.
