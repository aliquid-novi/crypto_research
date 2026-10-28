# %% [markdown]
# # 01 · Data hygiene
#
# **Purpose.** Before trusting any markout, check that the data is complete and consistent: positions
# reconcile, fees and funding recompute, no duplicate fills, no bad ticks, sane books, no unexplained feed
# gaps, known noise level, synchronised clocks and known staleness.
#
# **Tables.** fills (+ positions, funding, market_trades, book_l1, instruments when available).
#
# **What to look at first.** The position reconciliation (a persistent break means missing or duplicated
# fills, which corrupts everything downstream), then the fee recompute.
#
# Everything here **flags**; nothing is deleted except exact duplicate fills, which are shown before they
# are dropped.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posttrade import _util, hygiene, markouts, plots, reference, store

plots.style()
T, META = store.load_tables()
fills, book, trades = T["fills"], T["book_l1"], T["market_trades"]
NS = _util.NS

# %% [markdown]
# ## Duplicate fills
# Key: (venue, trade_id, order_id, side). Without trade_id the key falls back to (venue, order_id, ts, price, qty, side).

# %%
fills_dd, dups = hygiene.dedupe_fills(fills)
print(f"{len(dups)} duplicate fills removed of {len(fills):,}")
dups.head()

# %% [markdown]
# ## Position reconciliation

# %%
if T["positions"] is not None:
    rec = hygiene.reconcile_positions(fills_dd, T["positions"], tol=1e-6)
    fig, ax = plt.subplots()
    for i, ((v, s), g) in enumerate(rec.groupby(["venue", "symbol"])):
        ax.plot(_util.ns_to_dt(g["ts"]), g["break"], color=plots.SERIES[i], label=f"{v}:{s}", marker="o", ms=3)
    ax.axhline(0, color=plots.BASE, lw=1)
    ax.set(title="Position break: venue snapshot − position implied by fills", ylabel="qty")
    ax.legend()
    plt.show()
    display(rec[rec["flag"]].head(10))
else:
    print("UNAVAILABLE: no positions table")

# %% [markdown]
# **How to read this.** A flat line at zero is good. A step that **persists** means a fill is missing from
# your file (step = its signed qty) or duplicated (step of the opposite sign); the first flagged snapshot
# brackets when. A break that appears and then disappears usually means the two clocks disagree around a fill,
# so compare exchange vs local timestamps. Noise around zero suggests rounding: raise `tol`. Fix breaks before
# reading markouts; a missing toxic fill makes you look better than you are.

# %% [markdown]
# ## Fee recompute
# The schedule comes from `config.FEE_SCHEDULES`, with each venue's `fee_model`: bps of notional or per contract.

# %%
fee = hygiene.recompute_fees(fills_dd)
print(f"{fee['flag'].sum()} fills with |reported − expected| > 0.05 bps; total diff {fee['diff'].sum():.4f} quote ccy")
plots.hist(fee["diff_bps"], title="Reported − recomputed fee", xlabel="bps of notional")
plt.show()
fee[fee["flag"]].head(10)

# %% [markdown]
# **How to read this.** Ideally the histogram is a spike at 0. A cluster away from zero usually means a tier
# change (update the schedule, or make it time-dependent), a sign-convention flip (your file may report
# rebates as positive), or fees charged in a different currency. A few scattered outliers are worth asking the
# venue about. Notebook 02's after-fee markouts use the **reported** fee.

# %% [markdown]
# ## Funding recompute
# expected payment = −position × mark × rate × multiplier (positive rate means longs pay). Payments are signed as our cash (+ = received).

# %%
if T["funding"] is not None:
    fu = hygiene.recompute_funding(T["funding"], fills=fills_dd, instruments=T["instruments"])
    print(f"{fu['flag'].sum()} of {len(fu)} settlements disagree")
    display(fu[fu["flag"]])
else:
    print("UNAVAILABLE: no funding table (normal for dated futures such as CME)")

# %% [markdown]
# Disagreements come from four sources: a wrong sign convention (payment vs rate), the venue using a
# different mark (index vs mark price), a position snapshot that differs from your fills, or funding paid on
# a different schedule. Recomputing from fills (as here) also catches ledger positions that don't match your
# fills.

# %% [markdown]
# ## Bad ticks in market trades (Brownlees–Gallo)

# %%
if trades is not None:
    bt = hygiene.bad_ticks(trades, k=50, gamma_ticks=2, instruments=T["instruments"])
    print(f"{bt.sum()} of {len(trades):,} trades flagged ({bt.mean():.3%})")
    v, s = trades.loc[bt].iloc[0][["venue", "symbol"]] if bt.any() else trades.iloc[0][["venue", "symbol"]]
    g = trades[(trades["venue"] == v) & (trades["symbol"] == s)]
    fig, ax = plt.subplots()
    ax.plot(_util.ns_to_dt(g["ts"]), g["price"], lw=0.6, color=plots.SERIES[0], label="trades")
    gb = g[bt.loc[g.index]]
    ax.scatter(_util.ns_to_dt(gb["ts"]), gb["price"], color=plots.SERIES[1], s=25, zorder=3, label="flagged")
    ax.set(title=f"Trade prices with bad-tick flags: {v}:{s}", ylabel="price")
    ax.legend()
    plt.show()

# %% [markdown]
# **How to read this.** Flagged points that sit far from the price path are bad prints, so exclude them from
# reference prices and from the toxicity scores. Flags clustered in fast markets are usually real moves:
# raise `gamma_ticks` or `k`. The filter uses a centred window, so use it to flag data only, never to build a
# feature.

# %% [markdown]
# ## Book sanity and feed gaps

# %%
bs = hygiene.book_sanity(book, T["instruments"])
display(bs.drop(columns="any").sum().to_frame("rows flagged"))
gaps = hygiene.feed_gaps(book, rate_window_s=300, alpha=1e-6)
display(gaps.assign(start=lambda d: _util.ns_to_dt(d["start"]), end=lambda d: _util.ns_to_dt(d["end"])))

# %% [markdown]
# Crossed or locked books and zero sizes are almost always feed artefacts, often from a missed update. Off-tick
# prices need real tick sizes in `instruments` to be trusted. A feed gap is a silence that is implausible
# given the trailing message rate. Scheduled closures (CME halt, weekends) are excluded through
# `config.VENUES[venue].sessions`. During a gap every as-of value goes stale, so treat markouts and spreads
# computed there with suspicion (see staleness below).

# %% [markdown]
# ## Staleness of as-of values
# Every as-of join has an age. Below: the age of each fill's reference mid, and of the other leg's/lead venue's mid.

# %%
ref = reference.mid_ref(book)
ages = {}
for v, s in book[["venue", "symbol"]].drop_duplicates().itertuples(index=False):
    q = fills_dd[["ts"]].assign(venue=v, symbol=s)
    ages[f"{v}:{s}"] = hygiene.staleness(q, ref, ["value"])["age_s"].to_numpy()
fig, ax = plt.subplots()
for i, (k, a) in enumerate(ages.items()):
    a = np.sort(a[np.isfinite(a)])
    ax.plot(a, np.linspace(0, 1, len(a)), color=plots.SERIES[i], label=k)
ax.set_xscale("symlog", linthresh=0.01)
ax.set(title="Age of the latest book update at our fill times (CDF)", xlabel="age (s)", ylabel="share of fills")
ax.legend()
plt.show()

# %% [markdown]
# **How to read this.** Read along the y-axis. If 99% of fills see a book younger than ~1 update interval, joins
# are fine. A long right tail means stale inputs (feed gaps, quiet instruments). Markouts measured against a
# stale lead venue overstate how much the lead "knew". Consider excluding fills where the reference is older than
# a threshold.

# %% [markdown]
# ## Microstructure noise: signature plot

# %%
our_v, our_s = fills_dd.groupby(["venue", "symbol"]).size().idxmax()
our_ref = ref[(ref["venue"] == our_v) & (ref["symbol"] == our_s)]
sig = hygiene.signature(our_ref)
plots.signature_plot(sig)
plt.show()
print(f"noise variance omega^2 = RV/(2n) at tick level: {hygiene.noise_variance(our_ref):.3f} bps^2")

# %% [markdown]
# **How to read this.** Realised variance that rises at short intervals is microstructure noise (bid/ask
# bounce); a flat curve means the mid is close to efficient at that frequency. The interval where the curve
# flattens is a sensible minimum horizon for markouts and for sampling spreads (e.g. for the OU fit in 04). ω²
# is the per-observation noise variance in bps². Compare √ω² with your markout effects at short horizons.

# %% [markdown]
# ## Robust outlier flags on fills (MAD) and CUSUM of mean markout

# %%
mo = markouts.compute_markouts(fills_dd, ref, [60.0])[60.0]
fill_dev = 1e4 * fills_dd["side"] * (markouts.ref_at_fill(fills_dd, ref) - fills_dd["price"]) / fills_dd["price"]
mf = hygiene.mad_flags(fill_dev, thresh=6)
print(f"{mf['flag'].sum()} fills whose price is a robust outlier vs the pre-fill mid (|z|>6): check them, don't delete")
n_days = fills_dd["ts"].floordiv(86400 * NS).nunique()
bucket = "D" if n_days >= 20 else "h"  # CUSUM wants many periods; with little history use hours
per = pd.Series(mo.to_numpy(), index=_util.ns_to_dt(fills_dd["ts"])).resample(bucket).mean().dropna()
cs = hygiene.cusum(per, k=0.5, h=5.0, baseline_n=max(len(per) // 3, 5))
plots.cusum_plot(cs, h=5.0, title=f"CUSUM of {'daily' if bucket == 'D' else 'hourly'} mean 60 s markout")
plt.show()

# %% [markdown]
# **How to read this.** CUSUM accumulates standardised deviations from a baseline (the first third of
# periods). S− crossing the threshold means mean markout has shifted **down**: more toxic flow, a competitor,
# or a broken parameter. S+ means it shifted up. The date of the crossing is the place to look. Follow up with
# 02's conditional curves split before and after it. With synthetic data there is no regime change, so no alarm
# is the expected outcome.

# %% [markdown]
# ## Clock sync (NTP four-timestamp) and forward-fill-only resampling
# If you log request/response timestamps against the venue (t0 client send, t1 server recv, t2 server send,
# t3 client recv), estimate your clock offset and drift. Below is a synthetic example; replace it with your own log.

# %%
t0 = np.arange(0, 3600, 5, dtype="int64") * NS
true_off = 3e6 + 1500 * (t0 / NS)  # 3 ms offset, +1.5 us/s drift
q = np.random.default_rng(0).exponential(2e5, (2, len(t0)))  # queueing on each leg
ex = pd.DataFrame({"t0": t0, "t1": t0 + 5e5 + q[0] + true_off})
ex["t2"] = ex["t1"] + 2e4
ex["t3"] = ex["t2"] - true_off + 5e5 + q[1]
ntp = hygiene.ntp_offsets(ex)
print(f"offset at start {ntp['offset_ns_at_start'] / 1e6:.3f} ms, drift {ntp['slope_ns_per_s']:.0f} ns/s")
fig, ax = plt.subplots()
tb = ntp["table"]
ax.scatter(tb["t0"] / NS, tb["offset_ns"] / 1e6, s=6, color=plots.MUTED, label="all samples")
ax.scatter(tb.loc[tb["used_in_fit"], "t0"] / NS, tb.loc[tb["used_in_fit"], "offset_ns"] / 1e6, s=6, color=plots.SERIES[0],
           label="low-delay samples (fit)")
ax.plot(tb["t0"] / NS, (ntp["intercept"] + ntp["slope_ns_per_s"] * tb["t0"] / NS) / 1e6, color=plots.SERIES[1], label="drift fit")
ax.set(title="Clock offset (server − client)", xlabel="seconds", ylabel="ms")
ax.legend()
plt.show()

# %% [markdown]
# **How to read this.** Queueing makes offsets noisy and biased, so the fit uses only the lowest-delay
# samples. A non-zero slope means your clock drifts: correct `ts_local` before computing latency hops (06), or
# hops will drift through the day for reasons that have nothing to do with your system.

# %%
grid = hygiene.resample_ffill(our_ref, 1.0)
print(grid.head())
print("max age on the 1 s grid:", grid["age_s"].max(), "s  (forward fill only; values are never interpolated)")
