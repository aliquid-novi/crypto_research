# %% [markdown]
# # 05 · Models
#
# **Purpose.** Move from describing markouts to explaining and predicting them. The notebook covers markout
# regressions (OLS with Newey–West, Huber, quantile), a toxicity classifier judged by avoided loss, fill
# probability and hazards, EV per quote, selection correction, a two-state HMM, Hawkes self- and
# cross-excitation, the response function and propagator, Kyle's λ, Roll, Huang–Stoll, the Hasbrouck VAR and
# the square-root impact law.
#
# **Tables.** fills + book_l1 (regressions, toxicity, HMM), orders (fill probability, hazards, selection),
# market_trades (Hawkes, impact models).
#
# **What to look at first.** The coefficient-vs-horizon plot and the toxicity backtest.
#
# **Optional packages.** lifelines (Cox PH) and hmmlearn. Without them the discrete-time hazard and the
# built-in Baum–Welch are used, and the output says which.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from posttrade import _util, config, hygiene, markouts, models, plots, reference, store

plots.style()
T, META = store.load_tables()
fills, _ = hygiene.dedupe_fills(T["fills"])
book, trades, orders = T["book_l1"], T["market_trades"], T["orders"]
ref = reference.mid_ref(book)
maker = fills[fills["liquidity"] == "maker"]
events = markouts.aggregate_events(maker)
our_v, our_s = events.groupby(["venue", "symbol"]).size().idxmax()
H = [1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0]
mo = markouts.compute_markouts(events, ref, H)
print(f"{len(events):,} passive events on {our_v}:{our_s}; optional packages: lifelines={models.HAVE_LIFELINES}, hmmlearn={models.HAVE_HMMLEARN}")

# %% [markdown]
# ## Features at fill time (all strictly pre-fill)
# Directional features are oriented as "positive = pressure for the price to go up", then multiplied by our
# side, so **positive = favourable for our fill**. Non-directional features are left unsigned.

# %%
imb = _util.asof_values(events, book.assign(imb=reference.imbalance(book)), ["imb"], direction="backward", allow_exact=False)["imb"]
ofi = reference.ofi(book)
ofi_30 = reference.windowed_sum(ofi, "ofi", 30.0, query=events.assign(ts=events["ts"] - 1))
vol = reference.trailing_vol(ref, 300)
m0 = markouts.ref_at_fill(events, ref)
X_dir = pd.DataFrame({"imbalance": imb.to_numpy() - 0.5, "ofi_30s": ofi_30 / np.nanstd(ofi_30)})
lead = reference.lead_mid(book)
if len(lead):
    lm = _util.asof_values(events, lead, ["value"], direction="backward", allow_exact=False)["value"].to_numpy()
    X_dir["lead_gap_bps"] = 1e4 * (lm - m0) / m0
X = models.signed_features(X_dir, list(X_dir.columns), events["side"].to_numpy())
X["vol_bps"] = _util.asof_values(events, vol, ["vol_bps"], direction="backward", allow_exact=False)["vol_bps"].to_numpy()
X["log_qty"] = np.log(events["qty"].to_numpy())
X.describe().T.round(3)

# %% [markdown]
# ## Markout regressions per horizon

# %%
reg = models.markout_regression(mo, X, ts=events["ts"].to_numpy(), quantiles=(0.05, 0.10))
feat = "lead_gap_bps" if "lead_gap_bps" in X else "imbalance"
plots.coef_vs_horizon(reg, feat, models=["ols_nw", "huber", "q05", "q10"])
plt.show()
reg[reg["model"] == "ols_nw"].pivot(index="feature", columns="horizon_s", values="coef").round(3)

# %% [markdown]
# **How to read this.** Each point is the effect of a one-unit change in the feature on the markout at that
# horizon. A coefficient that **grows** with horizon is information that the market prices in gradually; one
# that peaks and decays is a transient effect (bounce, inventory). The Newey–West bands account for the overlap
# of horizons across nearby fills; an iid OLS band would be too narrow. Quantile coefficients (q05/q10) larger
# in magnitude than OLS mean the feature mainly predicts the **bad tail**, which is what a filter needs.
# Huber far from OLS means a few outliers drive the OLS estimate.

# %% [markdown]
# ## Toxicity classifier: logistic, threshold from costs, judged by avoided loss

# %%
TOX_H, TOX_THRESH = 60.0, 0.0  # <- horizon and "toxic" threshold (bps)
tox = models.toxicity_model(X, mo[TOX_H].to_numpy(), events["ts"].to_numpy(), threshold_bps=TOX_THRESH, train_frac=0.6)
print(f"p* = g/(g+l) = {tox['p_star']:.3f}  (g = {tox['g']:.2f} bps kept on good fills, l = {tox['l']:.2f} bps lost on toxic)")
bt = tox["backtest"]
fig, ax = plt.subplots()
ax.plot(bt["cutoff"], bt["avoided_loss_bps"], color=plots.SERIES[0], label="avoided loss")
ax.plot(bt["cutoff"], bt["forgone_gain_bps"], color=plots.SERIES[1], label="forgone gain")
ax.plot(bt["cutoff"], bt["net_bps"], color=plots.INK, label="net")
ax.axvline(tox["p_star"], color=plots.MUTED, ls="--", lw=1, label="p*")
ax.axhline(0, color=plots.BASE, lw=1)
ax.set(title="Out-of-sample filter backtest (later 40% of fills)", xlabel="skip if P(toxic) > cutoff", ylabel="sum of bps")
ax.legend()
plt.show()
pd.Series(tox["test_summary"]).drop("is_p_star").astype(float).round(3)

# %% [markdown]
# **How to read this.** For each cutoff, the blue line is the losses you would have avoided by not trading
# fills predicted toxic, orange is the profit you would have given up, and black is the difference. A filter
# is worth having only if the black line is clearly positive near p*. AUC can look fine while the net is zero,
# because the model may separate small winners from small losers. This backtest is a single time split, so
# repeat it on more days before trusting it. Cancelling quotes also changes which fills you get, which this
# backtest cannot see.

# %% [markdown]
# ## Fill probability: λ(δ) = A·e^{−kδ}, hazards and EV per quote

# %%
if orders is not None:
    po = orders[(orders["venue"] == our_v) & (orders["symbol"] == our_s) & orders["final_status"].isin(["filled", "partial", "cancelled"])]
    smp = models.quote_distance_samples(po, ref, step_s=1.0)
    fi = models.fit_fill_intensity(smp)
    t = fi["table"]
    fig, ax = plt.subplots()
    ax.scatter(t["delta_bps"], t["lambda_per_s"], color=plots.SERIES[0], label="empirical fills / quoted second")
    ax.plot(t["delta_bps"], t["lambda_fit"], color=plots.SERIES[1], label=f"A·exp(−kδ): A={fi['A']:.3f}, k={fi['k']:.3f}")
    ax.set_yscale("log")
    ax.set(title="Fill intensity vs distance from mid", xlabel="δ (bps behind mid)", ylabel="fills per second")
    ax.legend()
    plt.show()
else:
    print("UNAVAILABLE: needs orders")

# %% [markdown]
# **How to read this.** On a log scale the exponential model is a straight line. k is how fast fill
# probability decays per bp you quote away from the mid. Points bending away from the line at small δ often
# mean queue position matters more than distance there. This is the λ(δ) input to Avellaneda–Stoikov-style
# quoting. Negative δ are quotes through the mid (rare, or crossed books).

# %%
if orders is not None:
    sv = models.survival_frame(po, ref)
    sv["vol_bps"] = _util.asof_values(po.loc[sv.index].assign(ts=po.loc[sv.index, "ts_ack"].astype("int64")), vol, ["vol_bps"],
                                      direction="backward", allow_exact=False)["vol_bps"].to_numpy()
    sv = sv.dropna()
    hz = models.fit_hazard(sv, ["dist_bps", "vol_bps"])
    print("hazard method:", hz["method"])
    display(hz["params"].round(4))
    fill_mo = markouts.compute_markouts(maker, ref, [TOX_H])[TOX_H]
    first = maker.assign(mo=fill_mo.to_numpy()).groupby("order_id")["mo"].first()
    mf = pd.DataFrame({"dist_bps": sv.set_index("order_id")["dist_bps"].reindex(first.index), "markout_bps": first}).dropna()
    fee_bps = float(np.median(1e4 * maker["fee"] / maker["notional"]))
    ev = models.ev_per_quote(sv, mf, fee_bps=fee_bps)
    display(ev.round(3))

# %% [markdown]
# **How to read this.** Hazard coefficients are log hazard ratios (Cox) or per-bin log-odds (fallback). A
# negative `dist_bps` means quotes further away fill more slowly, as expected. The EV table combines how
# often each distance fills, how good those fills are, and fees. The best distance is where `ev_bps` peaks.
# Often that is not the touch, because touch fills fill most but are the most adversely selected.

# %% [markdown]
# ## Selection correction: are fills a biased sample of quotes?

# %%
if orders is not None:
    sel = sv["event"].to_numpy()
    filled = sv[sel == 1]
    y = mf["markout_bps"].reindex(filled["order_id"]).to_numpy()
    Z = sv[["dist_bps", "vol_bps"]]
    hk = models.heckman_two_step(Z, sel, filled[["vol_bps"]], y)  # rows with NaN markout are dropped in step 2
    print(f"Heckman: inverse-Mills coefficient {hk['imr_coef']:.3f} (t = {hk['imr_t']:.2f})")
    ok = np.isfinite(y)
    ipw = models.ipw_regression(Z, sel, filled[["vol_bps"]], y)
    naive = sm.OLS(y[ok], sm.add_constant(filled[["vol_bps"]][ok])).fit()
    w = ipw["weights"][ok]
    print(f"IPW weights: max/median {w.max() / np.median(w):.0f}x, Kish effective n {w.sum() ** 2 / (w ** 2).sum():.0f} of {ok.sum()}")
    display(pd.DataFrame({"naive OLS": naive.params, "IPW": ipw["outcome"].params}).round(4))

# %% [markdown]
# **How to read this.** A significant inverse-Mills coefficient means the fills you observe are not a random
# draw of the quotes you posted, so naive markout regressions are biased. Here the exclusion restriction is
# distance: it moves fill probability and is assumed not to move the markout directly. That assumption is
# questionable, so treat this as a diagnostic. IPW re-weights fills toward the quotes that rarely fill. If
# IPW and naive OLS agree, selection doesn't change the conclusion. Check the Kish effective n first: when a
# few rarely-filled far quotes carry most of the weight (as on the synthetic data), the IPW estimate is
# theirs, not the population's. Raise the `clip` or trim far quotes before reading it.

# %% [markdown]
# ## Two-state HMM: filtered P(toxic regime)

# %%
x = mo[TOX_H].to_numpy()
x = x[np.isfinite(x)]
hmm = models.fit_hmm2(x)
print("state means (bps):", hmm["means"].round(2), " sds:", hmm["sds"].round(2), "\ntransition matrix:\n", hmm["P"].round(3))
fig, ax = plt.subplots()
ax.plot(hmm["p_toxic_filtered"], lw=0.8, color=plots.SERIES[1], label="filtered (real-time)")
ax.plot(pd.Series(hmm["p_toxic_smoothed"]).rolling(25, min_periods=1).mean(), lw=1.2, color=plots.SERIES[0], label="smoothed (hindsight, 25-event mean)")
ax.set(title="P(toxic regime) by fill", xlabel="fill number", ylabel="probability")
ax.legend()
plt.show()

# %% [markdown]
# **How to read this.** The HMM splits markouts into two regimes with different means. If the two means
# barely differ, or the transition matrix has no persistence (diagonal ≈ 0.5), there are no regimes and
# markouts are just noise around one mean. If the toxic state is persistent (diagonal near 1), the filtered
# probability is a usable real-time signal for widening.
#
# ⚠ **Overlap creates fake regimes.** Markouts of fills closer together than h share most of their price path,
# so consecutive markouts are correlated and the HMM reports "persistent" states even when toxicity is iid.
# That is what happens on the synthetic data here: toxicity is iid per fill, yet the diagonal is ≈ 0.9. Before
# believing regimes, refit on markouts at a horizon shorter than the typical gap between fills, or on one
# fill per non-overlapping window, and check whether the persistence survives.

# %% [markdown]
# ## Hawkes: self-excitation and cross-excitation with the lead venue

# %%
if trades is not None and len(reference.lead_mid(book)):
    (lv, ls), _ = next(iter(config.LEAD_MAP.items()))
    lt = np.sort(trades.loc[(trades["venue"] == lv) & (trades["symbol"] == ls), "ts"].to_numpy())
    ft = np.sort(events["ts"].to_numpy())
    t0 = min(lt.min(), ft.min())
    TT = (max(lt.max(), ft.max()) - t0) / 1e9
    h1 = models.fit_hawkes_1d((lt - t0) / 1e9, T=TT)
    print(f"lead-venue trades: mu={h1['mu']:.3f}/s, branching alpha={h1['alpha']:.3f}, beta={h1['beta']:.3f}/s (se {np.round(h1['se'], 3)})")
    h2 = models.fit_hawkes_2d([(ft - t0) / 1e9, (lt - t0) / 1e9], TT)
    print("bivariate A[i,j] (effect of j on i; 0 = our fills, 1 = lead trades):\n", h2["A"].round(3))
    r1 = models.hawkes_rescaled_1d((lt - t0) / 1e9, h1["mu"], h1["alpha"], h1["beta"])
    r2 = models.hawkes_rescaled_2d([(ft - t0) / 1e9, (lt - t0) / 1e9], h2)
    plots.qq_exponential({"lead trades (1-d)": r1, "our fills (2-d)": r2[0], "lead trades (2-d)": r2[1]})
    plt.show()
else:
    print("UNAVAILABLE: needs market_trades on a lead venue in config.LEAD_MAP")

# %% [markdown]
# **How to read this.** The branching ratio α is the share of events triggered by earlier events: near 1 means
# trading is mostly self-generated bursts. In the bivariate fit, A[0,1] > 0 means lead-venue trades trigger
# our fills, which is the footprint of takers arbitraging our quotes against the lead. A[1,0] > 0 means our
# fills trigger lead trades (we are the information). The QQ plot checks the fit: points on the diagonal are
# good. Points above it in the tail mean the model misses bursts; try a sum of exponentials or a power-law
# kernel.

# %% [markdown]
# ## Price impact: response function, propagator, Kyle λ, Roll, Huang–Stoll, Hasbrouck VAR

# %%
if trades is not None:
    tape = trades[(trades["venue"] == our_v) & (trades["symbol"] == our_s)]
    ev_ = models.trade_event_series(tape, ref)
    resp = models.response_function(ev_, 50)
    G = models.propagator(resp, 30)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    plots.lines(resp[["R_bps"]].iloc[1:], ax=axs[0], title="Response R(ℓ)", xlabel="trades later", ylabel="bps")
    plots.lines(G, ax=axs[1], title="Propagator G(ℓ)", xlabel="lag (trades)", ylabel="bps")
    plt.show()
    k = models.kyle_lambda(tape, ref, 60.0)
    roll = models.roll_spread(tape["price"].to_numpy())
    hs = models.huang_stoll(ev_["price"].to_numpy(), ev_["eps"].to_numpy())
    print(f"Kyle lambda: {k['lambda']:.4f} bps per unit signed volume (t={k['t']:.1f}, R2={k['r2']:.3f})")
    print(f"Roll spread: {roll['spread_bps']:.3f} bps   Huang-Stoll: S={hs['S']:.5f} (price), lambda={hs['lam']:.3f}")

# %% [markdown]
# **How to read this.** R(ℓ) is the average move ℓ trades after a buy. It rises and plateaus at the permanent
# impact. G(ℓ) is the impact of one isolated trade. G decaying while R stays flat means impact is transient
# and sign autocorrelation (traders splitting orders) holds the price up. Kyle λ is price impact per unit of
# net order flow. Roll and Huang–Stoll estimate the spread from trade prices alone. Huang–Stoll's λ is the
# share of the spread due to information and inventory, so a larger value means more informed flow.

# %%
if trades is not None:
    hv = models.hasbrouck_var(ev_, p=5, horizon=30)
    plots.lines(pd.DataFrame({"cumulative impulse response": hv["irf_cum"]}), title="Hasbrouck VAR: price response to one buy",
                xlabel="trades after the shock", ylabel="bps")
    plt.show()
    print(f"permanent impact {hv['permanent_bps']:.3f} bps; trade share of permanent price variance {hv['trade_info_share']:.2%}")

# %% [markdown]
# **How to read this.** The curve's plateau is the permanent (information) impact of a trade. Overshoot that
# reverts is transient (inventory or bounce). The trade information share is the fraction of long-run price
# variance explained by trades. A high value means order flow on this venue carries the information. A low
# value means prices are driven by public news or by another venue (compare with the lead venue).

# %% [markdown]
# ## Square-root impact on our own metaorders

# %%
legs = fills[fills["decision_id"].notna()] if "decision_id" in fills else fills.iloc[0:0]
meta = models.metaorders(legs if len(legs) else maker, ref, ["decision_id", "leg", "leg_role"] if len(legs) else ["order_id"])
if trades is not None and len(meta) > 20:
    days = (trades["ts"].max() - trades["ts"].min()) / 86400e9
    V = {k: g["qty"].sum() / max(days, 1e-9) for k, g in trades.groupby(["venue", "symbol"])}
    sig = {k: np.nanmean(g["vol_bps"]) * np.sqrt(86400) for k, g in vol.groupby(["venue", "symbol"])}
    vv = np.array([V.get((a, b), np.nan) for a, b in zip(meta["venue"], meta["symbol"])])
    ss = np.array([sig.get((a, b), np.nan) for a, b in zip(meta["venue"], meta["symbol"])])
    sq = models.sqrt_impact_fit(meta, vv, ss, n_bins=6)
    t = sq["table"]
    fig, ax = plt.subplots()
    ax.errorbar(t["x"], t["y"], yerr=1.96 * t["y_se"], fmt="o", color=plots.SERIES[0], label="binned mean I/σ")
    if np.isfinite(sq["delta"]):
        xx = np.geomspace(t["x"].min(), t["x"].max(), 50)
        ax.plot(xx, sq["Y"] * xx ** sq["delta"], color=plots.SERIES[1], label=f"Y(Q/V)^δ: Y={sq['Y']:.2f}, δ={sq['delta']:.2f}")
    ax.set_xscale("log")
    ax.set(title="Impact vs metaorder size (our own)", xlabel="Q / daily volume", ylabel="impact / daily σ")
    ax.legend()
    plt.show()

# %% [markdown]
# **How to read this.** The square-root law predicts δ ≈ 0.5 and Y of order 1 for metaorders that are a
# meaningful fraction of daily volume. Our clips are tiny relative to daily volume, so their impact is mostly
# spread and noise. Expect a flat or noisy fit unless you trade size. Binned means with error bars crossing
# zero mean we cannot measure impact at that size: say so rather than reporting δ.
