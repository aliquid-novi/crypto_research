# %% [markdown]
# # 04 · Spread-trade implementation shortfall
#
# **Purpose.** For each spread decision: what the paper trade would have earned at decision-time mids, what
# you actually earned, and **exactly** where the difference went. The components are delay, legging,
# execution cost per leg, opportunity (unfilled quantity), exit-side equivalents, residual (hung) position,
# fees and funding carry. They sum exactly to realised − paper (asserted). Also: spread dynamics (OU
# half-life, Kalman hedge ratio, alpha decay).
#
# **Conventions.** `intended_qty` is signed in leg-A units (+ = buy A / sell B); `hedge_ratio` = units of B per
# unit of A; components are in quote currency and in bps of intended gross notional; positive = good.
#
# **Tables.** decisions, fills with decision_id/leg/leg_role, book_l1 (+ funding, instruments for multipliers).
#
# **What to look at first.** The waterfall, then the cost share of gross edge.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posttrade import _util, dynamics, hygiene, markouts, plots, reference, spreads_tca, store

plots.style()
T, META = store.load_tables()
fills, _ = hygiene.dedupe_fills(T["fills"])
book, decisions = T["book_l1"], T["decisions"]
assert decisions is not None, "UNAVAILABLE: this notebook needs the decisions table (see 00)"
fills = spreads_tca.link_fills(fills, decisions)  # no-op when fills already carry decision_id / leg / leg_role
res = spreads_tca.shortfall(decisions, fills, book, T["funding"], T["instruments"], default_hold_s=None)
trades, legs = res["trades"], res["legs"]
print(f"{len(trades)} decisions, {trades['filled_any'].sum()} with fills, {trades['open'].sum()} still open; "
      f"max |bridge error| = {trades['bridge_error'].abs().max():.2e}")

# %% [markdown]
# ## The bridge: paper → realised (mean bps per filled decision)

# %%
ft = trades[trades["filled_any"]]
comp = ft[[f"{c}_bps" for c in spreads_tca.COMPONENTS]].mean()
comp.index = spreads_tca.COMPONENTS
plots.waterfall(comp, start=ft["paper_bps"].mean(), start_label="paper", end_label="realised net",
                title="Implementation-shortfall bridge (mean per filled decision, bps of gross notional)")
plt.show()
pd.DataFrame({"mean_bps": comp, "se_bps": ft[[f"{c}_bps" for c in spreads_tca.COMPONENTS]].sem().to_numpy()}).round(3)

# %% [markdown]
# **How to read this.** The first bar is the gross edge your signal had on paper. Each following bar is a
# cost (orange) or a gain (blue). The last bar is what you kept. Typical patterns:
# - large **entry/exit cost**: crossing spreads, so consider passive entry on the less urgent leg;
# - large **legging**: the second leg's price moves while the first leg is on, so consider trading the
#   faster-reverting or more liquid leg first;
# - large **delay**: the signal decays faster than you act (see alpha decay below);
# - large **opportunity**: missed fills cost real edge (see the missed-decision report).
#
# Compare each bar with its standard error before acting on it.

# %% [markdown]
# ## Cost share of gross edge and net Sharpe

# %%
cs = spreads_tca.cost_share(trades)
pd.Series(cs).round(3)

# %% [markdown]
# `cost_share` = C/α. With costs that do not vary much between trades, SR_net ≈ SR_gross × (1 − C/α), so at a
# cost share of 0.8 you keep only a fifth of the paper Sharpe. A large gap between `sr_net_approx` and
# `sr_net_actual` means costs are variable and correlated with the edge (e.g. you pay most on your best
# signals).

# %% [markdown]
# ## Aggregates by pair, leg order, signal bucket, vol regime

# %%
vol = reference.trailing_vol(reference.mid_ref(book), window_s=1800)
first_leg_v = decisions.set_index("decision_id").loc[trades["decision_id"], "leg_A_venue"].to_numpy()
first_leg_s = decisions.set_index("decision_id").loc[trades["decision_id"], "leg_A_symbol"].to_numpy()
q = pd.DataFrame({"ts": trades["ts"].to_numpy(), "venue": first_leg_v, "symbol": first_leg_s})
trades["vol_regime"] = pd.qcut(_util.asof_values(q, vol, ["vol_bps"], direction="backward")["vol_bps"], 2, labels=["low vol", "high vol"])
trades["signal_bucket"] = markouts.quantile_buckets(trades["signal"].abs(), 3).to_numpy()
for by in ["pair", "leg_order", "signal_bucket", "vol_regime"]:
    display(spreads_tca.aggregate(trades, by).round(2))

# %% [markdown]
# **How to read this.** Read across rows for the component that changes most between groups. If B-first
# trades have much smaller legging than A-first ones, that is a rule you can adopt. If the strongest-signal
# bucket also has the largest delay cost, the signal is fast and you are late. With few trades per group
# the means are noisy: check `n`.

# %% [markdown]
# ## Legging cost vs time between legs; hung legs

# %%
plots.scatter(trades["legging_s"], trades["legging_bps"], title="Legging cost vs time between legs",
              xlabel="seconds between first fills of the two legs", ylabel="legging (bps)", logx=True)
plt.show()
spreads_tca.hung_legs(trades, legs, hung_s=10.0)[["decision_id", "legging_s", "reason", "residual_bps", "realised_net_bps"]]

# %% [markdown]
# **How to read this.** Legging cost should be centred on zero for short gaps and spread out as the gap grows
# (the leg is exposed for longer). A binned mean that drifts **negative** with the gap means the second leg
# moves against you systematically while you wait, which is informed competition on the same signal. Hung-leg
# episodes are where the tail risk is. Check whether they cluster in time (venue issues) or by leg.

# %% [markdown]
# ## Missed decisions

# %%
spreads_tca.missed_report(trades).round(3)

# %% [markdown]
# If missed decisions had **better** paper PnL than filled ones, your fill process is adversely selected: you
# get filled when the signal is about to fail. Equal means within error says missing fills only costs
# volume, not edge.

# %% [markdown]
# ## Markouts anchored at decision time vs fill time

# %%
H = [0.0, 1.0, 5.0, 30.0, 60.0, 300.0]
dv = spreads_tca.decision_vs_fill_markouts(fills, decisions, reference.mid_ref(book), H)
plots.markout_curves({"anchored at fill": markouts.markout_curve(dv["fill_time"], n_boot=200),
                      "anchored at decision": markouts.markout_curve(dv["decision_time"], n_boot=200)},
                     title="Entry-leg markouts: decision-time vs fill-time anchor")
plt.show()

# %% [markdown]
# **How to read this.** Both curves use the fill price. The decision-anchored curve at horizon h compares it
# with the mid h seconds after the **decision**. A gap between the curves at small h is the price move during
# your reaction time. If the decision-anchored curve is the higher of the two, the market moved your way while
# you were acting, and your signal is slower than you are.

# %% [markdown]
# ## MAE / MFE per trade

# %%
mm = spreads_tca.mae_mfe(legs, fills, book, step_s=1.0)
fig, ax = plt.subplots()
ax.scatter(mm["mae"], mm["mfe"], c=np.where(mm["final"] >= 0, plots.GOOD, plots.BAD), s=18)
ax.set(title="Max adverse vs max favourable excursion (quote ccy); blue = winner", xlabel="MAE", ylabel="MFE")
plt.show()

# %% [markdown]
# **How to read this.** Winners with large MAE mean you sit through deep drawdowns to get paid; a stop would
# cut winners. Losers with large MFE are trades that were in profit and gave it back, which is the case for a
# profit-taking exit rule. A clean split (winners top-left, losers bottom-right) means the exit is fine.

# %% [markdown]
# ## Spread dynamics: OU half-life, Kalman hedge ratio, alpha decay

# %%
d0 = decisions.iloc[0]
legA, legB = (d0["leg_A_venue"], d0["leg_A_symbol"]), (d0["leg_B_venue"], d0["leg_B_symbol"])
SAMPLE_S = 5.0  # <- sampling for the spread (choose using the signature plot in 01)
HEDGE_LOG_BETA = 1.0  # <- your log hedge ratio; or use the causal Kalman estimate below
sp = reference.composite_spread(book, legA, legB, HEDGE_LOG_BETA, log=True, on="A")
g = hygiene.resample_ffill(sp.assign(venue="pair", symbol="pair"), SAMPLE_S, ["value", "mid_A", "mid_B"])
fits = pd.DataFrame([vars(dynamics.ou_fit(g["value"], SAMPLE_S, lag=L)) for L in [1, 3, 6, 12, 24]], index=[1, 3, 6, 12, 24])
fits.index.name = "AR lag (samples)"
fits[["theta", "half_life_s", "b", "se_b", "n"]].round(5)

# %% [markdown]
# **How to read this.** θ should be roughly stable across lags. If it falls as the lag grows, measurement
# noise (bid/ask bounce, legs sampled at slightly different times) is inflating lag-1 estimates, so use the
# plateau. If it keeps changing, the spread is not a single OU process (regimes, drifting hedge ratio). With
# few half-lives in the sample, θ has a large sampling error (roughly √(2θ/T)).
#
# **Published reference points** (details in `REFERENCES.md`).
#
# - Makarov and Schoar (2020) found gaps between crypto exchanges that recur and can persist for weeks, so one
#   stationary OU mean may not hold. Fit per day or per regime and see whether θ is stable.
# - Which venue leads also moves over time. BitMEX, then Huobi and OKEx perpetuals, then Binance's USDT
#   perpetual led bitcoin price discovery, and CME mostly reacted (Alexander et al.). So re-estimate the lead
#   venue from your own data rather than assuming it.

# %%
kf = dynamics.kalman_hedge_ratio(np.log(g["mid_A"].to_numpy()), np.log(g["mid_B"].to_numpy()), q=(1e-7, 1e-7), r=1e-6, init_var=1e-4,
                                 init_beta=HEDGE_LOG_BETA, init_alpha=float(np.log(g["mid_A"].iat[0]) - HEDGE_LOG_BETA * np.log(g["mid_B"].iat[0])))
plots.time_series(g["ts"], {"Kalman β (prior: uses data up to t−1 only)": kf["beta_prior"].to_numpy()},
                  title="Time-varying log hedge ratio", ylabel="β")
plt.show()

# %% [markdown]
# **How to read this.** `beta_prior` at time t uses only data up to t−1, so it is the one you could have
# traded on. A β that wanders far from your fixed hedge ratio means your spread contains leftover directional
# exposure, and some "spread PnL" is really beta PnL. `q` controls how fast β may move; tune it so the
# innovation z-scores look stationary. (On the synthetic pair β is constant at 1, so the line should stay flat
# near 1.)

# %%
best = fits.loc[6]
mu = float(g["value"].mean())
g["z"] = (g["value"] - mu) / g["value"].std()
HZ = [10, 30, 60, 120, 300, 600, 900]
emp = dynamics.empirical_alpha_decay(g, HZ, mu=mu, n_bins=3, min_abs_signal=0.5)
fig, ax = plt.subplots()
for i, (b, e) in enumerate(emp.groupby("bucket", observed=True)):
    e = e.sort_values("horizon_s")
    ax.plot(e["horizon_s"], 1e4 * e["mean_captured"], color=plots.SERIES[i], marker="o", ms=3, label=f"|z| {b} empirical")
    D = e["mean_deviation"].mean()
    ax.plot(HZ, 1e4 * dynamics.model_alpha_decay(D, best["theta"], HZ), color=plots.SERIES[i], ls="--", lw=1, label=f"|z| {b} OU model")
ax.set(title="Alpha decay: reversion captured vs horizon", xlabel="horizon (s)", ylabel="bps")
ax.legend(fontsize=8)
plt.show()
dynamics.alpha_decay_regression(g, HZ).round(5)

# %% [markdown]
# **How to read this.** Dashed lines are the OU prediction D(1 − e^{−θt}) for each signal bucket. Solid lines
# are what actually happened. Empirical curves that rise faster than the model and then flatten mean the edge
# arrives early, so holding past the knee adds risk without return. Curves below the model at all horizons
# mean the OU fit overstates reversion. The regression gives the same information as one coefficient per
# horizon, with Newey–West SEs for the overlapping windows.
