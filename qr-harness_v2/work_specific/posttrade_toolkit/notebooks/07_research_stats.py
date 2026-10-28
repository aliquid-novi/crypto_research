# %% [markdown]
# # 07 · Research statistics
#
# **Purpose.** Before believing a result, ask how much data it needs, how many effectively independent
# observations it really has, how many things were tried before it, and whether it survives multiple testing.
#
# **Tables.** None required. The examples use markouts from the cached data; swap in any return series.
#
# **What to look at first.** The deflated Sharpe ratio, with the number of trials taken from your research log.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from posttrade import hygiene, markouts, plots, reference, stats, store

plots.style()
T, META = store.load_tables()
fills, _ = hygiene.dedupe_fills(T["fills"])
ref = reference.mid_ref(T["book_l1"])
maker = fills[fills["liquidity"] == "maker"].sort_values("ts")
mo = markouts.net_of_fees(markouts.compute_markouts(maker, ref, [60.0]), maker)[60.0].dropna()
print(f"{len(mo):,} fills; mean net 60 s markout {mo.mean():.3f} bps, sd {mo.std():.3f}")

# %% [markdown]
# ## How much data do you need?

# %%
mu, sd = mo.mean(), mo.std()
sr = mu / sd
print(f"per-fill Sharpe {sr:.4f} -> t = SR·√n = {stats.sharpe_tstat(sr, len(mo)):.2f}")
for target in [0.1, 0.25, 0.5]:
    print(f"fills needed to detect a mean of {target} bps at 2 sigma (sd {sd:.1f}): {stats.required_n(target, sd, 2.0):,.0f}")

# %% [markdown]
# `required_n = (zσ/μ)²` assumes independent observations. Overlapping markouts and clustered fills are not
# independent, so the next cell shows how many effective observations you actually have.

# %%
n_eff_ac = stats.effective_n_autocorr(mo.to_numpy())
cl = markouts.time_clusters(maker.loc[mo.index], 60)
sizes = pd.Series(cl).value_counts()
same = cl[1:] == cl[:-1]  # consecutive fills in the same 60 s cluster
x = mo.to_numpy()
icc = np.corrcoef(x[:-1][same], x[1:][same])[0, 1] if same.sum() > 10 else 0.0  # intra-cluster correlation estimate
print(f"n = {len(mo):,}; effective n (autocorrelation) = {n_eff_ac:,.0f}; "
      f"effective n (60 s clusters, ICC {icc:.3f}, mean size {sizes.mean():.1f}) = {stats.effective_n_cluster(len(mo), sizes.mean(), max(icc, 0)):,.0f}")
print(f"Newey-West SE of the mean (lag 20): {stats.newey_west_se(mo.to_numpy(), 20):.4f} vs iid {mo.std() / np.sqrt(len(mo)):.4f}")

# %% [markdown]
# ## Multiple trials: expected max Sharpe, PSR and deflated Sharpe

# %%
N = np.arange(1, 501)
fig, ax = plt.subplots()
for i, s_ in enumerate([0.25, 0.5, 1.0]):
    ax.plot(N, [stats.expected_max_sharpe(n, s_) if n > 1 else 0 for n in N], color=plots.SERIES[i], label=f"σ(SR across trials) = {s_}")
ax.set(title="Expected best Sharpe from N skill-less trials", xlabel="number of trials N", ylabel="E[max SR]")
ax.legend()
plt.show()
print("check: N=100, sigma=0.5 ->", round(stats.expected_max_sharpe(100, 0.5), 3))

# %% [markdown]
# **How to read this.** If you tried N variants and kept the best, its Sharpe would look like the curve even
# with **zero** skill. With σ = 0.5 and 100 trials the best is ~1.26 by luck alone. Compare your chosen
# variant's Sharpe with the curve at your actual N (from the research log below), on the same per-period
# scale as σ.

# %%
LOG = stats.ResearchLog(store.LOG_PATH)
n_trials_before = LOG.n_trials()
daily = mo.groupby(maker.loc[mo.index, "ts"].floordiv(3600 * 10 ** 9)).mean()  # hourly here; use daily with more data
n_trials = max(n_trials_before, 10)  # <- use your real number of trials
d = stats.deflated_sharpe(daily.to_numpy(), n_trials=n_trials, sr_std_trials=0.2)
psr0 = stats.probabilistic_sharpe(d["sr"], 0.0, d["n"], d["skew"], d["kurt"])
print(f"per-period SR {d['sr']:.3f} over n={d['n']}; PSR(SR*=0) = {psr0:.3f}; "
      f"SR* after {n_trials} trials = {d['sr_star']:.3f}; DSR = {d['dsr']:.3f}")

# %% [markdown]
# **How to read this.** PSR is the probability that the true Sharpe exceeds 0, given the sample length,
# skew and fat tails. DSR replaces 0 with the best Sharpe you would expect from your number of trials by luck.
# A PSR of 0.97 with a DSR of 0.4 means the result is plausibly the winner of a search, not an edge.
# `sr_std_trials` should be the spread of Sharpes across the variants you tried, so record them.

# %% [markdown]
# ## Block bootstrap and Benjamini–Hochberg

# %%
bb = stats.block_bootstrap(mo.to_numpy(), block=50, n_boot=1000)
plots.hist(bb["draws"], bins=50, title="Block-bootstrap distribution of the mean net markout", xlabel="bps", vline=0.0)
plt.show()
print(f"mean {bb['estimate']:.3f} bps, 95% CI [{bb['lo']:.3f}, {bb['hi']:.3f}]")

# %% [markdown]
# **How to read this.** Blocks keep autocorrelated runs together, so the interval is honest about
# dependence. Choose the block length near the autocorrelation length (e.g. the horizon divided by the typical
# spacing between fills). If 0 is inside the interval, the mean markout is not distinguishable from zero.

# %%
# p-values for "mean markout differs from 0" in each hour of the day: a typical many-tests situation
hours = pd.to_datetime(maker.loc[mo.index, "ts"], utc=True).dt.hour.to_numpy()
rows = []
for h in np.unique(hours):
    x = mo.to_numpy()[hours == h]
    se = stats.newey_west_se(x, 10)
    rows.append({"hour": h, "n": len(x), "mean": x.mean(), "p": 2 * sps.norm.sf(abs(x.mean() / se))})
ph = pd.DataFrame(rows)
ph = ph.join(stats.benjamini_hochberg(ph["p"].to_numpy(), q=0.10)[["q_value", "reject"]])
ph.round(4)

# %% [markdown]
# **How to read this.** With 24 hourly tests, a couple of "significant" hours at p < 0.05 are expected by
# chance. Benjamini–Hochberg controls the false discovery rate: only rows with `reject = True` survive at
# q = 10%. Report the q-value, not the raw p.

# %% [markdown]
# ## Research log
# Every test you run gets a row: hypothesis, parameters, result. The trial count feeds the deflated Sharpe
# above. Never delete failed tests from the log.

# %%
n_now = LOG.log("passive MM 60s net markout > 0", {"horizon_s": 60, "fills": "maker", "fees": "net"},
                {"mean_bps": round(float(mo.mean()), 4), "ci": [round(bb["lo"], 4), round(bb["hi"], 4)], "dsr": round(d["dsr"], 4)})
print(f"logged to {LOG.path}; total trials now {n_now}")
LOG.read().tail()
