"""Research statistics: Newey-West, sample sizes, Sharpe inference, bootstrap, multiple testing, research log.

Sharpe ratios here are PER PERIOD (per observation) unless a function says otherwise; convert
with SR_ann = SR_period * sqrt(periods_per_year).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------------------------
# Newey-West
# ---------------------------------------------------------------------------------------------
def nw_lags_for_overlap(horizon_s: float, sample_spacing_s: float) -> int:
    """Default NW lag for overlapping horizons: ceil(horizon / spacing) (the overlap length), >= 1."""
    if sample_spacing_s <= 0:
        return 1
    return max(int(np.ceil(abs(horizon_s) / sample_spacing_s)), 1)


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the sample mean with Bartlett weights.

    Var(mean) = [g0 + 2 sum_{k=1..L} (1 - k/(L+1)) g_k] / n, g_k = autocovariance at lag k.
    Units: same as x.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(s, 0) / n))


def ols_nw(y: np.ndarray, X: np.ndarray | pd.DataFrame, lags: int, add_const: bool = True):
    """OLS with Newey-West (HAC, Bartlett) covariance via statsmodels. Returns the fitted results."""
    import statsmodels.api as sm

    Xm = sm.add_constant(X, has_constant="add") if add_const else X
    return sm.OLS(np.asarray(y, float), Xm, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})


# ---------------------------------------------------------------------------------------------
# Sample size / Sharpe inference
# ---------------------------------------------------------------------------------------------
def sharpe_tstat(sr: float, n: int) -> float:
    """t-stat of a per-period Sharpe ratio over n periods: t = SR * sqrt(n) (iid, normal approx)."""
    return float(sr * np.sqrt(n))


def required_n(mu: float, sigma: float, z: float = 1.96) -> float:
    """Observations needed for the mean mu to be z standard errors from 0: n >= (z sigma / mu)^2."""
    return float((z * sigma / mu) ** 2)


def effective_n_cluster(n: int, cluster_size: float, icc: float) -> float:
    """Effective sample size with clustering: n_eff = n / (1 + (m - 1) rho), m = mean cluster size,
    rho = intra-cluster correlation."""
    return float(n / (1 + (cluster_size - 1) * icc))


def effective_n_autocorr(x: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size under autocorrelation: n / (1 + 2 sum_k rho_k), summed until rho_k <= 0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    n = len(x)
    max_lag = max_lag or min(n // 4, 1000)
    v = x @ x
    s = 0.0
    for k in range(1, max_lag + 1):
        r = (x[k:] @ x[:-k]) / v
        if r <= 0:
            break
        s += r
    return float(n / (1 + 2 * s))


def expected_max_sharpe(n_trials: int, sr_std: float, sr_mean: float = 0.0) -> float:
    """Expected maximum Sharpe of N independent no-skill trials (Bailey & Lopez de Prado):

        E[max SR] ~ mean + std * [(1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))], g = Euler-Mascheroni.

    Units: those of ``sr_std``. Example: N=100, std=0.5 -> ~1.26.
    """
    N = float(n_trials)
    return float(sr_mean + sr_std * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1 / N)
                                     + EULER_GAMMA * sps.norm.ppf(1 - 1 / (N * np.e))))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR = Phi( (SR - SR*) sqrt(n - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) ).

    SR per period, n observations, kurt = raw (non-excess) kurtosis. Probability the true SR > SR*.
    """
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / den))


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_std_trials: float) -> dict:
    """Deflated Sharpe ratio: PSR against SR* = expected max Sharpe of ``n_trials`` no-skill trials.

    ``sr_std_trials`` = std of the per-period Sharpes across the trials you ran. Returns dict with
    sr, sr_star, dsr, n, skew, kurt.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    sr = r.mean() / r.std(ddof=1)
    sk = float(sps.skew(r))
    ku = float(sps.kurtosis(r, fisher=False))
    sr_star = expected_max_sharpe(n_trials, sr_std_trials)
    return {"sr": float(sr), "sr_star": sr_star, "dsr": probabilistic_sharpe(sr, sr_star, len(r), sk, ku),
            "n": len(r), "skew": sk, "kurt": ku}


# ---------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------
def block_bootstrap(x: np.ndarray, stat=np.mean, block: int = 20, n_boot: int = 2000, ci: float = 0.95,
                    seed: int = 0) -> dict:
    """Circular moving-block bootstrap CI for ``stat`` of a time series.

    Returns dict(estimate, lo, hi, se, draws). Block length ~ the autocorrelation length.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, (n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    draws = np.array([stat(x[i]) for i in idx])
    a = (1 - ci) / 2
    return {"estimate": float(stat(x)), "lo": float(np.quantile(draws, a)), "hi": float(np.quantile(draws, 1 - a)),
            "se": float(draws.std(ddof=1)), "draws": draws}


def cluster_bootstrap_mean(values: np.ndarray, weights: np.ndarray | None = None, clusters: np.ndarray | None = None,
                           n_boot: int = 500, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Weighted mean of each column of ``values`` (n x H) with cluster-bootstrap CIs.

    Clusters (e.g. time buckets, decision ids) are resampled with replacement; the mean is
    sum(w x) / sum(w) over non-NaN entries. Returns DataFrame[mean, lo, hi, se, n] per column.
    """
    V = np.asarray(values, float)
    if V.ndim == 1:
        V = V[:, None]
    n, H = V.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    cl = np.arange(n) if clusters is None else pd.factorize(np.asarray(clusters))[0]
    nc = cl.max() + 1
    ok = np.isfinite(V)
    WX = np.where(ok, V * w[:, None], 0.0)
    W = np.where(ok, w[:, None], 0.0)
    SX = np.zeros((nc, H))
    SW = np.zeros((nc, H))
    np.add.at(SX, cl, WX)
    np.add.at(SW, cl, W)
    est = SX.sum(0) / np.where(SW.sum(0) > 0, SW.sum(0), np.nan)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(nc, np.full(nc, 1 / nc), size=n_boot).astype(float)  # (B, nc)
    bx = cnt @ SX
    bw = cnt @ SW
    draws = bx / np.where(bw > 0, bw, np.nan)
    a = (1 - ci) / 2
    return pd.DataFrame({"mean": est, "lo": np.nanquantile(draws, a, axis=0), "hi": np.nanquantile(draws, 1 - a, axis=0),
                         "se": np.nanstd(draws, axis=0, ddof=1), "n": ok.sum(0)})


# ---------------------------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------------------------
def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR. Returns DataFrame[p, rank, threshold, q_value, reject] in input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, int)
    ranks[order] = np.arange(1, m + 1)
    thr = ranks / m * q
    passed = p[order] <= thr[order]
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    reject = ranks <= k
    qv_sorted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.minimum(qv_sorted, 1)
    return pd.DataFrame({"p": p, "rank": ranks, "threshold": thr, "q_value": qv, "reject": reject})


# ---------------------------------------------------------------------------------------------
# Research log
# ---------------------------------------------------------------------------------------------
class ResearchLog:
    """Append-only CSV of every test you run: timestamp, hypothesis, params (JSON), result (JSON).

    ``n_trials()`` = number of rows (optionally for one hypothesis) - feed it to deflated_sharpe.
    Never edit the file by hand to remove failed tests; that is the point of it.
    """

    FIELDS = ["ts_utc", "hypothesis", "params", "result", "note"]

    def __init__(self, path: str):
        self.path = path

    def log(self, hypothesis: str, params: dict, result: dict, note: str = "") -> int:
        new = not os.path.exists(self.path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow({"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hypothesis": hypothesis,
                        "params": json.dumps(params, default=str, sort_keys=True),
                        "result": json.dumps(result, default=str, sort_keys=True), "note": note})
        return self.n_trials()

    def read(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.FIELDS)
        return pd.read_csv(self.path)

    def n_trials(self, hypothesis: str | None = None) -> int:
        df = self.read()
        return int(len(df) if hypothesis is None else (df["hypothesis"] == hypothesis).sum())
