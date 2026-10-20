"""Uncertainty helpers: moving-block bootstrap and the deflated Sharpe ratio."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def block_bootstrap(x, stat=np.mean, block: int = 50, reps: int = 1000, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap distribution of stat(x) for time-ordered, autocorrelated x."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.full(reps, np.nan)
    block = int(max(1, min(block, n)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    if stat is np.mean:
        return x[idx].mean(axis=1)
    return np.array([stat(x[i]) for i in idx])


def block_bootstrap_ci(x, stat=np.mean, block: int = 50, reps: int = 1000, level: float = 0.95, seed: int = 0):
    d = block_bootstrap(x, stat, block, reps, seed)
    a = (1 - level) / 2
    return float(np.nanquantile(d, a)), float(np.nanquantile(d, 1 - a))


def sharpe(returns, periods_per_year: float = 1.0) -> float:
    r = np.asarray(returns, float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] across n_trials independent zero-skill trials (Bailey & Lopez de Prado 2014)."""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    return float(np.sqrt(var_sr) * ((1 - g) * st.norm.ppf(1 - 1 / n_trials) + g * st.norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns, n_trials: int, var_sr: float | None = None, sr_benchmark: float | None = None) -> dict:
    """Probability that the true (per-period) Sharpe exceeds the best you'd expect from n_trials of noise.
    returns: per-period returns of the chosen variant. var_sr: variance of SR across the trials (if unknown,
    1/T is used, the zero-skill sampling variance)."""
    r = np.asarray(returns, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = st.skew(r), st.kurtosis(r, fisher=False)
    var_sr = 1.0 / T if var_sr is None else var_sr
    sr0 = expected_max_sharpe(n_trials, var_sr) if sr_benchmark is None else sr_benchmark
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return {"sr": float(sr), "sr_benchmark": float(sr0), "dsr": float(st.norm.cdf(z)), "n_trials": int(n_trials), "T": T}
