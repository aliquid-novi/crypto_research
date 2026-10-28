"""Spread dynamics: OU fit and half-life, model vs empirical alpha decay, causal Kalman hedge ratio.

Time units are seconds throughout; spreads are in whatever units you pass (price or log).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._util import NS


@dataclass
class OUFit:
    """x_{t+dt} = a + b x_t + e  <=>  OU with theta = -ln(b)/dt, mu = a/(1-b),
    sigma (diffusion) = sd(e) * sqrt(2 theta / (1 - b^2)). half_life = ln 2 / theta (seconds)."""

    theta: float
    mu: float
    sigma: float
    half_life_s: float
    b: float
    se_b: float
    n: int
    dt_s: float


def ou_fit(x: np.ndarray | pd.Series, dt_s: float, lag: int = 1) -> OUFit:
    """Fit an OU process to an evenly sampled series via AR OLS of x_{t+lag} on x_t.

    ``dt_s`` = sampling interval in seconds (resample irregular data with hygiene.resample_ffill
    first). theta = -ln(b) / (lag * dt_s), in 1/seconds. If b >= 1 there is no mean reversion (theta NaN).

    Measurement noise (bid/ask bounce, tick rounding, legs observed at slightly different times)
    attenuates b and BIASES THETA UP by about ln(1 + noise_var / spread_var) / (lag * dt_s).
    With lag=1 on a noisy spread this can be large; use a lag of a few percent of the half-life
    (e.g. lag * dt_s ~ 60 s for a 5 min half-life) and check theta is stable across lags.
    """
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    y, z = v[lag:], v[:-lag]
    dt_s = dt_s * lag
    Z = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    a, b = coef
    e = y - Z @ coef
    s2 = e @ e / (len(y) - 2)
    se_b = float(np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[1, 1]))
    if not 0 < b < 1:
        return OUFit(np.nan, np.nan, np.nan, np.nan, float(b), se_b, len(y), dt_s)
    theta = -np.log(b) / dt_s
    sigma = np.sqrt(s2) * np.sqrt(2 * theta / (1 - b * b))
    return OUFit(float(theta), float(a / (1 - b)), float(sigma), float(np.log(2) / theta), float(b), se_b, len(y), dt_s)


def model_alpha_decay(deviation: float | np.ndarray, theta: float, horizons_s) -> np.ndarray:
    """Expected reversion captured by horizon t for a deviation D from the mean:
    D * (1 - exp(-theta t)). Same units as D."""
    h = np.asarray(horizons_s, float)
    return np.outer(np.atleast_1d(deviation), 1 - np.exp(-theta * h)).squeeze()


def empirical_alpha_decay(series: pd.DataFrame, horizons_s, signal_col: str = "z", value_col: str = "value",
                          mu: float = 0.0, n_bins: int = 4, min_abs_signal: float = 0.0) -> pd.DataFrame:
    """Empirical reversion by horizon per signal bucket on an evenly sampled spread.

    For each row t with |signal| >= min_abs_signal: captured_h = -sign(x_t - mu) * (x_{t+h} - x_t)
    (positive = the spread moved back toward mu). Buckets = quantiles of |signal|. Returns long
    DataFrame[bucket, horizon_s, mean_captured, mean_deviation, n] with the model curve
    D(1 - e^{-theta t}) to compare using the bucket's mean deviation D = |x_t - mu|.
    Uses future values only as the OUTCOME, never as an input.
    """
    s = series.sort_values("ts").reset_index(drop=True)
    ts = s["ts"].to_numpy(dtype="int64")
    x = s[value_col].to_numpy(dtype=float)
    sig = s[signal_col].to_numpy(dtype=float)
    dev = x - mu
    keep = np.abs(sig) >= min_abs_signal
    b = pd.qcut(np.abs(sig[keep]), n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)], duplicates="drop")
    out = []
    for h in horizons_s:
        j = np.searchsorted(ts, ts + int(h * NS), side="left")
        ok = j < len(ts)
        fut = np.full(len(x), np.nan)
        fut[ok] = x[j[ok]]
        cap = -np.sign(dev) * (fut - x)
        df = pd.DataFrame({"bucket": np.asarray(b), "cap": cap[keep], "dev": np.abs(dev[keep])})
        g = df.groupby("bucket", observed=True)
        r = pd.DataFrame({"mean_captured": g["cap"].mean(), "mean_deviation": g["dev"].mean(), "n": g["cap"].count()})
        out.append(r.reset_index().assign(horizon_s=float(h)))
    return pd.concat(out, ignore_index=True)


def alpha_decay_regression(series: pd.DataFrame, horizons_s, signal_col: str = "z", value_col: str = "value",
                           nw_lags: int | None = None, sample_spacing_s: float | None = None) -> pd.DataFrame:
    """Per horizon h, regress the forward paper PnL of a unit spread position, -(x_{t+h} - x_t),
    on the signal: pnl_h = a + b_h * signal + e, Newey-West SE (lags default = overlap length).

    b_h traces how much of the signal's predictive power has been realised by h; the curve
    flattening tells you the horizon beyond which holding adds nothing. Returns [horizon_s, coef, se, t, n].
    """
    from .stats import nw_lags_for_overlap, ols_nw

    s = series.sort_values("ts").reset_index(drop=True)
    ts = s["ts"].to_numpy(dtype="int64")
    x = s[value_col].to_numpy(dtype=float)
    sig = s[signal_col].to_numpy(dtype=float)
    spacing = sample_spacing_s or float(np.median(np.diff(ts)) / NS)
    rows = []
    for h in horizons_s:
        j = np.searchsorted(ts, ts + int(h * NS), side="left")
        ok = j < len(ts)
        y = np.full(len(x), np.nan)
        y[ok] = -(x[j[ok]] - x[ok])
        lags = nw_lags if nw_lags is not None else nw_lags_for_overlap(h, spacing)
        m = np.isfinite(y) & np.isfinite(sig)
        res = ols_nw(y[m], sig[m], lags)
        rows.append({"horizon_s": float(h), "coef": float(res.params[1]), "se": float(res.bse[1]),
                     "t": float(res.tvalues[1]), "n": int(m.sum())})
    return pd.DataFrame(rows)


def kalman_hedge_ratio(y: np.ndarray, x: np.ndarray, q: float | tuple[float, float] = 1e-5, r: float = 1e-3,
                       init_beta: float = 0.0, init_alpha: float = 0.0, init_var: float = 1.0) -> pd.DataFrame:
    """Time-varying regression y_t = alpha_t + beta_t x_t + e_t, random-walk states, Kalman filter.

    q = state noise variance per step, scalar or (q_alpha, q_beta) - scale it with your units
    (beta moves by ~sqrt(q_beta) per step), r = obs
    noise variance. NO LOOK-AHEAD: ``beta_prior``/``alpha_prior`` at t use observations up to t-1
    only - use these for trading/hedging decisions at t. ``beta_post`` also uses y_t (fine for
    description, not for decisions at t). ``innov`` = y_t - prior prediction; ``innov_var`` its
    variance (z = innov / sqrt(innov_var) is a causal spread z-score).
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    th = np.array([init_alpha, init_beta], float)
    P = np.eye(2) * init_var
    Qm = np.diag(np.broadcast_to(np.asarray(q, float), (2,)))
    out = np.zeros((n, 6))
    for t in range(n):
        P = P + Qm
        H = np.array([1.0, x[t]])
        pred = H @ th
        S = H @ P @ H + r
        out[t, 0:2] = th
        out[t, 4] = y[t] - pred
        out[t, 5] = S
        if np.isfinite(y[t]) and np.isfinite(x[t]):
            K = P @ H / S
            th = th + K * (y[t] - pred)
            P = P - np.outer(K, H) @ P
        out[t, 2:4] = th
    return pd.DataFrame(out, columns=["alpha_prior", "beta_prior", "alpha_post", "beta_post", "innov", "innov_var"])
