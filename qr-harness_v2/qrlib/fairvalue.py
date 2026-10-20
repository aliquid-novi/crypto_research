"""Fair-value baselines and a common scorer.

Every fair value is a function of the top of book known at t. `evaluate` scores any fair value the same way:
forecast error of the future mid at a horizon, skill vs the mid with a block-bootstrap CI, and a calibration slope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .pit import asof_features, make_target

L1_COLS = ["bid_px", "ask_px", "bid_sz", "ask_sz"]


def l1_at(quotes: pd.DataFrame, t, latency="0ms", ts_col: str = "ts_recv") -> pd.DataFrame:
    """Top of book known at each t (bid_px, ask_px, bid_sz, ask_sz)."""
    return asof_features(t, quotes, L1_COLS, latency=latency, ts_col=ts_col)


def mid(book: pd.DataFrame) -> pd.Series:
    return (book["bid_px"] + book["ask_px"]) / 2


def weighted_mid(book: pd.DataFrame) -> pd.Series:
    """Size-weighted touch: I*ask + (1-I)*bid with I = Qb / (Qb + Qa)."""
    imb = book["bid_sz"] / (book["bid_sz"] + book["ask_sz"])
    return imb * book["ask_px"] + (1 - imb) * book["bid_px"]


def imbalance(book: pd.DataFrame) -> pd.Series:
    return book["bid_sz"] / (book["bid_sz"] + book["ask_sz"])


def evaluate(fv: pd.DataFrame | pd.Series, quotes: pd.DataFrame, t, horizons=("250ms", "500ms", "1s"),
             latency="0ms", ts_col: str = "ts_recv", boot_block: int = 600, reps: int = 500, seed: int = 0) -> pd.DataFrame:
    """Score fair-value columns against the mid known at t + h (target), relative to the mid known at t.
    skill = 1 - MSE(fv) / MSE(mid); CI from a moving-block bootstrap over time-ordered rows."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    fv = fv.to_frame() if isinstance(fv, pd.Series) else fv
    m0 = mid(l1_at(quotes, t, latency, ts_col)).to_numpy()
    qm = quotes.assign(_mid=(quotes["bid_px"] + quotes["ask_px"]) / 2)
    rows = []
    for h in horizons:
        fut = make_target(t, qm, "_mid", h, ts_col=ts_col).to_numpy()
        e_mid = fut - m0
        for c in fv.columns:
            f = fv[c].to_numpy(dtype=float)
            ok = np.isfinite(f) & np.isfinite(e_mid) & np.isfinite(m0)
            e = fut[ok] - f[ok]
            base = np.mean(e_mid[ok] ** 2)
            d = (e_mid[ok] ** 2 - e ** 2) / base
            lo, hi = stats.block_bootstrap_ci(d, block=boot_block, reps=reps, seed=seed)
            sig = f[ok] - m0[ok]
            slope = float(np.sum(sig * e_mid[ok]) / np.sum(sig ** 2)) if np.sum(sig ** 2) > 0 else np.nan
            rows.append({"horizon": h, "fair value": c, "n": int(ok.sum()), "skill vs mid": float(d.mean()),
                         "CI lo": lo, "CI hi": hi, "calib. slope": slope,
                         "share mid moved": float(np.mean(e_mid[ok] != 0))})
    return pd.DataFrame(rows).set_index(["horizon", "fair value"])
