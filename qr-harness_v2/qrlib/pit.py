"""Point-in-time joins, targets and walk-forward splits.

The one rule: a feature at decision time t may only use events that were *known* by t,
where known = ts_recv + latency (strictly backward, never forward or nearest).

Targets are the only place the future is allowed, and they live here so notebooks never
need a negative shift. Target columns are prefixed `target_` and are refused as features.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TARGET_PREFIX = "target_"


def _td(x) -> pd.Timedelta:
    return x if isinstance(x, pd.Timedelta) else pd.Timedelta(x)


def known_times(events: pd.DataFrame, latency="0ms", ts_col: str = "ts_recv") -> pd.Series:
    """When each event becomes usable: ts_col + latency, made monotone (an event is never known before an
    earlier one in arrival order)."""
    if ts_col == "ts_exch":
        warnings.warn("joining on ts_exch breaks CLAUDE.md rule 3 (features join on ts_recv + latency)", stacklevel=3)
    k = pd.to_datetime(events[ts_col]) + _td(latency)
    return pd.Series(np.maximum.accumulate(k.to_numpy()), index=events.index)


def asof_features(t, events: pd.DataFrame, cols, latency="0ms", ts_col: str = "ts_recv") -> pd.DataFrame:
    """Last known value of `cols` at each time in `t`. An event is known at t iff ts_col + latency <= t.
    Returns a frame indexed by t (NaN before the first known event)."""
    cols = list(cols)
    bad = [c for c in cols if str(c).startswith(TARGET_PREFIX)]
    if bad:
        raise ValueError(f"target columns can't be features: {bad}")
    t = pd.DatetimeIndex(pd.to_datetime(t))
    order = np.argsort(pd.to_datetime(events[ts_col]).to_numpy(), kind="stable")
    ev = events.iloc[order]
    k = known_times(ev, latency, ts_col).to_numpy()
    i = np.searchsorted(k, t.to_numpy(), side="right") - 1
    out = pd.DataFrame(index=t)
    for c in cols:
        v = ev[c].to_numpy()
        col = np.where(i >= 0, v[np.clip(i, 0, None)], np.nan) if v.dtype.kind in "fiu" else \
            pd.Series(v[np.clip(i, 0, None)]).where(i >= 0).to_numpy()
        out[c] = col
    return out


def window_count(t, events: pd.DataFrame, window, latency="0ms", ts_col: str = "ts_recv") -> pd.Series:
    """Number of events known in (t - window, t]."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    k = np.sort(known_times(events, latency, ts_col).to_numpy())
    hi = np.searchsorted(k, t.to_numpy(), side="right")
    lo = np.searchsorted(k, (t - _td(window)).to_numpy(), side="right")
    return pd.Series(hi - lo, index=t)


def make_target(t, events: pd.DataFrame, value_col: str, horizon, ts_col: str = "ts_recv", name: str | None = None) -> pd.Series:
    """Future value for evaluation only: the last `value_col` known at t + horizon. Named target_*."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    fut = asof_features(t + _td(horizon), events, [value_col], ts_col=ts_col)[value_col].to_numpy()
    return pd.Series(fut, index=t, name=name or f"{TARGET_PREFIX}{value_col}_{horizon}")


def walk_forward(t, n_folds: int = 4, min_train_frac: float = 0.4, embargo="0s"):
    """Expanding-window folds over sorted times: yields (train_mask, test_mask). `embargo` drops test rows within
    that distance after the train end (use >= the target horizon)."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    t0, t1 = t.min(), t.max()
    span = t1 - t0
    edges = [t0 + span * (min_train_frac + (1 - min_train_frac) * k / n_folds) for k in range(n_folds + 1)]
    for a, b in zip(edges[:-1], edges[1:]):
        train = np.asarray(t <= a)
        test = np.asarray((t > a + _td(embargo)) & (t <= b))
        yield train, test


def chrono_split(t, train_frac: float = 0.6):
    """Single chronological split: (train_mask, test_mask, split_time)."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    cut = t.min() + (t.max() - t.min()) * train_frac
    return np.asarray(t <= cut), np.asarray(t > cut), cut
