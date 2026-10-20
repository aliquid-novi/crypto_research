"""Truncation test + data hygiene checks.

truncation_test(fn, data, times): run `fn(data, times)` on the full data, then again with every event that was
not yet known at a cut deleted (and only the times <= cut). Every output row at or before the cut must be identical.
A feature that peeks forward, normalises on the full sample, fits on everything, uses a centred window, or ignores
latency changes when the future is removed, and is caught.

`fn` must contain the *whole* estimator (fitting, priors, normalisation). That's CLAUDE.md rule 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pit import known_times


class LeakageError(AssertionError):
    pass


@dataclass
class Report:
    passed: bool
    cuts: list = field(default_factory=list)
    detail: list = field(default_factory=list)

    def __str__(self):
        head = "PASSED" if self.passed else "FAILED"
        return f"truncation test {head} at {len(self.cuts)} cuts\n" + "\n".join(self.detail)


def _truncate(data: dict, cut, latency, ts_col):
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame) and ts_col in v.columns:
            out[k] = v.loc[(known_times(v, latency, ts_col) <= cut).to_numpy()].copy()
        else:
            out[k] = v
    return out


def _as_frame(x, times):
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(np.asarray(x), index=times[: len(x)])
    return x


def _probe_cuts(data, times, latency, ts_col, n):
    """Cuts that sit exactly on a decision time with an event in (t - latency, t]: the only place where ignoring
    latency changes a feature. Without these, a latency leak is caught only by luck."""
    if pd.Timedelta(latency) <= pd.Timedelta(0):
        return []
    tt = times.to_numpy()
    hits = np.zeros(len(tt), bool)
    for v in data.values():
        if isinstance(v, pd.DataFrame) and ts_col in v.columns:
            r = np.sort(pd.to_datetime(v[ts_col]).to_numpy())
            lo = np.searchsorted(r, (times - pd.Timedelta(latency)).to_numpy(), side="right")
            hi = np.searchsorted(r, tt, side="right")
            hits |= hi > lo
    idx = np.where(hits)[0]
    idx = idx[(idx > len(tt) * 0.2) & (idx < len(tt) * 0.95)]
    return list(times[idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]]) if len(idx) else []


def truncation_test(fn, data: dict, times, latency="0ms", ts_col: str = "ts_recv",
                    fracs=(0.3, 0.5, 0.7, 0.9), n_probe: int = 8, rtol: float = 1e-9, atol: float = 1e-12,
                    raise_on_fail: bool = True) -> Report:
    """Raise LeakageError (or return a failed Report) if any output at time <= cut changes when the future is
    removed. `latency` is the delay at which the pipeline *claims* events become known."""
    times = pd.DatetimeIndex(pd.to_datetime(times))
    full = _as_frame(fn(data, times), times)
    if len(full) != len(times):
        raise LeakageError(f"fn returned {len(full)} rows for {len(times)} times; return one row per time")
    full.index = times
    cuts = [times[int(f * (len(times) - 1))] for f in fracs] + _probe_cuts(data, times, latency, ts_col, n_probe)
    rep = Report(passed=True)
    for cut in cuts:
        keep = times <= cut
        part = _as_frame(fn(_truncate(data, cut, latency, ts_col), times[keep]), times[keep])
        part.index = times[keep]
        a = full.loc[keep].to_numpy(dtype=float)
        b = part.reindex(columns=full.columns).to_numpy(dtype=float)
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        rep.cuts.append(cut)
        if bad.any():
            rows, cols_ = np.where(bad)
            first = times[keep][rows[0]]
            rep.passed = False
            rep.detail.append(f"cut {cut}: {bad.sum()} values changed; first at {first} in column "
                              f"'{full.columns[cols_[0]]}' ({a[rows[0], cols_[0]]!r} -> {b[rows[0], cols_[0]]!r}); "
                              f"changed columns: {sorted(set(full.columns[cols_]))}")
    if not rep.passed and raise_on_fail:
        raise LeakageError(str(rep))
    return rep


# ---------------------------------------------------------------- hygiene

def hygiene(df: pd.DataFrame, ts_exch: str = "ts_exch", ts_recv: str = "ts_recv", name: str = "") -> dict:
    """Cheap checks before anything else: ordering, duplicates, negative feed latency, gaps."""
    out = {"name": name, "rows": len(df)}
    if ts_recv in df:
        r = pd.to_datetime(df[ts_recv])
        out["recv_backsteps"] = int((r.diff() < pd.Timedelta(0)).sum())
        g = r.sort_values().diff().dropna()
        out["max_gap_s"] = float(g.max().total_seconds()) if len(g) else np.nan
    if ts_exch in df:
        e = pd.to_datetime(df[ts_exch])
        out["exch_backsteps"] = int((e.diff() < pd.Timedelta(0)).sum())
    if ts_exch in df and ts_recv in df:
        lat = (pd.to_datetime(df[ts_recv]) - pd.to_datetime(df[ts_exch])).dt.total_seconds() * 1e3
        out |= {"latency_ms_min": float(lat.min()), "latency_ms_median": float(lat.median()),
                "latency_ms_p99": float(lat.quantile(0.99)), "negative_latency_rows": int((lat < 0).sum())}
    out["duplicate_rows"] = int(df.duplicated().sum())
    return out
