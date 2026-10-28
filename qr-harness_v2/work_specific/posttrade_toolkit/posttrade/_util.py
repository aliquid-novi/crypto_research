"""Shared low-level helpers: timestamp conversion, as-of joins, sessions, instrument lookups.

As-of joins are done with numpy.searchsorted per (venue, symbol) group rather than
pandas.merge_asof: identical semantics, no global sort requirement, and the direction /
exactness rules are explicit in one place.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config

NS = 1_000_000_000


def to_ns(x: pd.Series | np.ndarray | Iterable, unit: str = "ns", naive_tz: str = "UTC") -> np.ndarray:
    """Convert timestamps to int64 nanoseconds since epoch, UTC.

    Accepts integers (interpreted in ``unit``: 's','ms','us','ns'), floats (same), tz-aware or
    naive datetimes (naive are localised to ``naive_tz``) and ISO strings.
    """
    s = pd.Series(x) if not isinstance(x, pd.Series) else x
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        mult = {"s": NS, "ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
        v = s.to_numpy(dtype="float64") * mult if pd.api.types.is_float_dtype(s) else s.to_numpy(dtype="int64") * mult
        return np.asarray(v).astype("int64")
    dt = pd.to_datetime(s, utc=False)
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(naive_tz)
    dt = dt.dt.tz_convert("UTC")
    return dt.dt.tz_localize(None).astype("datetime64[ns]").to_numpy().astype("int64")


def ns_to_dt(ns: np.ndarray | pd.Series) -> pd.DatetimeIndex:
    """int64 ns -> tz-aware UTC DatetimeIndex (for plotting / display only)."""
    return pd.to_datetime(np.asarray(ns, dtype="int64"), utc=True)


def asof_idx(query_ts: np.ndarray, ref_ts: np.ndarray, direction: str = "backward",
             allow_exact: bool = True) -> np.ndarray:
    """Index into sorted ``ref_ts`` for each query time; -1 where no match.

    backward, allow_exact=True : last ref with ref_ts <= q
    backward, allow_exact=False: last ref with ref_ts <  q
    forward,  allow_exact=True : first ref with ref_ts >= q
    forward,  allow_exact=False: first ref with ref_ts >  q
    """
    q = np.asarray(query_ts)
    r = np.asarray(ref_ts)
    if direction == "backward":
        idx = np.searchsorted(r, q, side="right" if allow_exact else "left") - 1
        idx[idx < 0] = -1
    elif direction == "forward":
        idx = np.searchsorted(r, q, side="left" if allow_exact else "right")
        idx[idx >= len(r)] = -1
    else:
        raise ValueError("direction must be 'backward' or 'forward'")
    return idx


def group_keys(df: pd.DataFrame, by: tuple[str, ...] = ("venue", "symbol")) -> pd.Series:
    """Single string key per row for grouping on several columns (fast hashing)."""
    if len(by) == 1:
        return df[by[0]].astype(str)
    out = df[by[0]].astype(str)
    for c in by[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def asof_values(query: pd.DataFrame, ref: pd.DataFrame, value_cols: list[str], ts_query: np.ndarray | None = None,
                direction: str = "backward", allow_exact: bool = True,
                by: tuple[str, ...] = ("venue", "symbol"), ref_ts_col: str = "ts") -> pd.DataFrame:
    """As-of join of ``ref[value_cols]`` onto ``query`` rows, per ``by`` group.

    ``ts_query`` overrides query['ts'] (e.g. ts + h). Returns a DataFrame aligned to query.index
    with the value columns plus ``<ref_ts_col>_ref`` (matched ref ts, NaN-able int as float) and
    ``age_ns`` = query ts - matched ts (positive for backward joins). No interpolation, ever.
    """
    qts = np.asarray(query["ts"].to_numpy() if ts_query is None else ts_query, dtype="int64")
    out = {c: np.full(len(query), np.nan) for c in value_cols}
    mts = np.full(len(query), np.nan)
    qk = group_keys(query, by).to_numpy()
    rk = group_keys(ref, by).to_numpy()
    ref_groups = pd.Series(np.arange(len(ref))).groupby(rk).indices
    q_groups = pd.Series(np.arange(len(query))).groupby(qk).indices
    rts_all = ref[ref_ts_col].to_numpy(dtype="int64")
    vals_all = {c: ref[c].to_numpy(dtype="float64") for c in value_cols}
    for key, qi in q_groups.items():
        ri = ref_groups.get(key)
        if ri is None:
            continue
        rts = rts_all[ri]
        order = np.argsort(rts, kind="stable")
        ri = ri[order]
        rts = rts[order]
        j = asof_idx(qts[qi], rts, direction, allow_exact)
        ok = j >= 0
        for c in value_cols:
            out[c][qi[ok]] = vals_all[c][ri][j[ok]]
        mts[qi[ok]] = rts[j[ok]]
    res = pd.DataFrame(out, index=query.index)
    res[f"{ref_ts_col}_ref"] = mts
    res["age_ns"] = qts - mts
    return res


# ---------------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------------
def _hhmm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def closure_intervals(spec: config.SessionSpec | None, start_ns: int, end_ns: int) -> np.ndarray:
    """Closed intervals [start, end) in UTC ns overlapping [start_ns, end_ns]. Shape (k, 2)."""
    if spec is None:
        return np.zeros((0, 2), dtype="int64")
    tz = spec.tz
    d0 = pd.Timestamp(start_ns, tz="UTC").tz_convert(tz).normalize().tz_localize(None) - pd.Timedelta(days=1)
    d1 = pd.Timestamp(end_ns, tz="UTC").tz_convert(tz).normalize().tz_localize(None) + pd.Timedelta(days=1)
    days = pd.date_range(d0, d1, freq="D")
    hol = set(spec.holidays)
    rows = []
    for day in days:
        wd = day.weekday()
        if day.strftime("%Y-%m-%d") in hol:
            rows.append((day, day + pd.Timedelta(days=1)))
            continue
        for w, a, b in spec.closed:
            if w == wd:
                rows.append((day + pd.Timedelta(minutes=_hhmm_to_min(a)), day + pd.Timedelta(minutes=_hhmm_to_min(b))))
    if not rows:
        return np.zeros((0, 2), dtype="int64")
    loc = pd.DataFrame(rows, columns=["a", "b"]).sort_values("a")
    a = loc["a"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=False).dt.tz_convert("UTC")
    b = loc["b"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=False).dt.tz_convert("UTC")
    iv = np.column_stack([a.dt.tz_localize(None).astype("datetime64[ns]").astype("int64"),
                          b.dt.tz_localize(None).astype("datetime64[ns]").astype("int64")])
    # merge touching windows (e.g. Fri 16:00-24:00 + Sat + Sun 00:00-17:00)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    m = np.array(merged, dtype="int64")
    return m[(m[:, 1] >= start_ns) & (m[:, 0] <= end_ns)]


def session_key(ts_ns: np.ndarray, closures: np.ndarray) -> np.ndarray:
    """Session id per timestamp: number of closures that ended at or before ts.

    Two timestamps are in the same session iff their keys match and neither is inside a closure.
    Returns float array; NaN for timestamps inside a closure.
    """
    ts = np.asarray(ts_ns, dtype="int64")
    if len(closures) == 0:
        return np.zeros(len(ts))
    k = np.searchsorted(closures[:, 1], ts, side="right").astype(float)
    # inside closure i if closures[i,0] <= ts < closures[i,1]; i = number of closures with end<=ts
    i = k.astype(int)
    inside = np.zeros(len(ts), dtype=bool)
    valid = i < len(closures)
    inside[valid] = ts[valid] >= closures[i[valid], 0]
    k[inside] = np.nan
    return k


def same_session_mask(venue: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """True where t0 and t1 lie in the same open session of their venue (always True for 24/7)."""
    venue = np.asarray(venue)
    out = np.ones(len(t0), dtype=bool)
    if len(t0) == 0:
        return out
    for v in pd.unique(venue):
        spec = config.venue_spec(str(v)).sessions
        if spec is None:
            continue
        m = venue == v
        lo = int(min(np.nanmin(t0[m]), np.nanmin(t1[m])))
        hi = int(max(np.nanmax(t0[m]), np.nanmax(t1[m])))
        cl = closure_intervals(spec, lo, hi)
        k0 = session_key(t0[m], cl)
        k1 = session_key(t1[m], cl)
        out[m] = (k0 == k1) & ~np.isnan(k0)
    return out


# ---------------------------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------------------------
def instrument_lookup(df: pd.DataFrame, instruments: pd.DataFrame | None, col: str,
                      default: float | None) -> np.ndarray:
    """Per-row value of instruments[col] matched on (venue, symbol); ``default`` where missing."""
    out = np.full(len(df), np.nan if default is None else float(default))
    if instruments is None or col not in instruments.columns or len(instruments) == 0:
        return out
    m = dict(zip(zip(instruments["venue"].astype(str), instruments["symbol"].astype(str)),
                 instruments[col].astype(float)))
    keys = list(zip(df["venue"].astype(str), df["symbol"].astype(str)))
    vals = np.array([m.get(k, np.nan) for k in keys], dtype=float)
    return np.where(np.isnan(vals), out, vals)


def infer_tick(prices: np.ndarray, decimals: int = 10) -> float:
    """Smallest positive increment among distinct prices (rounded), as a tick-size guess."""
    p = np.unique(np.round(np.asarray(prices, dtype=float), decimals))
    d = np.diff(p)
    d = d[d > 0]
    return float(np.round(d.min(), decimals)) if len(d) else np.nan


def bps(x: np.ndarray | pd.Series, ref: np.ndarray | pd.Series) -> np.ndarray:
    """1e4 * x / ref."""
    return 1e4 * np.asarray(x, dtype=float) / np.asarray(ref, dtype=float)
