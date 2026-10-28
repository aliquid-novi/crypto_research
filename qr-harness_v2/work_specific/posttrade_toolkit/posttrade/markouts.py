"""Markouts: wall-clock and event-time, multiple references, decomposition, taker-view spreads,
conditional curves with bootstrap CIs, and day-by-day processing for large fill sets.

Sign convention: d = +1 our buy, -1 our sell.  markout_h = d * (m_{t+h} - p), in bps of p
(= bps of notional), positive = good for us.

Join rules (fixed here, used by every function):
* h > 0  : m_{t+h} = first reference update STRICTLY AFTER t+h  (forward, allow_exact=False).
           Conservative: never uses a quote that could pre-date the horizon.
           ``future_join='backward'`` switches to "last update at or before t+h".
* h <= 0 : m_{t+h} = last reference update STRICTLY BEFORE t+h (backward, allow_exact=False),
           so m_t (h=0) is the book before our fill, not the book our fill created.
* A horizon whose t and t+h fall in different trading sessions (config.VENUES[...].sessions,
  e.g. across the CME daily halt or a weekend) is NaN, never carried across the closure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from ._util import NS, asof_idx, asof_values, group_keys, same_session_mask
from .stats import cluster_bootstrap_mean


def _ref_at(fills: pd.DataFrame, ref: pd.DataFrame, ts_q: np.ndarray, h: float, future_join: str) -> np.ndarray:
    if h > 0:
        direction, exact = ("forward", False) if future_join == "forward" else ("backward", True)
    else:
        direction, exact = "backward", False
    return asof_values(fills, ref, ["value"], ts_query=ts_q, direction=direction, allow_exact=exact)["value"].to_numpy()


def ref_at_fill(fills: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    """m_t: reference value strictly before each fill ts (pre-fill reference), price units."""
    return _ref_at(fills, ref, fills["ts"].to_numpy(dtype="int64"), 0.0, "forward")


def compute_markouts(fills: pd.DataFrame, ref: pd.DataFrame, horizons_s=None, future_join: str = "forward",
                     ts_col: str = "ts") -> pd.DataFrame:
    """Wall-clock markouts in bps for every fill and horizon.

        markout_h = 1e4 * d * (m_{t+h} - p) / p

    fills: needs ts, venue, symbol, side, price. ref: reference series [ts, venue, symbol, value]
    keyed by the fills' (venue, symbol). ``ts_col`` lets you anchor at another time (e.g. decision
    ts). Returns DataFrame (index = fills.index, columns = horizons in seconds, float bps).
    """
    hs = config.HORIZONS_S if horizons_s is None else np.asarray(horizons_s, float)
    t = fills[ts_col].to_numpy(dtype="int64")
    d = fills["side"].to_numpy(dtype=float)
    p = fills["price"].to_numpy(dtype=float)
    venue = fills["venue"].to_numpy()
    q = fills.assign(ts=t) if ts_col != "ts" else fills
    out = {}
    for h in hs:
        tq = t + int(round(h * NS))
        m = _ref_at(q, ref, tq, h, future_join)
        mo = 1e4 * d * (m - p) / p
        if h != 0:
            mo[~same_session_mask(venue, t, tq)] = np.nan
        out[float(h)] = mo
    return pd.DataFrame(out, index=fills.index)


def compute_markouts_multi(fills: pd.DataFrame, refs: dict[str, pd.DataFrame], horizons_s=None,
                           future_join: str = "forward") -> pd.DataFrame:
    """Markouts (bps) against several references in one call. Columns = MultiIndex (ref, horizon)."""
    parts = {name: compute_markouts(fills, r, horizons_s, future_join) for name, r in refs.items()}
    return pd.concat(parts, axis=1, names=["ref", "horizon_s"])


def decompose(fills: pd.DataFrame, ref: pd.DataFrame, horizons_s=None, future_join: str = "forward") -> dict:
    """Split each markout into spread captured and adverse move:

        spread_captured = 1e4 * d * (m_t - p) / p
        adverse_h       = 1e4 * d * (m_{t+h} - m_t) / p
        markout_h       = spread_captured + adverse_h      (asserted exactly)

    Returns dict(markout=DataFrame, spread_captured=Series, adverse=DataFrame, m_t=Series). bps.
    """
    mo = compute_markouts(fills, ref, horizons_s, future_join)
    d = fills["side"].to_numpy(dtype=float)
    p = fills["price"].to_numpy(dtype=float)
    mt = ref_at_fill(fills, ref)
    sc = 1e4 * d * (mt - p) / p
    adv = mo.sub(sc, axis=0)
    ok = np.isfinite(mo.to_numpy())
    recon = adv.to_numpy() + sc[:, None]
    assert np.allclose(recon[ok], mo.to_numpy()[ok], atol=1e-9), "decomposition identity failed"
    return {"markout": mo, "spread_captured": pd.Series(sc, index=fills.index), "adverse": adv,
            "m_t": pd.Series(mt, index=fills.index)}


def fee_bps(fills: pd.DataFrame) -> np.ndarray:
    """Fee in bps of notional: 1e4 * fee / notional. Positive = cost, negative = rebate."""
    return 1e4 * fills["fee"].to_numpy(dtype=float) / fills["notional"].to_numpy(dtype=float)


def net_of_fees(mo: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Markout after fees (bps): markout_h - fee_bps. A rebate (negative fee) increases it."""
    return mo.sub(fee_bps(fills), axis=0)


def vol_scaled(mo: pd.DataFrame, fills: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Vol-scaled markout: markout_h / (sigma_t * sqrt(|h|)), dimensionless (h=0 -> NaN).

    vol: output of reference.trailing_vol [ts, venue, symbol, vol_bps] (bps per sqrt second),
    joined strictly backward at the fill time, so only pre-fill information is used.
    """
    s = asof_values(fills, vol, ["vol_bps"], direction="backward", allow_exact=False)["vol_bps"].to_numpy()
    hs = np.asarray(mo.columns, float)
    den = s[:, None] * np.sqrt(np.abs(hs))[None, :]
    den[:, hs == 0] = np.nan
    return pd.DataFrame(mo.to_numpy() / den, index=mo.index, columns=mo.columns)


def aggregate_events(fills: pd.DataFrame, keys: tuple[str, ...] = ("venue", "symbol", "ts", "side")) -> pd.DataFrame:
    """Collapse fills from one aggressor event (same keys, e.g. a taker sweeping several of our
    levels or partial fills of one order at one timestamp) into one row.

    price = qty-weighted average price; qty, notional, fee summed; n_fills counted. Use this
    before averaging markouts so one sweep does not count as several independent observations.
    """
    k = list(keys)
    f = fills.assign(_pq=fills["price"] * fills["qty"])
    agg = {"qty": "sum", "notional": "sum", "fee": "sum", "_pq": "sum", "fill_id": "first"}
    for c in ("liquidity", "counterparty", "order_id"):
        if c in f.columns and c not in k:
            agg[c] = "first"
    g = f.groupby(k, sort=False).agg(agg)
    g["n_fills"] = f.groupby(k, sort=False).size()
    g["price"] = g["_pq"] / g["qty"]
    return g.drop(columns="_pq").reset_index().sort_values("ts", kind="stable").reset_index(drop=True)


def markout_curve(mo: pd.DataFrame, weights: np.ndarray | None = None, clusters: np.ndarray | None = None,
                  n_boot: int = 500, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Average markout per horizon with cluster-bootstrap CI.

    weights: e.g. notional for a notional-weighted curve (None = per-fill average).
    clusters: resampling unit, e.g. ``time_clusters(fills, 60)`` so overlapping fills that share
    price paths are not treated as independent. Returns DataFrame index=horizon [mean, lo, hi, se, n].
    """
    res = cluster_bootstrap_mean(mo.to_numpy(), weights, clusters, n_boot, ci, seed)
    res.index = pd.Index(mo.columns, name="horizon_s")
    return res


def time_clusters(fills: pd.DataFrame, cluster_s: float = 60.0) -> np.ndarray:
    """Cluster label = floor(ts / cluster_s). Use for bootstrap resampling of overlapping markouts."""
    return (fills["ts"].to_numpy(dtype="int64") // int(cluster_s * NS)).astype("int64")


def quantile_buckets(x: pd.Series | np.ndarray, n_bins: int = 5) -> pd.Series:
    """Quantile bins of x labelled 'Q1'..'Qn' (NaN stays NaN). Ties collapse bins (duplicates='drop')."""
    s = pd.Series(np.asarray(x, float))
    b = pd.qcut(s, n_bins, duplicates="drop")
    labels = {iv: f"Q{i + 1}" for i, iv in enumerate(b.cat.categories)}
    return b.map(labels).astype(object)


def conditional_curves(mo: pd.DataFrame, bucket: pd.Series | np.ndarray, weights: np.ndarray | None = None,
                       clusters: np.ndarray | None = None, n_boot: int = 300, ci: float = 0.95) -> pd.DataFrame:
    """Markout curve per bucket (any label column, e.g. ``quantile_buckets(feature)``) with CIs.

    Returns long DataFrame [bucket, horizon_s, mean, lo, hi, se, n].
    """
    b = np.asarray(bucket, dtype=object)
    out = []
    for lab in sorted(pd.unique(b[pd.notna(b)]), key=str):
        m = b == lab
        r = markout_curve(mo[m], None if weights is None else np.asarray(weights)[m],
                          None if clusters is None else np.asarray(clusters)[m], n_boot, ci)
        out.append(r.reset_index().assign(bucket=lab))
    return pd.concat(out, ignore_index=True)[["bucket", "horizon_s", "mean", "lo", "hi", "se", "n"]]


# ---------------------------------------------------------------------------------------------
# Event time
# ---------------------------------------------------------------------------------------------
def mid_change_events(ref: pd.DataFrame) -> pd.DataFrame:
    """Rows of a reference series where the value changed (per venue/symbol) -> event series."""
    out = []
    for _, g in ref.groupby(group_keys(ref), sort=False):
        g = g.sort_values("ts", kind="stable")
        v = g["value"].to_numpy()
        ch = np.concatenate([[False], v[1:] != v[:-1]])
        out.append(g[ch][["ts", "venue", "symbol"]])
    return pd.concat(out, ignore_index=True)


def event_time_markouts(fills: pd.DataFrame, ref: pd.DataFrame, events: pd.DataFrame, n_list,
                        size_col: str | None = None) -> pd.DataFrame:
    """Markouts (bps) after the next n events instead of h seconds.

    events: [ts, venue, symbol] (+ ``size_col`` for volume time), keyed like the fills.
      * next n market trades       : events = market trades, size_col=None
      * next n mid changes         : events = mid_change_events(ref)
      * next v units of volume     : events = market trades, size_col='qty', n_list = volumes
    The target time is the ts of the n-th event strictly after the fill (or the event at which
    cumulative volume after the fill first reaches v); m = reference value at or before that time
    (it includes the event's own update). Returns DataFrame columns = n_list.
    """
    t = fills["ts"].to_numpy(dtype="int64")
    d = fills["side"].to_numpy(dtype=float)
    p = fills["price"].to_numpy(dtype=float)
    out = {float(n): np.full(len(fills), np.nan) for n in n_list}
    ek = group_keys(events).to_numpy()
    e_idx = pd.Series(np.arange(len(events))).groupby(ek).indices
    fk = group_keys(fills).to_numpy()
    for key, fi in pd.Series(np.arange(len(fills))).groupby(fk).indices.items():
        ei = e_idx.get(key)
        if ei is None:
            continue
        ets = events["ts"].to_numpy(dtype="int64")[ei]
        o = np.argsort(ets, kind="stable")
        ets = ets[o]
        i0 = np.searchsorted(ets, t[fi], side="right")  # first event strictly after fill
        if size_col is not None:
            cv = np.cumsum(events[size_col].to_numpy(dtype=float)[ei][o])
            base = np.where(i0 > 0, cv[np.maximum(i0 - 1, 0)], 0.0)
        for n in n_list:
            if size_col is None:
                j = i0 + int(n) - 1
            else:
                j = np.searchsorted(cv, base + float(n), side="left")
            ok = j < len(ets)
            tq = np.full(len(fi), -1, dtype="int64")
            tq[ok] = ets[j[ok]]
            sub = fills.iloc[fi]
            m = asof_values(sub, ref, ["value"], ts_query=tq, direction="backward", allow_exact=True)["value"].to_numpy()
            m[~ok] = np.nan
            out[float(n)][fi] = 1e4 * d[fi] * (m - p[fi]) / p[fi]
    return pd.DataFrame(out, index=fills.index)


# ---------------------------------------------------------------------------------------------
# Taker view
# ---------------------------------------------------------------------------------------------
def taker_spreads(trades: pd.DataFrame, ref: pd.DataFrame, horizon_s: float, taker_side: np.ndarray | None = None,
                  future_join: str = "forward") -> pd.DataFrame:
    """Effective / realised spread and price impact from the TAKER's view, bps of m_t:

        effective = 2 q (p - m_t) / m_t
        realised  = 2 q (p - m_{t+h}) / m_t
        impact    = 2 q (m_{t+h} - m_t) / m_t
        effective = realised + impact   (asserted)

    q = taker direction (+1 taker bought). Default: ``aggressor_side`` if present, else for our
    fills q = -side for maker fills and +side for taker fills. A large realised spread means
    liquidity providers kept the spread; large impact means takers were informed.
    """
    if taker_side is None:
        if "aggressor_side" in trades.columns:
            taker_side = trades["aggressor_side"].to_numpy(dtype=float)
        else:
            s = trades["side"].to_numpy(dtype=float)
            taker_side = np.where(trades["liquidity"].to_numpy() == "maker", -s, s)
    q = np.asarray(taker_side, float)
    p = trades["price"].to_numpy(dtype=float)
    t = trades["ts"].to_numpy(dtype="int64")
    mt = _ref_at(trades, ref, t, 0.0, future_join)
    tq = t + int(round(horizon_s * NS))
    mh = _ref_at(trades, ref, tq, horizon_s, future_join)
    mh[~same_session_mask(trades["venue"].to_numpy(), t, tq)] = np.nan
    eff = 2e4 * q * (p - mt) / mt
    real = 2e4 * q * (p - mh) / mt
    imp = 2e4 * q * (mh - mt) / mt
    ok = np.isfinite(eff + real + imp)
    assert np.allclose(eff[ok], real[ok] + imp[ok], atol=1e-9), "effective != realised + impact"
    return pd.DataFrame({"effective_bps": eff, "realised_bps": real, "impact_bps": imp}, index=trades.index)


# ---------------------------------------------------------------------------------------------
# Day-by-day for large fill sets
# ---------------------------------------------------------------------------------------------
def daily_markout_means(fills: pd.DataFrame, ref: pd.DataFrame, horizons_s=None, weight_col: str | None = "notional",
                        net_fees: bool = False, future_join: str = "forward") -> pd.DataFrame:
    """Process one UTC day at a time; returns per-day weighted mean markout (bps) per horizon plus
    columns ``n`` and ``weight``. Memory is bounded by the largest day, so millions of fills are fine.

    Combine days with ``combine_daily`` (weighted by day weight); use the day-level series for
    CUSUM / day-clustered CIs.
    """
    hs = config.HORIZONS_S if horizons_s is None else np.asarray(horizons_s, float)
    day = fills["ts"].to_numpy(dtype="int64") // (86400 * NS)
    pad_lo, pad_hi = int((abs(min(hs.min(), 0)) + 60) * NS), int((max(hs.max(), 0) + 3600) * NS)
    rts = ref["ts"].to_numpy(dtype="int64")
    rows = []
    for dd in np.unique(day):
        f = fills[day == dd]
        lo, hi = int(dd) * 86400 * NS - pad_lo, (int(dd) + 1) * 86400 * NS + pad_hi
        r = ref[(rts >= lo) & (rts <= hi)]
        mo = compute_markouts(f, r, hs, future_join)
        if net_fees:
            mo = net_of_fees(mo, f)
        w = f[weight_col].to_numpy(dtype=float) if weight_col else np.ones(len(f))
        M = mo.to_numpy()
        ok = np.isfinite(M)
        num = np.where(ok, M * w[:, None], 0).sum(0)
        den = np.where(ok, w[:, None], 0).sum(0)
        row = dict(zip(hs, num / np.where(den > 0, den, np.nan)))
        row.update(day=pd.Timestamp(int(dd) * 86400 * NS, tz="UTC").date(), n=len(f), weight=float(w.sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("day")


def combine_daily(daily: pd.DataFrame) -> pd.Series:
    """Weight-averaged curve across days from ``daily_markout_means`` output."""
    hs = [c for c in daily.columns if c not in ("n", "weight")]
    w = daily["weight"].to_numpy()
    return pd.Series({h: np.nansum(daily[h].to_numpy() * w) / w[np.isfinite(daily[h].to_numpy())].sum() for h in hs})
