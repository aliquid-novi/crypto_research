"""Latency: per-hop decomposition, latency vs time of day / activity / volatility, latency as a
regressor for markouts and shortfall, and restart discontinuities.

Hops (all optional except sent->ack): md_recv -> signal -> sent -> ack. All in milliseconds.
Timestamps must be on ONE clock for a hop to mean anything: exchange ack times minus local send
times mix clocks (see hygiene.ntp_offsets). The functions do not fix that for you.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import NS, asof_values

HOPS = [("md_to_signal_ms", "ts_md_recv", "ts_signal"), ("signal_to_sent_ms", "ts_signal", "ts_sent"),
        ("sent_to_ack_ms", "ts_sent", "ts_ack")]


def hop_latencies(orders: pd.DataFrame) -> pd.DataFrame:
    """Per-order hop latencies in ms for every hop whose two timestamps exist, plus total_ms.

    hop = (ts_b - ts_a) / 1e6. Negative hops mean clock mismatch or bad data - they are kept (and
    counted in ``negative_hops``) so you can see them, not dropped.
    """
    out = pd.DataFrame(index=orders.index)
    for name, a, b in HOPS:
        if a in orders.columns and b in orders.columns:
            out[name] = (orders[b].astype("float64") - orders[a].astype("float64")) / 1e6
    out["total_ms"] = out.sum(axis=1, min_count=1)
    out["negative_hops"] = (out.drop(columns="total_ms") < 0).sum(axis=1)
    out["ts"] = orders["ts_sent"].to_numpy()
    return out


def summarise_by(lat: pd.DataFrame, key: pd.Series | np.ndarray, quantiles=(0.5, 0.9, 0.99)) -> pd.DataFrame:
    """Quantiles of each hop by a grouping key (e.g. hour of day, activity bucket)."""
    cols = [c for c in lat.columns if c.endswith("_ms")]
    g = lat[cols].groupby(np.asarray(key))
    return pd.concat({f"q{int(q * 100)}": g.quantile(q) for q in quantiles}, axis=1)


def hour_of_day(ts_ns: np.ndarray | pd.Series, tz: str = "UTC") -> np.ndarray:
    """Hour of day (0-23) in ``tz`` for int64 ns timestamps."""
    return pd.to_datetime(np.asarray(ts_ns, dtype="int64"), utc=True).tz_convert(tz).hour.to_numpy()


def message_rate(at_ts: np.ndarray, events_ts: np.ndarray, window_s: float = 60.0) -> np.ndarray:
    """Trailing message rate (events per second) over (t - window, t] at each time. Causal."""
    e = np.sort(np.asarray(events_ts, dtype="int64"))
    t = np.asarray(at_ts, dtype="int64")
    hi = np.searchsorted(e, t, side="right")
    lo = np.searchsorted(e, t - int(window_s * NS), side="right")
    return (hi - lo) / window_s


def attach_order_latency(fills: pd.DataFrame, orders: pd.DataFrame, lat: pd.DataFrame, col: str = "sent_to_ack_ms") -> np.ndarray:
    """Latency of the order behind each fill (matched on order_id). NaN where no order is found."""
    m = pd.Series(lat[col].to_numpy(), index=orders["order_id"].to_numpy())
    m = m[~m.index.duplicated()]
    return fills["order_id"].map(m).to_numpy(dtype=float)


def latency_regression(y: np.ndarray, latency_ms: np.ndarray, controls: pd.DataFrame | None, nw_lags: int = 10) -> pd.DataFrame:
    """OLS of an outcome (markout or shortfall, bps) on latency with controls (vol, activity...),
    Newey-West SE. Without controls, latency picks up whatever co-moves with it (busy, volatile
    periods are both slower and more toxic). Returns coefficient table [coef, se, t, lo, hi]."""
    from .stats import ols_nw

    X = pd.DataFrame({"latency_ms": np.asarray(latency_ms, float)})
    if controls is not None:
        X = pd.concat([X, controls.reset_index(drop=True)], axis=1)
    m = np.isfinite(np.asarray(y, float)) & np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
    r = ols_nw(np.asarray(y, float)[m], X[m].astype(float), nw_lags)
    return pd.DataFrame({"coef": r.params, "se": r.bse, "t": r.tvalues, "lo": r.conf_int()[0], "hi": r.conf_int()[1]})


def restart_discontinuity(ts_ns: np.ndarray, y: np.ndarray, restarts_ns: list[int], window_s: float = 1800.0) -> pd.DataFrame:
    """Local comparison around each restart: mean of y in (restart - window, restart) vs
    [restart, restart + window), difference with a Welch t-stat. Also a local-linear jump estimate
    (separate lines each side evaluated at the restart). Optional input: restart timestamps."""
    from scipy import stats as sps

    t = np.asarray(ts_ns, dtype="int64")
    y = np.asarray(y, float)
    rows = []
    w = int(window_s * NS)
    for r in restarts_ns:
        b = (t >= r - w) & (t < r) & np.isfinite(y)
        a = (t >= r) & (t < r + w) & np.isfinite(y)
        if b.sum() < 5 or a.sum() < 5:
            rows.append({"restart": r, "n_before": int(b.sum()), "n_after": int(a.sum())})
            continue
        tt = sps.ttest_ind(y[a], y[b], equal_var=False)
        jump = np.polyval(np.polyfit((t[a] - r) / NS, y[a], 1), 0) - np.polyval(np.polyfit((t[b] - r) / NS, y[b], 1), 0)
        rows.append({"restart": r, "n_before": int(b.sum()), "n_after": int(a.sum()), "mean_before": y[b].mean(),
                     "mean_after": y[a].mean(), "diff": y[a].mean() - y[b].mean(), "t": tt.statistic, "p": tt.pvalue,
                     "local_linear_jump": jump})
    return pd.DataFrame(rows)


def vol_at(fills_or_orders: pd.DataFrame, vol: pd.DataFrame, ts: np.ndarray | None = None) -> np.ndarray:
    """Trailing vol (reference.trailing_vol output) at each row, strictly before ts."""
    return asof_values(fills_or_orders, vol, ["vol_bps"], ts_query=ts, direction="backward", allow_exact=False)["vol_bps"].to_numpy()
