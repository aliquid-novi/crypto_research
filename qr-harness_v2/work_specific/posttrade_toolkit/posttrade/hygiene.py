"""Data hygiene: reconciliation, fee/funding recompute, dedupe, bad ticks, book sanity, feed gaps,
robust flags, CUSUM, microstructure noise, clock sync, staleness and forward-fill resampling.

Rule: hygiene functions FLAG, they never silently delete or alter values. The one exception is
``dedupe_fills``, which returns both the deduplicated table and the removed rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from ._util import NS, asof_values, closure_intervals, group_keys, instrument_lookup


# ---------------------------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------------------------
def cumulative_position(fills: pd.DataFrame, at: pd.DataFrame) -> np.ndarray:
    """Position implied by fills at each row of ``at`` [ts, venue, symbol]: sum(side * qty) over
    fills with ts <= at.ts (same venue/symbol). Units: qty units (contracts / coins)."""
    f = fills.sort_values("ts", kind="stable").assign(_sq=lambda x: x["side"] * x["qty"])
    f = f.assign(_cum=f.groupby(group_keys(f))["_sq"].cumsum())
    res = asof_values(at, f, ["_cum"], direction="backward", allow_exact=True)["_cum"].to_numpy()
    return np.nan_to_num(res, nan=0.0)


def reconcile_positions(fills: pd.DataFrame, positions: pd.DataFrame, tol: float = 1e-6,
                        anchor_first: bool = True) -> pd.DataFrame:
    """Compare venue position snapshots with the cumulative position from our fills.

    break = snapshot - (offset + cum_fills), offset = first snapshot - cum_fills at that time
    (``anchor_first``; covers positions opened before your fill history starts), else 0.
    Returns positions with columns implied, offset, break, flag (|break| > tol). A break that
    appears and persists marks a missing/duplicated fill near the first flagged snapshot.
    """
    p = positions.sort_values("ts", kind="stable").reset_index(drop=True).copy()
    p["implied"] = cumulative_position(fills, p)
    first = p.groupby(group_keys(p))[["position", "implied"]].transform("first")
    p["offset"] = (first["position"] - first["implied"]) if anchor_first else 0.0
    p["break"] = p["position"] - (p["offset"] + p["implied"])
    p["flag"] = p["break"].abs() > tol
    return p


def recompute_fees(fills: pd.DataFrame, fee_schedule: dict[str, dict] | None = None,
                   instruments: pd.DataFrame | None = None, tol_bps: float = 0.05,
                   venue_models: dict[str, str] | None = None) -> pd.DataFrame:
    """Recompute each fill's fee from a schedule and compare with the reported fee.

    Schedule entries (config.FEE_SCHEDULES): {'model': 'bps', 'maker': x, 'taker': y} -> fee =
    notional * rate / 1e4;  {'model': 'per_contract', ...} -> fee = qty * rate. Rates negative =
    rebate. Venue -> schedule via ``venue_models`` or config.VENUES[venue].fee_model.
    Returns DataFrame[fill_id, venue, liquidity, fee, fee_expected, diff, diff_bps, flag].
    """
    sched = fee_schedule or config.FEE_SCHEDULES
    exp = np.full(len(fills), np.nan)
    v = fills["venue"].astype(str).to_numpy()
    liq = fills["liquidity"].to_numpy()
    for ven in pd.unique(v):
        model_key = (venue_models or {}).get(ven) or config.venue_spec(ven).fee_model
        if model_key is None or model_key not in sched:
            continue
        s = sched[model_key]
        m = v == ven
        rate = np.where(liq[m] == "maker", float(s["maker"]), float(s["taker"]))
        if s["model"] == "bps":
            exp[m] = fills["notional"].to_numpy(dtype=float)[m] * rate / 1e4
        elif s["model"] == "per_contract":
            exp[m] = fills["qty"].to_numpy(dtype=float)[m] * rate
        else:
            raise ValueError(f"unknown fee model {s['model']!r}")
    out = fills[["fill_id", "venue", "symbol", "liquidity", "notional", "fee"]].copy()
    out["fee_expected"] = exp
    out["diff"] = out["fee"] - exp
    out["diff_bps"] = 1e4 * out["diff"] / out["notional"]
    out["flag"] = out["diff_bps"].abs() > tol_bps
    return out


def recompute_funding(funding: pd.DataFrame, fills: pd.DataFrame | None = None,
                      instruments: pd.DataFrame | None = None, tol: float = 1e-6) -> pd.DataFrame:
    """Recompute funding payments: expected = -position * mark * rate * multiplier.

    Sign: rate > 0 means longs pay; ``payment`` is our cash (+ received). If ``fills`` is given,
    the position is recomputed from fills (catches ledger positions that disagree with fills),
    otherwise the ledger's ``position_at_settlement`` is used. Returns funding with
    position_used, payment_expected, diff, flag.
    """
    f = funding.copy()
    mult = instrument_lookup(f, instruments, "multiplier", config.DEFAULT_MULTIPLIER)
    f["position_used"] = cumulative_position(fills, f) if fills is not None else f["position_at_settlement"]
    f["payment_expected"] = -f["position_used"] * f["mark"] * f["rate"] * mult
    f["diff"] = f["payment"] - f["payment_expected"]
    f["flag"] = f["diff"].abs() > tol + 1e-9 * f["payment_expected"].abs()
    return f


def dedupe_fills(fills: pd.DataFrame, keys: tuple[str, ...] = ("venue", "trade_id", "order_id", "side")
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicate fills on ``keys`` (keeps first). Returns (deduped, removed_rows).

    Falls back to (venue, order_id, ts, price, qty, side) when trade_id is absent.
    """
    k = [c for c in keys if c in fills.columns]
    if "trade_id" not in fills.columns or fills["trade_id"].isna().all():
        k = [c for c in ("venue", "order_id", "ts", "price", "qty", "side") if c in fills.columns]
    dup = fills.duplicated(subset=k, keep="first")
    return fills[~dup].reset_index(drop=True), fills[dup]


# ---------------------------------------------------------------------------------------------
# Price / book checks
# ---------------------------------------------------------------------------------------------
def brownlees_gallo(prices: pd.Series | np.ndarray, k: int = 50, gamma: float = 0.0, trim: float = 0.1,
                    n_sd: float = 3.0, chunk: int = 200_000) -> np.ndarray:
    """Brownlees-Gallo (2006) outlier flag for one instrument's trade prices (time-ordered).

    For each i, take the k nearest neighbours (k/2 each side, excluding i), compute the trimmed
    mean and sd (``trim`` fraction cut from each tail); flag if |p_i - mean| >= n_sd * sd + gamma.
    gamma (price units, ~ a few ticks) stops flagging when the neighbourhood has zero variance.
    Uses a CENTRED window: fine for flagging bad data, never use it to build features.
    Returns boolean array (True = suspicious).
    """
    p = np.asarray(prices, float)
    n = len(p)
    h = k // 2
    pad = np.concatenate([np.full(h, np.nan), p, np.full(h, np.nan)])
    flags = np.zeros(n, dtype=bool)
    cut = int(np.floor(trim * k))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        win = np.lib.stride_tricks.sliding_window_view(pad[s: e + 2 * h], 2 * h + 1)
        nb = np.delete(win, h, axis=1)
        nb = np.sort(nb, axis=1)  # NaN sort to the end
        cnt = np.sum(np.isfinite(nb), axis=1)
        out_mean = np.full(e - s, np.nan)
        out_sd = np.full(e - s, np.nan)
        for c in np.unique(cnt):
            r = cnt == c
            lo, hi = cut, c - cut
            if hi - lo < 3:
                continue
            seg = nb[r, lo:hi]
            out_mean[r] = seg.mean(1)
            out_sd[r] = seg.std(1, ddof=1)
        flags[s:e] = np.abs(p[s:e] - out_mean) >= n_sd * out_sd + gamma
    return flags


def bad_ticks(trades: pd.DataFrame, k: int = 50, gamma_ticks: float = 2.0,
              instruments: pd.DataFrame | None = None) -> pd.Series:
    """Brownlees-Gallo flag per (venue, symbol), gamma = gamma_ticks * tick size (inferred if needed)."""
    from ._util import infer_tick

    out = pd.Series(False, index=trades.index)
    for _, g in trades.groupby(group_keys(trades), sort=False):
        g = g.sort_values("ts", kind="stable")
        tick = instrument_lookup(g.head(1), instruments, "tick_size", None)[0]
        tick = tick if np.isfinite(tick) else infer_tick(g["price"].to_numpy())
        out.loc[g.index] = brownlees_gallo(g["price"].to_numpy(), k=k, gamma=gamma_ticks * tick)
    return out


def book_sanity(book: pd.DataFrame, instruments: pd.DataFrame | None = None, tick_tol: float = 1e-6) -> pd.DataFrame:
    """Flag crossed (bid > ask), locked (bid == ask), non-positive sizes/prices and off-tick prices.

    Off-tick needs a tick size: instruments.tick_size, else inferred from the instrument's prices
    (an inferred tick cannot detect off-tick prints reliably - supply tick sizes for real data).
    Returns DataFrame of boolean columns aligned to book.index plus ``any``.
    """
    from .reference import tick_sizes

    ticks = tick_sizes(book, instruments)
    tk = np.array([ticks.get((v, s), np.nan) for v, s in zip(book["venue"], book["symbol"])])
    def off(px):
        r = px / tk
        return np.abs(r - np.round(r)) > tick_tol * np.maximum(1, np.abs(r)) + 1e-9
    out = pd.DataFrame({
        "crossed": book["bid_px"] > book["ask_px"],
        "locked": book["bid_px"] == book["ask_px"],
        "nonpos_size": (book["bid_sz"] <= 0) | (book["ask_sz"] <= 0),
        "nonpos_price": (book["bid_px"] <= 0) | (book["ask_px"] <= 0),
        "off_tick": off(book["bid_px"].to_numpy()) | off(book["ask_px"].to_numpy()),
    }, index=book.index)
    out["any"] = out.any(axis=1)
    return out


def feed_gaps(events: pd.DataFrame, rate_window_s: float = 300.0, alpha: float = 1e-6,
              min_gap_s: float = 1.0) -> pd.DataFrame:
    """Detect suspicious silences per venue/symbol.

    For each inter-arrival gap g starting at t, the trailing message rate lambda = (#messages in
    (t - W, t]) / W. Under Poisson arrivals P(silence >= g) = exp(-lambda g). Flag when < alpha.
    Silence inside a scheduled closure (config.VENUES[venue].sessions) is not a gap.
    Returns DataFrame[venue, symbol, start, end, gap_s, rate_per_s, p_silence].
    """
    rows = []
    W = int(rate_window_s * NS)
    for (ven, sym), g in events.groupby(["venue", "symbol"], sort=False):
        ts = np.sort(g["ts"].to_numpy(dtype="int64"))
        if len(ts) < 3:
            continue
        gap = np.diff(ts)
        start = ts[:-1]
        cnt = np.arange(1, len(ts)) - np.searchsorted(ts, start - W, side="right")
        elapsed = np.minimum(W, start - ts[0]).astype(float)
        rate = cnt / np.maximum(elapsed / NS, 1e-9)
        p = np.exp(-rate * gap / NS)
        m = (p < alpha) & (gap >= min_gap_s * NS) & (elapsed >= W / 4)
        cl = closure_intervals(config.venue_spec(str(ven)).sessions, int(ts[0]), int(ts[-1]))
        for i in np.flatnonzero(m):
            a, b = start[i], ts[i + 1]
            if len(cl) and np.any((cl[:, 0] < b) & (cl[:, 1] > a)):
                continue
            rows.append({"venue": ven, "symbol": sym, "start": a, "end": b, "gap_s": gap[i] / NS,
                         "rate_per_s": rate[i], "p_silence": p[i]})
    return pd.DataFrame(rows, columns=["venue", "symbol", "start", "end", "gap_s", "rate_per_s", "p_silence"])


def mad_flags(x: pd.Series | np.ndarray, thresh: float = 5.0) -> pd.DataFrame:
    """Robust z = 0.6745 (x - median) / MAD; flag |z| > thresh. Flags only - nothing is removed."""
    v = np.asarray(x, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    z = 0.6745 * (v - med) / mad if mad > 0 else np.zeros_like(v)
    return pd.DataFrame({"robust_z": z, "flag": np.abs(z) > thresh})


def cusum(x: pd.Series, k: float = 0.5, h: float = 4.0, baseline_n: int | None = None) -> pd.DataFrame:
    """Two-sided CUSUM on a (daily) series standardised by a baseline mean/sd.

    z_t = (x_t - mu0)/sd0;  S+_t = max(0, S+_{t-1} + z_t - k);  S-_t = max(0, S-_{t-1} - z_t - k).
    Alarm when S+ or S- > h. Baseline = first ``baseline_n`` points (default: all - sample is
    then only descriptive). Use on daily mean markout to spot a regime change in toxicity.
    """
    v = np.asarray(x, float)
    base = v[: baseline_n] if baseline_n else v
    mu, sd = np.nanmean(base), np.nanstd(base, ddof=1)
    z = (v - mu) / sd
    sp = np.zeros(len(v))
    sn = np.zeros(len(v))
    for i in range(len(v)):
        zi = 0.0 if np.isnan(z[i]) else z[i]
        sp[i] = max(0.0, (sp[i - 1] if i else 0.0) + zi - k)
        sn[i] = max(0.0, (sn[i - 1] if i else 0.0) - zi - k)
    return pd.DataFrame({"x": v, "z": z, "s_pos": sp, "s_neg": sn, "alarm": (sp > h) | (sn > h)}, index=x.index)


# ---------------------------------------------------------------------------------------------
# Microstructure noise
# ---------------------------------------------------------------------------------------------
def signature(ref: pd.DataFrame, intervals_s=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600)) -> pd.DataFrame:
    """Realised variance per day vs sampling interval (signature plot input) for ONE series.

    RV(dt) = sum of squared log returns sampled every dt (forward-filled grid). Noise inflates RV
    at small dt. Also returns n (returns per sample) and omega2 = RV/(2n) (the Bandi-Russell /
    Zhang noise-variance estimate, meaningful at the smallest dt). Units: RV in bps^2 per day.
    """
    ts = ref["ts"].to_numpy(dtype="int64")
    v = ref["value"].to_numpy(dtype=float)
    o = np.argsort(ts, kind="stable")
    ts, v = ts[o], v[o]
    days = (ts[-1] - ts[0]) / (86400 * NS)
    rows = []
    for dt in intervals_s:
        grid = np.arange(ts[0], ts[-1], int(dt * NS))
        j = np.searchsorted(ts, grid, side="right") - 1
        r = np.diff(np.log(v[j])) * 1e4
        rv = np.sum(r ** 2)
        rows.append({"interval_s": dt, "rv_per_day": rv / days, "n": len(r), "omega2": rv / (2 * len(r))})
    return pd.DataFrame(rows)


def noise_variance(ref: pd.DataFrame) -> float:
    """omega^2 = RV / (2 n) using every observed change (tick-by-tick), bps^2. ~ noise variance."""
    v = ref.sort_values("ts", kind="stable")["value"].to_numpy(dtype=float)
    r = np.diff(np.log(v)) * 1e4
    r = r[r != 0]
    return float(np.sum(r ** 2) / (2 * len(r)))


# ---------------------------------------------------------------------------------------------
# Clocks and staleness
# ---------------------------------------------------------------------------------------------
def ntp_offsets(df: pd.DataFrame, min_delay_quantile: float = 0.25) -> dict:
    """Clock offset/delay from four-timestamp exchanges (NTP style), plus a linear drift fit.

    Columns t0 (client send), t1 (server receive), t2 (server send), t3 (client receive), ns.
        offset = ((t1 - t0) + (t2 - t3)) / 2     (server clock - client clock)
        delay  = (t3 - t0) - (t2 - t1)          (round trip net of server time)
    Drift: OLS of offset on t0 using only the lowest-delay samples (least queueing).
    Returns dict(table=DataFrame, offset_ns_at_start, drift_ppm, intercept, slope).
    """
    t0, t1, t2, t3 = (df[c].to_numpy(dtype="float64") for c in ("t0", "t1", "t2", "t3"))
    off = ((t1 - t0) + (t2 - t3)) / 2
    dl = (t3 - t0) - (t2 - t1)
    tab = df.copy()
    tab["offset_ns"], tab["delay_ns"] = off, dl
    good = dl <= np.quantile(dl, min_delay_quantile)
    x = (t0 - t0.min()) / NS
    slope, icpt = np.polyfit(x[good], off[good], 1)
    tab["used_in_fit"] = good
    return {"table": tab, "offset_ns_at_start": float(icpt), "drift_ppm": float(slope / NS * 1e6 * 1),
            "intercept": float(icpt), "slope_ns_per_s": float(slope)}


def staleness(query: pd.DataFrame, ref: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """As-of (backward, inclusive) values with their age: age_s = query ts - source ts, seconds.

    Every as-of value you use should carry this, so stale inputs can be excluded or studied.
    """
    r = asof_values(query, ref, value_cols, direction="backward", allow_exact=True)
    r["age_s"] = r["age_ns"] / NS
    return r.drop(columns=["age_ns"])


def resample_ffill(ref: pd.DataFrame, freq_s: float, value_cols: list[str] | None = None,
                   start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """Forward-fill-only resampling onto a regular grid per venue/symbol (NEVER interpolates).

    Each grid point gets the last observation at or before it, plus ``age_s``. Grid points before
    the first observation are dropped.
    """
    value_cols = value_cols or ["value"]
    out = []
    step = int(freq_s * NS)
    for (v, s), g in ref.groupby(["venue", "symbol"], sort=False):
        lo = start if start is not None else int(g["ts"].min())
        hi = end if end is not None else int(g["ts"].max())
        grid = np.arange(lo - lo % step + step, hi + 1, step, dtype="int64")
        q = pd.DataFrame({"ts": grid, "venue": v, "symbol": s})
        r = asof_values(q, g, value_cols, direction="backward", allow_exact=True)
        q[value_cols] = r[value_cols].to_numpy()
        q["age_s"] = r["age_ns"].to_numpy() / NS
        out.append(q.dropna(subset=value_cols))
    return pd.concat(out, ignore_index=True)
