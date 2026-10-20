"""Passive fill simulation from public trades, with two bounds, plus markouts.

A quote row at time q (my_bid / my_ask, NaN = no quote) is live from q + live_delay until the next quote row's
time + live_delay (a replacement also takes the delay to land). It fills at most once per side.

mode="through": a seller-initiated trade strictly below my_bid (buyer strictly above my_ask). The level was
                swept, so the fill happened wherever we sat in the queue. Pessimistic.
mode="touch":   at or beyond our price. Assumes we were first in the queue. Optimistic.

CLAUDE.md rule 5: report both, and say which conclusions survive only under touch.
Ignored by design (flag it if it matters): partial fills, our own impact, queue position, fees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pit import asof_features

AGGRESSOR_SELL = {"sell", "s", "ask", -1}
AGGRESSOR_BUY = {"buy", "b", "bid", 1}


def simulate_fills(my_quotes: pd.DataFrame, trades: pd.DataFrame, live_delay="0ms", mode: str = "through",
                   ts_col: str = "ts_recv", quote_size: float = 1.0) -> pd.DataFrame:
    """Returns one row per fill: ts (trade time), our_side ('buy' = our bid filled, 'sell' = our ask filled),
    price (our quote price), quote_ts, trade_price, trade_size."""
    if mode not in ("through", "touch"):
        raise ValueError("mode must be 'through' or 'touch'")
    q = my_quotes.sort_index()
    qt = pd.DatetimeIndex(q.index)
    delay = pd.Timedelta(live_delay)
    live_from = qt + delay
    live_to = np.r_[qt[1:].to_numpy(), np.datetime64("NaT")]
    live_to = pd.DatetimeIndex(live_to) + delay
    tr = trades.sort_values(ts_col)
    tt = pd.DatetimeIndex(pd.to_datetime(tr[ts_col]))
    side = tr["side"].map(lambda s: -1 if s in AGGRESSOR_SELL else (1 if s in AGGRESSOR_BUY else 0)).to_numpy()
    px, sz = tr["price"].to_numpy(float), tr["size"].to_numpy(float)
    out = []
    for i in range(len(q)):
        lo = np.searchsorted(tt.to_numpy(), live_from[i].to_datetime64(), side="right")
        hi = len(tt) if pd.isna(live_to[i]) else np.searchsorted(tt.to_numpy(), live_to[i].to_datetime64(), side="right")
        if hi <= lo:
            continue
        s_, p_, z_, t_ = side[lo:hi], px[lo:hi], sz[lo:hi], tt[lo:hi]
        b, a = q["my_bid"].iloc[i], q["my_ask"].iloc[i]
        if np.isfinite(b):
            m = (s_ == -1) & ((p_ < b) if mode == "through" else (p_ <= b))
            if m.any():
                j = int(np.argmax(m))
                out.append((t_[j], "buy", b, qt[i], p_[j], z_[j]))
        if np.isfinite(a):
            m = (s_ == 1) & ((p_ > a) if mode == "through" else (p_ >= a))
            if m.any():
                j = int(np.argmax(m))
                out.append((t_[j], "sell", a, qt[i], p_[j], z_[j]))
    f = pd.DataFrame(out, columns=["ts", "our_side", "price", "quote_ts", "trade_price", "trade_size"])
    f["qty"] = quote_size
    return f.sort_values("ts").reset_index(drop=True)


def markouts(f: pd.DataFrame, quotes: pd.DataFrame, horizons=("0ms", "250ms", "1s", "5s"), ts_col: str = "ts_recv",
             in_bps: bool = True) -> pd.DataFrame:
    """Per-fill markout = sign * (mid known at fill + h - fill price); sign +1 for our buys. Positive = good for us.
    The '0ms' column is the spread captured at the fill."""
    if len(f) == 0:
        return pd.DataFrame(columns=list(horizons))
    qm = quotes.assign(_mid=(quotes["bid_px"] + quotes["ask_px"]) / 2)
    sign = np.where(f["our_side"].to_numpy() == "buy", 1.0, -1.0)
    out = {}
    for h in horizons:
        m = asof_features(pd.DatetimeIndex(f["ts"]) + pd.Timedelta(h), qm, ["_mid"], ts_col=ts_col)["_mid"].to_numpy()
        v = sign * (m - f["price"].to_numpy(float))
        out[h] = v / f["price"].to_numpy(float) * 1e4 if in_bps else v
    return pd.DataFrame(out, index=f.index)


def pnl_path(f: pd.DataFrame, quotes: pd.DataFrame, t, ts_col: str = "ts_recv", fee_bps: float = 0.0) -> pd.DataFrame:
    """Inventory, cash and mark-to-mid equity at times t from a fill list (fees in bps of notional, + = cost)."""
    t = pd.DatetimeIndex(pd.to_datetime(t))
    sign = np.where(f["our_side"].to_numpy() == "buy", 1.0, -1.0)
    qty = f["qty"].to_numpy(float) * sign
    cash = -qty * f["price"].to_numpy(float) - np.abs(qty) * f["price"].to_numpy(float) * fee_bps * 1e-4
    k = np.searchsorted(pd.DatetimeIndex(f["ts"]).to_numpy(), t.to_numpy(), side="right")
    inv = np.r_[0, np.cumsum(qty)][k]
    cash_t = np.r_[0, np.cumsum(cash)][k]
    m = asof_features(t, quotes.assign(_mid=(quotes["bid_px"] + quotes["ask_px"]) / 2), ["_mid"], ts_col=ts_col)["_mid"].to_numpy()
    return pd.DataFrame({"inventory": inv, "cash": cash_t, "equity": cash_t + inv * m}, index=t)
