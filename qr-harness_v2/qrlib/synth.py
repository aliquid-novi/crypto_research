"""Synthetic market with a known latent price, for testing estimators and guardrails.

The latent (efficient) price is a random walk. Quotes sit around it on a tick grid; the queue imbalance carries real
information about the latent price, so a good fair value beats the mid by construction, and you know by how much.
Trades arrive as a Poisson stream and are buyer-initiated when the latent price sits above the mid.
Feed latency is ts_recv - ts_exch ~ base + exponential jitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(n: int = 20_000, seed: int = 0, tick: float = 0.01, start="2026-01-01", mean_gap_ms: float = 50.0,
                vol_ticks_per_sqrt_s: float = 1.0, base_latency_ms: float = 2.0, jitter_ms: float = 1.0,
                trade_rate_hz: float = 5.0, imbalance_info: float = 0.8) -> dict:
    """Returns {"quotes", "trades", "latent", "params"}. Quotes/trades have ts_exch, ts_recv (datetime64[ns])."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_gap_ms, n)
    t_ms = np.cumsum(gaps)
    dW = rng.normal(0, 1, n) * vol_ticks_per_sqrt_s * tick * np.sqrt(gaps / 1000)
    latent = 100.0 + np.cumsum(dW)
    # one-tick market around the latent price; the bid sits at floor(latent)
    bid = np.floor(latent / tick) * tick
    ask = bid + tick
    frac = (latent - bid) / tick                       # where the latent price sits inside the spread, in [0, 1)
    # imbalance Qb/(Qb+Qa) is an informative but noisy read of frac
    imb = np.clip(imbalance_info * frac + (1 - imbalance_info) * rng.uniform(0, 1, n), 0.01, 0.99)
    tot = rng.lognormal(3, 0.6, n)
    bid_sz, ask_sz = tot * imb, tot * (1 - imb)
    t0 = pd.Timestamp(start)
    ts_exch = t0 + pd.to_timedelta(t_ms, unit="ms")
    lat = base_latency_ms + rng.exponential(jitter_ms, n)
    ts_recv = ts_exch + pd.to_timedelta(lat, unit="ms")
    quotes = pd.DataFrame({"ts_exch": ts_exch, "ts_recv": ts_recv, "bid_px": np.round(bid, 10), "ask_px": np.round(ask, 10),
                           "bid_sz": bid_sz, "ask_sz": ask_sz}).sort_values("ts_recv", kind="stable").reset_index(drop=True)
    # trades
    span_ms = t_ms[-1]
    m = rng.poisson(trade_rate_hz * span_ms / 1000)
    tt = np.sort(rng.uniform(0, span_ms, m))
    k = np.clip(np.searchsorted(t_ms, tt, side="right") - 1, 0, n - 1)
    buy = rng.uniform(size=m) < frac[k]                 # latent above mid -> buyers more likely
    price = np.where(buy, ask[k], bid[k])
    tex = t0 + pd.to_timedelta(tt, unit="ms")
    trec = tex + pd.to_timedelta(base_latency_ms + rng.exponential(jitter_ms, m), unit="ms")
    trades = pd.DataFrame({"ts_exch": tex, "ts_recv": trec, "price": np.round(price, 10),
                           "size": rng.lognormal(0, 1, m), "side": np.where(buy, "buy", "sell")}
                          ).sort_values("ts_recv", kind="stable").reset_index(drop=True)
    lat_df = pd.DataFrame({"ts_exch": ts_exch, "latent": latent})
    return {"quotes": quotes, "trades": trades, "latent": lat_df,
            "params": dict(n=n, seed=seed, tick=tick, mean_gap_ms=mean_gap_ms, vol=vol_ticks_per_sqrt_s,
                           base_latency_ms=base_latency_ms, jitter_ms=jitter_ms, trade_rate_hz=trade_rate_hz,
                           imbalance_info=imbalance_info)}
