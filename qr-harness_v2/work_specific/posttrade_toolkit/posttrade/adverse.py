"""Adverse selection: AS ratio from orders, pre-fill drift, late cancels, full vs partial fills,
counterparty concentration and venue-wide per-taker toxicity.

Sign convention: d = +1 our buy. Markouts in bps, positive = good for us. For takers in the
public tape, q = aggressor_side and "taker markout" = 1e4 * q * (m_{t+h} - p) / p (positive =
the taker gained, i.e. the taker was informed / toxic for whoever provided liquidity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import NS, asof_values, group_keys
from .markouts import compute_markouts


def adverse_selection_ratio(orders: pd.DataFrame, ref: pd.DataFrame, horizon_s: float | None = None) -> dict:
    """AS ratio = P(fill | next move against) / P(fill | next move in favour), from all resting orders.

    For each order (status filled/partial = filled, cancelled = not), the move after ts_done is
    either the first reference change strictly after ts_done (``horizon_s=None``) or
    m(ts_done + h) - m(ts_done). "Against" = d * move < 0 (a bid followed by a down-move).
    Ratio > 1: you get filled preferentially before adverse moves. Orders with no move are dropped.
    Returns dict(ratio, table, per_order).
    """
    o = orders[orders["final_status"].isin(["filled", "partial", "cancelled"]) & orders["ts_done"].notna()].copy()
    o["ts"] = o["ts_done"].astype("int64")
    m0 = asof_values(o, ref, ["value"], direction="backward", allow_exact=True)["value"].to_numpy()
    if horizon_s is None:
        chg = []
        for _, g in ref.groupby(group_keys(ref), sort=False):
            g = g.sort_values("ts", kind="stable")
            v = g["value"].to_numpy()
            chg.append(g[np.concatenate([[False], v[1:] != v[:-1]])])
        ch = pd.concat(chg, ignore_index=True)
        m1 = asof_values(o, ch, ["value"], direction="forward", allow_exact=False)["value"].to_numpy()
    else:
        m1 = asof_values(o, ref, ["value"], ts_query=o["ts"].to_numpy() + int(horizon_s * NS), direction="backward")["value"].to_numpy()
    mv = o["side"].to_numpy(dtype=float) * (m1 - m0)
    o["move"] = np.where(mv < 0, "against", np.where(mv > 0, "favour", "none"))
    o["filled"] = o["final_status"].isin(["filled", "partial"])
    if "filled_qty" in o.columns:
        o["filled"] &= o["filled_qty"].fillna(1) > 0
    o = o[o["move"] != "none"]
    tab = o.groupby("move")["filled"].agg(["mean", "size"]).rename(columns={"mean": "p_fill", "size": "n_orders"})
    ratio = float(tab.loc["against", "p_fill"] / tab.loc["favour", "p_fill"]) if {"against", "favour"} <= set(tab.index) else np.nan
    return {"ratio": ratio, "table": tab, "per_order": o[["order_id", "side", "move", "filled"]]}


def pre_fill_drift(fills: pd.DataFrame, ref: pd.DataFrame, lookback_s=(-60, -30, -10, -5, -2, -1, -0.5)) -> pd.DataFrame:
    """Pre-fill drift of the reference: 1e4 * d * (m_t - m_{t+h}) / p for h < 0 (bps, per fill).

    Positive = the price was moving in our direction before we traded (we bought into a rise);
    negative = moving against us (we bought into a fall - typical of being picked off by momentum).
    Returns DataFrame columns = lookback (s, negative).
    """
    lb = np.asarray(lookback_s, float)
    mo = compute_markouts(fills, ref, np.concatenate([lb, [0.0]]))
    return mo[lb].rsub(mo[0.0], axis=0)


def late_cancels(fills: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Fills whose order had a cancel SENT before the fill time (ts_cancel_sent < fill ts): we tried
    to pull the quote and lost the race. Returns the fills with a boolean ``late_cancel`` and
    ``cancel_lead_ms`` = fill ts - cancel sent (ms)."""
    c = orders.drop_duplicates("order_id").set_index("order_id")["ts_cancel_sent"]
    cs = fills["order_id"].map(c).astype("float64")
    out = fills.copy()
    out["cancel_lead_ms"] = (out["ts"].astype("float64") - cs) / 1e6
    out["late_cancel"] = out["cancel_lead_ms"] > 0
    return out


def fill_completeness(fills: pd.DataFrame, orders: pd.DataFrame) -> pd.Series:
    """'full' if the order's total filled qty >= its qty, else 'partial' (per fill, by order_id).

    Uses orders.filled_qty when present, else the sum of fills per order."""
    o = orders.drop_duplicates("order_id").set_index("order_id")
    filled = o["filled_qty"] if "filled_qty" in o.columns else fills.groupby("order_id")["qty"].sum()
    ratio = fills["order_id"].map(filled) / fills["order_id"].map(o["qty"])
    return pd.Series(np.where(ratio >= 1 - 1e-9, "full", np.where(ratio.notna(), "partial", None)), index=fills.index)


def counterparty_concentration(fills: pd.DataFrame, markout_bps: np.ndarray, min_fills: int = 1) -> pd.DataFrame:
    """Rank counterparties by total adverse PnL against us.

    pnl = markout_bps * notional / 1e4 (quote ccy, positive = good for us). Per counterparty:
    n, notional, total pnl, total negative pnl (loss), mean markout bps. Sorted by loss (largest
    first) with the cumulative share of total loss (Lorenz input). Needs fills.counterparty.
    """
    f = fills.assign(pnl=np.asarray(markout_bps, float) * fills["notional"].to_numpy(dtype=float) / 1e4,
                     mo=np.asarray(markout_bps, float))
    f = f[np.isfinite(f["pnl"])]
    g = f.groupby("counterparty")
    t = pd.DataFrame({"n": g.size(), "notional": g["notional"].sum(), "pnl": g["pnl"].sum(),
                      "loss": g["pnl"].apply(lambda x: -x[x < 0].sum()), "mean_markout_bps": g["mo"].mean()})
    t = t[t["n"] >= min_fills].sort_values("pnl")
    tot_loss = max(-t["pnl"].clip(upper=0).sum(), 1e-12)
    t["cum_share_of_net_loss"] = (-t["pnl"].clip(upper=0)).cumsum() / tot_loss
    return t


def taker_toxicity(trades: pd.DataFrame, ref: pd.DataFrame, horizon_s: float = 30.0, prior_n: float | None = None,
                   exclude: tuple[str, ...] = ("OURS",), min_n: int = 5) -> pd.DataFrame:
    """Venue-wide toxicity score per taker address from the public tape (needs buyer/seller).

    Taker = buyer if aggressor_side = +1 else seller. Taker markout per trade
    = 1e4 * q * (m_{t+h} - p) / p (bps). Per address mean is shrunk toward the grand mean:
        score = (n * mean + k * grand) / (n + k)
    with k = prior_n, or (default) the empirical-Bayes k = within-address variance / between-address
    variance of true means (method of moments, floored at 1). Small-sample addresses are pulled to
    the average so they cannot top the ranking by luck. Sorted by score descending (most toxic first).
    """
    t = trades[trades["buyer"].notna() & trades["seller"].notna()].copy()
    t["taker"] = np.where(t["aggressor_side"] > 0, t["buyer"], t["seller"])
    t = t[~t["taker"].isin(exclude)]
    t["side"] = t["aggressor_side"]
    mo = compute_markouts(t, ref, [horizon_s])[float(horizon_s)].to_numpy()
    t["mo"] = mo
    t = t[np.isfinite(t["mo"])]
    g = t.groupby("taker")["mo"]
    s = pd.DataFrame({"n": g.size(), "mean_bps": g.mean(), "var": g.var(ddof=1)})
    s = s[s["n"] >= min_n]
    grand = float(t["mo"].mean())
    if prior_n is None:
        within = float(np.nanmean(s["var"]))
        between = float(np.var(s["mean_bps"], ddof=1) - np.mean(within / s["n"]))
        prior_n = max(within / between, 1.0) if between > 0 else float(s["n"].median())
    s["score_bps"] = (s["n"] * s["mean_bps"] + prior_n * grand) / (s["n"] + prior_n)
    s["se_bps"] = np.sqrt(s["var"] / s["n"])
    s.attrs["prior_n"] = prior_n
    s.attrs["grand_mean_bps"] = grand
    return s.drop(columns="var").sort_values("score_bps", ascending=False)
