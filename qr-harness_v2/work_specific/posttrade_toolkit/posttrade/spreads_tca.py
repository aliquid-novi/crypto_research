"""Implementation shortfall for two-leg spread trades: decision -> entry legs -> exit legs.

Conventions
-----------
* decisions.intended_qty is SIGNED in units of leg A: + = buy A / sell B (long spread).
* decisions.hedge_ratio = units of leg B per unit of leg A (always traded on the opposite side).
  For a notional or beta hedge, convert to units yourself (h_units = beta * P_A / P_B).
* Leg sign s_A = sign(intended_qty), s_B = -s_A. Q_L = intended units, E_L = executed entry units,
  X_L = executed exit units, R_L = E_L - X_L residual. mult_L = contract multiplier.
* Reference prices are mids from book_l1. m0 = mid at decision ts (at or before), m1 = mid at exit
  decision (``ts_exit_decision`` column, else the first exit fill). Arrival mids are taken strictly
  BEFORE the first fill of that leg (the pre-trade book).

Bridge (quote currency, positive = good for us), per leg:
    paper        =  s Q (m1 - m0)
    delay        = -s E (m_L(t_first) - m0)           t_first = first entry fill of EITHER leg
    legging      = -s E (m_L(t_arr_L) - m_L(t_first))  zero for the leg traded first
    entry_cost   = -s E (pe - m_L(t_arr_L))            pe = entry VWAP (half-spread + impact + slippage)
    opportunity  = -s (Q - E) (m1 - m0)                the unfilled part of the paper trade
    exit_delay   =  s X (m_L(t_xfirst) - m1)
    exit_legging =  s X (m_L(t_xarr_L) - m_L(t_xfirst))
    exit_cost    =  s X (px - m_L(t_xarr_L))
    residual     =  s R (m_end - m1)                   hung position marked at the end
    fees         = -sum(fee)
    carry        =  sum over funding settlements while the leg is open of -pos * mark * rate * mult
  realised_net = s (X px + R m_end - E pe) * mult + fees + carry
  realised_net - paper = sum of all components      (asserted to 1e-9 relative)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from ._util import NS, asof_values, instrument_lookup
from .reference import mid

COMPONENTS = ["delay", "legging", "entry_cost", "opportunity", "exit_delay", "exit_legging", "exit_cost",
              "residual", "fees", "carry"]


def _mids_at(book: pd.DataFrame, venue: np.ndarray, symbol: np.ndarray, ts: np.ndarray, strict: bool) -> np.ndarray:
    ref = book[["ts", "venue", "symbol"]].assign(value=mid(book))
    q = pd.DataFrame({"ts": np.nan_to_num(ts, nan=0).astype("int64"), "venue": venue, "symbol": symbol})
    v = asof_values(q, ref, ["value"], direction="backward", allow_exact=not strict)["value"].to_numpy()
    v[~np.isfinite(np.asarray(ts, float))] = np.nan
    return v


def link_fills(fills: pd.DataFrame, decisions: pd.DataFrame, max_lag_s: float = 60.0) -> pd.DataFrame:
    """Return fills carrying decision_id / leg / leg_role.

    If fills already have decision_id, leg and leg_role, they are returned unchanged. Otherwise a
    HEURISTIC links each fill on a decision's leg instruments to the latest decision at or before
    it within ``max_lag_s`` (entry) and assigns leg by instrument; leg_role = 'entry' until the
    decision's ts_exit_decision, 'exit' after. Check the result - real books need real ids.
    """
    if {"decision_id", "leg", "leg_role"} <= set(fills.columns) and fills["decision_id"].notna().any():
        return fills
    out = fills.copy()
    out["decision_id"], out["leg"], out["leg_role"] = None, None, None
    d = decisions.sort_values("ts")
    for leg in ("A", "B"):
        for (v, s), dd in d.groupby([f"leg_{leg}_venue", f"leg_{leg}_symbol"]):
            m = (out["venue"] == v) & (out["symbol"] == s)
            ft = out.loc[m, "ts"].to_numpy(dtype="int64")
            dts = dd["ts"].to_numpy(dtype="int64")
            j = np.searchsorted(dts, ft, side="right") - 1
            ok = j >= 0
            dex = dd["ts_exit_decision"].to_numpy(dtype="float64") if "ts_exit_decision" in dd else np.full(len(dd), np.nan)
            lag = np.where(ok, ft - dts[np.maximum(j, 0)], np.inf)
            exit_end = np.where(np.isfinite(dex[np.maximum(j, 0)]), dex[np.maximum(j, 0)] + max_lag_s * NS, np.inf)
            ok &= (lag <= max_lag_s * NS) | (ft <= exit_end)
            idx = out.index[m][ok]
            out.loc[idx, "decision_id"] = dd["decision_id"].to_numpy()[j[ok]]
            out.loc[idx, "leg"] = leg
            is_exit = np.isfinite(dex[j[ok]]) & (ft[ok] >= dex[j[ok]])
            out.loc[idx, "leg_role"] = np.where(is_exit, "exit", "entry")
    return out


def shortfall(decisions: pd.DataFrame, fills: pd.DataFrame, book: pd.DataFrame, funding: pd.DataFrame | None = None,
              instruments: pd.DataFrame | None = None, default_hold_s: float | None = None) -> dict[str, pd.DataFrame]:
    """Implementation-shortfall bridge per decision and leg (see module docstring for formulas).

    Returns dict(legs=per decision x leg, trades=per decision). ``trades`` has every component,
    paper, realised_net, the check column ``bridge_error`` (should be ~0), gross notional and bps
    versions (bps of intended gross notional at decision mids), legging_s, leg_order, filled flags.
    """
    dec = decisions.copy()
    if "ts_exit_decision" not in dec.columns:
        dec["ts_exit_decision"] = np.nan
    f = fills[fills["decision_id"].notna()].copy()
    f["decision_id"] = f["decision_id"].astype(str)
    dec["decision_id"] = dec["decision_id"].astype(str)
    rows = []
    for leg in ("A", "B"):
        sgn = np.sign(dec["intended_qty"].to_numpy(dtype=float)) * (1 if leg == "A" else -1)
        Q = np.abs(dec["intended_qty"].to_numpy(dtype=float)) * (1 if leg == "A" else dec["hedge_ratio"].to_numpy(dtype=float))
        rows.append(pd.DataFrame({"decision_id": dec["decision_id"].to_numpy(), "leg": leg,
                                  "venue": dec[f"leg_{leg}_venue"].to_numpy(), "symbol": dec[f"leg_{leg}_symbol"].to_numpy(),
                                  "s": sgn, "Q": Q, "t0": dec["ts"].to_numpy(dtype="int64"),
                                  "t1": dec["ts_exit_decision"].to_numpy(dtype="float64")}))
    L = pd.concat(rows, ignore_index=True)
    L["mult"] = instrument_lookup(L, instruments, "multiplier", config.DEFAULT_MULTIPLIER)

    f["pq"] = f["price"] * f["qty"]
    g = f.groupby(["decision_id", "leg", "leg_role"])
    agg = g.agg(qty=("qty", "sum"), pq=("pq", "sum"), fee=("fee", "sum"), t_first=("ts", "min"), t_last=("ts", "max"))
    agg = agg.reset_index()
    for role, pre in (("entry", "e"), ("exit", "x")):
        a = agg[agg["leg_role"] == role].drop(columns="leg_role")
        a = a.rename(columns={"qty": f"{pre}_qty", "pq": f"{pre}_pq", "fee": f"{pre}_fee", "t_first": f"{pre}_t_first",
                              "t_last": f"{pre}_t_last"})
        L = L.merge(a, on=["decision_id", "leg"], how="left")
    for c in ("e_qty", "e_pq", "e_fee", "x_qty", "x_pq", "x_fee"):
        L[c] = L[c].fillna(0.0)
    # exit decision time fallback: first exit fill of either leg, else default hold
    first_exit = L.groupby("decision_id")["x_t_first"].transform("min")
    L["t1"] = L["t1"].fillna(first_exit)
    if default_hold_s is not None:
        L["t1"] = L["t1"].fillna(L["t0"] + default_hold_s * NS)
    L["t_first"] = L.groupby("decision_id")["e_t_first"].transform("min")
    L["t_xfirst"] = L.groupby("decision_id")["x_t_first"].transform("min")
    L["t_end"] = np.fmax(L.groupby("decision_id")["x_t_last"].transform("max").to_numpy(dtype=float), L["t1"].to_numpy(dtype=float))

    v, s_ = L["venue"].to_numpy(), L["symbol"].to_numpy()
    L["m0"] = _mids_at(book, v, s_, L["t0"].to_numpy(dtype=float), strict=False)
    L["m1"] = _mids_at(book, v, s_, L["t1"].to_numpy(dtype=float), strict=False)
    L["m_first"] = _mids_at(book, v, s_, L["t_first"].to_numpy(dtype=float), strict=True)
    L["m_arr"] = _mids_at(book, v, s_, L["e_t_first"].to_numpy(dtype=float), strict=True)
    L["m_xfirst"] = _mids_at(book, v, s_, L["t_xfirst"].to_numpy(dtype=float), strict=True)
    L["m_xarr"] = _mids_at(book, v, s_, L["x_t_first"].to_numpy(dtype=float), strict=True)
    L["m_end"] = _mids_at(book, v, s_, L["t_end"].to_numpy(dtype=float), strict=False)

    s, E, X, Q, mu = (L[c].to_numpy(dtype=float) for c in ("s", "e_qty", "x_qty", "Q", "mult"))
    pe = np.divide(L["e_pq"].to_numpy(), E, out=np.zeros(len(L)), where=E > 0)
    px = np.divide(L["x_pq"].to_numpy(), X, out=np.zeros(len(L)), where=X > 0)
    R = E - X
    z = lambda a: np.nan_to_num(a, nan=0.0)  # noqa: E731  (a missing mid only matters when its qty is 0)
    m0, m1 = L["m0"].to_numpy(), L["m1"].to_numpy()
    L["entry_vwap"], L["exit_vwap"], L["residual_qty"] = np.where(E > 0, pe, np.nan), np.where(X > 0, px, np.nan), R
    L["paper"] = s * Q * (m1 - m0) * mu
    L["delay"] = np.where(E > 0, -s * E * z(L["m_first"] - m0), 0) * mu
    L["legging"] = np.where(E > 0, -s * E * z(L["m_arr"] - L["m_first"]), 0) * mu
    L["entry_cost"] = np.where(E > 0, -s * E * z(pe - L["m_arr"]), 0) * mu
    L["opportunity"] = -s * (Q - E) * (m1 - m0) * mu
    L["exit_delay"] = np.where(X > 0, s * X * z(L["m_xfirst"] - m1), 0) * mu
    L["exit_legging"] = np.where(X > 0, s * X * z(L["m_xarr"] - L["m_xfirst"]), 0) * mu
    L["exit_cost"] = np.where(X > 0, s * X * z(px - L["m_xarr"]), 0) * mu
    L["residual"] = np.where(R != 0, s * R * z(L["m_end"] - m1), 0) * mu
    L["fees"] = -(L["e_fee"] + L["x_fee"])
    L["carry"] = _carry(L, f, funding, mu)
    L["realised_gross"] = s * (X * px + R * z(L["m_end"]) - E * pe) * mu
    L["realised_net"] = L["realised_gross"] + L["fees"] + L["carry"]
    L["bridge_error"] = (L["realised_net"] - L["paper"]) - L[COMPONENTS].sum(axis=1)

    keep = ["paper", "realised_gross", "realised_net", *COMPONENTS, "bridge_error"]
    T = L.groupby("decision_id", sort=False)[keep].sum()
    gross = (L["Q"] * L["m0"] * L["mult"]).groupby(L["decision_id"], sort=False).sum()
    T["gross_notional"] = gross
    for c in ["paper", "realised_net", *COMPONENTS]:
        T[f"{c}_bps"] = 1e4 * T[c] / gross
    piv = L.pivot(index="decision_id", columns="leg", values="e_t_first")
    T["legging_s"] = (piv["B"] - piv["A"]).abs() / NS
    T["leg_order"] = np.where(piv["A"].isna() & piv["B"].isna(), "none",
                              np.where(piv["B"].isna() | (piv["A"] <= piv["B"]), "A_first", "B_first"))
    fq = L.pivot(index="decision_id", columns="leg", values="e_qty")
    T["filled_any"] = (fq > 0).any(axis=1)
    T["filled_full"] = (L.assign(full=L["e_qty"] >= L["Q"] * (1 - 1e-6)).groupby("decision_id")["full"].all())
    T["residual_any"] = (L.assign(r=L["residual_qty"].abs() > 1e-12).groupby("decision_id")["r"].any())
    T = T.join(dec.set_index("decision_id")[["ts", "pair", "signal", "intended_qty", "hedge_ratio"]])
    T["open"] = ~np.isfinite(T["paper"].to_numpy())  # no exit time known -> paper undefined
    scale = np.maximum(np.nanmax(np.abs(T[keep[:-1]].to_numpy()), axis=1), 1.0)
    err = np.abs(T["bridge_error"].to_numpy())
    assert np.all(err[~T["open"].to_numpy()] <= 1e-9 * scale[~T["open"].to_numpy()]), \
        "shortfall components do not sum to realised - paper"
    return {"legs": L, "trades": T.reset_index()}


def _carry(L: pd.DataFrame, f: pd.DataFrame, funding: pd.DataFrame | None, mult: np.ndarray) -> np.ndarray:
    """Funding carry per leg: sum over settlements in (first fill, last fill] of -pos * mark * rate * mult,
    pos = signed leg position from that leg's own fills at the settlement time."""
    out = np.zeros(len(L))
    if funding is None or len(funding) == 0:
        return out
    fu = {k: g.sort_values("ts") for k, g in funding.groupby(["venue", "symbol"])}
    fg = {k: g.sort_values("ts") for k, g in f.groupby(["decision_id", "leg"])}
    for i, r in enumerate(L.itertuples(index=False)):
        g = fg.get((r.decision_id, r.leg))
        fr = fu.get((r.venue, r.symbol))
        if g is None or fr is None:
            continue
        ts = g["ts"].to_numpy(dtype="int64")
        sq = (g["side"] * g["qty"]).to_numpy(dtype=float)
        st = fr["ts"].to_numpy(dtype="int64")
        m = (st > ts.min()) & (st <= ts.max())
        if not m.any():
            continue
        cum = np.cumsum(sq)
        j = np.searchsorted(ts, st[m], side="right") - 1
        pos = np.where(j >= 0, cum[np.maximum(j, 0)], 0.0)
        out[i] = float(np.sum(-pos * fr["mark"].to_numpy()[m] * fr["rate"].to_numpy()[m]) * mult[i])
    return out


# ---------------------------------------------------------------------------------------------
# Aggregations and reports
# ---------------------------------------------------------------------------------------------
def aggregate(trades: pd.DataFrame, by: str | list[str], bps: bool = True) -> pd.DataFrame:
    """Mean of each component (bps of gross notional if ``bps``) and count, grouped by ``by``
    (e.g. 'pair', 'leg_order', a signal bucket or vol-regime column you add)."""
    cols = [f"{c}_bps" for c in ["paper", "realised_net", *COMPONENTS]] if bps else ["paper", "realised_net", *COMPONENTS]
    g = trades[trades["filled_any"]].groupby(by, observed=True)
    out = g[cols].mean()
    out["n"] = g.size()
    return out


def hung_legs(trades: pd.DataFrame, legs: pd.DataFrame, hung_s: float = 10.0) -> pd.DataFrame:
    """Episodes where one leg was exposed alone: time between legs > hung_s, a second leg that
    never/partly filled, or residual position after exit. Returns the affected trades with reason."""
    t = trades.copy()
    reasons = []
    for r in t.itertuples(index=False):
        rs = []
        if np.isfinite(r.legging_s) and r.legging_s > hung_s:
            rs.append(f"legs {r.legging_s:.1f}s apart")
        if r.filled_any and not r.filled_full:
            rs.append("partial/missing leg")
        if r.residual_any:
            rs.append("residual after exit")
        reasons.append("; ".join(rs))
    t["reason"] = reasons
    return t[t["reason"] != ""]


def missed_report(trades: pd.DataFrame) -> pd.DataFrame:
    """Paper PnL (bps) of decisions that filled vs those that did not. If missed decisions have
    better paper PnL than filled ones, the fill process is selecting against you."""
    g = trades.groupby(np.where(trades["filled_any"], "filled", "missed"))["paper_bps"]
    return pd.DataFrame({"n": g.size(), "mean_paper_bps": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.size())})


def cost_share(trades: pd.DataFrame) -> dict:
    """Cost share of gross edge and the net-Sharpe approximation, on filled trades.

    alpha = mean paper PnL (bps), C = mean (paper - realised_net) (bps),
    SR_gross = mean(paper)/sd(paper), SR_net_approx = SR_gross * (1 - C/alpha) (holds when costs are
    roughly constant), SR_net_actual = mean(realised_net)/sd(realised_net). Per trade.
    """
    t = trades[trades["filled_any"]]
    a = t["paper_bps"].mean()
    c = (t["paper_bps"] - t["realised_net_bps"]).mean()
    sr_g = a / t["paper_bps"].std(ddof=1)
    return {"alpha_bps": float(a), "cost_bps": float(c), "cost_share": float(c / a) if a else np.nan,
            "sr_gross": float(sr_g), "sr_net_approx": float(sr_g * (1 - c / a)) if a else np.nan,
            "sr_net_actual": float(t["realised_net_bps"].mean() / t["realised_net_bps"].std(ddof=1)), "n": len(t)}


def decision_vs_fill_markouts(fills: pd.DataFrame, decisions: pd.DataFrame, ref: pd.DataFrame, horizons_s,
                              role: str = "entry") -> dict[str, pd.DataFrame]:
    """Markouts (bps) of leg fills anchored at decision time vs at fill time.

    Both use the fill price p:  1e4 * d * (m_{anchor+h} - p) / p. The gap between the two curves at
    small h is the cost of the delay between deciding and filling.
    """
    from .markouts import compute_markouts

    f = fills[(fills["leg_role"] == role) & fills["decision_id"].notna()].copy()
    f = f.merge(decisions[["decision_id", "ts"]].rename(columns={"ts": "ts_decision"}), on="decision_id", how="left")
    return {"fill_time": compute_markouts(f, ref, horizons_s),
            "decision_time": compute_markouts(f, ref, horizons_s, ts_col="ts_decision"), "fills": f}


def mae_mfe(legs: pd.DataFrame, fills: pd.DataFrame, book: pd.DataFrame, step_s: float = 1.0) -> pd.DataFrame:
    """Max adverse / favourable excursion per trade from the marked-to-mid PnL path.

    PnL(t) = sum over legs of [pos_L(t) * m_L(t) + cash_L(t)] * mult, pos from fills up to t,
    cash = -sum side*qty*price. Path on a ``step_s`` grid from first entry to last exit fill.
    Returns DataFrame[decision_id, mae, mfe, final] in quote ccy (mae <= 0 <= mfe usually).
    """
    ref = book[["ts", "venue", "symbol"]].assign(value=mid(book))
    f = fills[fills["decision_id"].notna()]
    out = []
    mult = legs.set_index(["decision_id", "leg"])["mult"].to_dict()
    for did, g in f.groupby("decision_id"):
        t_lo, t_hi = int(g["ts"].min()), int(g["ts"].max())
        grid = np.arange(t_lo, t_hi + int(step_s * NS), int(step_s * NS), dtype="int64")
        pnl = np.zeros(len(grid))
        for (leg, v, s), gl in g.groupby(["leg", "venue", "symbol"]):
            gl = gl.sort_values("ts")
            ts = gl["ts"].to_numpy(dtype="int64")
            j = np.searchsorted(ts, grid, side="right") - 1
            sq = (gl["side"] * gl["qty"]).to_numpy(dtype=float)
            pos = np.where(j >= 0, np.cumsum(sq)[np.maximum(j, 0)], 0.0)
            cash = np.where(j >= 0, np.cumsum(-sq * gl["price"].to_numpy(dtype=float))[np.maximum(j, 0)], 0.0)
            q = pd.DataFrame({"ts": grid, "venue": v, "symbol": s})
            m = asof_values(q, ref, ["value"])["value"].to_numpy()
            pnl += (pos * np.nan_to_num(m) + cash) * mult.get((did, leg), 1.0)
        out.append({"decision_id": did, "mae": pnl.min(), "mfe": pnl.max(), "final": pnl[-1]})
    return pd.DataFrame(out)
