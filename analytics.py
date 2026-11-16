"""What happened, and why: the run's records as DataFrames plus the standard execution measures.

Sign convention: every "edge", "markout" and PnL number is positive when it is good for us.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd


class Result:
    """Records of one run.

    fills     one row per fill: id, order_id, inst, side, price, qty, maker, fee, t (exchange time),
              t_local (when we learned of it), mid/bid/ask/bid_qty/ask_qty at the exchange just before, tag, parent
    orders    one row per order with its timestamps (submit, arrive, ack, cancel request/arrival, done),
              the mid when we decided and when it arrived, and queue ahead / level size when it joined
    bbo       {inst: DataFrame(t, bid, ask, mid)} every change of the true best bid/ask
    trades    {inst: DataFrame(t, side, price, qty)} the market's trade tape (side = aggressor)
    equity    DataFrame(t, equity, pos_<inst>, mid_<inst>, ...) sampled every sample_dt, base currency
    funding   one row per funding settlement: t, inst, rate, pos, price, payment
    logs      {key: DataFrame(t, value)} whatever the strategy passed to sim.log
    truth     latent series if the market was synthetic (see synth.py), else None
    """

    def __init__(self, sim):
        self.instruments = sim.instruments
        self.names = sim.names
        self.t0, self.t_end = sim.t0, sim.t_end
        self.truth = sim.truth
        self.cancel_rejects = sim.cancel_rejects
        self.fills = pd.DataFrame(sim._r_fills, columns=[
            "id", "order_id", "inst", "side", "price", "qty", "maker", "fee", "t", "t_local", "mid", "bid",
            "ask", "bid_qty", "ask_qty", "tag", "parent"])
        self.orders = pd.DataFrame([{
            "id": o.id, "inst": o.inst, "side": o.side, "price": o.price, "qty": o.qty, "filled": o.qty - o.x_rem,
            "tif": o.tif, "post_only": o.post_only, "status": _final_status(o), "reason": o.reason, "tag": o.tag,
            "parent": o.parent, "t_submit": o.t_submit, "t_arrive": o.t_arrive, "t_ack": o.t_ack,
            "t_cancel_req": o.t_cancel_req, "t_cancel_arrive": o.t_cancel_arrive, "t_done": o.t_done,
            "mid_decision": o.mid_decision, "mid_arrive": o.mid_arrive, "ahead0": o.ahead0, "level0": o.level0,
            "cancel_reject": bool(o.meta and isinstance(o.meta, dict) and o.meta.get("cancel_reject")),
            "k": (o.meta.get("k") if isinstance(o.meta, dict) else None),
        } for o in sim._all_orders])
        self.bbo = {}
        for i, x in enumerate(sim.instruments):
            t, b, a = sim._r_bbo[i]
            b = np.asarray(b, dtype=float)
            a = np.asarray(a, dtype=float)
            b[b < 0] = np.nan
            a[a < 0] = np.nan
            df = pd.DataFrame({"t": np.asarray(t, dtype=float), "bid": b * x.tick, "ask": a * x.tick})
            df["mid"] = 0.5 * (df.bid + df.ask)
            self.bbo[x.name] = df
        self.trades = {}
        for i, x in enumerate(sim.instruments):
            t, sd, tk, q = sim._r_trades[i]
            self.trades[x.name] = pd.DataFrame({"t": np.asarray(t, dtype=float), "side": np.asarray(sd, dtype=int),
                                                "price": np.asarray(tk, dtype=float) * x.tick, "qty": np.asarray(q, dtype=float)})
        rows = []
        for t, eq, pos, mid, cash, fees, fund in sim._r_equity:
            r = {"t": t, "equity": eq}
            for i, nm in enumerate(sim.names):
                r[f"pos_{nm}"], r[f"mid_{nm}"] = pos[i], mid[i]
                r[f"cash_{nm}"], r[f"fees_{nm}"], r[f"funding_{nm}"] = cash[i], fees[i], fund[i]
            rows.append(r)
        self.equity = pd.DataFrame(rows)
        self.funding = pd.DataFrame(sim._r_funding, columns=["t", "inst", "rate", "pos", "price", "payment"])
        self.logs = {k: pd.DataFrame({"t": t, "value": v}) for k, (t, v) in sim._r_logs.items()}
        self._fx_end = {x.name: sim._fx(i) for i, x in enumerate(sim.instruments)}
        self._spec = {x.name: x for x in sim.instruments}

    # ---- lookups ------------------------------------------------------------------------------
    def mid_at(self, inst: str, t) -> np.ndarray:
        """True mid of `inst` at times t (last value at or before t). NaN before the first quote or after the run."""
        b = self.bbo[inst]
        t = np.asarray(t, dtype=float)
        k = np.searchsorted(b.t.values, t, side="right") - 1
        out = np.where(k >= 0, b.mid.values[np.clip(k, 0, len(b) - 1)], np.nan)
        return np.where(t > self.t_end, np.nan, out)

    def ref_at(self, inst: str, t) -> np.ndarray:
        """Latent fair price of `inst` at times t (synthetic markets only)."""
        if self.truth is None:
            raise ValueError("no truth attached: this run was not on a synthetic market")
        tt = self.truth["t"]
        k = np.clip(np.searchsorted(tt, np.asarray(t, dtype=float), side="right") - 1, 0, len(tt) - 1)
        return self.truth["ref"][inst][k]

    @property
    def pnl(self) -> float:
        return float(self.equity.equity.iloc[-1]) if len(self.equity) else 0.0


def _final_status(o) -> str:
    if o.x_rem <= 1e-9:
        return "filled"
    if o.status in ("sent", "live", "cancelling"):
        return "open_at_end"
    return o.status


def _sel(res: Result, inst=None, tag=None, maker=None) -> pd.DataFrame:
    f = res.fills
    if inst is not None:
        f = f[f.inst == inst]
    if tag is not None:
        f = f[f.tag == tag]
    if maker is not None:
        f = f[f.maker == maker]
    return f


# ------------------------------------------------------------------------------------- markouts
def markouts(res: Result, horizons=(0.1, 1.0, 5.0, 30.0), inst=None, tag=None, maker=None,
             vs: str = "mid", unit: str = "bps") -> pd.DataFrame:
    """Per-fill markout at each horizon: side x (benchmark(t + h) - fill price).

    vs="mid" uses the true mid; vs="ref" uses the latent fair price (synthetic only), which removes
    bid-ask bounce from the picture. unit: "bps", "ticks" or "px". Horizon 0 is the edge at the fill.
    Fills too close to the end of the run to have a horizon get NaN."""
    f = _sel(res, inst, tag, maker)
    out = f[["id", "inst", "side", "price", "qty", "maker", "tag", "t"]].copy()
    look = res.mid_at if vs == "mid" else res.ref_at
    for h in horizons:
        col = np.full(len(f), np.nan)
        for nm in f.inst.unique():
            m = (f.inst == nm).values
            g = f[m]
            t = g.t.values + h
            bench = np.asarray(look(nm, t), dtype=float)
            bench = np.where(t > res.t_end, np.nan, bench)
            raw = g.side.values * (bench - g.price.values)
            if unit == "bps":
                raw = raw / g.price.values * 1e4
            elif unit == "ticks":
                raw = raw / res._spec[nm].tick
            col[m] = raw
        out[f"m{h:g}"] = col
    return out


def markout_curve(res: Result, horizons=(0.0, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0), **kw) -> pd.DataFrame:
    """Mean, median, 10th percentile and count of markouts at each horizon (quantity-weighted mean)."""
    m = markouts(res, horizons, **kw)
    rows = []
    for h in horizons:
        v = m[f"m{h:g}"].values
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            rows.append({"horizon": h, "n": 0})
            continue
        w = m.qty.values[ok]
        rows.append({"horizon": h, "n": int(ok.sum()), "mean": float(np.average(v[ok], weights=w)),
                     "median": float(np.median(v[ok])), "p10": float(np.percentile(v[ok], 10)),
                     "p90": float(np.percentile(v[ok], 90))})
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------------- attribution
def pnl_attribution(res: Result, horizon: float = 5.0) -> pd.DataFrame:
    """Where the money came from, per instrument and in total (base currency).

        spread_capture      passive fills: fill price vs mid at the fill
        hedge_cost          aggressive fills: fill price vs mid at the fill (normally negative)
        adverse_selection   mid move over the `horizon` seconds after each fill, in our position's direction
        inventory           mid move after that, while the position was held
        fees, funding       as paid
        total               = the sum = cash + position marked to the final mid

    The six components add up to `total` exactly for each instrument. `fx_translation` on the TOTAL row is
    the small remainder between the summed rows and the equity curve when a leg's currency rate moved."""
    rows = []
    for nm in res.names:
        x, f = res._spec[nm], res.fills[res.fills.inst == nm]
        fx, cs = res._fx_end[nm], x.contract_size
        b = res.bbo[nm]
        mid_end = b.mid.dropna().values[-1] if b.mid.notna().any() else np.nan
        if len(f):
            s, p, q = f.side.values, f.price.values, f.qty.values
            m0 = f.mid.values
            th = np.minimum(f.t.values + horizon, res.t_end)
            mh = res.mid_at(nm, th)
            mh = np.where(np.isnan(mh), m0, mh)
            edge = s * (m0 - p) * q * cs
            mk = f.maker.values.astype(bool)
            sc, hc = edge[mk].sum(), edge[~mk].sum()
            adv = (s * (mh - m0) * q * cs).sum()
            inv = (s * (mid_end - mh) * q * cs).sum()
            fees = -f.fee.sum()
        else:
            sc = hc = adv = inv = fees = 0.0
        fund = res.funding[res.funding.inst == nm].payment.sum() if len(res.funding) else 0.0
        comp = np.array([sc, hc, adv, inv, fees, fund]) * fx
        rows.append([nm, *comp, comp.sum()])
    df = pd.DataFrame(rows, columns=["inst", "spread_capture", "hedge_cost", "adverse_selection", "inventory",
                                     "fees", "funding", "total"]).set_index("inst")
    tot = df.sum()
    tot.name = "TOTAL"
    df = pd.concat([df, tot.to_frame().T])
    df["fx_translation"] = 0.0
    df.loc["TOTAL", "fx_translation"] = res.pnl - df.loc["TOTAL", "total"]
    return df


# ---------------------------------------------------------------------------- aggressive orders
def slippage(res: Result, tag=None, inst=None) -> pd.DataFrame:
    """Per aggressive (IOC/taker) order, in bps of the decision mid, positive = cost:

        delay_cost   the mid moved between deciding and the order reaching the exchange
        cross_cost   fill price vs the mid on arrival (half the spread plus walking the book)
        total        fill price vs the mid when we decided
        fill_ratio   filled / requested
    """
    o = res.orders
    o = o[o.tif == "IOC"]
    if tag is not None:
        o = o[o.tag == tag]
    if inst is not None:
        o = o[o.inst == inst]
    g = res.fills.assign(notional=res.fills.price * res.fills.qty).groupby("order_id")
    f = pd.DataFrame({"vwap": g.notional.sum() / g.qty.sum(), "t_fill": g.t.max()})
    o = o.join(f, on="id")
    d = o.mid_decision
    out = o[["id", "inst", "side", "qty", "filled", "tag", "parent", "t_submit", "t_arrive"]].copy()
    out["delay_ms"] = (o.t_arrive - o.t_submit) * 1e3
    out["delay_cost"] = o.side * (o.mid_arrive - d) / d * 1e4
    out["cross_cost"] = o.side * (o.vwap - o.mid_arrive) / d * 1e4
    out["total"] = o.side * (o.vwap - d) / d * 1e4
    out["fill_ratio"] = o.filled / o.qty
    return out


def spread_pairs(res: Result, quote_tag: str = "quote", hedge_tag: str = "hedge") -> pd.DataFrame:
    """One row per passive fill that was hedged (hedge orders carry parent = the quote fill's id).

        quote_edge_bps    passive fill vs its own mid at the fill
        hedge_cost_bps    hedge fill price vs the hedge leg's mid AT THE MOMENT OF THE QUOTE FILL, as a cost
        hedge_delay_ms    quote fill (exchange time) -> last hedge fill (exchange time)
        net_bps           quote_edge_bps - hedge_cost_bps: the spread actually locked in, before fees
    """
    q = res.fills[res.fills.tag == quote_tag]
    h = res.fills[(res.fills.tag == hedge_tag) & res.fills.parent.notna()]
    rows = []
    hg = {k: g for k, g in h.groupby("parent")}
    for r in q.itertuples():
        g = hg.get(r.id)
        row = {"fill_id": r.id, "t": r.t, "inst": r.inst, "side": r.side, "qty": r.qty,
               "quote_edge_bps": r.side * (r.mid - r.price) / r.price * 1e4}
        if g is not None and len(g):
            row["hedge_inst"] = g.inst.iloc[0]
            for nm, gg in g.groupby("inst"):
                vwap = np.average(gg.price, weights=gg.qty)
                m0 = float(res.mid_at(nm, [r.t])[0])
                cost = gg.side.iloc[0] * (vwap - m0) / m0 * 1e4
                row["hedge_cost_bps"] = row.get("hedge_cost_bps", 0.0) + cost
            row["hedge_delay_ms"] = (g.t.max() - r.t) * 1e3
            row["net_bps"] = row["quote_edge_bps"] - row["hedge_cost_bps"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- order stats
def maker_stats(res: Result, tag=None) -> pd.DataFrame:
    """Per instrument: passive orders sent, share filled, queue position on joining, time to first fill."""
    o = res.orders
    o = o[(o.tif == "GTC")]
    if tag is not None:
        o = o[o.tag == tag]
    first = res.fills.groupby("order_id").t.min()
    o = o.join(first.rename("t_first_fill"), on="id")
    rows = []
    for nm, g in o.groupby("inst"):
        ttf = (g.t_first_fill - g.t_arrive).dropna()
        rows.append({"inst": nm, "orders": len(g), "rejected": int((g.status == "rejected").sum()),
                     "filled_qty_share": g.filled.sum() / g.qty.sum(),
                     "orders_with_fill": float((g.filled > 0).mean()),
                     "queue_ahead_at_join": g.ahead0.mean(), "level_size_at_join": g.level0.mean(),
                     "median_secs_to_fill": ttf.median() if len(ttf) else np.nan,
                     "cancel_rejects": int(g.cancel_reject.sum())})
    return pd.DataFrame(rows).set_index("inst") if rows else pd.DataFrame()


def latency_stats(res: Result) -> pd.DataFrame:
    """Milliseconds: order entry (submit -> exchange), ack round trip, cancel entry. p50 / p95 / p99."""
    o = res.orders
    parts = {"entry": (o.t_arrive - o.t_submit), "ack_round_trip": (o.t_ack - o.t_submit),
             "cancel_entry": (o.t_cancel_arrive - o.t_cancel_req)}
    rows = []
    for k, v in parts.items():
        v = v.dropna().values * 1e3
        if len(v):
            rows.append({"hop": k, "n": len(v), "p50": np.percentile(v, 50), "p95": np.percentile(v, 95),
                         "p99": np.percentile(v, 99)})
    return pd.DataFrame(rows).set_index("hop") if rows else pd.DataFrame()


def fair_value_error(res: Result, key: str, inst: str) -> dict:
    """How far a fair value the strategy logged (sim.log(key, value)) was from the latent fair price."""
    lg = res.logs[key].dropna()
    ref = res.ref_at(inst, lg.t.values)
    e = (lg.value.values - ref) / ref * 1e4
    return {"n": len(e), "bias_bps": float(np.mean(e)), "rmse_bps": float(np.sqrt(np.mean(e ** 2))),
            "p95_abs_bps": float(np.percentile(np.abs(e), 95))}


# ------------------------------------------------------------------------------------- summary
def summary(res: Result, horizon: float = 5.0) -> dict:
    """One row of headline numbers for a run. Money in base currency, edges in bps."""
    f, e = res.fills, res.equity
    att = pnl_attribution(res, horizon).loc["TOTAL"]
    mk = f[f.maker]
    out = {"pnl": res.pnl, "fills": len(f), "maker_fills": len(mk)}
    for c in ["spread_capture", "hedge_cost", "adverse_selection", "inventory", "fees", "funding"]:
        out[c] = float(att[c])
    if len(mk):
        m = markouts(res, (0.0, horizon), maker=True)
        w = m.qty.values
        for h, nm in ((0.0, "maker_edge_bps"), (horizon, f"maker_markout_{horizon:g}s_bps")):
            v = m[f"m{h:g}"].values
            ok = ~np.isnan(v)
            out[nm] = float(np.average(v[ok], weights=w[ok])) if ok.any() else math.nan
    hd = slippage(res)
    if len(hd):
        out["taker_slippage_bps"] = float(np.average(hd.total.fillna(0), weights=hd.qty))
        out["taker_fill_ratio"] = float(hd.filled.sum() / hd.qty.sum())
    if len(e) > 2:
        d = np.diff(e.equity.values)
        out["pnl_per_step_sharpe"] = float(d.mean() / d.std()) if d.std() > 0 else math.nan
        out["max_drawdown"] = float((e.equity.cummax() - e.equity).max())
        for nm in res.names:
            out[f"max_abs_pos_{nm}"] = float(e[f"pos_{nm}"].abs().max())
    out["cancel_rejects"] = res.cancel_rejects
    return out


def compare(runs: dict[str, Result], horizon: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame({k: summary(v, horizon) for k, v in runs.items()}).T


def sweep(run: Callable[..., Result], grid: dict[str, list], seeds=(0,), horizon: float = 5.0,
          metrics: Callable[[Result], dict] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Run `run(seed=..., **params)` over every combination in `grid` and every seed.

    Returns one row per parameter combination: the mean of each metric across seeds, plus `<metric>_se`,
    its standard error. One seed is one sample; a difference smaller than a couple of standard errors
    is noise. Pass the same seeds to every configuration so they face the same markets."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        per = []
        for s in seeds:
            r = run(seed=s, **params)
            per.append(metrics(r) if metrics else summary(r, horizon))
        df = pd.DataFrame(per)
        row = dict(params)
        for c in df.columns:
            row[c] = df[c].mean()
            if len(seeds) > 1:
                row[c + "_se"] = df[c].std(ddof=1) / math.sqrt(len(seeds))
        rows.append(row)
        if verbose:
            print("  done", params, flush=True)
    return pd.DataFrame(rows).set_index(keys)
