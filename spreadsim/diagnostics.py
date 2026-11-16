"""Per-fill diagnostics: what the market did around each of our fills, and what would have happened
had the order rested somewhere else.

    fill_table            one row per passive fill, with markouts, excursions, volatility state, queue, ...
    shift_counterfactual  "what if every clip had been k ticks further out?" answered from one run's records
    conditional           any column of the fill table, averaged within bins of any other column
    vol_series, premium_series   the observable series the bins are usually built from

Everything here uses only observable records (fills, quotes, the trade tape), so the same functions run
on a replay of real data. Columns that need the synthetic truth are added only when it exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import Result


def vol_series(res: Result, inst: str, halflife: float = 60.0, dt: float = 1.0) -> pd.DataFrame:
    """Recent volatility of an instrument's mid, in bps per sqrt(second): EWMA of squared dt-second returns."""
    grid = np.arange(res.t0 + dt, res.t_end, dt)
    mid = pd.Series(res.mid_at(inst, grid)).ffill().bfill().values
    r2 = np.r_[0.0, np.diff(np.log(mid)) * 1e4] ** 2 / dt
    return pd.DataFrame({"t": grid, "vol": np.sqrt(pd.Series(r2).ewm(halflife=halflife / dt).mean().values)})


def premium_series(res: Result, a: str, b: str, ratio: float = 1.0, dt: float = 1.0) -> pd.DataFrame:
    """ln(mid of a / (ratio x mid of b)) in bps on a regular grid: the observable spread."""
    grid = np.arange(res.t0 + dt, res.t_end, dt)
    return pd.DataFrame({"t": grid, "premium": np.log(res.mid_at(a, grid) / (ratio * res.mid_at(b, grid))) * 1e4})


def _at(t_series, v_series, t):
    k = np.clip(np.searchsorted(t_series, t, side="right") - 1, 0, len(t_series) - 1)
    return v_series[k]


def fill_table(res: Result, inst: str | None = None, tag: str | None = "quote", horizons=(1.0, 5.0, 30.0, 120.0),
               window: float = 20.0, vol_halflife: float = 60.0, fair_key: str = "fair", hedge: str | None = None,
               ratio: float = 1.0) -> pd.DataFrame:
    """One row per passive fill. All price distances are in ticks of the filled instrument and signed
    so that positive is good for us.

        k                 ladder level of the order (0 = nearest fair value), when the strategy set one
        m<h>              markout h seconds after the fill, against the filled instrument's own mid
        s<h>              (when `hedge` is given) SPREAD markout: the same, minus what the hedge leg did over
                          those h seconds. This is the markout of a hedged fill; m<h> also contains the
                          market move that the hedge cancels, so it is much noisier. Hedge costs not included.
        mae, mfe          worst and best the mid got within `window` seconds, relative to the fill price
        through, touch    how many ticks further out an order would still have filled within `window`,
                          letting that order follow the strategy's logged fair value as a real quote does:
                          `through` is certain (the market traded or quoted through that price),
                          `touch` is the optimistic bound (a trade printed at that price)
        vol, vol_state    recent volatility at the fill (of the spread when `hedge` is given, else of the
                          instrument); vol_state is 1 if above the run's median
        dev               distance of the fill from the strategy's logged fair value
        lean              book imbalance at the fill, signed toward our side (+1 = all size on our side)
        ahead0            queue ahead when the order joined
        edge_ref, regime  (synthetic only) distance from the latent fair price; true regime
    """
    f = res.fills[res.fills.maker]
    if tag is not None:
        f = f[f.tag == tag]
    if inst is not None:
        f = f[f.inst == inst]
    o = res.orders.set_index("id")
    out = f[["id", "order_id", "inst", "side", "price", "qty", "t"]].copy().reset_index(drop=True)
    out["k"] = o.k.reindex(f.order_id).values
    out["ahead0"] = o.ahead0.reindex(f.order_id).values
    n = len(out)
    cols = {f"m{h:g}": np.full(n, np.nan) for h in horizons}
    if hedge is not None:
        cols.update({f"s{h:g}": np.full(n, np.nan) for h in horizons})
    for c in ("mae", "mfe", "through", "touch", "vol", "dev", "lean", "edge_ref", "regime"):
        cols[c] = np.full(n, np.nan)
    fair = res.logs.get(fair_key)
    for nm in out.inst.unique():
        idx = np.flatnonzero((out.inst == nm).values)
        g = out.iloc[idx]
        tick = res._spec[nm].tick
        s, p, t = g.side.values, g.price.values, g.t.values
        for h in horizons:
            m = s * (res.mid_at(nm, t + h) - p) / tick
            cols[f"m{h:g}"][idx] = np.where(t + h > res.t_end, np.nan, m)
            if hedge is not None:
                hr = res.mid_at(hedge, t + h) / res.mid_at(hedge, t) - 1.0
                cols[f"s{h:g}"][idx] = np.where(t + h > res.t_end, np.nan, m - s * p * hr / tick)
        b = res.bbo[nm]
        bt, bmid, bbid, bask = b.t.values, b.mid.values, b.bid.values, b.ask.values
        tr = res.trades[nm]
        tt, ts, tp = tr.t.values, tr.side.values, tr.price.values
        if hedge is not None:
            ps = premium_series(res, nm, hedge, ratio)
            d2 = pd.Series(np.r_[0.0, np.diff(ps.premium.ffill().bfill().values)] ** 2)
            vs = pd.DataFrame({"t": ps.t, "vol": np.sqrt(d2.ewm(halflife=vol_halflife).mean().values)})
        else:
            vs = vol_series(res, nm, vol_halflife)
        cols["vol"][idx] = _at(vs.t.values, vs.vol.values, t)
        if fair is not None and len(fair):
            ft_, fv_ = fair.t.values, fair.value.values
        ff = f[f.inst == nm]
        cols["lean"][idx] = s * (ff.bid_qty.values - ff.ask_qty.values) / np.maximum(ff.bid_qty.values + ff.ask_qty.values, 1e-12)
        if fair is not None and len(fair):
            cols["dev"][idx] = s * (_at(fair.t.values, fair.value.values, t) - p) / tick
        if res.truth is not None:
            cols["edge_ref"][idx] = s * (res.ref_at(nm, t) - p) / tick
            cols["regime"][idx] = _at(res.truth["t"], res.truth["regime"], t)
        for j, i in enumerate(idx):
            if t[j] + window > res.t_end:
                continue
            a, z = np.searchsorted(bt, t[j], side="right"), np.searchsorted(bt, t[j] + window, side="right")
            mids = bmid[max(a - 1, 0):z]
            mids = mids[~np.isnan(mids)]
            if len(mids):
                cols["mae"][i] = min(0.0, (s[j] * (mids - p[j])).min()) / tick
                cols["mfe"][i] = max(0.0, (s[j] * (mids - p[j])).max()) / tick
            # How far past our price did the market go?  (buy: how far below; sell: how far above.)
            # A resting quote follows the strategy's fair value, so "our price" moves with it over the window.
            a2, z2 = np.searchsorted(tt, t[j], side="right"), np.searchsorted(tt, t[j] + window, side="right")
            m = ts[a2:z2] == -s[j]
            opp, opp_t = (bask if s[j] > 0 else bbid)[a:z], bt[a:z]
            ok = ~np.isnan(opp)
            opp, opp_t = opp[ok], opp_t[ok]
            if fair is not None and len(fair):
                f0 = _at(ft_, fv_, t[j])
                ours_tr = p[j] + (_at(ft_, fv_, tt[a2:z2][m]) - f0) if m.any() else None
                ours_q = p[j] + (_at(ft_, fv_, opp_t) - f0) if len(opp) else None
            else:
                ours_tr, ours_q = p[j], p[j]
            quoted = (s[j] * (ours_q - opp)).max() / tick if len(opp) else -np.inf        # opposite quote at or through us
            traded = (s[j] * (ours_tr - tp[a2:z2][m])).max() / tick if m.any() else -np.inf
            cols["touch"][i] = max(0.0, np.floor(traded + 1e-6), np.floor(quoted + 1e-6))
            cols["through"][i] = max(0.0, np.floor(traded + 1e-6) - 1, np.floor(quoted + 1e-6))
    for c, v in cols.items():
        out[c] = v
    med = np.nanmedian(out.vol.values) if n else np.nan
    out["vol_state"] = (out.vol > med).astype(int)
    if res.truth is None:
        out = out.drop(columns=["edge_ref", "regime"])
    return out


def shift_counterfactual(ft: pd.DataFrame, shifts=(1, 2, 3), value: str = "m120", short: str = "m5",
                         tick_value: float = 1.0, rule: str = "through", cost_ticks: float = 0.0) -> pd.DataFrame:
    """What moving every clip k ticks further out would have done, estimated from one run's fills.

    A fill is KEPT if an order k ticks further out would still have filled (`rule`: "through" is the
    cautious count, "touch" the generous one); it then earns k ticks more. Otherwise it is LOST, and with
    it whatever that fill went on to make: its `value` markout less `cost_ticks`, the per-fill cost of
    hedging and fees that a missed fill never pays. So, per original fill:

        net = share_kept x k  -  share_lost x (average value of the lost fills, after costs)

    For a hedged trade use the spread markouts (value="s120", short="s5").
    (`gain_kept` and `value_lost` in the table, both per original fill.) The table also shows the
    short-horizon markout of kept and lost fills separately, which is the selection at work: the fills
    you keep are the ones the market ran through.

    This is a description of one run's fills, not a forecast. It cannot see the fills a different ladder
    would have taken instead, or what the position capacity freed by a missed fill was worth. Check any
    conclusion drawn from it by re-running with the ladder actually moved.
    `tick_value` converts ticks x quantity into money (tick size x contract size x fx)."""
    d = ft.dropna(subset=[value, rule])
    rows = []
    for k in shifts:
        kept = d[rule].values >= k
        lost = ~kept
        v, q = d[value].values - cost_ticks, d.qty.values
        row = {"shift": k, "fills": len(d), "share_kept": kept.mean() if len(d) else np.nan,
               f"{short}_kept": d[short].values[kept].mean() if kept.any() else np.nan,
               f"{short}_lost": d[short].values[lost].mean() if lost.any() else np.nan,
               f"{value}_kept": d[value].values[kept].mean() if kept.any() else np.nan,
               f"{value}_lost": d[value].values[lost].mean() if lost.any() else np.nan}
        row["gain_kept"] = k * kept.mean() if len(d) else np.nan
        row["value_lost"] = (v[lost].sum() / len(d)) if len(d) else np.nan
        row["net_ticks_per_fill"] = row["gain_kept"] - row["value_lost"]
        row["net_money"] = float(((k * kept - v * lost) * q).sum() * tick_value)
        rows.append(row)
    return pd.DataFrame(rows).set_index("shift")


def conditional(df: pd.DataFrame, by, cols, bins: int | None = None) -> pd.DataFrame:
    """Mean, standard error, median and count of `cols` within groups of `by`.
    `by` may be one column or a list; pass `bins` to cut a numeric `by` column into quantile buckets."""
    by = [by] if isinstance(by, str) else list(by)
    d = df.copy()
    if bins is not None:
        d[by[0]] = pd.qcut(d[by[0]], bins, duplicates="drop")
    cols = [cols] if isinstance(cols, str) else list(cols)
    g = d.groupby(by, observed=True)
    parts = {"n": g[cols[0]].count()}
    for c in cols:
        parts[f"{c}_mean"] = g[c].mean()
        parts[f"{c}_se"] = g[c].std(ddof=1) / np.sqrt(g[c].count())
        parts[f"{c}_median"] = g[c].median()
    return pd.DataFrame(parts)
