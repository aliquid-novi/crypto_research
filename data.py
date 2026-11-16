"""Turning your own market data into the event stream the simulator replays.

The simulator needs, per instrument, level updates ("this price now holds this much") and trades.
Index prices and funding prints are optional. Three common starting points are covered:

  events_from_frames    you have incremental L2 updates and a trade tape
  snapshots_to_frame    you only have periodic top-N book snapshots; this diffs them into level updates
  mid_index             no index feed: use another leg's mid as the perp's index
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import DEPTH, FUNDING, INDEX, TRADE, Instrument

_ORDER = {TRADE: 0, DEPTH: 1, INDEX: 2, FUNDING: 3}


def events_from_frames(instruments: list[Instrument], depth: pd.DataFrame | None = None,
                       trades: pd.DataFrame | None = None, index: pd.DataFrame | None = None,
                       funding: pd.DataFrame | None = None) -> list[tuple]:
    """Merge DataFrames into one time-sorted event list.

        depth    columns: t, inst, side (+1 bid / -1 ask), price, qty   (qty = new size at that price; 0 removes it)
        trades   columns: t, inst, side (+1 buyer aggressor / -1 seller aggressor), price, qty
        index    columns: t, inst, price
        funding  columns: t, inst, rate

    `t` is seconds (float). At equal timestamps trades are placed before the depth updates that reflect
    them, which is what the queue logic expects. Start the depth frame with a full snapshot of each book."""
    idx = {x.name: i for i, x in enumerate(instruments)}
    parts = []

    def add(df, kind):
        if df is None or len(df) == 0:
            return
        t = df["t"].to_numpy(dtype=float)
        i = df["inst"].map(idx).to_numpy()
        if np.isnan(i.astype(float)).any():
            raise ValueError("unknown instrument name in data")
        if kind in (DEPTH, TRADE):
            tick = np.array([instruments[int(j)].tick for j in i])
            a = df["side"].to_numpy(dtype=int)
            b = np.rint(df["price"].to_numpy(dtype=float) / tick).astype(np.int64)
            c = df["qty"].to_numpy(dtype=float)
        else:
            a = df["price" if kind == INDEX else "rate"].to_numpy(dtype=float)
            b = np.zeros(len(df), dtype=np.int64)
            c = np.zeros(len(df))
        parts.append(pd.DataFrame({"t": t, "o": _ORDER[kind], "kind": kind, "i": i.astype(int), "a": a, "b": b, "c": c,
                                   "n": np.arange(len(df))}))

    add(trades, TRADE), add(depth, DEPTH), add(index, INDEX), add(funding, FUNDING)
    if not parts:
        return []
    ev = pd.concat(parts).sort_values(["t", "o", "n"], kind="stable")
    out = []
    for t, kind, i, a, b, c in zip(ev.t.values, ev.kind.values, ev.i.values, ev.a.values, ev.b.values, ev.c.values):
        if kind == DEPTH or kind == TRADE:
            out.append((float(t), int(kind), int(i), int(a), int(b), float(c)))
        else:
            out.append((float(t), int(kind), int(i), float(a), 0, 0.0))
    return out


def snapshots_to_frame(inst: str, t, bid_px, bid_qty, ask_px, ask_qty) -> pd.DataFrame:
    """Diff periodic top-N snapshots into a `depth` frame for events_from_frames.

    t: (n,) seconds. bid_px, bid_qty, ask_px, ask_qty: (n, N) arrays, best level first.
    A price that disappears is set to zero only if it lies inside the newly visible price range; a level
    that merely scrolled out of the top N is left as it was, because the data no longer says anything
    about it. Queue estimates from snapshot data are coarse: everything between two snapshots is netted."""
    t = np.asarray(t, dtype=float)
    rows = []
    for side, px, qty in ((1, np.asarray(bid_px, float), np.asarray(bid_qty, float)),
                          (-1, np.asarray(ask_px, float), np.asarray(ask_qty, float))):
        prev: dict[float, float] = {}
        for k in range(len(t)):
            cur = {p: q for p, q in zip(px[k], qty[k]) if q > 0 and p == p}
            if not cur:
                continue
            lo, hi = min(cur), max(cur)
            for p, q in cur.items():
                if prev.get(p) != q:
                    rows.append((t[k], inst, side, p, q))
            for p in prev:
                if p not in cur and lo <= p <= hi:
                    rows.append((t[k], inst, side, p, 0.0))
            # levels now through the new best are gone too (the market moved past them)
            for p in prev:
                if p not in cur and ((side == 1 and p > hi) or (side == -1 and p < lo)):
                    rows.append((t[k], inst, side, p, 0.0))
            prev = {p: q for p, q in prev.items() if not (lo <= p <= hi) and not ((side == 1 and p > hi) or (side == -1 and p < lo))}
            prev.update(cur)
    return pd.DataFrame(rows, columns=["t", "inst", "side", "price", "qty"]).sort_values("t", kind="stable")


def mid_index(other: str, scale: float = 1.0):
    """index_fn for Sim: the perp's index is another instrument's mid times `scale`."""
    return lambda sim: sim.true_book(other).mid * scale
