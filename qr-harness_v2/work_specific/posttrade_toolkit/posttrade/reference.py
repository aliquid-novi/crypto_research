"""Reference prices: mid, weighted mid, Stoikov microprice, OFI, composite spread value, lead-venue mid.

A *reference series* is a DataFrame with columns [ts, venue, symbol, value] (ts int64 ns UTC,
value in price units), keyed by the (venue, symbol) of the FILLS it will be compared against.
Markout functions accept any reference series, so you can add your own fair value by building
a frame in that shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from ._util import NS, asof_values, group_keys, infer_tick, instrument_lookup


def mid(book: pd.DataFrame) -> pd.Series:
    """Mid price m = (bid_px + ask_px) / 2, price units."""
    return (book["bid_px"] + book["ask_px"]) / 2


def weighted_mid(book: pd.DataFrame) -> pd.Series:
    """Size-weighted mid  w = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz).

    Leans toward the side with LESS size (heavy bid -> price closer to ask). Falls back to mid
    when both sizes are zero.
    """
    den = book["bid_sz"] + book["ask_sz"]
    w = (book["bid_px"] * book["ask_sz"] + book["ask_px"] * book["bid_sz"]) / den.where(den > 0)
    return w.fillna(mid(book))


def imbalance(book: pd.DataFrame) -> pd.Series:
    """Queue imbalance I = bid_sz / (bid_sz + ask_sz) in [0, 1]; 0.5 when both are zero."""
    den = book["bid_sz"] + book["ask_sz"]
    return (book["bid_sz"] / den.where(den > 0)).fillna(0.5)


def to_ref(book: pd.DataFrame, value: pd.Series | np.ndarray) -> pd.DataFrame:
    """Package values computed on book rows as a reference series [ts, venue, symbol, value]."""
    out = book[["ts", "venue", "symbol"]].copy()
    out["value"] = np.asarray(value, dtype=float)
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def mid_ref(book: pd.DataFrame) -> pd.DataFrame:
    """Reference series of the mid."""
    return to_ref(book, mid(book))


# ---------------------------------------------------------------------------------------------
# Stoikov (2018) microprice
# ---------------------------------------------------------------------------------------------
@dataclass
class MicropriceModel:
    """Fitted Stoikov microprice. ``table`` = adjustment (price units) to add to the mid, indexed
    by spread in ticks (rows) and imbalance bin (cols). ``tick`` = tick size used."""

    table: pd.DataFrame
    tick: float
    n_imb: int
    n_spread: int
    G1: np.ndarray
    B: np.ndarray

    def adjustment(self, book: pd.DataFrame) -> np.ndarray:
        """Microprice adjustment per book row (price units). Spreads beyond the table use the widest row."""
        imb = imbalance(book).to_numpy()
        b = np.minimum((imb * self.n_imb).astype(int), self.n_imb - 1)
        sp = np.clip(np.round((book["ask_px"] - book["bid_px"]).to_numpy() / self.tick).astype(int), 1, self.n_spread)
        return self.table.to_numpy()[sp - 1, b]

    def apply(self, book: pd.DataFrame) -> pd.Series:
        """Microprice P_micro = mid + G*(spread, imbalance), price units."""
        return mid(book) + self.adjustment(book)


def fit_microprice(book: pd.DataFrame, tick: float | None = None, n_imb: int = 10, n_spread: int = 2,
                   max_iter: int = 30, tol: float = 1e-10) -> MicropriceModel:
    """Estimate the Stoikov microprice adjustment G*(I, S) from consecutive L1 updates.

    State x = (spread in ticks clipped to 1..n_spread, imbalance bin of I = bid_sz/(bid_sz+ask_sz)).
    From consecutive rows (per venue/symbol) count transitions with no mid change (Q), with a mid
    change to state y (T) and the mid change K (only |dM| <= 1 tick kept). Data are symmetrised
    (I -> 1-I, dM -> -dM). Then
        G1 = (I - Q)^-1 R K,   B = (I - Q)^-1 T,   G* = sum_{i>=0} B^i G1.
    Units: G* in price units (add to the mid). Fit one model per instrument family: pass only that
    instrument's rows. Note: fitted on the sample you pass, so for out-of-sample use, fit on a
    previous day and ``apply`` to the next.
    """
    tick = tick or infer_tick(np.concatenate([book["bid_px"].to_numpy(), book["ask_px"].to_numpy()]))
    rows = []
    for _, g in book.groupby(group_keys(book), sort=False):
        g = g.sort_values("ts", kind="stable")
        imb = imbalance(g).to_numpy()
        sp = np.clip(np.round((g["ask_px"] - g["bid_px"]).to_numpy() / tick).astype(int), 1, n_spread)
        b = np.minimum((imb * n_imb).astype(int), n_imb - 1)
        m = mid(g).to_numpy()
        dm = np.round((m[1:] - m[:-1]) / (tick / 2)).astype(int)  # half ticks
        rows.append(pd.DataFrame({"s0": sp[:-1], "b0": b[:-1], "s1": sp[1:], "b1": b[1:], "dm": dm}))
    d = pd.concat(rows, ignore_index=True)
    d = d[d["dm"].abs() <= 2]
    flip = d.assign(b0=n_imb - 1 - d["b0"], b1=n_imb - 1 - d["b1"], dm=-d["dm"])
    d = pd.concat([d, flip], ignore_index=True)
    nS = n_spread * n_imb
    x = ((d["s0"] - 1) * n_imb + d["b0"]).to_numpy()
    y = ((d["s1"] - 1) * n_imb + d["b1"]).to_numpy()
    cnt = np.bincount(x, minlength=nS).astype(float)
    cnt[cnt == 0] = np.inf
    z = d["dm"].to_numpy() == 0
    Q = np.zeros((nS, nS))
    np.add.at(Q, (x[z], y[z]), 1)
    T = np.zeros((nS, nS))
    np.add.at(T, (x[~z], y[~z]), 1)
    Ks = np.array([-2, -1, 1, 2])
    R = np.zeros((nS, len(Ks)))
    for j, kv in enumerate(Ks):
        mk = d["dm"].to_numpy() == kv
        np.add.at(R[:, j], x[mk], 1)
    Q /= cnt[:, None]
    T /= cnt[:, None]
    R /= cnt[:, None]
    K = Ks * tick / 2
    inv = np.linalg.inv(np.eye(nS) - Q)
    G1 = inv @ (R @ K)
    B = inv @ T
    G = G1.copy()
    Gk = G1.copy()
    for _ in range(max_iter):
        Gk = B @ Gk
        G += Gk
        if np.max(np.abs(Gk)) < tol:
            break
    table = pd.DataFrame(G.reshape(n_spread, n_imb), index=pd.Index(range(1, n_spread + 1), name="spread_ticks"),
                         columns=pd.Index(np.round((np.arange(n_imb) + 0.5) / n_imb, 3), name="imbalance_bin_mid"))
    return MicropriceModel(table, tick, n_imb, n_spread, G1, B)


# ---------------------------------------------------------------------------------------------
# Order-flow imbalance (Cont, Kukanov, Stoikov 2014)
# ---------------------------------------------------------------------------------------------
def ofi(book: pd.DataFrame) -> pd.DataFrame:
    """Per-update OFI e_n (size units), per venue/symbol:

        e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
            - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}

    Positive = net buying pressure. First row of each group is 0. Returns [ts, venue, symbol, ofi].
    """
    out = []
    for _, g in book.groupby(group_keys(book), sort=False):
        g = g.sort_values("ts", kind="stable")
        pb, qb, pa, qa = (g[c].to_numpy(dtype=float) for c in ("bid_px", "bid_sz", "ask_px", "ask_sz"))
        e = np.zeros(len(g))
        e[1:] = ((pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1]
                 - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1])
        out.append(pd.DataFrame({"ts": g["ts"].to_numpy(), "venue": g["venue"].to_numpy(),
                                 "symbol": g["symbol"].to_numpy(), "ofi": e}))
    return pd.concat(out, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)


def windowed_sum(events: pd.DataFrame, col: str, window_s: float, at_ts: np.ndarray | None = None,
                 query: pd.DataFrame | None = None) -> np.ndarray:
    """Trailing (backward-looking) sum of ``events[col]`` over (t - window, t], per venue/symbol.

    If ``query`` (DataFrame with ts, venue, symbol) is given, evaluates at those rows; otherwise at
    each event row. Strictly causal: only events with ts <= t are included.
    """
    q = events if query is None else query
    out = np.full(len(q), np.nan)
    w = int(window_s * NS)
    ek = group_keys(events).to_numpy()
    qk = group_keys(q).to_numpy()
    e_idx = pd.Series(np.arange(len(events))).groupby(ek).indices
    for key, qi in pd.Series(np.arange(len(q))).groupby(qk).indices.items():
        ei = e_idx.get(key)
        if ei is None:
            continue
        ts = events["ts"].to_numpy()[ei]
        o = np.argsort(ts, kind="stable")
        ts = ts[o]
        cs = np.concatenate([[0.0], np.cumsum(events[col].to_numpy(dtype=float)[ei][o])])
        t = q["ts"].to_numpy()[qi]
        hi = np.searchsorted(ts, t, side="right")
        lo = np.searchsorted(ts, t - w, side="right")
        out[qi] = cs[hi] - cs[lo]
    return out


# ---------------------------------------------------------------------------------------------
# Spread / pair references
# ---------------------------------------------------------------------------------------------
def composite_spread(book: pd.DataFrame, leg_a: tuple[str, str], leg_b: tuple[str, str],
                     hedge_ratio: float | pd.DataFrame, log: bool = False, on: str = "union") -> pd.DataFrame:
    """Two-leg spread value S_t = m_A(t) - h * m_B(t)  (or log m_A - h log m_B if ``log``).

    leg_a / leg_b = (venue, symbol). ``hedge_ratio`` is a scalar h (units of B per unit of A) or a
    DataFrame [ts, hedge_ratio] (e.g. a causal Kalman estimate) joined backward as-of.
    Evaluated at every update of either leg (``on='union'``) or of leg A only (``on='A'``).
    Each leg is taken backward as-of (latest value at or before t, never interpolated).
    Returns [ts, value, mid_A, mid_B, age_A_ns, age_B_ns, hedge_ratio].
    """
    def leg(v, s):
        g = book[(book["venue"] == v) & (book["symbol"] == s)]
        return g.assign(m=mid(g))[["ts", "m"]].sort_values("ts", kind="stable")

    a, b = leg(*leg_a), leg(*leg_b)
    ts = np.unique(np.concatenate([a["ts"].to_numpy(), b["ts"].to_numpy()])) if on == "union" else a["ts"].to_numpy()
    q = pd.DataFrame({"ts": ts, "venue": "_", "symbol": "_"})
    ja = asof_values(q, a.assign(venue="_", symbol="_"), ["m"])
    jb = asof_values(q, b.assign(venue="_", symbol="_"), ["m"])
    if isinstance(hedge_ratio, pd.DataFrame):
        h = asof_values(q, hedge_ratio.assign(venue="_", symbol="_"), ["hedge_ratio"])["hedge_ratio"].to_numpy()
    else:
        h = np.full(len(q), float(hedge_ratio))
    ma, mb = ja["m"].to_numpy(), jb["m"].to_numpy()
    val = np.log(ma) - h * np.log(mb) if log else ma - h * mb
    return pd.DataFrame({"ts": ts, "value": val, "mid_A": ma, "mid_B": mb, "age_A_ns": ja["age_ns"].to_numpy(),
                         "age_B_ns": jb["age_ns"].to_numpy(), "hedge_ratio": h}).dropna(subset=["value"])


def lead_mid(book: pd.DataFrame, lead_map: dict[tuple[str, str], tuple[str, str]] | None = None,
             delay_s: float = 0.0, scale: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    """Lead-venue mid relabelled with OUR (venue, symbol) so it can be used as a markout reference.

    lead_map: {(lead_venue, lead_symbol): (our_venue, our_symbol)} (default config.LEAD_MAP).
    delay_s: added to lead timestamps = how long after the lead's exchange time we could have
    seen it (one-way latency). Use 0 for "the lead as it truly was"; use your measured latency
    for "the lead as we could have known it".
    scale: optional multiplicative basis per lead key (e.g. contract size or FX) so prices compare.
    Returns reference series [ts, venue, symbol, value].
    """
    lead_map = lead_map or config.LEAD_MAP
    out = []
    for (lv, ls), (ov, os_) in lead_map.items():
        g = book[(book["venue"] == lv) & (book["symbol"] == ls)]
        if g.empty:
            continue
        k = (scale or {}).get((lv, ls), 1.0)
        out.append(pd.DataFrame({"ts": g["ts"].to_numpy() + int(delay_s * NS), "venue": ov, "symbol": os_,
                                 "value": mid(g).to_numpy() * k}))
    if not out:
        return pd.DataFrame(columns=["ts", "venue", "symbol", "value"])
    return pd.concat(out, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)


def trailing_vol(ref: pd.DataFrame, window_s: float = 300.0, sample_s: float = 1.0) -> pd.DataFrame:
    """Trailing realised vol of a reference series, bps per sqrt(second), per venue/symbol.

    The ref is sampled on a ``sample_s`` grid by forward fill (never interpolated); log returns;
    rolling std over the trailing ``window_s`` (backward only); divided by sqrt(sample_s).
    Returns [ts, venue, symbol, vol_bps] on the grid.
    """
    out = []
    step = int(sample_s * NS)
    for _, g in ref.groupby(group_keys(ref), sort=False):
        g = g.sort_values("ts", kind="stable")
        ts = g["ts"].to_numpy()
        grid = np.arange(ts[0] - ts[0] % step + step, ts[-1] + 1, step, dtype="int64")
        j = np.searchsorted(ts, grid, side="right") - 1
        v = g["value"].to_numpy(dtype=float)[j]
        r = pd.Series(np.diff(np.log(v), prepend=np.nan))
        n = max(int(window_s / sample_s), 2)
        vol = r.rolling(n, min_periods=max(n // 4, 2)).std().to_numpy() * 1e4 / np.sqrt(sample_s)
        out.append(pd.DataFrame({"ts": grid, "venue": g["venue"].iat[0], "symbol": g["symbol"].iat[0], "vol_bps": vol}))
    return pd.concat(out, ignore_index=True)


def tick_sizes(book: pd.DataFrame, instruments: pd.DataFrame | None = None) -> dict[tuple[str, str], float]:
    """Tick size per (venue, symbol): instruments.tick_size if given, else inferred from book prices."""
    out = {}
    for (v, s), g in book.groupby(["venue", "symbol"]):
        t = instrument_lookup(g.head(1), instruments, "tick_size", None)[0]
        out[(v, s)] = t if np.isfinite(t) else infer_tick(np.concatenate([g["bid_px"].to_numpy(), g["ask_px"].to_numpy()]))
    return out
