"""Functions behind mm_feasibility.ipynb: quote Hyperliquid perps, hedge on Binance perps.

The notebook holds the config and calls one function per step; everything computational lives here.
Usage:  import mm_feasibility_lib as mm;  mm.setup(**CONFIG_NAMES);  then the section functions in order.

Config is passed once to `setup()`, which stores it as module globals (the functions read them by name) and
resets the per-run state: RESULTS (gate metrics), CHECKS (synthetic ground-truth checks), WARNINGS_LOG, DQ.

Only pandas, numpy, matplotlib and pyarrow. All cross-venue joins are strictly backward as-of (`asof_idx`).
"""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  parquet engine
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from IPython.display import display

MS, S = 1_000_000, 1_000_000_000
HOUR_MS = 3_600_000
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False})

CONFIG_KEYS = [
    "DATA_SOURCE", "DATA_FORMAT", "DATES", "COINS", "SYMBOLS", "PATHS", "COLMAP", "TS_TZ", "AGGRESSOR_MAP",
    "FEES_BPS", "EXIT_LEG", "TAU_MS", "MARKOUT_HORIZONS_MS", "FV_LAG_MS", "JOIN_CLOCK", "THRESHOLDS",
    "SIZE_BUCKETS_USD", "PRE_WINDOW_MS", "BN_MOVE_BUCKETS_BPS", "OFFSET_TARGET", "OFFSET_MAX_WAIT_S",
    "OFFSET_BN_WINDOW_MS", "BASIS_RESAMPLE_S", "HEDGE_SIZE_USD", "XCORR_GRID_MS", "XCORR_LAGS_MS",
    "REFLECT_WINDOW_MS", "REFLECT_H_MS", "CAPACITY_SHARE", "QUOTE_SIZE_USD", "BIG_MOVE_Q", "CLOCK_WARN_MS",
    "BOOT_BLOCK_S", "N_BOOT", "SEED", "CACHE_DIR", "REBUILD_CACHE", "SYNTH",
]

SCHEMA = {
    "book": ["ts_exchange", "ts_local", "venue", "symbol", "bid_px", "bid_sz", "ask_px", "ask_sz"],
    "trades": ["ts_exchange", "ts_local", "venue", "symbol", "px", "sz", "aggressor_side"],
    "own_orders": ["order_id", "send_ts", "ack_ts", "cancel_send_ts", "cancel_ack_ts", "fill_ts",
                   "fill_notified_ts", "px", "sz", "side"],
}
TS_COLS = {"ts_exchange", "ts_local", "send_ts", "ack_ts", "cancel_send_ts", "cancel_ack_ts", "fill_ts",
           "fill_notified_ts"}
GATES = {"1 latency": ["ack_latency_p99_ms"], "2 room": ["room_median_bps"], "3 adverse sel.": ["net_edge_tau_bps"],
         "4 getting out": ["basis_half_life_s", "offset_median_s"],
         "5 lead-lag": ["frac_reflected_tau", "bn_move_tau_p99_bps"],
         "6 capacity": ["capacity_usd_day_touch", "capacity_usd_day_through"]}


# ============================================ setup & helpers ============================================

def setup(**cfg):
    """Store the notebook's config as module globals, derive TAU-dependent settings, reset run state."""
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    assert not missing, f"config missing: {missing}"
    g = globals()
    g.update({k: cfg[k] for k in CONFIG_KEYS})
    g["SYNTHETIC"] = DATA_SOURCE == "synthetic"
    tau = TAU_MS if TAU_MS is not None else (SYNTH["tau_ms"] if SYNTHETIC else None)
    assert tau is not None, "Set TAU_MS (your stale window) before running on real data."
    assert all(l % XCORR_GRID_MS == 0 for l in XCORR_LAGS_MS), "XCORR_LAGS_MS must be multiples of XCORR_GRID_MS"
    _tau = lambda v: tau if v == "TAU" else v
    g.update(TAU=tau, HORIZONS=[_tau(h) for h in MARKOUT_HORIZONS_MS], PRE_W=_tau(PRE_WINDOW_MS),
             OFF_BN_W=_tau(OFFSET_BN_WINDOW_MS), REFLECT_H=[_tau(h) for h in REFLECT_H_MS],
             CACHE=Path(CACHE_DIR) / DATA_SOURCE,
             COIN_COLOR={c: SERIES[i % len(SERIES)] for i, c in enumerate(COINS)},
             WAIT_EDGES=np.concatenate([[0], np.geomspace(0.01, OFFSET_MAX_WAIT_S, 80)]),
             MOVE_EDGES=np.concatenate([[0], np.geomspace(1e-3, 1e3, 400)]),
             RESULTS={k: {} for k in THRESHOLDS}, CHECKS={}, WARNINGS_LOG=[], DQ=[], TRUTH={})
    g["FEE_RT"] = roundtrip_fee_bps()
    print(f"source={DATA_SOURCE}  TAU={TAU} ms  horizons={HORIZONS}  join clock={JOIN_CLOCK}  "
          f"round-trip fees={FEE_RT:.2f} bps")


def loud(msg):
    """Print a warning you can't miss and keep it for the end-of-notebook list."""
    WARNINGS_LOG.append(msg)
    print("\n" + "!" * 100 + f"\n!!! WARNING: {msg}\n" + "!" * 100)
    warnings.warn(msg)


def check(name, ok, detail):
    """Record a synthetic ground-truth check and assert it. No-op on real data (no ground truth)."""
    if not SYNTHETIC:
        return
    CHECKS[name] = (bool(ok), detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    assert ok, f"synthetic check failed: {name}: {detail}"


def asof_idx(right_ts, left_ts, strict=False):
    """Index of the last right row with ts <= left (< left if strict); -1 if none. right_ts must be sorted."""
    return np.searchsorted(right_ts, left_ts, side="left" if strict else "right") - 1


def asof_vals(right_ts, right_vals, left_ts, strict=False):
    """Backward as-of lookup of right_vals at left_ts; NaN before the first right row."""
    i = asof_idx(right_ts, left_ts, strict)
    out = right_vals[np.clip(i, 0, None)].astype(float)
    out[i < 0] = np.nan
    return out


def block_sums(ts, x, w=None, keys=None):
    """Per (keys, time-block) sums of w*x and w: enough to bootstrap means across coin-days without keeping rows."""
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w)
    d = pd.DataFrame({"blk": ts[ok] // (BOOT_BLOCK_S * S), "swx": (w * x)[ok], "sw": w[ok], "n": 1})
    for k, v in (keys or {}).items():
        d[k] = np.asarray(v)[ok]
    return d.groupby([*(keys or {}), "blk"], observed=True).sum().reset_index()


def mean_ci(b, seed=None):
    """Weighted mean and 95% block-bootstrap CI from block sums (columns swx, sw, n)."""
    swx, sw = b["swx"].to_numpy(float), b["sw"].to_numpy(float)
    if len(sw) == 0 or sw.sum() == 0:
        return pd.Series({"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0})
    idx = np.random.default_rng(SEED if seed is None else seed).integers(0, len(sw), (N_BOOT, len(sw)))
    lo, hi = np.percentile(swx[idx].sum(1) / sw[idx].sum(1), [2.5, 97.5])
    return pd.Series({"mean": swx.sum() / sw.sum(), "lo": lo, "hi": hi, "n": int(b["n"].sum())})


def roundtrip_fee_bps():
    """Fees per unit: HL maker entry plus the exit leg (EXIT_LEG)."""
    exit_fee = FEES_BPS["binance"]["taker"] if EXIT_LEG == "binance_taker" else FEES_BPS["hyperliquid"]["maker"]
    return FEES_BPS["hyperliquid"]["maker"] + exit_fee


def coin_days():
    """All (coin, date) pairs: the unit of work, one in memory at a time."""
    return [(c, d) for c in COINS for d in DATES]


def date_axis(ax):
    """Compact date ticks so time-series labels don't collide."""
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# ============================================ adapters & cache ============================================

def to_utc_ns(s, unit):
    """Raw timestamps -> UTC int64 ns (nullable Int64 if there are gaps). unit: s|ms|us|ns|datetime."""
    if unit == "datetime":
        d = pd.to_datetime(s)
        if d.dt.tz is None:
            if TS_TZ is None:
                raise ValueError(f"{s.name}: tz-naive datetimes. Set TS_TZ explicitly; I won't guess.")
            d = d.dt.tz_localize(TS_TZ)
        return d.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64")
    mult = {"s": S, "ms": MS, "us": 1_000, "ns": 1}[unit]
    if pd.api.types.is_integer_dtype(s) and not s.isna().any():
        return s.astype("int64") * mult
    return pd.Series(np.round(s.astype(float) * mult), index=s.index).astype("Int64")


def read_raw(table, venue, coin, date):
    """Read one coin-day of one raw table, only the mapped columns. None if not configured."""
    tpl = PATHS[table].get(venue)
    if tpl is None:
        return None
    sym = SYMBOLS.get(venue, {}).get(coin, coin)
    path = tpl.format(date=date, coin=coin, symbol=sym)
    cmap = COLMAP[table][venue]
    cols = sorted({v[0] if isinstance(v, tuple) else v for v in cmap.values() if v})
    sym_col = cmap.get("symbol")
    need_filter = "{coin}" not in tpl and "{symbol}" not in tpl and sym_col
    if DATA_FORMAT == "parquet":
        df = pd.read_parquet(path, columns=cols, filters=[(sym_col, "==", sym)] if need_filter else None)
    else:
        df = pd.read_csv(path, usecols=cols)
        df = df[df[sym_col] == sym] if need_filter else df
    return df.reset_index(drop=True)


def adapt(raw, table, venue, coin):
    """Map a raw frame into the canonical schema for `table` (column mappings live in COLMAP)."""
    cmap, out = COLMAP[table][venue], {}
    for col in SCHEMA[table]:
        src = cmap.get(col)
        if col in TS_COLS:
            out[col] = to_utc_ns(raw[src[0]], src[1])
        elif col == "venue":
            out[col] = venue
        elif col == "symbol":
            out[col] = SYMBOLS.get(venue, {}).get(coin, coin)
        elif col in ("aggressor_side", "side"):
            mapped = raw[src].map(AGGRESSOR_MAP)
            if mapped.isna().any():
                raise ValueError(f"{table}.{col}: values {raw[src][mapped.isna()].unique()[:5]} not in AGGRESSOR_MAP")
            out[col] = mapped.astype("int8")
        else:
            out[col] = raw[src].to_numpy() if col == "order_id" else raw[src].astype(float).to_numpy()
    return pd.DataFrame(out)


def clean_and_write(df, table, venue, coin, date):
    """Drop unusable book rows, sort by ts_local, write canonical parquet, log counts."""
    n0 = len(df)
    if table == "book":
        df = df[(df.bid_px > 0) & (df.ask_px > df.bid_px)]
    sort_col = "send_ts" if table == "own_orders" else "ts_local"
    df = df.sort_values(sort_col, kind="stable").reset_index(drop=True)
    name = table if table == "own_orders" else f"{table}_{venue}"
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE / f"{coin}_{date}_{name}.parquet", index=False)
    DQ.append({"coin": coin, "date": date, "table": name, "rows": n0, "dropped": n0 - len(df)})


def load(coin, date, name, columns=None):
    """Canonical cached table for one coin-day (name e.g. 'book_binance'); None if absent."""
    p = CACHE / f"{coin}_{date}_{name}.parquet"
    return pd.read_parquet(p, columns=columns) if p.exists() else None


def load_frames(coin, date):
    """The three canonical market tables for one coin-day."""
    return {n: load(coin, date, n) for n in ("book_hyperliquid", "book_binance", "trades_hyperliquid")}


def build_data():
    """Fill the canonical cache one coin-day at a time (synthetic generator or adapters). Returns DQ counts."""
    if REBUILD_CACHE or not CACHE.exists():
        DQ.clear()
        for k, (coin, date) in enumerate(coin_days()):
            if SYNTHETIC:
                frames, TRUTH[(coin, date)] = synth_coin_day(coin, date, SYNTH["seed"] + k)
                for name, df in frames.items():
                    table, venue = ("own_orders", "hyperliquid") if name == "own_orders" else name.split("_")
                    clean_and_write(df, table, venue, coin, date)
                del frames   # one coin-day in memory at a time
            else:
                for table, venues in PATHS.items():
                    for venue in venues:
                        raw = read_raw(table, venue, coin, date)
                        if raw is not None:
                            clean_and_write(adapt(raw, table, venue, coin), table, venue, coin, date)
        print(f"cache built in {CACHE.resolve()}")
    elif SYNTHETIC:
        for k, (coin, date) in enumerate(coin_days()):   # ground truth is cheap to regenerate
            TRUTH[(coin, date)] = synth_coin_day(coin, date, SYNTH["seed"] + k)[1]
    return pd.DataFrame(DQ)


# ============================================ synthetic market ============================================

def _local(ts_exch, venue, rng):
    """Exchange -> local timestamp: fixed latency + exponential jitter, FIFO (monotone) like a real feed."""
    fixed, jit = SYNTH["lat_ms"][venue]
    return np.maximum.accumulate(ts_exch + ((fixed + rng.exponential(jit, len(ts_exch))) * MS).astype(np.int64))


def _ou_path(n, dt_s, half_life_s, sd, rng):
    """Exact OU discretisation on a regular grid, stationary sd `sd`, zero mean."""
    phi = math.exp(-math.log(2) * dt_s / half_life_s)
    eps = (rng.standard_normal(n) * sd * math.sqrt(1 - phi * phi)).tolist()
    out, b = [0.0] * n, rng.standard_normal() * sd
    for k in range(n):           # sequential recursion; numpy has no vectorised AR(1) filter
        b = phi * b + eps[k]
        out[k] = b
    return np.array(out)


def synth_coin_day(coin, date, seed):
    """Generate canonical book/trade/own_order frames for one coin-day plus its ground truth."""
    p, rng = SYNTH, np.random.default_rng(seed)
    t0, T = pd.Timestamp(date, tz="UTC").value, int(SYNTH["hours"] * 3600 * S)
    p0, sym = p["p0"].get(coin, 100.0), {v: SYMBOLS[v].get(coin, coin) for v in SYMBOLS}

    # HL taker arrivals; informed ones trigger a later BN jump in their direction
    t_tr = t0 + np.sort(rng.integers(0, T, rng.poisson(p["hl_trade_hz"] * T / S)))
    side = rng.choice(np.array([-1, 1], dtype=np.int8), len(t_tr))
    informed = rng.random(len(t_tr)) < p["informed_frac"]

    # BN log-mid: diffusion at Poisson update times + planted jumps
    t_diff = t0 + np.sort(rng.integers(0, T, rng.poisson(p["bn_update_hz"] * T / S)))
    t_jump = t_tr[informed] + int(p["jump_delay_ms"] * MS)
    jumps = side[informed] * p["informed_markout_bps"] / 1e4
    t_bn = np.concatenate([t_diff, t_jump])
    order = np.argsort(t_bn, kind="stable")
    t_bn, jump_all = t_bn[order], np.concatenate([np.zeros(len(t_diff)), jumps])[order]
    keep = t_bn < t0 + T
    t_bn, jump_all = t_bn[keep], jump_all[keep]
    dt_s = np.diff(t_bn, prepend=t0) / S
    sig = p["bn_vol_bps_sqrt_s"] / 1e4 * (1 + p["vol_diurnal"] * np.sin(2 * np.pi * (t_bn - t0) / (24 * 3600 * S)))
    lm_bn = math.log(p0) + np.cumsum(sig * np.sqrt(dt_s) * rng.standard_normal(len(t_bn)) + jump_all)

    # OU basis on a 100 ms grid
    g_dt = 100 * MS
    basis = p["basis_mean_bps"] / 1e4 + _ou_path(T // g_dt + 1, g_dt / S, p["basis_half_life_s"],
                                                 p["basis_sd_bps"] / 1e4, rng)

    # HL log-mid = BN log-mid delayed by hl_lag_ms + basis
    t_hl = t0 + np.sort(rng.integers(0, T, rng.poisson(p["hl_update_hz"] * T / S)))
    i_bn = np.clip(asof_idx(t_bn, t_hl - int(p["hl_lag_ms"] * MS)), 0, None)
    lm_hl = lm_bn[i_bn] + basis[(t_hl - t0) // g_dt]

    def book(ts, lm, hs_bps, touch_usd, venue):
        hs = hs_bps * np.exp(p["spread_noise"] * rng.standard_normal(len(ts))) / 1e4
        mid = np.exp(lm)
        return pd.DataFrame({"ts_exchange": ts, "ts_local": _local(ts, venue, rng), "venue": venue,
                             "symbol": sym[venue], "bid_px": mid * (1 - hs),
                             "bid_sz": touch_usd * rng.lognormal(0, 0.5, len(ts)) / mid,
                             "ask_px": mid * (1 + hs), "ask_sz": touch_usd * rng.lognormal(0, 0.5, len(ts)) / mid})

    bn = book(t_bn, lm_bn, p["bn_half_spread_bps"], p["bn_touch_usd"], "binance")
    hl = book(t_hl, lm_hl, p["hl_half_spread_bps"], p["hl_touch_usd"], "hyperliquid")

    # HL prints at the touch just before the trade; size beyond the touch prints one tick through
    j = asof_idx(t_hl, t_tr, strict=True)
    ok = j >= 0
    t_tr, side, j = t_tr[ok], side[ok], j[ok]
    px = np.where(side == 1, hl.ask_px.to_numpy()[j], hl.bid_px.to_numpy()[j])
    touch = np.where(side == 1, hl.ask_sz.to_numpy()[j], hl.bid_sz.to_numpy()[j])
    sz = p["trade_usd_median"] * rng.lognormal(0, p["trade_usd_sigma"], len(t_tr)) / px
    sweep = sz > touch
    px2 = px[sweep] * (1 + side[sweep] * p["tick_bps"] / 1e4)
    tr = pd.DataFrame({"ts_exchange": np.concatenate([t_tr, t_tr[sweep]]),
                       "px": np.concatenate([px, px2]),
                       "sz": np.concatenate([np.minimum(sz, touch), (sz - touch)[sweep]]),
                       "aggressor_side": np.concatenate([side, side[sweep]]).astype(np.int8)})
    tr = tr.sort_values("ts_exchange", kind="stable").reset_index(drop=True)
    tr.insert(1, "ts_local", _local(tr.ts_exchange.to_numpy(), "hyperliquid", rng))
    tr.insert(2, "venue", "hyperliquid")
    tr.insert(3, "symbol", sym["hyperliquid"])

    # own orders: planted lognormal latency medians
    n = p["own_orders_per_day"]
    send = t0 + np.sort(rng.integers(0, T, n))
    lat = lambda med: (med * rng.lognormal(0, 0.5, n) * MS).astype(np.int64)
    ack = send + lat(p["ack_ms"])
    cancelled = rng.random(n) < 0.7
    c_send = ack + (rng.exponential(2.0, n) * S).astype(np.int64)
    c_ack = c_send + lat(p["cancel_ms"])
    f_ts = ack + (rng.exponential(5.0, n) * S).astype(np.int64)
    f_not = f_ts + lat(p["fill_notify_ms"])
    na = lambda v, m: pd.Series(v, dtype="Int64").mask(~m)   # missing for orders that never got there
    oo = pd.DataFrame({"order_id": np.arange(n), "send_ts": send, "ack_ts": ack,
                       "cancel_send_ts": na(c_send, cancelled), "cancel_ack_ts": na(c_ack, cancelled),
                       "fill_ts": na(f_ts, ~cancelled), "fill_notified_ts": na(f_not, ~cancelled),
                       "px": p0, "sz": 1.0, "side": rng.choice(np.array([-1, 1], dtype=np.int8), n)})

    hl_lat, bn_lat = SYNTH["lat_ms"]["hyperliquid"], SYNTH["lat_ms"]["binance"]
    truth = {"through_usd": float((px2 * (sz - touch)[sweep]).sum()),
             "touch_usd": float((tr.px * tr.sz).sum()),
             "lag_local_ms": p["hl_lag_ms"] + sum(hl_lat) - sum(bn_lat)}
    return {"book_hyperliquid": hl, "book_binance": bn, "trades_hyperliquid": tr, "own_orders": oo}, truth


# ============================================ 0. timestamp sanity ============================================

def ts_table_checks(frames, date):
    """Per-table timestamp diagnostics + warning strings for one coin-day (frames: name -> canonical df)."""
    day0 = pd.Timestamp(date, tz="UTC").value
    lo, hi = pd.Timestamp("2015-01-01", tz="UTC").value, pd.Timestamp.now(tz="UTC").value + 86_400 * S
    rows, warns = [], []
    for name in ("book_hyperliquid", "book_binance", "trades_hyperliquid"):
        d = frames[name]
        for c in ("ts_exchange", "ts_local"):
            v = d[c].to_numpy()
            if v.min() < lo or v.max() > hi:
                warns.append(f"{name}.{c}: values outside 2015..now. Wrong unit (s/ms/us/ns)?")
        te = d.ts_exchange.to_numpy()
        outside = np.mean((te < day0) | (te >= day0 + 86_400 * S))
        if outside > 0.01:
            warns.append(f"{name}: {outside:.1%} of rows outside UTC {date}. Files partitioned by local date?")
        off = (d.ts_local.to_numpy() - te) / MS
        med = float(np.median(off))
        h = round(med / HOUR_MS)
        if h != 0 and abs(med - h * HOUR_MS) < 5 * 60_000:
            warns.append(f"TIMEZONE? {name}: ts_local - ts_exchange = {med / HOUR_MS:+.3f} h, about {h:+d} whole hours")
        elif abs(med) > CLOCK_WARN_MS:
            warns.append(f"{name}: median ts_local - ts_exchange = {med:,.0f} ms (> CLOCK_WARN_MS)")
        if np.mean(off < 0) > 0.01:
            warns.append(f"{name}: local before exchange on {np.mean(off < 0):.1%} of rows. Clock skew?")
        rows.append({"table": name, "rows": len(d), "first": pd.Timestamp(te.min(), tz="UTC"),
                     "last": pd.Timestamp(te.max(), tz="UTC"), "outside_day": outside,
                     "lat_p1_ms": np.percentile(off, 1), "lat_med_ms": med, "lat_p99_ms": np.percentile(off, 99),
                     "frac_negative": np.mean(off < 0),
                     "frac_nonmono_local": np.mean(np.diff(d.ts_local.to_numpy()) < 0)})
    return rows, warns


def cross_venue_alignment(frames, clock):
    """Best lag (minutes) between BN and HL 1-minute mid returns, over whole-hour and whole-minute lags."""
    def minute_mid(d):
        o = np.argsort(d[clock].to_numpy(), kind="stable")
        return d[clock].to_numpy()[o], np.log(((d.bid_px + d.ask_px) / 2).to_numpy()[o])
    tb, lb = minute_mid(frames["book_binance"])
    th, lh = minute_mid(frames["book_hyperliquid"])
    lo_t, hi_t = min(tb[0], th[0]), max(tb[-1], th[-1])
    g = np.arange(lo_t - lo_t % (60 * S), hi_t + 60 * S, 60 * S)
    rb, rh = np.diff(asof_vals(tb, lb, g)), np.diff(asof_vals(th, lh, g))
    corr = {}
    for L in sorted(set(range(-720, 721, 60)) | set(range(-5, 6))):
        x, y = (rb[: len(rb) - L], rh[L:]) if L >= 0 else (rb[-L:], rh[: len(rh) + L])
        ok = np.isfinite(x) & np.isfinite(y)
        corr[L] = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 30 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
    s = pd.Series(corr)
    return int(s.idxmax()), s


def run_sanity(frames, date):
    """All timestamp checks for one coin-day. Returns (table rows, warnings)."""
    rows, warns = ts_table_checks(frames, date)
    for clock in ("ts_exchange", "ts_local"):
        best, _ = cross_venue_alignment(frames, clock)
        if best != 0:
            warns.append(f"TIMEZONE? BN vs HL 1-min returns align best at {best:+d} min on {clock} (should be 0)")
    return rows, warns


def sanity():
    """Run the timestamp checks on every coin-day, raise loud warnings, return the diagnostics table."""
    out = []
    for coin, date in coin_days():
        rows, warns = run_sanity(load_frames(coin, date), date)
        out += [{"coin": coin, "date": date, **r} for r in rows]
        for w in warns:
            loud(f"{coin} {date}: {w}")
    if not WARNINGS_LOG:
        print("No timestamp warnings.")
    return pd.DataFrame(out)


def sanity_selftest():
    """Shift copies of the first coin-day by whole hours and assert both timezone checks fire."""
    c0, d0 = coin_days()[0]
    f = load_frames(c0, d0)
    hb = f["book_hyperliquid"]
    tz_local = {**f, "book_hyperliquid": hb.assign(ts_local=hb.ts_local + 10 * 3600 * S)}
    tz_venue = {**f, "book_hyperliquid": hb.assign(ts_exchange=hb.ts_exchange + 3600 * S,
                                                   ts_local=hb.ts_local + 3600 * S)}
    w_local, w_venue = run_sanity(tz_local, d0)[1], run_sanity(tz_venue, d0)[1]
    assert any("TIMEZONE" in w and "ts_local - ts_exchange" in w for w in w_local), "local-vs-exchange check silent"
    assert any("TIMEZONE" in w and "align best" in w for w in w_venue), "cross-venue check silent"
    print("Self-test passed: a +10 h local shift and a +1 h venue shift both trigger TIMEZONE warnings.")


def plot_alignment():
    """Correlation of BN vs HL 1-min returns against HL lag, first coin-day, JOIN_CLOCK."""
    c0, d0 = coin_days()[0]
    s = cross_venue_alignment(load_frames(c0, d0), JOIN_CLOCK)[1]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s.index / 60, s.values, marker="o", ms=4, color=SERIES[0])
    ax.set(xlabel="HL lag vs BN (hours)", ylabel="corr of 1-min returns",
           title=f"Cross-venue alignment, {c0} {d0}, {JOIN_CLOCK} (peak should be at 0)")
    plt.show()


# ============================================ 1. latency ============================================

LAT_METRICS = ["ack", "cancel", "fill_notify"]
_pct = lambda q: (lambda x: np.nanpercentile(x, q) if np.isfinite(x).any() else np.nan)


def own_latencies(coin, date):
    """Per-order latencies (ms) and UTC send hour for one coin-day; None if no own_orders."""
    o = load(coin, date, "own_orders")
    if o is None or o.empty:
        return None
    lat = lambda a, b: (o[a] - o[b]).astype("Float64").to_numpy(dtype=float, na_value=np.nan) / MS
    return pd.DataFrame({"coin": coin, "hour": (o.send_ts.astype("int64") // (3600 * S)) % 24,
                         "ack": lat("ack_ts", "send_ts"), "cancel": lat("cancel_ack_ts", "cancel_send_ts"),
                         "fill_notify": lat("fill_notified_ts", "fill_ts")})


def latency():
    """Latency percentiles per coin and by UTC hour; None (gate 1 n/a) if there is no own_orders data."""
    parts = [x for x in (own_latencies(c, d) for c, d in coin_days()) if x is not None]
    if not parts:
        print("own_orders empty or not configured: section 1 skipped (gate 1 will show n/a).")
        return None
    lat = pd.concat(parts, ignore_index=True)
    long = lat.melt(id_vars=["coin", "hour"], value_vars=LAT_METRICS, var_name="metric", value_name="ms")
    summary = long.groupby(["coin", "metric"]).ms.agg(n="count", p50=_pct(50), p90=_pct(90), p99=_pct(99))
    for coin in summary.index.get_level_values(0).unique():
        RESULTS["ack_latency_p99_ms"][coin] = summary.loc[(coin, "ack"), "p99"]
    by_hour = long.groupby(["metric", "hour"]).ms.agg(p50=_pct(50), p90=_pct(90), p99=_pct(99))
    return {"raw": lat, "summary": summary, "by_hour": by_hour}


def plot_latency(r):
    if r is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, m in zip(axes, LAT_METRICS):
        for q, col in zip(["p50", "p90", "p99"], SERIES):
            ax.plot(r["by_hour"].loc[m].index, r["by_hour"].loc[m][q], label=q, color=col)
        ax.set(title=f"{m} latency (all coins)", xlabel="UTC hour", ylabel="ms")
    axes[0].legend()
    plt.tight_layout(); plt.show()


def check_latency(r):
    """Synthetic: p50 of each latency within 10% of its planted median."""
    if r is None or not SYNTHETIC:
        return
    for m, key in zip(LAT_METRICS, ["ack_ms", "cancel_ms", "fill_notify_ms"]):
        got = np.nanmedian(r["raw"][m])
        check(f"1. {m} p50", abs(got / SYNTH[key] - 1) < 0.10, f"{got:.2f} ms vs planted {SYNTH[key]} ms")


# ============================================ 2. room ============================================

def time_weighted(ts, v, bucket_s, cap_s=60):
    """Time-weighted mean of a step series per bucket; a value counts until the next update (capped).
    A value that spans a bucket edge is credited to its start bucket."""
    dt = np.minimum(np.diff(ts, append=ts[-1]), cap_s * S).astype(float)
    g = pd.DataFrame({"b": ts // (bucket_s * S), "vw": v * dt, "w": dt}).groupby("b").sum()
    g = g[g.w > 0]
    return pd.Series((g.vw / g.w).to_numpy(), index=pd.to_datetime(g.index.to_numpy() * bucket_s * S, utc=True))


def fv_series(coin, date):
    """Fair value: Binance log-mid on JOIN_CLOCK, timestamps shifted by FV_LAG_MS (so FV(t) = mid(t - lag))."""
    b = load(coin, date, "book_binance", [JOIN_CLOCK, "bid_px", "ask_px"])
    return b[JOIN_CLOCK].to_numpy() + FV_LAG_MS * MS, np.log(((b.bid_px + b.ask_px) / 2).to_numpy())


def hl_book(coin, date, clock=None):
    """HL top of book sorted by `clock` (default JOIN_CLOCK), with log-mid and half-spread in bps."""
    h = load(coin, date, "book_hyperliquid").sort_values(clock or JOIN_CLOCK, kind="stable")
    return h.assign(lm=np.log((h.bid_px + h.ask_px) / 2), hs_bps=(h.ask_px - h.bid_px) / (h.ask_px + h.bid_px) * 1e4)


def room_coin_day(coin, date):
    """Per-minute HL half-spread and 5-minute (half-spread, BN realised vol) for one coin-day."""
    h = hl_book(coin, date)
    ts, hs = h[JOIN_CLOCK].to_numpy(), h.hs_bps.to_numpy()
    fts, flog = fv_series(coin, date)
    g = np.arange(fts[0] - fts[0] % S + S, fts[-1], S)   # 1 s grid for realised vol
    r = np.diff(asof_vals(fts, flog, g)) * 1e4
    rv = pd.Series(r ** 2, index=pd.to_datetime(g[1:], utc=True)).resample("5min").sum() ** 0.5
    return time_weighted(ts, hs, 60), pd.DataFrame({"hs_bps": time_weighted(ts, hs, 300), "rv_bps_5m": rv}).dropna()


def room():
    """Room = HL half-spread - round-trip fees, per minute; summary per coin."""
    mins, rv = {}, {}
    for coin, date in coin_days():
        m, r5 = room_coin_day(coin, date)
        mins.setdefault(coin, []).append(m)
        rv.setdefault(coin, []).append(r5)
    mins = {c: pd.concat(v) for c, v in mins.items()}
    rv = {c: pd.concat(v) for c, v in rv.items()}
    rows = []
    for coin in COINS:
        rm = mins[coin] - FEE_RT
        RESULTS["room_median_bps"][coin] = rm.median()
        rows.append({"coin": coin, "minutes": len(rm), "half_spread_med_bps": mins[coin].median(),
                     "fees_rt_bps": FEE_RT, "room_p10": rm.quantile(.1), "room_med": rm.median(),
                     "room_p90": rm.quantile(.9), "frac_minutes_room_pos": (rm > 0).mean()})
    return {"summary": pd.DataFrame(rows).set_index("coin"), "half_spread_min": mins, "rv": rv}


def plot_room(r):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for coin in COINS:
        rm = r["half_spread_min"][coin] - FEE_RT
        axes[0].plot(rm.index, rm.rolling(15, min_periods=1).median(), label=coin, color=COIN_COLOR[coin], lw=1.5)
        axes[1].hist(rm, bins=60, histtype="step", label=coin, color=COIN_COLOR[coin], lw=2)
        axes[2].scatter(r["rv"][coin].rv_bps_5m, r["rv"][coin].hs_bps, s=8, alpha=.5, label=coin,
                        color=COIN_COLOR[coin])
    axes[0].axhline(0, color="0.4", lw=1)
    axes[1].axvline(0, color="0.4", lw=1)
    axes[0].set(title="room (15-min rolling median of 1-min)", ylabel="bps")
    date_axis(axes[0])
    axes[1].set(title="room distribution (1-min)", xlabel="bps")
    axes[2].set(title="HL half-spread vs BN realised vol (5 min)", xlabel="BN realised vol, bps per 5 min",
                ylabel="half-spread bps")
    axes[0].legend()
    plt.tight_layout(); plt.show()


def check_room(r):
    """Synthetic: median HL half-spread within 5% of the planted median."""
    if not SYNTHETIC:
        return
    for coin in COINS:
        got = r["half_spread_min"][coin].median()
        check(f"2. {coin} HL half-spread", abs(got / SYNTH["hl_half_spread_bps"] - 1) < 0.05,
              f"{got:.3f} bps vs planted median {SYNTH['hl_half_spread_bps']}")


# ============================================ 3. adverse selection ============================================

def trade_frame(coin, date):
    """HL trades plus the HL touch strictly before each print (same venue, exchange clock)."""
    t = load(coin, date, "trades_hyperliquid")
    hb = load(coin, date, "book_hyperliquid").sort_values("ts_exchange", kind="stable")
    i = asof_idx(hb.ts_exchange.to_numpy(), t.ts_exchange.to_numpy(), strict=True)
    t, i = t[i >= 0].reset_index(drop=True), i[i >= 0]
    bid, ask = hb.bid_px.to_numpy()[i], hb.ask_px.to_numpy()[i]
    return t.assign(bid_pre=bid, ask_pre=ask, hs_bps=(ask - bid) / (ask + bid) * 1e4, notional=t.px * t.sz)


def markout_frame(coin, date):
    """Trade-level markouts, prior BN move, cost and net edge at TAU for one coin-day."""
    t = trade_frame(coin, date)
    fts, flog = fv_series(coin, date)
    ts, side = t[JOIN_CLOCK].to_numpy(), t.aggressor_side.to_numpy()
    f0 = asof_vals(fts, flog, ts)
    valid = ts + max(HORIZONS) * MS <= fts[-1]          # no markout read off a frozen end-of-data FV
    for h in HORIZONS:
        t[f"mo_{h}"] = np.where(valid, side * (asof_vals(fts, flog, ts + h * MS) - f0) * 1e4, np.nan)
    t["bn_pre"] = side * (f0 - asof_vals(fts, flog, ts - PRE_W * MS)) * 1e4
    b = load(coin, date, "book_binance", [JOIN_CLOCK, "bid_px", "ask_px"])
    bn_hs = asof_vals(b[JOIN_CLOCK].to_numpy(), ((b.ask_px - b.bid_px) / (b.ask_px + b.bid_px) * 1e4).to_numpy(), ts)
    t["cost_bps"] = FEE_RT + (bn_hs if EXIT_LEG == "binance_taker" else 0.0)
    t["net_tau"] = t.hs_bps - t[f"mo_{TAU}"] - t.cost_bps
    t["size_b"] = pd.cut(t.notional, SIZE_BUCKETS_USD).astype(str)
    t["bn_b"] = pd.cut(t.bn_pre, BN_MOVE_BUCKETS_BPS).astype(str)
    return t


def markout_blocks(t, coin):
    """Block sums for every (metric, weight, split, bucket); trades themselves aren't kept."""
    ts, out = t[JOIN_CLOCK].to_numpy(), []
    for metric in [f"mo_{h}" for h in HORIZONS] + ["net_tau", "hs_bps", "cost_bps"]:
        x = t[metric].to_numpy(float)
        for wname, w in (("trade", None), ("notional", t.notional.to_numpy())):
            for split, b in (("all", np.full(len(t), "all")), ("size", t.size_b), ("bn_move", t.bn_b)):
                if split != "all" and not metric.startswith("mo_"):
                    continue
                out.append(block_sums(ts, x, w, {"bucket": b}).assign(coin=coin, metric=metric, weight=wname,
                                                                      split=split))
    return pd.concat(out, ignore_index=True)


def adverse():
    """Markouts (all / by size / by prior BN move) and net edge at TAU, with block-bootstrap CIs."""
    blocks = []
    for coin, date in coin_days():
        blocks.append(markout_blocks(markout_frame(coin, date), coin))
    blocks = pd.concat(blocks, ignore_index=True)
    mo = (blocks.groupby(["coin", "metric", "weight", "split", "bucket"], sort=False)
          .apply(mean_ci, include_groups=False).reset_index())
    mo_all = mo[mo.split == "all"]
    markouts = mo_all[mo_all.metric.str.startswith("mo_")].pivot_table(
        index=["coin", "weight"], columns="metric", values=["mean", "lo", "hi"])
    net_rows = []
    for coin in COINS:
        g = mo_all[mo_all.coin == coin].set_index(["metric", "weight"])
        for wname in ("trade", "notional"):
            net_rows.append({"coin": coin, "weight": wname, "half_spread": g.loc[("hs_bps", wname), "mean"],
                             f"markout_{TAU}ms": g.loc[(f"mo_{TAU}", wname), "mean"],
                             "cost": g.loc[("cost_bps", wname), "mean"],
                             "net_edge": g.loc[("net_tau", wname), "mean"],
                             "net_lo": g.loc[("net_tau", wname), "lo"], "net_hi": g.loc[("net_tau", wname), "hi"],
                             "n_trades": int(g.loc[("net_tau", wname), "n"])})
        RESULTS["net_edge_tau_bps"][coin] = g.loc[("net_tau", "notional"), "mean"]
    counts = (mo[(mo.split != "all") & (mo.metric == f"mo_{TAU}") & (mo.weight == "trade")]
              .pivot_table(index=["split", "bucket"], columns="coin", values="n"))
    return {"mo": mo, "markouts": markouts, "net": pd.DataFrame(net_rows).set_index(["coin", "weight"]),
            "counts": counts}


def plot_adverse(r):
    mo = r["mo"]
    mo_all = mo[mo.split == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for coin in COINS:
        for wname, ls in (("trade", "-"), ("notional", "--")):
            d = mo_all[(mo_all.coin == coin) & (mo_all.weight == wname) & mo_all.metric.str.startswith("mo_")]
            h = d.metric.str[3:].astype(int)
            axes[0].errorbar(h, d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]], ls=ls, marker="o", ms=5,
                             capsize=3, color=COIN_COLOR[coin], label=f"{coin} {wname}")
    axes[0].set(xscale="log", xlabel="horizon (ms)", ylabel="markout bps", title="Markout vs horizon (95% CI)")
    axes[0].axhline(0, color="0.4", lw=1)
    axes[0].legend(fontsize=8)
    for ax, split, title in ((axes[1], "size", "markout(TAU) by notional bucket"),
                             (axes[2], "bn_move", "markout(TAU) by prior BN move (aggressor-signed, bps)")):
        d = mo[(mo.split == split) & (mo.metric == f"mo_{TAU}") & (mo.weight == "trade")]
        buckets = sorted([b for b in d.bucket.unique() if b != "nan"],
                         key=lambda s: float(s.strip("(]").split(",")[0]))
        x = np.arange(len(buckets))
        for k, coin in enumerate(COINS):
            dc = d[d.coin == coin].set_index("bucket").reindex(buckets)
            ax.bar(x + (k - (len(COINS) - 1) / 2) * 0.8 / len(COINS), dc["mean"], width=0.8 / len(COINS) - 0.02,
                   yerr=[dc["mean"] - dc.lo, dc.hi - dc["mean"]], capsize=2, color=COIN_COLOR[coin], label=coin)
        ax.set_xticks(x, buckets, rotation=20, fontsize=8)
        ax.axhline(0, color="0.4", lw=1)
        ax.set(title=title, ylabel="bps")
    axes[1].legend()
    plt.tight_layout(); plt.show()


def check_adverse(r):
    """Synthetic: markout at TAU and at the longest horizon = informed_frac x markout (±0.15 bps); net edge =
    planted half-spread - that - fees - BN half-spread (±0.15 bps)."""
    if not SYNTHETIC:
        return
    mo_all = r["mo"][r["mo"].split == "all"]
    truth_mo = SYNTH["informed_frac"] * SYNTH["informed_markout_bps"]
    hs_mean = SYNTH["hl_half_spread_bps"] * math.exp(SYNTH["spread_noise"] ** 2 / 2)
    bn_mean = SYNTH["bn_half_spread_bps"] * math.exp(SYNTH["spread_noise"] ** 2 / 2)
    for coin in COINS:
        g = mo_all[(mo_all.coin == coin) & (mo_all.weight == "trade")].set_index("metric")
        for h in (TAU, max(HORIZONS)):
            got = g.loc[f"mo_{h}", "mean"]
            check(f"3. {coin} markout({h}ms)", abs(got - truth_mo) < 0.15,
                  f"{got:.3f} bps vs planted informed_frac*markout = {truth_mo:.2f}")
        want, got = hs_mean - truth_mo - FEE_RT - bn_mean, g.loc["net_tau", "mean"]
        check(f"3. {coin} net edge", abs(got - want) < 0.15, f"{got:.3f} bps vs planted {want:.3f}")


# ============================================ 4. getting out ============================================

def offset_waits(t, fts, flog):
    """Seconds to offset each entry (inf = not within max wait) and the BN with/against label."""
    t = t.sort_values("ts_exchange", kind="stable").reset_index(drop=True)
    tx, px, sd = t.ts_exchange.to_numpy(), t.px.to_numpy(), t.aggressor_side.to_numpy()
    ev = t.drop_duplicates(["ts_exchange", "aggressor_side"], keep="first")
    ev = ev[ev.ts_exchange <= tx[-1] - OFFSET_MAX_WAIT_S * S]
    ev_t, ev_s = ev.ts_exchange.to_numpy(), ev.aggressor_side.to_numpy()
    # maker position = -aggressor; long (+1) exits on a buyer-initiated print at >= target, short on a sell at <=
    target = (np.where(ev_s == -1, ev.ask_pre.to_numpy(), ev.bid_pre.to_numpy()) if OFFSET_TARGET == "entry_quote"
              else ev.px.to_numpy())
    wait = np.full(len(ev), np.inf)
    for s_in in (-1, 1):
        ex = sd == -s_in
        ex_t, ex_p = tx[ex], px[ex]
        sel = np.flatnonzero(ev_s == s_in)
        lo = np.searchsorted(ex_t, ev_t[sel], "right")
        hi = np.searchsorted(ex_t, ev_t[sel] + OFFSET_MAX_WAIT_S * S, "right")
        for j, a, b in zip(sel, lo, hi):   # one short window per entry; vectorised inside the window
            if b > a:
                seg = ex_p[a:b]
                hit = seg >= target[j] * (1 - 1e-9) if s_in == -1 else seg <= target[j] * (1 + 1e-9)
                k = int(np.argmax(hit))
                if hit[k]:
                    wait[j] = (ex_t[a + k] - ev_t[j]) / S
    tj = ev[JOIN_CLOCK].to_numpy()
    bn = -ev_s * (asof_vals(fts, flog, tj + OFF_BN_W * MS) - asof_vals(fts, flog, tj))
    return wait, np.where(bn > 0, "with", np.where(bn < 0, "against", "flat"))


def wait_hist(wait, label):
    """Histogram counts of waits per label (last column = censored): small enough to pool over days."""
    out = {}
    for lab in ("all", "with", "against"):
        w = wait if lab == "all" else wait[label == lab]
        out[lab] = np.append(np.histogram(w[np.isfinite(w)], WAIT_EDGES)[0], np.isinf(w).sum())
    return out


def survival(counts):
    """P(wait > edge) at each upper bin edge, from binned counts with censoring at the max wait."""
    return 1 - np.cumsum(counts[:-1]) / counts.sum()


def basis_series(coin, date, fts, flog):
    """Basis (bps) on a BASIS_RESAMPLE_S grid over the overlap of both feeds."""
    h = hl_book(coin, date)
    hts, hlm = h[JOIN_CLOCK].to_numpy(), h.lm.to_numpy()
    g = np.arange(max(hts[0], fts[0]) + BASIS_RESAMPLE_S * S, min(hts[-1], fts[-1]), BASIS_RESAMPLE_S * S)
    return g, (asof_vals(hts, hlm, g) - asof_vals(fts, flog, g)) * 1e4


def bn_cost_coin_day(coin, date):
    """Per-minute BN half-spread and share of time the touch is smaller than HEDGE_SIZE_USD."""
    b = load(coin, date, "book_binance").sort_values(JOIN_CLOCK, kind="stable")
    ts = b[JOIN_CLOCK].to_numpy()
    hs = ((b.ask_px - b.bid_px) / (b.ask_px + b.bid_px) * 1e4).to_numpy()
    thin = (np.minimum(b.bid_px * b.bid_sz, b.ask_px * b.ask_sz) < HEDGE_SIZE_USD).to_numpy(float)
    return time_weighted(ts, hs, 60), time_weighted(ts, thin, 60)


def ar1_half_life(series_list, dt_s, block=180, seed=None):
    """Pooled within-day AR(1): phi, half-life (s) with block-bootstrap CI, and the noise-robust phi."""
    rows, rob = [], []
    for _, _, b in series_list:
        x, y = b[:-1], b[1:]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        rows.append(pd.DataFrame({"blk": np.arange(len(x)) // block, "n": 1, "sx": x, "sy": y, "sxx": x * x,
                                  "sxy": x * y}).groupby("blk").sum())
        z = b[np.isfinite(b)] - np.nanmean(b)
        rob.append((np.dot(z[:-2], z[2:]), np.dot(z[:-1], z[1:])))
    B = pd.concat(rows, ignore_index=True).to_numpy()

    def fit(s):
        n, sx, sy, sxx, sxy = s
        return (sxy - sx * sy / n) / (sxx - sx * sx / n)
    hl = lambda ph: -math.log(2) * dt_s / math.log(ph) if 0 < ph < 1 else np.nan
    phi = fit(B.sum(0))
    idx = np.random.default_rng(SEED if seed is None else seed).integers(0, len(B), (N_BOOT, len(B)))
    boots = np.array([hl(fit(B[i].sum(0))) for i in idx])
    phi_r = sum(a for a, _ in rob) / sum(b for _, b in rob)
    return {"phi": phi, "half_life_s": hl(phi), "hl_lo": np.nanpercentile(boots, 2.5),
            "hl_hi": np.nanpercentile(boots, 97.5), "phi_robust": phi_r, "half_life_robust_s": hl(phi_r),
            "n_pairs": int(B[:, 0].sum())}


def getting_out():
    """Offset survival, basis AR(1) half-life and Binance taker cost per coin."""
    waits, basis, bnc = {}, {}, {}
    for coin, date in coin_days():
        fts, flog = fv_series(coin, date)
        w, lab = offset_waits(trade_frame(coin, date), fts, flog)
        waits[coin] = {k: waits.get(coin, {}).get(k, 0) + v for k, v in wait_hist(w, lab).items()}
        g, b = basis_series(coin, date, fts, flog)
        basis.setdefault(coin, []).append((date, g, b))
        bnc.setdefault(coin, []).append(bn_cost_coin_day(coin, date))
    rows = []
    for coin in COINS:
        surv = survival(waits[coin]["all"])
        med = WAIT_EDGES[1:][np.argmax(surv <= 0.5)] if (surv <= 0.5).any() else np.inf
        ar = ar1_half_life(basis[coin], BASIS_RESAMPLE_S)
        hs_min, thin = pd.concat([x[0] for x in bnc[coin]]), pd.concat([x[1] for x in bnc[coin]])
        RESULTS["offset_median_s"][coin] = med
        RESULTS["basis_half_life_s"][coin] = ar["half_life_s"]
        rows.append({"coin": coin, "entries": int(waits[coin]["all"].sum()), "offset_median_s": med,
                     f"P(no offset in {OFFSET_MAX_WAIT_S}s)": surv[-1],
                     **{k: ar[k] for k in ("half_life_s", "hl_lo", "hl_hi", "half_life_robust_s", "n_pairs")},
                     "bn_half_spread_med_bps": hs_min.median(),
                     "bn_taker_cost_med_bps": hs_min.median() + FEES_BPS["binance"]["taker"],
                     f"frac_time_touch<{HEDGE_SIZE_USD:,}$": thin.mean()})
    return {"summary": pd.DataFrame(rows).set_index("coin"), "waits": waits, "basis": basis}


def plot_getting_out(r):
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for coin in COINS:
        for lab, ls in (("all", "-"), ("with", "--"), ("against", ":")):
            axes[0].step(WAIT_EDGES[1:], survival(r["waits"][coin][lab]), where="post", ls=ls,
                         color=COIN_COLOR[coin], label=f"{coin} {lab}")
        date, g, b = r["basis"][coin][0]
        axes[1].plot(pd.to_datetime(g, utc=True), b, lw=0.8, color=COIN_COLOR[coin], label=f"{coin} {date}")
        lags = np.arange(1, 61)
        z = b[np.isfinite(b)] - np.nanmean(b)
        acf = [np.dot(z[:-k], z[k:]) / np.dot(z, z) for k in lags]
        hl = r["summary"].loc[coin, "half_life_s"]
        axes[2].plot(lags * BASIS_RESAMPLE_S, acf, color=COIN_COLOR[coin], label=f"{coin} empirical")
        axes[2].plot(lags * BASIS_RESAMPLE_S, 0.5 ** (lags * BASIS_RESAMPLE_S / hl), ls="--",
                     color=COIN_COLOR[coin], label=f"{coin} AR(1) fit")
    axes[0].set(xscale="log", xlabel="seconds since entry", ylabel="P(not yet offset)",
                title=f"Offset survival ({OFFSET_TARGET})")
    axes[0].legend(fontsize=7)
    axes[1].set(title="HL - BN basis (first day)", ylabel="bps")
    date_axis(axes[1])
    axes[1].legend(fontsize=8)
    axes[2].set(title="Basis autocorrelation", xlabel="lag (s)")
    axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.show()


def check_getting_out(r):
    """Synthetic: basis half-life within 20% of planted; BN median half-spread within 5% of planted."""
    if not SYNTHETIC:
        return
    s = r["summary"]
    for coin in COINS:
        got = s.loc[coin, "half_life_s"]
        check(f"4. {coin} basis half-life", abs(got / SYNTH["basis_half_life_s"] - 1) < 0.20,
              f"{got:.1f} s vs planted {SYNTH['basis_half_life_s']} s (robust {s.loc[coin, 'half_life_robust_s']:.1f} s)")
        got = s.loc[coin, "bn_half_spread_med_bps"]
        check(f"4. {coin} BN half-spread", abs(got / SYNTH["bn_half_spread_bps"] - 1) < 0.05,
              f"{got:.3f} bps vs planted {SYNTH['bn_half_spread_bps']}")


# ============================================ 5. lead-lag ============================================

def xcorr_sums(hts, hlm, fts, flog):
    """Pooled-correlation sums (n, sx, sy, sxx, syy, sxy) per lag between BN and HL grid returns."""
    step, K = XCORR_GRID_MS * MS, np.array(XCORR_LAGS_MS) // XCORR_GRID_MS
    kmax = int(np.abs(K).max())
    start, end = max(hts[0], fts[0]), min(hts[-1], fts[-1])
    sums = np.zeros((len(K), 6))
    for a in range(int(start), int(end), 3600 * S):
        nb = int(math.ceil((min(a + 3600 * S, end) - a) / step))
        g = a + (np.arange(nb + 2 * kmax + 1, dtype=np.int64) - kmax) * step   # int64: Windows default is int32
        rb, rh = np.diff(asof_vals(fts, flog, g)), np.diff(asof_vals(hts, hlm, g))
        x = rb[kmax: kmax + nb]                       # BN returns starting inside [a, a + 1h)
        for j, k in enumerate(K):
            y = rh[kmax + k: kmax + k + nb]           # HL return k steps later
            ok = np.isfinite(x) & np.isfinite(y)
            xo, yo = x[ok], y[ok]
            sums[j] += [ok.sum(), xo.sum(), yo.sum(), xo @ xo, yo @ yo, xo @ yo]
    return sums


def corr_from_sums(s):
    """Pearson correlation from pooled sums."""
    n, sx, sy, sxx, syy, sxy = s.T
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def reflect_sums(hts, hlm, fts, flog):
    """Regression sums of HL response on the BN move, per horizon h (n, sx, sy, sxx, sxy)."""
    W = REFLECT_WINDOW_MS * MS
    g = np.arange(max(hts[0], fts[0]) + W, min(hts[-1], fts[-1]) - max(REFLECT_H) * MS, W)
    x = (asof_vals(fts, flog, g) - asof_vals(fts, flog, g - W)) * 1e4
    out = []
    for h in REFLECT_H:
        y = (asof_vals(hts, hlm, g + h * MS) - asof_vals(hts, hlm, g - W)) * 1e4
        ok = np.isfinite(x) & np.isfinite(y)
        out.append([ok.sum(), x[ok].sum(), y[ok].sum(), x[ok] @ x[ok], x[ok] @ y[ok]])
    return np.array(out)


def slope_from_sums(s):
    """OLS slope per row from (n, sx, sy, sxx, sxy)."""
    n, sx, sy, sxx, sxy = s.T
    return (sxy - sx * sy / n) / (sxx - sx ** 2 / n)


def tau_moves_hist(fts, flog):
    """24 x len(MOVE_EDGES)-1 counts of |BN move| over non-overlapping TAU windows, by UTC hour."""
    g = np.arange(fts[0] + TAU * MS, fts[-1], TAU * MS)
    m = np.abs(np.diff(asof_vals(fts, flog, g))) * 1e4
    hour = ((g[1:] // (3600 * S)) % 24).astype(int)
    return np.histogram2d(hour, m, bins=[np.arange(25), MOVE_EDGES])[0]


def hist_quantile(counts, q):
    """Quantile from binned counts (upper bin edge)."""
    c = np.cumsum(counts) / counts.sum()
    return MOVE_EDGES[1:][np.searchsorted(c, q)]


def lead_lag():
    """Cross-correlation by lag, fraction of a BN move reflected in HL by t+h, and the TAU-move distribution."""
    xc, rf, mv = {}, {}, {}
    for coin, date in coin_days():
        fts, flog = fv_series(coin, date)
        h = hl_book(coin, date)
        hts, hlm = h[JOIN_CLOCK].to_numpy(), h.lm.to_numpy()
        xc[coin] = xc.get(coin, 0) + xcorr_sums(hts, hlm, fts, flog)
        rf[coin] = rf.get(coin, 0) + reflect_sums(hts, hlm, fts, flog)
        mv[coin] = mv.get(coin, 0) + tau_moves_hist(fts, flog)
    rows = []
    for coin in COINS:
        cc, frac, tot = corr_from_sums(xc[coin]), slope_from_sums(rf[coin]), mv[coin].sum(0)
        p90, p99 = hist_quantile(tot, .9), hist_quantile(tot, .99)
        RESULTS["frac_reflected_tau"][coin] = frac[REFLECT_H.index(TAU)]
        RESULTS["bn_move_tau_p99_bps"][coin] = p99
        rows.append({"coin": coin, "peak_lag_ms": XCORR_LAGS_MS[int(np.nanargmax(cc))], "peak_corr": np.nanmax(cc),
                     **{f"frac_reflected_h{h}": f for h, f in zip(REFLECT_H, frac)},
                     f"bn_move_{TAU}ms_p90_bps": p90, f"bn_move_{TAU}ms_p99_bps": p99})
    return {"summary": pd.DataFrame(rows).set_index("coin"), "xc": xc, "rf": rf, "mv": mv}


def plot_lead_lag(r):
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.8))
    centers = np.sqrt(MOVE_EDGES[1:-1] * MOVE_EDGES[2:])
    for coin in COINS:
        axes[0].plot(XCORR_LAGS_MS, corr_from_sums(r["xc"][coin]), color=COIN_COLOR[coin], label=coin)
        axes[1].plot(REFLECT_H, slope_from_sums(r["rf"][coin]), marker="o", color=COIN_COLOR[coin], label=coin)
        tot = r["mv"][coin].sum(0)
        axes[2].step(centers, tot[1:] / tot.sum(), where="mid", color=COIN_COLOR[coin], label=coin)
        for q, ls in ((.9, "--"), (.99, ":")):
            axes[2].axvline(hist_quantile(tot, q), color=COIN_COLOR[coin], ls=ls, lw=1)
        big = r["mv"][coin][:, MOVE_EDGES[:-1] >= hist_quantile(tot, BIG_MOVE_Q)].sum(1)
        axes[3].plot(np.arange(24), big / big.sum(), marker="o", ms=4, color=COIN_COLOR[coin], label=coin)
    for ax in axes[:2]:
        ax.axvline(TAU, color="0.4", lw=1, ls="--")
    axes[0].set(xlabel="lag (ms), + = BN leads", ylabel="corr",
                title=f"Return cross-correlation ({XCORR_GRID_MS} ms grid)")
    axes[1].set(xlabel="h (ms)", ylabel="slope", title=f"Fraction of BN move (W={REFLECT_WINDOW_MS} ms) in HL by t+h")
    axes[2].set(xscale="log", yscale="log", xlabel="|BN move| bps", title=f"|BN move| over {TAU} ms (p90 --, p99 :)")
    axes[3].set(xlabel="UTC hour", ylabel="share of big moves", title=f"When moves > q{BIG_MOVE_Q} land")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def check_lead_lag(r):
    """Synthetic: xcorr peak within 30 ms of the planted lag (on JOIN_CLOCK); fraction reflected at the longest h
    within 0.05 of 1."""
    if not SYNTHETIC:
        return
    s = r["summary"]
    for coin in COINS:
        want = TRUTH[(coin, DATES[0])]["lag_local_ms"] if JOIN_CLOCK == "ts_local" else SYNTH["hl_lag_ms"]
        got = s.loc[coin, "peak_lag_ms"]
        check(f"5. {coin} lead-lag peak", abs(got - want) <= 30, f"{got} ms vs planted {want:.1f} ms on {JOIN_CLOCK}")
        f_long = s.loc[coin, f"frac_reflected_h{max(REFLECT_H)}"]
        check(f"5. {coin} fully reflected at {max(REFLECT_H)} ms", abs(f_long - 1) < 0.05, f"{f_long:.3f} vs 1")


# ============================================ 6. capacity ============================================

def capacity_coin_day(coin, date):
    """Touch and through notional (USD) for one coin-day."""
    t = trade_frame(coin, date)
    buy = t.aggressor_side.to_numpy() == 1
    px, bid, ask = t.px.to_numpy(), t.bid_pre.to_numpy(), t.ask_pre.to_numpy()
    touch = np.where(buy, px >= ask * (1 - 1e-9), px <= bid * (1 + 1e-9))
    through = np.where(buy, px > ask * (1 + 1e-9), px < bid * (1 - 1e-9))
    v = t.notional.to_numpy() if QUOTE_SIZE_USD is None else np.minimum(t.notional.to_numpy(), QUOTE_SIZE_USD)
    return {"coin": coin, "date": date, "touch_usd": v[touch].sum(), "through_usd": v[through].sum(),
            "total_usd": t.notional.sum()}


def capacity(net):
    """$/day per coin under both fill bounds, from daily at/through volume x net edge (with its CI)."""
    days = pd.DataFrame([capacity_coin_day(c, d) for c, d in coin_days()])
    rows = []
    for coin in COINS:
        d, ne = days[days.coin == coin], net.loc[(coin, "notional")]
        row = {"coin": coin, "days": len(d), "touch_usd_day": d.touch_usd.mean(),
               "through_usd_day": d.through_usd.mean(), "net_edge_bps": ne.net_edge}
        for bound in ("touch", "through"):
            vol = row[f"{bound}_usd_day"] * CAPACITY_SHARE / 1e4
            row[f"usd_day_{bound}"], row[f"usd_day_{bound}_lo"], row[f"usd_day_{bound}_hi"] = \
                vol * ne.net_edge, vol * ne.net_lo, vol * ne.net_hi
            RESULTS[f"capacity_usd_day_{bound}"][coin] = row[f"usd_day_{bound}"]
        rows.append(row)
    return {"summary": pd.DataFrame(rows).set_index("coin"), "days": days}


def plot_capacity(r):
    cap = r["summary"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(COINS))
    for k, bound in enumerate(("touch", "through")):
        y = cap[f"usd_day_{bound}"]
        ax.bar(x + (k - .5) * 0.38, y, width=0.36, color=SERIES[k], label=f"{bound} bound",
               yerr=[y - cap[f"usd_day_{bound}_lo"], cap[f"usd_day_{bound}_hi"] - y], capsize=3)
    ax.set_xticks(x, COINS)
    ax.axhline(0, color="0.4", lw=1)
    ax.set(ylabel="$/day", title=f"Rough capacity (share={CAPACITY_SHARE}, net-edge 95% CI)")
    ax.legend()
    plt.show()


def check_capacity(r):
    """Synthetic (uncapped): touch volume = all generated volume; through volume = the generated sweep remainder."""
    if not SYNTHETIC or QUOTE_SIZE_USD is not None:
        return
    for coin in COINS:
        d = r["days"][r["days"].coin == coin].iloc[0]
        tr = TRUTH[(coin, d.date)]
        check(f"6. {coin} touch volume", abs(d.touch_usd / tr["touch_usd"] - 1) < 1e-6,
              f"{d.touch_usd:,.0f} vs planted {tr['touch_usd']:,.0f}")
        check(f"6. {coin} through volume", abs(d.through_usd / tr["through_usd"] - 1) < 1e-6,
              f"{d.through_usd:,.0f} vs planted {tr['through_usd']:,.0f}")


# ============================================ 7. competition (stubs) ============================================

def top_of_book_owners(coin, date):
    """TODO: share of time each wallet holds the HL best bid/ask (needs order-level data with addresses).
    Output: DataFrame[wallet, side, share_of_time_at_touch, median_size_usd]."""
    raise NotImplementedError("needs wallet-level book data")


def reprice_speed_after_bn_move(coin, date, move_bps):
    """TODO: per wallet, time from a Binance mid move > move_bps to that wallet's HL cancel/replace.
    Output: DataFrame[wallet, n_events, p50_ms, p90_ms]."""
    raise NotImplementedError("needs wallet-level order events")


def cancel_to_fill(coin, date):
    """TODO: per wallet cancel-to-fill ratio and fill share at the touch.
    Output: DataFrame[wallet, n_orders, n_cancels, n_fills, cancel_to_fill]."""
    raise NotImplementedError("needs wallet-level order events")


# ============================================ decision ============================================

def decide(thresholds):
    """Per-coin PASS/FAIL per gate, or (None, missing thresholds) if any is unset."""
    missing = [k for k, v in thresholds.items() if v is None]
    if missing:
        return None, missing
    ops = {">=": np.greater_equal, "<=": np.less_equal}
    table = {}
    for coin in COINS:
        row = {}
        for gate, metrics in GATES.items():
            vals = [RESULTS[m].get(coin, np.nan) for m in metrics]
            if any(not np.isfinite(v) for v in vals):
                row[gate] = "n/a"
            else:
                row[gate] = "PASS" if all(ops[thresholds[m][0]](v, thresholds[m][1])
                                          for m, v in zip(metrics, vals)) else "FAIL"
        table[coin] = row
    return pd.DataFrame(table).T, []


def decision():
    """Print verdicts, or refuse if any threshold is None; then show the metric values the gates compare."""
    verdicts, missing = decide(THRESHOLDS)
    if verdicts is None:
        print("REFUSING TO PRINT VERDICTS. These thresholds are still None:\n  " + "\n  ".join(missing) +
              "\nSet each to (op, value) in CONFIG *before* looking at real results.")
    else:
        display(verdicts)
    print("\nMetric values per coin (what the thresholds are compared against):")
    display(pd.DataFrame(RESULTS).T.astype(float).round(3))


def decision_selftest():
    """Synthetic: decide() refuses with None thresholds, all-pass / all-fail with dummy ones. Prints no verdicts."""
    if not SYNTHETIC:
        return
    assert decide({k: None for k in THRESHOLDS})[0] is None
    v = decide({k: (">=", -np.inf) for k in THRESHOLDS})[0]
    assert set(v.columns) == set(GATES) and v.isin(["PASS", "n/a"]).all().all()
    assert decide({k: ("<=", -np.inf) for k in THRESHOLDS})[0].isin(["FAIL", "n/a"]).all().all()
    print("Decision logic self-test passed.")


def finish():
    """List every warning raised and (synthetic) assert that all ground-truth checks passed."""
    print(f"Warnings raised: {len(WARNINGS_LOG)}")
    for w in WARNINGS_LOG:
        print("  -", w)
    if SYNTHETIC:
        print(f"\nSynthetic ground-truth checks: {sum(ok for ok, _ in CHECKS.values())}/{len(CHECKS)} passed")
        assert all(ok for ok, _ in CHECKS.values())
