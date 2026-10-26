# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MM feasibility: quote Hyperliquid perps, hedge on Binance perps
#
# **Summary.** Seven gates, cheapest first, decide whether medium-frequency market making on Hyperliquid (HL),
# hedged on Binance (BN), is worth building: (1) latency, (2) room before adverse selection, (3) adverse selection
# (kill test), (4) getting out, (5) lead-lag, (6) capacity, (7) competition (stubs). Fair value (FV) = Binance mid,
# optionally lagged. Every cross-venue join is a strictly backward as-of on `JOIN_CLOCK`: nothing looks forward.
# Markouts read FV at `t + h`, but those are outcomes measured afterwards, never features.
#
# **How this notebook is laid out.** All the code lives in `mm_feasibility_lib.py` (keep it next to this file).
# Each section below says what is measured and how it's defined, then makes one call to compute, one to plot,
# and one to check against the synthetic ground truth. Processing is one coin-day at a time.
#
# **Validation.** With `DATA_SOURCE = "synthetic"`, a generator plants known values: HL lags BN, a set fraction of
# informed HL takers with a set markout, an OU basis with a set half-life, known spreads and latencies. Each
# section asserts that it recovers them. The checks are switched off on real data because there's no ground truth.
#
# **Switching to real data.** In CONFIG: `DATA_SOURCE = "real"`, set `TAU_MS`, fill every `TODO` in `PATHS` /
# `COLMAP` / `SYMBOLS` / `AGGRESSOR_MAP`, and put in your fees. Run section 0 first and read every loud warning.
#
# **Not in here:** conclusions or threshold values. Set `THRESHOLDS` before looking at real results. The
# decision cell refuses to print verdicts until every threshold is set. Assumptions and open questions are at the end.

# %%
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
import mm_feasibility_lib as mm
importlib.reload(mm)   # pick up edits to the lib without restarting the kernel
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

# %%
# ================================ CONFIG: the only cell you edit ================================
DATA_SOURCE = "synthetic"      # "synthetic" = validate the notebook | "real" = your data through the adapters
DATA_FORMAT = "parquet"        # "parquet" | "csv"
DATES = ["2026-10-01"]         # UTC calendar dates; processed one coin-day at a time
COINS = ["BTC", "ETH"]
SYMBOLS = {                    # TODO(real): venue symbol for each coin
    "hyperliquid": {"BTC": "BTC", "ETH": "ETH"},
    "binance": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
}
# TODO(real): path templates; {date} {coin} {symbol} are filled in. A template with neither {coin} nor {symbol}
# is filtered on the mapped `symbol` column. own_orders None -> section 1 is skipped.
PATHS = {
    "book":       {"hyperliquid": "TODO/hyperliquid/book/{date}/{coin}.parquet",
                   "binance":     "TODO/binance/book/{date}/{symbol}.parquet"},
    "trades":     {"hyperliquid": "TODO/hyperliquid/trades/{date}/{coin}.parquet"},
    "own_orders": {"hyperliquid": None},
}
# TODO(real): canonical column -> your column. Timestamps are (column, unit), unit in s | ms | us | ns | datetime.
# "symbol": None means the file holds one symbol only.
COLMAP = {
    "book": {
        "hyperliquid": {"ts_exchange": ("TODO", "ms"), "ts_local": ("TODO", "ns"), "symbol": None,
                        "bid_px": "TODO", "bid_sz": "TODO", "ask_px": "TODO", "ask_sz": "TODO"},
        "binance":     {"ts_exchange": ("TODO", "ms"), "ts_local": ("TODO", "ns"), "symbol": None,
                        "bid_px": "TODO", "bid_sz": "TODO", "ask_px": "TODO", "ask_sz": "TODO"},
    },
    "trades": {
        "hyperliquid": {"ts_exchange": ("TODO", "ms"), "ts_local": ("TODO", "ns"), "symbol": None,
                        "px": "TODO", "sz": "TODO", "aggressor_side": "TODO"},
    },
    "own_orders": {
        "hyperliquid": {"order_id": "TODO", "send_ts": ("TODO", "ns"), "ack_ts": ("TODO", "ns"),
                        "cancel_send_ts": ("TODO", "ns"), "cancel_ack_ts": ("TODO", "ns"),
                        "fill_ts": ("TODO", "ns"), "fill_notified_ts": ("TODO", "ns"),
                        "px": "TODO", "sz": "TODO", "side": "TODO"},
    },
}
TS_TZ = None                   # only for unit="datetime" with tz-naive strings, e.g. "UTC". Never guessed.
AGGRESSOR_MAP = {"buy": 1, "sell": -1}   # TODO(real): raw value -> +1 buyer-initiated / -1 seller-initiated.
                                          # Check the convention (e.g. Binance `m`=True means the BUYER was maker).

FEES_BPS = {"hyperliquid": {"maker": 1.5, "taker": 4.5}, "binance": {"maker": 2.0, "taker": 5.0}}
# ^ ASSUMPTION: public base-tier rates, only as placeholders. Put yours in on the work machine.
EXIT_LEG = "binance_taker"     # how a HL maker fill is closed, for costs: "binance_taker" (hedge) | "hl_maker"

TAU_MS = None                  # YOUR stale window (ms). Required for real data; synthetic uses SYNTH["tau_ms"].
MARKOUT_HORIZONS_MS = ["TAU", 1000, 5000, 30000]
FV_LAG_MS = 0                  # fair value at t = Binance mid as of (t - FV_LAG_MS)
JOIN_CLOCK = "ts_local"        # cross-venue joins use this clock, strictly backward as-of

# Gate thresholds: (op, value) with op in {">=", "<="}. Set them BEFORE looking at real results. None = no verdicts.
THRESHOLDS = {
    "ack_latency_p99_ms":       None,   # gate 1
    "room_median_bps":          None,   # gate 2
    "net_edge_tau_bps":         None,   # gate 3 (notional-weighted point estimate)
    "basis_half_life_s":        None,   # gate 4
    "offset_median_s":          None,   # gate 4
    "frac_reflected_tau":       None,   # gate 5
    "bn_move_tau_p99_bps":      None,   # gate 5
    "capacity_usd_day_touch":   None,   # gate 6, optimistic fill bound
    "capacity_usd_day_through": None,   # gate 6, conservative fill bound
}

# Analysis knobs ("TAU" means TAU_MS)
SIZE_BUCKETS_USD = [0, 1e3, 1e4, 1e5, np.inf]            # section 3 trade-size split (notional)
PRE_WINDOW_MS = "TAU"                                     # section 3: Binance move window before each trade
BN_MOVE_BUCKETS_BPS = [-np.inf, -5, -1, 1, 5, np.inf]     # signed by aggressor: + = trade follows the prior BN move
OFFSET_TARGET = "entry_quote"  # section 4 exit price: "entry_quote" = opposite touch at entry | "entry_px" = scratch
OFFSET_MAX_WAIT_S = 60         # waits beyond this are censored
OFFSET_BN_WINDOW_MS = "TAU"    # BN move after entry that labels a wait "with" / "against" your position
BASIS_RESAMPLE_S = 10          # basis sampling interval for the AR(1)
HEDGE_SIZE_USD = 50_000        # Binance taker cost size
XCORR_GRID_MS = 10             # lead-lag grid; lags must be multiples of it
XCORR_LAGS_MS = sorted(set(range(-500, 501, 10)) | {-2000, -1500, -1000, -750, 750, 1000, 1500, 2000})
REFLECT_WINDOW_MS = 1000       # BN move measured over [t - W, t]; HL response over [t - W, t + h]
REFLECT_H_MS = [0, "TAU", 1000, 5000]
CAPACITY_SHARE = 1.0           # share of at/through volume you'd get; 1.0 = upper bound
QUOTE_SIZE_USD = None          # cap each print at your quote size; None = uncapped
BIG_MOVE_Q = 0.99              # "big" Binance TAU move = above this quantile
CLOCK_WARN_MS = 1000           # sanity: warn if |median(ts_local - ts_exchange)| exceeds this
BOOT_BLOCK_S = 300             # block length for bootstrap CIs (trades cluster in time)
N_BOOT = 500
SEED = 7
CACHE_DIR = Path("mm_cache")   # canonical per-coin-day parquet; subfolder per DATA_SOURCE
REBUILD_CACHE = True

SYNTH = dict(                  # synthetic market; PLANTED values are what the checks must recover
    tau_ms=200,                # stand-in for TAU_MS in synthetic runs only
    hours=24, p0={"BTC": 60_000.0, "ETH": 3_000.0}, seed=11,
    bn_update_hz=10, hl_update_hz=10, hl_trade_hz=0.5,
    bn_vol_bps_sqrt_s=0.5, vol_diurnal=0.5,
    hl_lag_ms=250,                       # PLANTED: HL mid = BN mid delayed by this (exchange time)
    informed_frac=0.2,                   # PLANTED: share of informed HL takers
    informed_markout_bps=3.0,            # PLANTED: BN jumps this much in their direction ...
    jump_delay_ms=50,                    # ... this long after their trade
    basis_half_life_s=60.0, basis_sd_bps=8.0, basis_mean_bps=1.0,   # PLANTED OU basis
    hl_half_spread_bps=1.2, bn_half_spread_bps=0.3, spread_noise=0.2,  # PLANTED median half-spreads
    hl_touch_usd=30_000, bn_touch_usd=200_000, trade_usd_median=3_000, trade_usd_sigma=1.3, tick_bps=0.5,
    lat_ms={"binance": (2.0, 0.5), "hyperliquid": (30.0, 5.0)},     # local - exchange: fixed + exp jitter mean
    own_orders_per_day=5000, ack_ms=25.0, cancel_ms=30.0, fill_notify_ms=15.0,   # PLANTED latency medians
)

# %%
mm.setup(**{k: globals()[k] for k in mm.CONFIG_KEYS})   # hand the config above to the lib; resets run state

# %% [markdown]
# ## Data: canonical schemas and the cache
#
# Every source is mapped into these schemas, with all timestamps as UTC int64 nanoseconds:
#
# | table | columns |
# |---|---|
# | book (top of book) | ts_exchange, ts_local, venue, symbol, bid_px, bid_sz, ask_px, ask_sz |
# | trades | ts_exchange, ts_local, venue, symbol, px, sz, aggressor_side (+1 buy / -1 sell) |
# | own_orders (optional) | order_id, send_ts, ack_ts, cancel_send_ts, cancel_ack_ts, fill_ts, fill_notified_ts, px, sz, side |
#
# `build_data()` processes one coin-day at a time. **Real:** it reads only the mapped columns, renames them per
# `COLMAP`, and converts timestamps (a tz-naive datetime column without `TS_TZ` is an error, not a guess).
# **Synthetic:** the generator works in exchange time. HL mid = BN mid delayed `hl_lag_ms` × exp(OU basis).
# Informed HL takers are followed by a BN jump of `informed_markout_bps` in their direction. Prints are at the HL
# touch, and size beyond it prints one tick through. Local timestamps add per-venue latency and stay FIFO.
# Either way it then drops crossed or non-positive books, sorts by `ts_local` and writes parquet to the cache.
# The table below shows rows and dropped rows per coin-day and table.

# %%
display(mm.build_data())

# %% [markdown]
# ## 0. Timestamp sanity: timezone and clock alignment
#
# **What and why.** A timezone or unit bug quietly ruins everything after this. Per table: are timestamps in a
# plausible epoch range (catches s/ms/us/ns mix-ups)? What share of rows fall outside the requested UTC date
# (catches files partitioned by *local* date)? And what does `ts_local - ts_exchange` look like (median, tails,
# share negative)? It warns loudly when that offset is near a whole number of hours. Across venues, 1-minute mid
# returns are correlated at whole-hour and whole-minute lags on both clocks, and the best lag must be 0.
# **Self-test:** copies shifted by +10 h (local only) and +1 h (one venue) must both raise TIMEZONE warnings.

# %%
display(mm.sanity().round(3))
mm.sanity_selftest()
mm.plot_alignment()

# %% [markdown]
# ### My read:
# 1. Is the local-minus-exchange offset per venue what I expect from where my collectors run, and is it stable
#    across days?
# 2. Do any tables have a meaningful share of rows outside the UTC date, or non-monotone local timestamps?
#
#

# %% [markdown]
# ## 1. Latency / stale window
#
# **What and why.** The stale window is how long after a Binance move my HL quote can still be picked off. It's
# bounded by order-ack, cancel and fill-notification latency, shown at p50/p90/p99 overall and by UTC hour (of send).
# `ack = ack_ts - send_ts` and `cancel = cancel_ack_ts - cancel_send_ts` are on one clock. `fill notification =
# fill_notified_ts - fill_ts` mixes exchange and local clocks unless both are local, so any clock offset ends up in
# it. Skipped (gate 1 = n/a) without own_orders data. **Check:** each p50 within 10% of its planted median.

# %%
lat = mm.latency()
if lat is not None:
    display(lat["summary"].round(1))
mm.plot_latency(lat)
mm.check_latency(lat)

# %% [markdown]
# ### My read:
# 1. Is my p99 cancel latency inside TAU, and in which hours does it stop being?
# 2. Does fill notification lag enough that I'd hedge late on the moves that matter?
# 3. Is the hour-of-day pattern mine (infra) or the venue's (load)?
#
#

# %% [markdown]
# ## 2. Room before adverse selection
#
# **What and why.** If the quoted HL half-spread doesn't cover round-trip fees, nothing downstream matters.
# `room = HL half-spread (bps of mid) - round-trip fees`, where the round trip is the HL maker fee plus the
# `EXIT_LEG` fee. The half-spread is time-weighted per minute (a quote counts until the next update, capped at
# 60 s). Plots: room over time, its distribution, and half-spread against 5-minute realised Binance volatility (from
# 1 s returns). **Check:** median HL half-spread within 5% of planted.

# %%
rm = mm.room()
display(rm["summary"].round(3))
mm.plot_room(rm)
mm.check_room(rm)

# %% [markdown]
# ### My read:
# 1. Is there room left after fees at my fee tier, and in what share of minutes?
# 2. Does the spread widen with volatility enough to pay for the extra adverse selection, or does it stay flat?
#
#

# %% [markdown]
# ## 3. Adverse selection (kill test)
#
# **What and why.** The markout of a public HL trade at horizon h is `side × (FV(t+h) - FV(t))`, in bps, where
# `side` is the aggressor (+1 buy). It's what the aggressor gained, so it's what a maker on the other side lost
# relative to FV. `t` is the trade's `JOIN_CLOCK` time. Trades whose `t + max horizon` runs past the end of the FV
# data are dropped. Splits: notional size bucket, and the Binance move over `[t - PRE_WINDOW, t]` signed by
# aggressor (+ = the taker is following a Binance move I'd be stale on).
#
# `net edge = half-spread at the trade - markout(TAU) - cost`, where cost = round-trip fees, plus the Binance
# half-spread at the trade when hedging there. The half-spread is the HL touch strictly before the print (same venue,
# exchange clock). Means are trade-weighted and notional-weighted, with 95% block-bootstrap CIs (`BOOT_BLOCK_S`
# blocks, because trades cluster in time). **Check:** markout at TAU and 30 s ≈ `informed_frac × informed_markout`
# (±0.15 bps), and net edge ≈ its planted value. Public data identifies only that product, not the two separately.

# %%
adv = mm.adverse()
print(f"Markouts by horizon (bps; + = aggressor gained = maker lost). TAU = {mm.TAU} ms")
display(adv["markouts"].round(3))
print("Net edge per unit at TAU (bps), with 95% block-bootstrap CI")
display(adv["net"].round(3))
mm.plot_adverse(adv)
display(adv["counts"])   # trades per bucket: read the bucket bars with these in mind
mm.check_adverse(adv)

# %% [markdown]
# ### My read:
# 1. Is this coin's net edge positive at TAU, and how much worse should I expect my own fills to be than the
#    average public trade?
# 2. Which bucket carries the toxicity (large prints, or takers following a Binance move I'd be stale on)?
#    Could I avoid it with a size or move filter without losing most of the volume?
# 3. Does the markout keep growing past TAU (information) or flatten (pure latency)?
#
#

# %% [markdown]
# ## 4. Getting out
#
# **What and why.** After a passive fill I either wait for an offsetting trade on HL or hedge on Binance.
# - **Offset hazard.** Each aggressor event (first print per timestamp and side) makes me the maker on the other
#   side. Wait = time until an opposite-aggressor print at or better than the target. With
#   `OFFSET_TARGET="entry_quote"` the target is the opposite HL touch at entry (full spread); with `"entry_px"` it's
#   my fill price (scratch). Waits are censored at `OFFSET_MAX_WAIT_S`. Each wait is labelled by whether Binance moved
#   with or against my position over `OFFSET_BN_WINDOW_MS` after entry. That label is descriptive and conditions on
#   the future.
# - **Basis half-life.** `basis = log HL mid - log FV` (bps), sampled every `BASIS_RESAMPLE_S`. AR(1) by OLS on
#   pairs within each day, `half-life = -ln2 · dt / ln φ`, with a CI from 30-minute bootstrap blocks. HL lagging
#   Binance adds noise that biases this downward. `half_life_robust_s` (from γ(2)/γ(1)) is insensitive to that
#   noise and is shown as a diagnostic.
# - **Binance taker cost** at `HEDGE_SIZE_USD` = half-spread + taker fee. With L1 only, it's a lower bound when the
#   size exceeds the touch; the share of time that happens is shown.
#
# **Check:** basis half-life within 20% of planted; BN median half-spread within 5%.

# %%
out = mm.getting_out()
display(out["summary"].round(3))
mm.plot_getting_out(out)
mm.check_getting_out(out)

# %% [markdown]
# ### My read:
# 1. When Binance moves against me, how long do I sit on inventory before HL offers an exit, and is hedging on
#    Binance cheaper than that wait?
# 2. Is the basis half-life short enough that hedging and unwinding later costs less than crossing on HL?
# 3. How far apart are the OLS and robust half-lives, and what does that say about noise in my basis?
#
#

# %% [markdown]
# ## 5. Lead-lag
#
# **What and why.** If HL follows Binance with a delay, a Binance move tells me where HL is going, and tells
# everyone faster than me too.
# - **Cross-correlation** of mid log-returns on an `XCORR_GRID_MS` grid, by lag. Lag > 0 means the HL return comes
#   after the BN return, i.e. Binance leads.
# - **Fraction reflected:** the slope of `HL(t+h) - HL(t-W)` on `BN(t) - BN(t-W)` (`W = REFLECT_WINDOW_MS`,
#   non-overlapping windows) at each `h`. A value of 1 means HL has fully caught up with the Binance move.
# - **Binance moves over TAU windows:** the distribution with p90/p99, and the UTC hours where moves above the
#   `BIG_MOVE_Q` quantile land.
#
# **Check:** xcorr peak within 30 ms of the planted lag on `JOIN_CLOCK` (i.e. `hl_lag_ms` plus HL minus BN feed
# latency); fraction reflected at 5 s within 0.05 of 1.

# %%
ll = mm.lead_lag()
display(ll["summary"].round(3))
mm.plot_lead_lag(ll)
mm.check_lead_lag(ll)

# %% [markdown]
# ### My read:
# 1. Is the lag between Binance and HL longer than my TAU? If it is, who is repricing HL before me?
# 2. What share of a Binance move is still unreflected after TAU, and is that what my quotes get picked off on?
# 3. Do the big TAU moves cluster in hours I could just stop quoting?
#
#

# %% [markdown]
# ## 6. Capacity
#
# **What and why.** Edge per unit means little without volume. My quote = the HL touch strictly before each print.
# **Touch** volume = prints at or through my level (optimistic fill bound). **Through** volume = prints strictly
# through it (conservative). Each print is capped at `QUOTE_SIZE_USD` if set.
# `$/day = mean daily volume × CAPACITY_SHARE × net edge (notional-weighted, section 3) / 1e4`, with the net edge CI
# carried through. Through-fills are likely more toxic than average, so the through bound is optimistic on edge.
# **Check:** touch volume = all generated volume, and through volume = the generated sweep remainder.

# %%
cap = mm.capacity(adv["net"])
display(cap["summary"].round(1))
mm.plot_capacity(cap)
mm.check_capacity(cap)

# %% [markdown]
# ### My read:
# 1. Does capacity survive under the through bound, or only under touch?
# 2. What queue share is realistic for me at the HL touch, and what does that do to $/day?
#
#

# %% [markdown]
# ## 7. Competition (placeholder)
#
# **What and why.** Who I'd be competing with at the HL touch, and how fast they reprice after Binance moves. This
# needs wallet- or order-level data (e.g. HL `userFills` by address), which isn't assumed to exist. The lib has
# TODO stubs for the intended outputs: `top_of_book_owners` (who holds the touch and for how long),
# `reprice_speed_after_bn_move` (per wallet, time from a BN move to their HL cancel/replace) and `cancel_to_fill`.
# Nothing is computed, so there's no check.

# %% [markdown]
# ### My read:
# 1. If I had wallet-level data, which competitor behaviour would change my decision: repricing speed or
#    share of time at the touch?
# 2. Can I infer competition from public data alone, e.g. how fast the HL touch reprices after a BN move?
#
#

# %% [markdown]
# ## Decision
#
# Per coin, a gate passes only if all of its metrics meet their thresholds: 1 `ack_latency_p99_ms`;
# 2 `room_median_bps`; 3 `net_edge_tau_bps`; 4 `basis_half_life_s` and `offset_median_s`; 5 `frac_reflected_tau`
# and `bn_move_tau_p99_bps`; 6 `capacity_usd_day_touch` and `_through`. With any threshold still `None`, the cell
# refuses to print verdicts. A gate with no data shows `n/a`. The final cell lists every warning raised and, on
# synthetic data, asserts that every check passed.

# %%
mm.decision()
mm.decision_selftest()

# %%
mm.finish()

# %% [markdown]
# ## Assumptions
# Every modelling choice the brief left open. Each one is a CONFIG knob unless noted.
# 1. **Fees**: placeholder public base-tier rates (HL 1.5/4.5, Binance 2.0/5.0 bps maker/taker).
# 2. **Round-trip cost** = HL maker fee + exit leg (`EXIT_LEG`, default Binance taker). In section 3, the cost also
#    includes the Binance half-spread at the trade.
# 3. **Markout reference** is the change in FV from the trade time, not trade price vs FV. The half-spread enters
#    net edge separately so it isn't counted twice. The basis level is ignored (hedge on Binance locks it).
# 4. **Clocks**: cross-venue joins on `JOIN_CLOCK = ts_local`. Same-venue matching (a print vs the HL touch just
#    before it, offset waits) uses `ts_exchange`, because that's the order in which the exchange matched them.
# 5. **My quote level** = the HL touch strictly before each print (join, no improvement). Fill bounds: touch = at or
#    through, through = strictly through.
# 6. **Net edge for the gate** = notional-weighted point estimate. Capacity applies that same edge to touch and
#    through volume.
# 7. **Size buckets** `SIZE_BUCKETS_USD` and **prior-move buckets** `BN_MOVE_BUCKETS_BPS` over `PRE_WINDOW_MS = TAU`.
# 8. **Offset hazard**: entry = first print per (timestamp, side); exit target = opposite touch at entry
#    (`OFFSET_TARGET`); censored at `OFFSET_MAX_WAIT_S = 60`; with/against = BN move over TAU after entry.
# 9. **Basis AR(1)**: `BASIS_RESAMPLE_S = 10`, pairs pooled within days, 30-minute bootstrap blocks; HL−BN lag noise
#    biases the half-life down (robust φ shown as a diagnostic).
# 10. **Binance taker cost** from L1 only, so it's a lower bound when `HEDGE_SIZE_USD` exceeds the touch.
# 11. **Lead-lag**: 10 ms grid; `REFLECT_WINDOW_MS = 1000`; big move = above q0.99 of |TAU moves|.
# 12. **Capacity**: `CAPACITY_SHARE = 1.0` (upper bound), no quote-size cap.
# 13. **Time weighting**: a quote counts until the next update, capped at 60 s, and is credited to its start bucket.
# 14. **CIs**: block bootstrap with 5-minute blocks, 500 resamples.
# 15. **Gate metrics**: one or two per gate (above). Their direction (`>=`/`<=`) is yours to set in `THRESHOLDS`.
# 16. **Synthetic**: informed takers are the same size as uninformed ones; prices aren't on a tick grid; HL mid
#     doesn't react to HL trades; there are no stale-quote pickers beyond the planted informed flow.
#
# ## Open questions
# 1. "Cost" in net edge: should the Binance hedge crossing cost be in it (current), or only fees as in section 2?
# 2. Offset hazard target: "at that level or better". Is that the opposite touch at entry (full spread, current) or
#    my fill price (scratch)?
# 3. Should the gate 3 verdict use the CI lower bound instead of the point estimate?
# 4. For `fill_ts` in own_orders: is it exchange time? If so, the fill-notification latency includes the clock
#    offset between exchange and local.
# 5. Capacity: should through-volume use its own (likely worse) net edge instead of the all-trade average?
