"""
gen_market.py — synthetic perpetual-futures tape for fill-strategy research.

Generates a BTCUSDT-perp-like market with the stylised facts that matter for
execution research, then writes CSVs whose schemas match Binance Vision's real
public data files exactly (see fetch_real_data.sh).

Generative model
----------------
1. Latent regime: 3-state Markov chain (calm / normal / stressed) on a 1s grid.
   Governs volatility, trade intensity and spread.

2. Trade arrivals: bivariate Hawkes process (buy-initiated, sell-initiated)
   with exponential kernels, self- and cross-excitation, regime-modulated
   baseline. Simulated on a 10ms grid by recursive intensity update.

3. Efficient price S_t: driftless diffusion with regime-dependent vol, PLUS
   permanent impact from each trade (square-root in size). The impact term is
   what creates adverse selection: clustered buy flow both fills your resting
   sell AND pushes the price up.

4. L1 book: bid/ask pinned to S_t with a regime-dependent half-spread and
   hysteresis. Touch queue sizes are mean-reverting, consumed by trades, and
   redrawn when the touch moves. Deeper levels follow an increasing depth
   profile.

Nothing here is fitted to real data. Parameters are chosen to be plausible for
a liquid crypto perp; treat magnitudes as illustrative, structure as the point.

Usage:  python3 gen_market.py [--hours 6] [--seed 7]
"""

import argparse
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# Instrument / market constants (BTCUSDT-perp-like)
# --------------------------------------------------------------------------
TICK = 0.10            # USDT
LOT = 0.001            # BTC
S0 = 78_000.0          # starting price
START_MS = 1_758_000_000_000   # arbitrary epoch ms for the session start

# --------------------------------------------------------------------------
# Instrument presets.
#
# The single number that decides WHICH GAME a passive order is playing is the
# ratio of tick size to per-second volatility:
#
#     time for price to cross one tick  ~  (tick / sigma_1s) ** 2
#     time for the touch queue to clear ~  queue / volume_rate
#
# small tick  -> the price sweeps through you long before the queue clears.
#                Queue position is nearly worthless; adverse selection and
#                non-fills dominate. This is BTCUSDT.
# large tick  -> the queue binds first. Queue position is worth real money and
#                time-to-fill has a long right tail. This is a wide-spread
#                contract, and it is the world most of the queue-value
#                literature was written about.
#
# Generate both and compare. The contrast is the point.
# --------------------------------------------------------------------------
PRESETS = {
    "btc_perp": dict(
        tick=0.10, lot=0.001, s0=78_000.0, daily_vol=0.018,
        queue_log_mean=0.30, queue_log_sd=0.60, impact_kappa=0.95,
        size_log_mean=-3.6, size_log_sd=1.15, reprice_ticks=1.5,
    ),
    "wide_tick": dict(
        tick=20.0, lot=0.01, s0=78_000.0, daily_vol=0.010,
        queue_log_mean=1.10, queue_log_sd=0.55, impact_kappa=3.0,
        size_log_mean=-2.6, size_log_sd=1.05, reprice_ticks=0.6,
    ),
}


def apply_preset(name):
    """Overwrite module globals from a preset."""
    global TICK, LOT, S0, DAILY_VOL_NORMAL, QUEUE_LOG_MEAN, QUEUE_LOG_SD
    global IMPACT_KAPPA, SIZE_LOG_MEAN, SIZE_LOG_SD, REPRICE_TICKS
    p = PRESETS[name]
    TICK = p["tick"]; LOT = p["lot"]; S0 = p["s0"]
    DAILY_VOL_NORMAL = p["daily_vol"]
    QUEUE_LOG_MEAN = p["queue_log_mean"]; QUEUE_LOG_SD = p["queue_log_sd"]
    IMPACT_KAPPA = p["impact_kappa"]
    SIZE_LOG_MEAN = p["size_log_mean"]; SIZE_LOG_SD = p["size_log_sd"]
    REPRICE_TICKS = p["reprice_ticks"]

# Regime definitions: (vol multiplier, intensity multiplier, half-spread ticks)
REGIMES = {
    0: dict(name="calm",     vol_mult=0.55, int_mult=0.60, hs_ticks=0.15),
    1: dict(name="normal",   vol_mult=1.00, int_mult=1.00, hs_ticks=0.30),
    2: dict(name="stressed", vol_mult=2.60, int_mult=2.40, hs_ticks=1.50),
}

# Regime transition matrix on a 1s grid (persistent).
P_REGIME = np.array([
    [0.99940, 0.00055, 0.00005],
    [0.00050, 0.99900, 0.00050],
    [0.00010, 0.00390, 0.99600],
])

# Baseline daily vol in the "normal" regime (fraction of price).
DAILY_VOL_NORMAL = 0.018

# Hawkes parameters (per side)
MU_BASE = 0.60         # baseline intensity, events/sec/side
BETA = 2.0             # kernel decay rate (1/s) -> ~0.5s clustering timescale
ALPHA_SELF = 0.80      # self-excitation
ALPHA_CROSS = 0.40     # cross-excitation
# branching ratio n = (ALPHA_SELF + ALPHA_CROSS) / BETA = 0.60  (stable, < 1)

DT = 0.01              # simulation grid for the Hawkes recursion (s)

# Trade size distribution: lognormal body + occasional large clip
SIZE_LOG_MEAN = -3.6   # log(BTC)
SIZE_LOG_SD = 1.15
SIZE_BIG_PROB = 0.012
SIZE_BIG_SCALE = 1.8

# Permanent impact: kappa * sign * sqrt(qty) in USDT
IMPACT_KAPPA = 0.95     # permanent impact: kappa * sign * sqrt(qty), in USDT

# Touch queue sizes (BTC), lognormal, regime-scaled
QUEUE_LOG_MEAN = 0.30
QUEUE_LOG_SD = 0.60

# Depth profile away from touch: depth(d ticks) = touch_q * (1 + DEPTH_C * d) ** DEPTH_G
DEPTH_C = 0.15
DEPTH_G = 0.70

BOOK_SNAP_MS = 100     # heartbeat book publication cadence (on top of every trade)

# Quote inertia: makers do not reprice on every wiggle of the efficient price.
# The quoted mid is only moved once S has drifted this far from it. Larger =
# stickier quotes and more adverse selection for anyone resting at the touch.
REPRICE_TICKS = 1.5


# --------------------------------------------------------------------------
def simulate_regimes(n_sec, rng):
    """1s-grid Markov regime path."""
    path = np.empty(n_sec, dtype=np.int8)
    state = 1
    cum = np.cumsum(P_REGIME, axis=1)
    u = rng.random(n_sec)
    for i in range(n_sec):
        state = int(np.searchsorted(cum[state], u[i]))
        path[i] = state
    return path


def simulate_hawkes(n_steps, regime_at_step, rng):
    """
    Bivariate Hawkes on a DT grid via recursive intensity update.
    Returns (times_sec, sides) where side +1 = buy-initiated, -1 = sell-initiated.
    """
    decay = np.exp(-BETA * DT)
    int_mult = np.array([REGIMES[r]["int_mult"] for r in range(3)])

    s_bb = s_bs = s_sb = s_ss = 0.0   # excitation state variables
    times = []
    sides = []

    u_buy = rng.random(n_steps)
    u_sell = rng.random(n_steps)
    jitter = rng.random(n_steps) * DT

    for i in range(n_steps):
        m = int_mult[regime_at_step[i]]
        lam_b = MU_BASE * m + s_bb + s_bs
        lam_s = MU_BASE * m + s_ss + s_sb

        # decay first, then add any new excitation from this step's events
        s_bb *= decay
        s_bs *= decay
        s_sb *= decay
        s_ss *= decay

        t = i * DT
        if u_buy[i] < lam_b * DT:
            times.append(t + jitter[i])
            sides.append(1)
            s_bb += ALPHA_SELF
            s_sb += ALPHA_CROSS
        if u_sell[i] < lam_s * DT:
            times.append(t + jitter[i])
            sides.append(-1)
            s_ss += ALPHA_SELF
            s_bs += ALPHA_CROSS

    order = np.argsort(times, kind="stable")
    return np.asarray(times)[order], np.asarray(sides, dtype=np.int8)[order]


def simulate_sizes(n, rng):
    sizes = np.exp(rng.normal(SIZE_LOG_MEAN, SIZE_LOG_SD, n))
    big = rng.random(n) < SIZE_BIG_PROB
    sizes[big] += rng.exponential(SIZE_BIG_SCALE, big.sum())
    return np.maximum(np.round(sizes / LOT) * LOT, LOT)


def simulate_efficient_price(n_sec, regime_sec, trade_t, trade_side, trade_qty, rng):
    """
    Efficient price on the DT grid: regime-dependent diffusion + permanent
    trade impact. Returns (grid_times, S).
    """
    n_steps = int(n_sec / DT)
    grid_t = np.arange(n_steps) * DT
    regime_at_step = regime_sec[np.minimum((grid_t).astype(int), n_sec - 1)]

    vol_mult = np.array([REGIMES[r]["vol_mult"] for r in range(3)])
    sig_per_sec = S0 * DAILY_VOL_NORMAL / np.sqrt(86_400.0)
    sig_step = sig_per_sec * vol_mult[regime_at_step] * np.sqrt(DT)

    diffusion = np.cumsum(rng.normal(0.0, 1.0, n_steps) * sig_step)

    impact = IMPACT_KAPPA * trade_side * np.sqrt(trade_qty)
    idx = np.minimum((trade_t / DT).astype(np.int64), n_steps - 1)
    impact_grid = np.zeros(n_steps)
    np.add.at(impact_grid, idx, impact)

    S = S0 + diffusion + np.cumsum(impact_grid)
    return grid_t, S, regime_at_step


def build_book(grid_t, S, regime_at_step, trade_t, trade_side, trade_qty, rng):
    """
    Evolve a consistent L1 book alongside the trade tape.

    The touch is pinned to S with a regime half-spread and one-tick hysteresis.
    Touch queues are consumed by same-side aggressing trades and redrawn when
    the level moves. Trades print at the touch they consume.

    Returns book update arrays and the realised trade print prices.
    """
    hs_ticks = np.array([REGIMES[r]["hs_ticks"] for r in range(3)])

    n_trades = len(trade_t)
    trade_idx_on_grid = np.minimum((trade_t / DT).astype(np.int64), len(grid_t) - 1)

    # heartbeat book snapshot steps
    snap_every = max(1, int((BOOK_SNAP_MS / 1000.0) / DT))
    snap_steps = set(range(0, len(grid_t), snap_every))

    # initial book
    quoted_mid = float(S[0])
    bid = np.floor((quoted_mid - hs_ticks[regime_at_step[0]] * TICK) / TICK) * TICK
    ask = bid + TICK * max(1, int(round(2 * hs_ticks[regime_at_step[0]])))
    bid_q = float(np.exp(rng.normal(QUEUE_LOG_MEAN, QUEUE_LOG_SD)))
    ask_q = float(np.exp(rng.normal(QUEUE_LOG_MEAN, QUEUE_LOG_SD)))

    bu_t, bu_bid, bu_bq, bu_ask, bu_aq = [], [], [], [], []
    trade_px = np.empty(n_trades)

    # map grid step -> trade indices starting there
    trade_ptr = 0

    def redraw(regime):
        q = float(np.exp(rng.normal(QUEUE_LOG_MEAN, QUEUE_LOG_SD)))
        # stressed regimes hold thinner queues
        return max(LOT, q * (0.45 if regime == 2 else 1.0))

    for i in range(len(grid_t)):
        r = regime_at_step[i]
        hs = hs_ticks[r] * TICK

        # ---- sticky quoted mid, then symmetric touch around it ------------
        # Repricing the mid (rather than each side independently) is what keeps
        # the spread controlled; pinning bid and ask separately lets the spread
        # random-walk to several ticks.
        moved = False
        if abs(S[i] - quoted_mid) > REPRICE_TICKS * TICK:
            quoted_mid = float(S[i])
            new_bid = np.floor((quoted_mid - hs) / TICK) * TICK
            new_ask = np.ceil((quoted_mid + hs) / TICK) * TICK
            if new_ask <= new_bid:
                new_ask = new_bid + TICK
            if new_bid != bid:
                bid = new_bid
                bid_q = redraw(r)
            if new_ask != ask:
                ask = new_ask
                ask_q = redraw(r)
            moved = True
        if ask <= bid:
            ask = bid + TICK
            ask_q = redraw(r)
            moved = True

        # ---- queue replenishment (mean reversion toward a target) ---------
        tgt = np.exp(QUEUE_LOG_MEAN) * (0.45 if r == 2 else 1.0)
        bid_q += 0.02 * (tgt - bid_q)
        ask_q += 0.02 * (tgt - ask_q)

        # ---- process trades landing on this grid step ---------------------
        traded = False
        while trade_ptr < n_trades and trade_idx_on_grid[trade_ptr] == i:
            traded = True
            side = trade_side[trade_ptr]
            qty = trade_qty[trade_ptr]
            if side > 0:                       # buy-initiated: lifts the ask
                trade_px[trade_ptr] = ask
                ask_q -= qty
                if ask_q <= 0:
                    ask += TICK
                    ask_q = redraw(r) + ask_q  # carry the overflow
                    if ask_q <= 0:
                        ask_q = redraw(r)
                    if ask <= bid:
                        ask = bid + TICK
            else:                              # sell-initiated: hits the bid
                trade_px[trade_ptr] = bid
                bid_q -= qty
                if bid_q <= 0:
                    bid -= TICK
                    bid_q = redraw(r) + bid_q
                    if bid_q <= 0:
                        bid_q = redraw(r)
                    if ask <= bid:
                        bid = ask - TICK
            moved = True
            trade_ptr += 1

        # Publish on every trade plus a fixed heartbeat. This mirrors a real
        # throttled bookTicker stream: it is a SAMPLED view of the book, not
        # every internal change. The fill model below drives off the trade
        # tape, which is complete, so the sampling does not bias fills.
        if traded or i in snap_steps:
            bu_t.append(grid_t[i])
            bu_bid.append(bid)
            bu_bq.append(max(bid_q, LOT))
            bu_ask.append(ask)
            bu_aq.append(max(ask_q, LOT))

    return (
        np.asarray(bu_t), np.asarray(bu_bid), np.asarray(bu_bq),
        np.asarray(bu_ask), np.asarray(bu_aq), trade_px,
    )


def build_episodes(n_sec, book_t, mid, bid, ask, rng, every_sec=600, length_sec=480):
    """
    Flatten episodes: at start_time you hold `quantity` and must be flat by
    `deadline_time`. side = the side you must TRADE to get flat.
    """
    rows = []
    eid = 0
    for start in range(300, n_sec - length_sec - 300, every_sec):
        deadline = start + length_sec
        k = int(np.searchsorted(book_t, start, side="right") - 1)
        side = "sell" if rng.random() < 0.5 else "buy"
        qty = float(np.round(np.exp(rng.normal(1.15, 0.6)) / LOT) * LOT)
        rows.append(dict(
            episode_id=eid,
            start_time=START_MS + int(start * 1000),
            deadline_time=START_MS + int(deadline * 1000),
            horizon_s=length_sec,
            side=side,
            quantity=round(qty, 3),
            arrival_mid=round(float(mid[k]), 2),
            arrival_bid=round(float(bid[k]), 4),
            arrival_ask=round(float(ask[k]), 4),
        ))
        eid += 1
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--preset", default="btc_perp", choices=sorted(PRESETS))
    args = ap.parse_args()
    apply_preset(args.preset)

    rng = np.random.default_rng(args.seed)
    n_sec = int(args.hours * 3600)
    n_steps = int(n_sec / DT)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"simulating {args.hours}h of '{args.preset}'  "
          f"(tick={TICK}, daily_vol={DAILY_VOL_NORMAL:.3f}, {n_steps:,} steps)")

    regime_sec = simulate_regimes(n_sec, rng)
    regime_at_step_pre = regime_sec[np.minimum(
        (np.arange(n_steps) * DT).astype(int), n_sec - 1)]

    trade_t, trade_side = simulate_hawkes(n_steps, regime_at_step_pre, rng)
    trade_qty = simulate_sizes(len(trade_t), rng)
    print(f"  trades: {len(trade_t):,}  ({len(trade_t)/n_sec:.2f}/s)")

    grid_t, S, regime_at_step = simulate_efficient_price(
        n_sec, regime_sec, trade_t, trade_side, trade_qty, rng)

    bu_t, bu_bid, bu_bq, bu_ask, bu_aq, trade_px = build_book(
        grid_t, S, regime_at_step, trade_t, trade_side, trade_qty, rng)
    print(f"  book updates: {len(bu_t):,}")

    mid = (bu_bid + bu_ask) / 2.0

    pdp = max(0, -int(np.floor(np.log10(TICK))))   # price decimals from tick
    qdp = max(0, -int(np.floor(np.log10(LOT))))     # qty decimals from lot

    # ---------------- write l1_book.csv (Binance bookTicker schema) --------
    path = os.path.join(args.outdir, "l1_book.csv")
    with open(path, "w") as f:
        f.write("update_id,best_bid_price,best_bid_qty,best_ask_price,"
                "best_ask_qty,transaction_time,event_time\n")
        for i in range(len(bu_t)):
            ts = START_MS + int(round(bu_t[i] * 1000))
            f.write(f"{1000000+i},{bu_bid[i]:.{pdp}f},{bu_bq[i]:.{qdp}f},"
                    f"{bu_ask[i]:.{pdp}f},{bu_aq[i]:.{qdp}f},{ts},{ts}\n")
    print(f"  wrote {path}")

    # ---------------- write trades.csv (Binance aggTrades schema) ----------
    # is_buyer_maker = true  => the BUYER was the maker => SELL-initiated trade
    path = os.path.join(args.outdir, "trades.csv")
    with open(path, "w") as f:
        f.write("agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
                "transact_time,is_buyer_maker\n")
        for i in range(len(trade_t)):
            ts = START_MS + int(round(trade_t[i] * 1000))
            ibm = "true" if trade_side[i] < 0 else "false"
            f.write(f"{2000000+i},{trade_px[i]:.{pdp}f},{trade_qty[i]:.{qdp}f},"
                    f"{3000000+i},{3000000+i},{ts},{ibm}\n")
    print(f"  wrote {path}")

    # ---------------- write episodes.csv -----------------------------------
    eps = build_episodes(n_sec, bu_t, mid, bu_bid, bu_ask, rng)
    path = os.path.join(args.outdir, "episodes.csv")
    with open(path, "w") as f:
        cols = ["episode_id", "start_time", "deadline_time", "horizon_s",
                "side", "quantity", "arrival_mid", "arrival_bid", "arrival_ask"]
        f.write(",".join(cols) + "\n")
        for r in eps:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  wrote {path}  ({len(eps)} episodes)")

    # ---------------- write ground_truth.csv (1s grid) ---------------------
    sec_idx = np.arange(n_sec)
    step_of_sec = np.minimum((sec_idx / DT).astype(np.int64), n_steps - 1)
    book_k = np.maximum(np.searchsorted(bu_t, sec_idx, side="right") - 1, 0)
    vol_mult = np.array([REGIMES[r]["vol_mult"] for r in range(3)])
    int_mult = np.array([REGIMES[r]["int_mult"] for r in range(3)])
    sig_bps = (S0 * DAILY_VOL_NORMAL / np.sqrt(86_400.0)) / S0 * 1e4

    path = os.path.join(args.outdir, "ground_truth.csv")
    with open(path, "w") as f:
        f.write("time,regime,regime_name,efficient_price,observed_mid,"
                "sigma_bps_per_sqrt_s,baseline_intensity_per_side,"
                "half_spread_ticks\n")
        for i in range(n_sec):
            r = int(regime_sec[i])
            ts = START_MS + i * 1000
            f.write(f"{ts},{r},{REGIMES[r]['name']},{S[step_of_sec[i]]:.2f},"
                    f"{mid[book_k[i]]:.2f},{sig_bps*vol_mult[r]:.4f},"
                    f"{MU_BASE*int_mult[r]:.4f},{REGIMES[r]['hs_ticks']:.2f}\n")
    print(f"  wrote {path}")

    # ---------------- params.json ------------------------------------------
    params = dict(
        preset=args.preset,
        seed=args.seed, hours=args.hours, tick=TICK, lot=LOT, s0=S0,
        start_ms=START_MS, daily_vol_normal=DAILY_VOL_NORMAL,
        hawkes=dict(mu_base=MU_BASE, beta=BETA, alpha_self=ALPHA_SELF,
                    alpha_cross=ALPHA_CROSS,
                    branching_ratio=(ALPHA_SELF + ALPHA_CROSS) / BETA),
        impact_kappa=IMPACT_KAPPA,
        regimes={str(k): v for k, v in REGIMES.items()},
        regime_transition=P_REGIME.tolist(),
        depth_profile=dict(c=DEPTH_C, gamma=DEPTH_G),
        n_trades=int(len(trade_t)), n_book_updates=int(len(bu_t)),
    )
    with open(os.path.join(args.outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    np.save(os.path.join(args.outdir, "_regime_sec.npy"), regime_sec)
    print("done")


if __name__ == "__main__":
    main()
