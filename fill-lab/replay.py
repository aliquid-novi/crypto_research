"""
replay.py — fill simulator and policy harness.

Takes a tape (l1_book.csv + trades.csv) and replays hypothetical passive orders
against it to produce fill outcomes with right-censoring and markouts.

FILL MODEL
----------
A resting SELL at price p, placed at t0 behind a queue of Q0, fills at the
first time t where

    cumulative BUY-initiated volume at prices >= p  over (t0, t]   >   Q0

Symmetrically for a BUY at prices <= p. The ">= p" (rather than "== p") matters:
for the market to trade above p it must first clear everything resting at p,
including you.

This is the standard conservative rule (it is what hftbacktest calls a
volume-based queue model). It is conservative because it ignores cancellations
ahead of you, which in reality pull you up the queue and make fills FASTER.
`cancel_rate` exposes that assumption: the queue ahead decays exponentially at
that rate per second. cancel_rate=0.0 is the pure trades-only bound.

Do not treat either end of that range as truth. The gap between them is a real
source of uncertainty in any passive-fill study, and quantifying it on your own
venue is a more useful exercise than picking a number.

MARKOUT SIGN CONVENTION
-----------------------
Signed so POSITIVE = favourable to you, for both sides:
    sell:  markout_h = (mid(t_fill) - mid(t_fill + h)) / mid(t_fill) * 1e4
    buy:   markout_h = (mid(t_fill + h) - mid(t_fill)) / mid(t_fill) * 1e4
A passive fill that is immediately followed by the price running against you
produces a NEGATIVE markout. That is adverse selection.
"""

import csv
import json
import os

import numpy as np

TICK = 0.10


# --------------------------------------------------------------------------
class Tape:
    """Loaded market data with fast time lookups."""

    def __init__(self, datadir):
        self.datadir = datadir
        self._load_book(os.path.join(datadir, "l1_book.csv"))
        self._load_trades(os.path.join(datadir, "trades.csv"))
        p = os.path.join(datadir, "params.json")
        self.params = json.load(open(p)) if os.path.exists(p) else {}
        self.t0_ms = int(self.book_ts[0])
        self.tick = float(self.params.get("tick", TICK))

    def _load_book(self, path):
        ts, bp, bq, ap, aq = [], [], [], [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                ts.append(int(r["transaction_time"]))
                bp.append(float(r["best_bid_price"]))
                bq.append(float(r["best_bid_qty"]))
                ap.append(float(r["best_ask_price"]))
                aq.append(float(r["best_ask_qty"]))
        self.book_ts = np.asarray(ts, dtype=np.int64)
        self.bid = np.asarray(bp)
        self.bid_qty = np.asarray(bq)
        self.ask = np.asarray(ap)
        self.ask_qty = np.asarray(aq)
        self.mid = (self.bid + self.ask) / 2.0

    def _load_trades(self, path):
        ts, px, qty, sell_init = [], [], [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                ts.append(int(r["transact_time"]))
                px.append(float(r["price"]))
                qty.append(float(r["quantity"]))
                # is_buyer_maker == true  =>  buyer was the maker
                #                         =>  the AGGRESSOR sold
                sell_init.append(r["is_buyer_maker"].strip().lower() == "true")
        self.tr_ts = np.asarray(ts, dtype=np.int64)
        self.tr_px = np.asarray(px)
        self.tr_qty = np.asarray(qty)
        self.tr_sell_init = np.asarray(sell_init, dtype=bool)
        self.tr_buy_init = ~self.tr_sell_init

    # -- lookups -----------------------------------------------------------
    def book_idx(self, ts_ms):
        """Index of the most recent book update at or before ts_ms."""
        return max(int(np.searchsorted(self.book_ts, ts_ms, side="right")) - 1, 0)

    def mid_at(self, ts_ms):
        return float(self.mid[self.book_idx(ts_ms)])

    def touch_at(self, ts_ms):
        k = self.book_idx(ts_ms)
        return float(self.bid[k]), float(self.bid_qty[k]), \
               float(self.ask[k]), float(self.ask_qty[k])

    def trade_intensity(self, ts_ms, window_s=60):
        lo = int(np.searchsorted(self.tr_ts, ts_ms - window_s * 1000, side="left"))
        hi = int(np.searchsorted(self.tr_ts, ts_ms, side="right"))
        return (hi - lo) / float(window_s)

    def realised_vol_bps(self, ts_ms, window_s=60):
        """Std of log mid returns over the trailing window, in bps."""
        lo = int(np.searchsorted(self.book_ts, ts_ms - window_s * 1000, side="left"))
        hi = int(np.searchsorted(self.book_ts, ts_ms, side="right"))
        if hi - lo < 10:
            return float("nan")
        m = self.mid[lo:hi]
        return float(np.std(np.diff(np.log(m))) * 1e4)


# --------------------------------------------------------------------------
def queue_ahead(tape, ts_ms, side, limit_price, depth_c=0.15, depth_g=0.70):
    """
    Size resting ahead of a newly placed order at `limit_price`.

    At the touch  -> the published touch quantity.
    Behind it     -> per-level size at that price, which grows modestly with
                     distance from the touch.
    Inside spread -> 0 (you are alone at that level).
    Marketable    -> 0 (it crosses).

    NOTE the distance is measured in BPS, not ticks. On a small-tick instrument
    like BTCUSDT one tick is 0.013 bps, so a normal few-bps price excursion is
    hundreds of ticks; a tick-indexed depth profile would demand an absurd queue
    and nothing would ever fill. What sits ahead of you is the size at YOUR one
    price level, not the cumulative depth between the touch and you.
    """
    bid, bqty, ask, aqty = tape.touch_at(ts_ms)
    mid = (bid + ask) / 2.0
    if side == "sell":
        if limit_price <= bid:
            return 0.0, "marketable"
        if limit_price < ask:
            return 0.0, "inside_spread"
        d_bps = (limit_price - ask) / mid * 1e4
        lvl = aqty * (1.0 + depth_c * d_bps) ** depth_g
        return lvl, ("at_touch" if d_bps < 1e-9 else "behind")
    else:
        if limit_price >= ask:
            return 0.0, "marketable"
        if limit_price > bid:
            return 0.0, "inside_spread"
        d_bps = (bid - limit_price) / mid * 1e4
        lvl = bqty * (1.0 + depth_c * d_bps) ** depth_g
        return lvl, ("at_touch" if d_bps < 1e-9 else "behind")


def simulate_order(tape, place_ts, side, limit_price, horizon_s,
                   cancel_rate=0.0, q_ahead=None):
    """
    Replay one resting limit order. Returns a dict of outcome fields.

    cancel_rate: exponential decay (per second) of the queue ahead, standing in
    for cancellations. 0.0 = trades-only (conservative lower bound on fills).
    """
    end_ts = place_ts + int(horizon_s * 1000)
    if q_ahead is None:
        q_ahead, placement = queue_ahead(tape, place_ts, side, limit_price)
    else:
        _, placement = queue_ahead(tape, place_ts, side, limit_price)

    lo = int(np.searchsorted(tape.tr_ts, place_ts, side="right"))
    hi = int(np.searchsorted(tape.tr_ts, end_ts, side="right"))

    fill_ts = None
    fill_reason = ""
    if hi > lo:
        ts = tape.tr_ts[lo:hi]
        px = tape.tr_px[lo:hi]
        qty = tape.tr_qty[lo:hi]

        # (a) queue rule: enough same-direction volume traded at or through you
        if side == "sell":
            m = tape.tr_buy_init[lo:hi] & (px >= limit_price - 1e-9)
        else:
            m = tape.tr_sell_init[lo:hi] & (px <= limit_price + 1e-9)

        t_queue = None
        if m.any():
            ets = ts[m]
            cum = np.cumsum(qty[m])
            if cancel_rate > 0:
                elapsed = (ets - place_ts) / 1000.0
                need = q_ahead * np.exp(-cancel_rate * elapsed)
            else:
                need = q_ahead
            hit = np.nonzero(cum > need)[0]
            if hit.size:
                t_queue = int(ets[hit[0]])

        # (b) sweep rule: the market cannot print THROUGH your price while you
        # are still resting there -- clearing your level is what let it print.
        # Without this the price can run away from a stale limit that never
        # fills, which silently flips the sign of every markout.
        eps = tape.tick * 1e-6
        if side == "sell":
            sweep = px > limit_price + eps
        else:
            sweep = px < limit_price - eps
        t_sweep = int(ts[np.argmax(sweep)]) if sweep.any() else None

        cands = [t for t in (t_queue, t_sweep) if t is not None]
        if cands:
            fill_ts = min(cands)
            if t_queue is not None and (t_sweep is None or t_queue <= t_sweep):
                fill_reason = "queue"
            else:
                fill_reason = "sweep"

    filled = fill_ts is not None
    out = dict(
        placement=placement,
        fill_reason=fill_reason,
        queue_ahead=round(float(q_ahead), 4),
        filled=int(filled),
        censored=int(not filled),
        fill_time=fill_ts if filled else "",
        time_to_fill_s=round((fill_ts - place_ts) / 1000.0, 3) if filled else "",
    )

    # ---- markouts, only defined for fills --------------------------------
    sign = 1.0 if side == "sell" else -1.0
    if filled:
        m_fill = tape.mid_at(fill_ts)
        out["mid_at_fill"] = round(m_fill, 2)
        for h in (1, 5, 30, 60):
            m_fut = tape.mid_at(fill_ts + h * 1000)
            out[f"markout_{h}s_bps"] = round(sign * (m_fill - m_fut) / m_fill * 1e4, 4)
    else:
        out["mid_at_fill"] = ""
        for h in (1, 5, 30, 60):
            out[f"markout_{h}s_bps"] = ""

    # ---- the counterfactual most analyses drop ---------------------------
    # If it did not fill, you still had to get the trade done. Price you would
    # get crossing at the horizon, and the honest realised price either way.
    bid_e, _, ask_e, _ = tape.touch_at(end_ts)
    cross_px = bid_e if side == "sell" else ask_e
    out["cross_price_at_horizon"] = round(float(cross_px), 2)
    realised = limit_price if filled else cross_px
    out["realised_price"] = round(float(realised), 2)

    m_place = tape.mid_at(place_ts)
    out["mid_at_place"] = round(m_place, 2)
    out["slippage_vs_place_mid_bps"] = round(sign * (realised - m_place) / m_place * 1e4, 4)
    return out


# --------------------------------------------------------------------------
def offset_price(tape, ts_ms, side, offset_bps):
    """Limit price at `offset_bps` outward from the same-side touch,
    rounded outward onto THIS instrument's tick grid."""
    tick = tape.tick
    bid, _, ask, _ = tape.touch_at(ts_ms)
    mid = (bid + ask) / 2.0
    delta = offset_bps / 1e4 * mid
    if side == "sell":
        return float(np.ceil((ask + delta) / tick - 1e-9) * tick)
    return float(np.floor((bid - delta) / tick + 1e-9) * tick)


def grid_sweep(tape, episodes, out_path, cadence_s=3, horizon_s=120,
               offsets_bps=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
               cancel_rate=0.0, ground_truth=None):
    """
    Systematic sweep: at a fixed cadence, place one order per (side, offset)
    and record the outcome. This is the survival dataset.
    """
    start = int(tape.book_ts[0]) + 60_000
    end = int(tape.book_ts[-1]) - (horizon_s + 61) * 1000
    times = np.arange(start, end, cadence_s * 1000, dtype=np.int64)

    ep_start = np.asarray([e["start_time"] for e in episodes], dtype=np.int64)
    ep_end = np.asarray([e["deadline_time"] for e in episodes], dtype=np.int64)
    ep_id = [e["episode_id"] for e in episodes]

    gt_ts, gt_reg = (None, None)
    if ground_truth is not None:
        gt_ts, gt_reg = ground_truth

    cols = ["order_id", "place_time", "episode_id", "time_left_in_episode_s",
            "side", "offset_bps", "limit_price", "distance_ticks", "placement",
            "queue_ahead", "horizon_s", "fill_reason",
            "mid_at_place", "spread_ticks", "queue_imbalance",
            "trade_intensity_60s", "realised_vol_60s_bps",
            "filled", "censored", "fill_time", "time_to_fill_s", "mid_at_fill",
            "markout_1s_bps", "markout_5s_bps", "markout_30s_bps",
            "markout_60s_bps",
            "cross_price_at_horizon", "realised_price",
            "slippage_vs_place_mid_bps", "true_regime"]

    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ts in times:
            bid, bq, ask, aq = tape.touch_at(int(ts))
            spread_ticks = round((ask - bid) / tape.tick)
            imb = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
            inten = tape.trade_intensity(int(ts), 60)
            rvol = tape.realised_vol_60s = tape.realised_vol_bps(int(ts), 60)

            k = int(np.searchsorted(ep_start, ts, side="right")) - 1
            if k >= 0 and ts < ep_end[k]:
                eid, tleft = ep_id[k], round((ep_end[k] - ts) / 1000.0, 1)
            else:
                eid, tleft = "", ""

            reg = ""
            if gt_ts is not None:
                reg = int(gt_reg[min(int(np.searchsorted(gt_ts, ts, side="right")) - 1,
                                     len(gt_reg) - 1)])

            for side in ("sell", "buy"):
                touch = ask if side == "sell" else bid
                for off in offsets_bps:
                    lp = offset_price(tape, int(ts), side, off)
                    d = abs(round((lp - touch) / tape.tick))
                    res = simulate_order(tape, int(ts), side, lp, horizon_s,
                                         cancel_rate=cancel_rate)
                    row = dict(order_id=n, place_time=int(ts), episode_id=eid,
                               time_left_in_episode_s=tleft, side=side,
                               offset_bps=off, limit_price=round(lp, 4),
                               distance_ticks=int(d), horizon_s=horizon_s,
                               spread_ticks=int(spread_ticks),
                               queue_imbalance=round(float(imb), 4),
                               trade_intensity_60s=round(inten, 3),
                               realised_vol_60s_bps=("" if rvol != rvol
                                                     else round(rvol, 4)),
                               true_regime=reg)
                    row.update(res)
                    w.writerow(row)
                    n += 1
    return n


# --------------------------------------------------------------------------
# Episode-level policies. Each returns a schedule of child orders.
# These are deliberately simple baselines, not recommendations.
# --------------------------------------------------------------------------
def policy_cross_immediately(tape, ep, **kw):
    """Benchmark: take the full size at the opposite touch, at t=0."""
    return [dict(place_ts=ep["start_time"], offset_bps=None, qty=ep["quantity"],
                 cross=True)]


def policy_twap_cross(tape, ep, n_slices=8, **kw):
    """Benchmark: equal slices crossing at even intervals."""
    span = ep["deadline_time"] - ep["start_time"]
    return [dict(place_ts=ep["start_time"] + int(i * span / n_slices),
                 offset_bps=None, qty=ep["quantity"] / n_slices, cross=True)
            for i in range(n_slices)]


def policy_passive_then_cross(tape, ep, offset_bps=0.5, n_slices=8, **kw):
    """
    Rest passively for one slice's worth of time, then cross whatever is left.
    The simplest thing that trades off fill risk against spread capture.
    """
    span = ep["deadline_time"] - ep["start_time"]
    slice_ms = int(span / n_slices)
    return [dict(place_ts=ep["start_time"] + i * slice_ms, offset_bps=offset_bps,
                 qty=ep["quantity"] / n_slices, cross=False,
                 rest_ms=slice_ms) for i in range(n_slices)]


def run_episode(tape, ep, policy, cancel_rate=0.0, **kw):
    """
    Execute one episode under a policy. Returns realised VWAP and its slippage
    against the arrival mid, plus how much ended up crossed.
    """
    side = ep["side"]
    sign = 1.0 if side == "sell" else -1.0
    arrival_mid = tape.mid_at(ep["start_time"])
    schedule = policy(tape, ep, **kw)

    filled_qty = crossed_qty = 0.0
    notional = 0.0
    for ch in schedule:
        q = ch["qty"]
        if ch.get("cross"):
            bid, _, ask, _ = tape.touch_at(ch["place_ts"])
            px = bid if side == "sell" else ask
            notional += px * q
            crossed_qty += q
            filled_qty += q
            continue

        lp = offset_price(tape, ch["place_ts"], side, ch["offset_bps"])
        res = simulate_order(tape, ch["place_ts"], side, lp,
                             ch["rest_ms"] / 1000.0, cancel_rate=cancel_rate)
        if res["filled"]:
            notional += lp * q
        else:
            # deadline discipline: cross the remainder at the slice end
            t_end = ch["place_ts"] + ch["rest_ms"]
            bid, _, ask, _ = tape.touch_at(t_end)
            px = bid if side == "sell" else ask
            notional += px * q
            crossed_qty += q
        filled_qty += q

    vwap = notional / filled_qty if filled_qty else float("nan")
    return dict(
        episode_id=ep["episode_id"], side=side, quantity=ep["quantity"],
        arrival_mid=round(arrival_mid, 2), realised_vwap=round(vwap, 2),
        slippage_bps=round(sign * (vwap - arrival_mid) / arrival_mid * 1e4, 4),
        crossed_fraction=round(crossed_qty / ep["quantity"], 4),
    )


# --------------------------------------------------------------------------
def load_episodes(path):
    eps = []
    with open(path) as f:
        for r in csv.DictReader(f):
            eps.append(dict(
                episode_id=int(r["episode_id"]),
                start_time=int(r["start_time"]),
                deadline_time=int(r["deadline_time"]),
                horizon_s=float(r["horizon_s"]),
                side=r["side"], quantity=float(r["quantity"]),
                arrival_mid=float(r["arrival_mid"]),
            ))
    return eps


def load_ground_truth(path):
    ts, reg = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ts.append(int(r["time"]))
            reg.append(int(r["regime"]))
    return np.asarray(ts, dtype=np.int64), np.asarray(reg, dtype=np.int8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--cadence", type=int, default=3)
    ap.add_argument("--horizon", type=float, default=120)
    ap.add_argument("--cancel-rate", type=float, default=0.0)
    ap.add_argument("--offsets", default="0,0.25,0.5,1,2,4,8",
                    help="comma-separated offsets from the touch, in bps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tape = Tape(a.datadir)
    eps = load_episodes(os.path.join(a.datadir, "episodes.csv"))
    gt_path = os.path.join(a.datadir, "ground_truth.csv")
    gt = load_ground_truth(gt_path) if os.path.exists(gt_path) else None
    out = a.out or os.path.join(a.datadir, "order_outcomes.csv")

    print(f"tape: {len(tape.book_ts):,} book rows, {len(tape.tr_ts):,} trades, "
          f"{len(eps)} episodes")
    offs = tuple(float(x) for x in a.offsets.split(","))
    n = grid_sweep(tape, eps, out, cadence_s=a.cadence, horizon_s=a.horizon,
                   offsets_bps=offs, cancel_rate=a.cancel_rate, ground_truth=gt)
    print(f"wrote {out}  ({n:,} orders)")
