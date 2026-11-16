"""Two deliberately plain baseline strategies.

They exist so there is something to measure and something to beat. Neither is tuned, and the obvious
weaknesses (a naive fair value, no reaction to book state, slow requotes, a blunt hedge) are left in
on purpose: finding and fixing them, and proving the fix with the analytics, is the exercise.
"""
from __future__ import annotations

import math

from .core import BUY, SELL
from .sim import Strategy


class _Quotes:
    """Keeps at most one resting order per (side, level) slot and moves it when the target price moves."""

    def _init_slots(self):
        self._slot: dict[tuple[int, int], object] = {}

    def _target(self, sim, inst, side, k, tick, qty, requote_ticks, tag="quote"):
        """Make slot (side, k) rest at `tick` (None = no order). A stale order is cancelled first and
        the replacement goes out on a later call, once the cancel is confirmed."""
        o = self._slot.get((side, k))
        if o is not None and not o.is_open:
            o = self._slot[(side, k)] = None
        if o is not None:
            if tick is None or abs(o.tick - tick) >= requote_ticks:
                sim.cancel(o)
            return
        if tick is not None:
            self._slot[(side, k)] = sim.submit(inst, side, tick * sim.spec(inst).tick, qty,
                                               post_only=True, tag=tag, meta={"k": k})


class SimpleMM(_Quotes, Strategy):
    """One bid and one ask around a fair value on a single instrument, skewed against inventory.

    fair: "mid" or "micro" (size-weighted mid), or any function f(sim, book) -> price."""

    def __init__(self, inst: str, half_spread_ticks: float = 1.0, size: float = 5.0, max_pos: float = 50.0,
                 skew_ticks: float = 0.5, interval: float = 0.1, fair="mid", requote_ticks: int = 1,
                 warmup: float = 5.0):
        self.inst, self.hs, self.size, self.max_pos = inst, half_spread_ticks, size, max_pos
        self.skew, self.interval, self.fair, self.requote = skew_ticks, interval, fair, requote_ticks
        self.warmup = warmup
        self._init_slots()

    def on_start(self, sim):
        sim.timer(self.interval, "quote")

    def on_timer(self, sim, tag):
        sim.timer(self.interval, "quote")
        b = sim.book(self.inst)
        if b.bb is None or b.ba is None or sim.now - sim.t0 < self.warmup:
            return
        fv = b.mid if self.fair == "mid" else b.microprice if self.fair == "micro" else self.fair(sim, b)
        sim.log("fair", fv)
        pos = sim.position(self.inst)
        center = fv / b.tick - self.skew * pos / self.size
        bid = min(math.floor(center - self.hs + 1e-9), b.ba - 1)
        ask = max(math.ceil(center + self.hs - 1e-9), b.bb + 1)
        self._target(sim, self.inst, BUY, 0, bid if pos < self.max_pos else None, self.size, self.requote)
        self._target(sim, self.inst, SELL, 0, ask if pos > -self.max_pos else None, self.size, self.requote)


class SpreadQuoter(_Quotes, Strategy):
    """Rests passive clips on the `quote` leg around a fair value implied by the `hedge` leg, and
    hedges each fill by taking liquidity on the hedge leg (and on `fx`, if given).

    Fair value: the quote leg's premium over the hedge leg, in logs, is tracked with an exponential
    moving average (half-life `halflife`); fair = hedge mid x exp(average premium + funding_adj).
    The strategy therefore buys the quote leg when it is cheap against its own recent relationship
    to the hedge leg and sells it when rich: it accumulates a spread position as the spread stretches.

    levels_bps        distances of the clips from fair value, nearest first
    clip              size of each clip on the quote leg
    max_clips         position limit, in clips, on each side
    hedge_per_unit    hedge-leg units per quote-leg unit
    ratio             quote price / hedge price at par, after any currency conversion
    skew_bps          shift of all quotes, per clip of inventory, against the position
    shift_ticks       push every clip this many ticks further from fair value (static)
    controller        optional object that adds a dynamic shift on top (see controllers.py)
    funding_adj       optional f(sim, strategy) -> log shift of fair value (see naive_funding_adj)
    fx, fx_invert     currency leg; fx_invert=True when it is quoted as quote-ccy per hedge-ccy (KRW per USDT)
    """

    def __init__(self, quote: str, hedge: str, fx: str | None = None, levels_bps=(6.0, 12.0), clip: float = 10.0,
                 max_clips: int = 4, hedge_per_unit: float = 1.0, ratio: float = 1.0, halflife: float = 120.0,
                 interval: float = 0.1, requote_ticks: int = 1, hedge_cross_ticks: int = 3, warmup: float = 60.0,
                 skew_bps: float = 0.0, funding_adj=None, fx_invert: bool = True, shift_ticks: int = 0,
                 controller=None, vol_halflife: float = 60.0):
        self.quote, self.hedge_leg, self.fx = quote, hedge, fx
        self.levels, self.clip, self.max_clips = tuple(levels_bps), clip, max_clips
        self.hpu, self.ratio, self.halflife, self.interval = hedge_per_unit, ratio, halflife, interval
        self.requote, self.cross, self.warmup = requote_ticks, hedge_cross_ticks, warmup
        self.skew, self.funding_adj, self.fx_invert = skew_bps, funding_adj, fx_invert
        self.alpha = 1 - math.exp(-interval * math.log(2) / halflife)
        self.shift_ticks, self.controller = shift_ticks, controller
        self.shift = shift_ticks        # ticks every clip is currently pushed away from fair value
        self.vol = math.nan             # EWMA volatility of the spread, bps per sqrt(second)
        self._var, self._vw, self._prev_rel, self._n = 0.0, 0.0, None, 0
        self._va = 1 - math.exp(-interval * math.log(2) / vol_halflife)
        self.premium = None             # EWMA of ln(quote mid in hedge ccy / (hedge mid x ratio))
        self.fair = math.nan
        self.due: dict[tuple[str, int], float] = {}      # (leg, quote fill id) -> signed hedge qty still to do
        self.working: set = set()
        self._init_slots()

    def on_start(self, sim):
        sim.timer(self.interval, "quote")

    # ---- fair value and quotes ------------------------------------------------------------
    def conv(self, sim) -> float:
        """Multiplier taking a quote-leg price into the hedge leg's currency."""
        if self.fx is None:
            return 1.0
        m = sim.book(self.fx).mid
        return (1.0 / m if self.fx_invert else m) if m == m else math.nan

    def on_timer(self, sim, tag):
        sim.timer(self.interval, "quote")
        qb, hb = sim.book(self.quote), sim.book(self.hedge_leg)
        qm, hm, cv = qb.mid, hb.mid, self.conv(sim)
        if qm != qm or hm != hm or cv != cv:
            return
        rel = math.log(qm * cv / (hm * self.ratio))
        self.premium = rel if self.premium is None else self.premium + self.alpha * (rel - self.premium)
        if self._prev_rel is not None:
            d2 = ((rel - self._prev_rel) * 1e4) ** 2 / self.interval
            self._var += self._va * (d2 - self._var)
            self._vw += self._va * (1.0 - self._vw)             # removes the start-up bias of the average
            self.vol = math.sqrt(self._var / self._vw)
        self._prev_rel = rel
        self._rehedge(sim)
        if sim.now - sim.t0 < self.warmup:
            return
        adj = self.funding_adj(sim, self) if self.funding_adj else 0.0
        self.fair = fair = hm * self.ratio * math.exp(self.premium + adj) / cv
        sim.log("fair", fair)
        shift = self.shift_ticks + (self.controller.shift(sim, self) if self.controller is not None else 0)
        self._n += 1
        if shift != self.shift or self._n % 20 == 0:
            sim.log("shift", shift)
            sim.log("vol", self.vol)
        self.shift = shift
        clips = sim.position(self.quote) / self.clip
        n_bid = int(min(len(self.levels), math.floor(self.max_clips - clips + 1e-9)))
        n_ask = int(min(len(self.levels), math.floor(self.max_clips + clips + 1e-9)))
        tick, last_b, last_a = qb.tick, None, None
        for k, lv in enumerate(self.levels):
            bt = math.floor(fair * (1 - (lv + self.skew * clips) / 1e4) / tick + 1e-9) - shift
            at = math.ceil(fair * (1 + (lv - self.skew * clips) / 1e4) / tick - 1e-9) + shift
            bt = min(bt, qb.ba - 1) if last_b is None else min(bt, last_b - 1)
            at = max(at, qb.bb + 1) if last_a is None else max(at, last_a + 1)
            last_b, last_a = bt, at
            self._target(sim, self.quote, BUY, k, bt if k < n_bid else None, self.clip, self.requote)
            self._target(sim, self.quote, SELL, k, at if k < n_ask else None, self.clip, self.requote)

    # ---- hedging --------------------------------------------------------------------------
    def on_fill(self, sim, fill):
        o = fill.order
        if self.controller is not None:
            self.controller.on_fill(sim, self, fill)
        if o.tag == "quote":
            key = (self.hedge_leg, fill.id)
            self.due[key] = -o.side * fill.qty * self.hpu
            self._send(sim, key)
        elif o.tag == "hedge":
            key = (o.inst, o.parent)
            if key in self.due:
                self.due[key] -= o.side * fill.qty
            if o.inst == self.hedge_leg and self.fx is not None:
                # the currency exposure created by this hedge fill, in hedge-ccy notional
                fkey = (self.fx, o.parent)
                self.due[fkey] = self.due.get(fkey, 0.0) + o.side * fill.qty * fill.price * sim.spec(o.inst).contract_size
                if fkey not in self.working:
                    self._send(sim, fkey)
            if not o.is_open:
                self.working.discard(key)

    def on_order(self, sim, order, event):
        if order.tag == "hedge" and event in ("cancelled", "rejected"):
            self.working.discard((order.inst, order.parent))

    def _send(self, sim, key):
        leg, parent = key
        q = self.due.get(key, 0.0)
        if abs(q) < sim.spec(leg).lot - 1e-9:
            self.due.pop(key, None)
            return
        o = sim.take(leg, BUY if q > 0 else SELL, abs(q), self.cross, tag="hedge", parent=parent)
        if o is not None:
            self.working.add(key)

    def _rehedge(self, sim):
        for key in [k for k in self.due if k not in self.working]:
            self._send(sim, key)


def naive_funding_adj(perp: str, perp_is: str = "hedge", window: float | None = None):
    """The simplest funding adjustment: shift fair value by the full currently-predicted funding rate,
    all the way to settlement (or only inside the last `window` seconds).

    Holding the perp short through a settlement with rate F receives F, so a quoter that hedges by
    shorting the perp can pay up to F more for the quote leg. This treats the prediction as certain
    and ignores how the perp itself behaves around the settlement. Improve on it."""
    sign = 1.0 if perp_is == "hedge" else -1.0

    def adj(sim, strat):
        c = sim.funding(perp)
        if c is None or (window is not None and c.time_to_settle(sim.now) > window):
            return 0.0
        return sign * c.predicted()

    return adj
