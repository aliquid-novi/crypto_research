"""One instrument's matching logic: the market's book plus our resting orders.

The market data is replayed as-is; our orders are overlaid on it. That is the standard backtest
assumption (our orders do not change what everyone else does), with two small corrections:
liquidity we take stays gone until that level next updates, and several of our own orders at one
price cannot be filled by the same traded quantity twice.

A resting order of ours is filled when:
  1. a trade prints THROUGH our price (worse for the aggressor than our price): up to the print size;
  2. a trade prints AT our price and is larger than the quantity ahead of us: the excess fills us;
  3. new opposite-side liquidity appears at or through our price: it would have traded with us;
  4. (fill_on_clear) a trade empties our level exactly: the aggressor wanted the whole level.
Rule 2 is where the queue model matters. Rules 1 and 3 are where adverse selection comes from.
"""
from __future__ import annotations

import math

from .book import Book
from .core import EPS, IOC, Instrument, Order
from .models import QueueModel


class Venue:
    def __init__(self, inst: Instrument, queue: QueueModel, fill_on_clear: bool = True):
        self.inst, self.queue, self.fill_on_clear = inst, queue, fill_on_clear
        self.book = Book(inst)
        self.resting: list[Order] = []
        self.taken: dict[tuple[int, int], float] = {}

    def _fill(self, o: Order, tick: int, qty: float, maker: bool, fills: list) -> float:
        """Fill whole lots only (queue models can leave fractional quantities ahead of us)."""
        lot = self.inst.lot
        qty = math.floor(qty / lot + 1e-7) * lot
        if qty <= 0:
            return 0.0
        o.x_rem -= qty
        fills.append((o, tick, qty, maker))
        return qty

    def _sweep(self) -> None:
        keep = []
        for o in self.resting:
            if o.x_rem > EPS:
                keep.append(o)
            else:
                o.x_live = False
        self.resting = keep

    # ---- market data ------------------------------------------------------------------------
    def on_depth(self, side: int, tick: int, qty: float, fills: list) -> None:
        prev = self.book.set(side, tick, qty)
        if self.taken:
            self.taken.pop((side, tick), None)
        if not self.resting:
            return
        delta = prev - qty
        n0, used = len(fills), 0.0
        for o in self.resting:
            if o.x_rem <= EPS:
                continue
            if o.side == side:
                if o.tick == tick:
                    if delta > 0:
                        from_trades = o.x_credit if o.x_credit < delta else delta
                        cancelled = delta - from_trades
                        if cancelled > EPS:
                            behind = prev - o.x_ahead
                            o.x_ahead = self.queue.on_cancel(o.x_ahead, behind if behind > 0 else 0.0, cancelled)
                        elif qty <= 0 and from_trades > 0 and self.fill_on_clear:
                            self._fill(o, o.tick, o.x_rem, True, fills)                 # rule 4
                    o.x_credit = 0.0
                    if o.x_ahead > qty:
                        o.x_ahead = qty
            elif qty > prev and ((side < 0 and o.tick >= tick) or (side > 0 and o.tick <= tick)):
                f = min(o.x_rem, qty - prev - used)                                     # rule 3
                if f > EPS:
                    used += self._fill(o, o.tick, f, True, fills)
        if len(fills) > n0:
            self._sweep()

    def on_trade(self, aggressor: int, tick: int, qty: float, fills: list) -> None:
        if not self.resting:
            return
        n0, ours = len(fills), 0.0
        for o in self.resting:                      # arrival order = our own time priority
            if o.side != -aggressor or o.x_rem <= EPS:
                continue
            if o.tick == tick:
                over = qty - o.x_ahead - ours
                o.x_ahead = o.x_ahead - qty if o.x_ahead > qty else 0.0
                o.x_credit += qty
                if over > EPS:                                                          # rule 2
                    ours += self._fill(o, o.tick, min(o.x_rem, over), True, fills)
            elif (aggressor < 0 and tick < o.tick) or (aggressor > 0 and tick > o.tick):
                f = min(o.x_rem, qty - ours)                                            # rule 1
                if f > EPS:
                    ours += self._fill(o, o.tick, f, True, fills)
        if len(fills) > n0:
            self._sweep()

    # ---- our orders -------------------------------------------------------------------------
    def arrive(self, o: Order, fills: list) -> tuple[str, str]:
        """An order reaches the exchange. Returns (status, reason)."""
        book = self.book
        if o.side > 0:
            marketable = book.ba is not None and o.tick >= book.ba
        else:
            marketable = book.bb is not None and o.tick <= book.bb
        if marketable:
            if o.post_only:
                return "rejected", "post_only_would_cross"
            opp = book.asks if o.side > 0 else book.bids
            for tk in sorted(opp, reverse=o.side < 0):
                if (o.side > 0 and tk > o.tick) or (o.side < 0 and tk < o.tick):
                    break
                key = (-o.side, tk)
                avail = opp[tk] - self.taken.get(key, 0.0)
                if avail <= EPS:
                    continue
                f = self._fill(o, tk, avail if avail < o.x_rem else o.x_rem, False, fills)
                self.taken[key] = self.taken.get(key, 0.0) + f
                if o.x_rem <= EPS:
                    return "filled", ""
            if o.tif == IOC:
                return "cancelled", "ioc_remainder"
            o.x_ahead = o.x_credit = 0.0
        else:
            if o.tif == IOC:
                return "cancelled", "ioc_no_liquidity_at_price"
            o.x_ahead = self.queue.join(book.qty(o.side, o.tick))
            o.x_credit = 0.0
        o.x_live = True
        self.resting.append(o)
        return "live", ""

    def cancel(self, o: Order) -> bool:
        """True if the order was still resting (cancel succeeds); False if it was already gone."""
        if o.x_live and o.x_rem > EPS:
            o.x_live = False
            self.resting.remove(o)
            return True
        return False
