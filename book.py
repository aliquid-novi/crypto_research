"""A price-level (L2) order book."""
from __future__ import annotations

import math

from .core import BUY, Instrument


def impact_price(levels: dict, descending: bool, tick: float, notional: float, contract_size: float = 1.0) -> float:
    """Average price paid/received to trade `notional` (quote ccy) walking the book from the best level.
    If the side is too thin, the average over everything available is returned. NaN if empty."""
    if not levels:
        return math.nan
    left, cost, units = notional, 0.0, 0.0
    for tk in sorted(levels, reverse=descending):
        px = tk * tick
        lvl_notional = levels[tk] * px * contract_size
        take = lvl_notional if lvl_notional < left else left
        cost += take
        units += take / px
        left -= take
        if left <= 0:
            break
    return cost / units if units > 0 else math.nan


class Book:
    __slots__ = ("inst", "tick", "bids", "asks", "bb", "ba")

    def __init__(self, inst: Instrument):
        self.inst, self.tick = inst, inst.tick
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self.bb: int | None = None      # best bid tick
        self.ba: int | None = None      # best ask tick

    # ---- mutation -------------------------------------------------------------------------
    def set(self, side: int, tick: int, qty: float) -> float:
        """Set the quantity at a level; returns the previous quantity."""
        if side > 0:
            d = self.bids
            prev = d.get(tick, 0.0)
            if qty > 0:
                d[tick] = qty
                if self.bb is None or tick > self.bb:
                    self.bb = tick
            elif prev:
                del d[tick]
                if tick == self.bb:
                    self.bb = max(d) if d else None
        else:
            d = self.asks
            prev = d.get(tick, 0.0)
            if qty > 0:
                d[tick] = qty
                if self.ba is None or tick < self.ba:
                    self.ba = tick
            elif prev:
                del d[tick]
                if tick == self.ba:
                    self.ba = min(d) if d else None
        return prev

    def qty(self, side: int, tick: int) -> float:
        return (self.bids if side > 0 else self.asks).get(tick, 0.0)

    # ---- views ----------------------------------------------------------------------------
    @property
    def best_bid(self) -> float:
        return self.bb * self.tick if self.bb is not None else math.nan

    @property
    def best_ask(self) -> float:
        return self.ba * self.tick if self.ba is not None else math.nan

    @property
    def bid_qty(self) -> float:
        return self.bids[self.bb] if self.bb is not None else 0.0

    @property
    def ask_qty(self) -> float:
        return self.asks[self.ba] if self.ba is not None else 0.0

    @property
    def mid(self) -> float:
        if self.bb is None or self.ba is None:
            return math.nan
        return 0.5 * (self.bb + self.ba) * self.tick

    @property
    def spread_ticks(self) -> float:
        if self.bb is None or self.ba is None:
            return math.nan
        return self.ba - self.bb

    @property
    def microprice(self) -> float:
        """Size-weighted mid: leans toward the side with less resting size."""
        if self.bb is None or self.ba is None:
            return math.nan
        bq, aq = self.bids[self.bb], self.asks[self.ba]
        return (self.bb * aq + self.ba * bq) / (aq + bq) * self.tick

    @property
    def imbalance(self) -> float:
        """(bid size - ask size) / (bid size + ask size) at the touch, in [-1, 1]."""
        if self.bb is None or self.ba is None:
            return math.nan
        bq, aq = self.bids[self.bb], self.asks[self.ba]
        return (bq - aq) / (bq + aq)

    def levels(self, side: int, n: int = 5) -> list[tuple[float, float]]:
        d = self.bids if side > 0 else self.asks
        return [(tk * self.tick, d[tk]) for tk in sorted(d, reverse=side > 0)[:n]]

    def depth(self, side: int, n_ticks: int = 5) -> float:
        """Total resting size within n_ticks of the touch on one side."""
        d, best = (self.bids, self.bb) if side > 0 else (self.asks, self.ba)
        if best is None:
            return 0.0
        return sum(d.get(best - side * k, 0.0) for k in range(n_ticks + 1))

    def impact(self, side: int, notional: float) -> float:
        """side=BUY: average price received selling `notional` into the bids (impact bid);
        side=SELL: average price paid buying `notional` from the asks (impact ask)."""
        if side == BUY:
            return impact_price(self.bids, True, self.tick, notional, self.inst.contract_size)
        return impact_price(self.asks, False, self.tick, notional, self.inst.contract_size)
