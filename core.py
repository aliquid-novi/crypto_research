"""Shared types and constants.

Conventions used everywhere:
  * time is float seconds (any origin; relative-to-session-start keeps precision),
  * side is +1 (buy / bid side) or -1 (sell / ask side),
  * prices inside the engine are integer ticks; the public API takes and returns floats,
  * a market event is a plain tuple ``(ts, kind, inst_idx, a, b, c)``:
        DEPTH   a=side      b=tick   c=qty     level (side, tick) now holds qty (0 = gone)
        TRADE   a=aggressor b=tick   c=qty     a trade printed; a=+1 means a buyer lifted the ask
        INDEX   a=price     -        -         index/oracle price published for a perp
        FUNDING a=rate      -        -         funding settles now at this rate (longs pay if > 0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

BUY, SELL = 1, -1
DEPTH, TRADE, INDEX, FUNDING = 0, 1, 2, 3
GTC, IOC = "GTC", "IOC"
EPS = 1e-9


@dataclass
class Instrument:
    name: str
    tick: float
    lot: float = 1.0
    contract_size: float = 1.0
    maker_fee: float = 0.0          # fraction of notional; negative = rebate
    taker_fee: float = 0.0
    kind: str = "spot"              # "spot" or "perp" (label only; funding is configured separately)
    fx: float = 1.0                 # base-currency units per 1 quote-currency unit (constant case)
    fx_inst: str | None = None      # or: take the conversion from this instrument's mid
    fx_invert: bool = False         # True when fx_inst is quoted as quote-ccy per base-ccy (e.g. KRW per USDT)

    def to_tick(self, price: float, side: int = 0) -> int:
        """Float price -> integer tick. With a side, never rounds to a more aggressive price."""
        x = price / self.tick
        if side > 0:
            return math.floor(x + 1e-7)
        if side < 0:
            return math.ceil(x - 1e-7)
        return int(round(x))

    def px(self, tick: int) -> float:
        return round(tick * self.tick, 10)

    def round_qty(self, qty: float) -> float:
        return math.floor(qty / self.lot + 1e-7) * self.lot


class Order:
    """One order. Fields without ``x_`` are what the strategy knows (delayed by latency).
    ``x_`` fields are the exchange's truth; strategies should not read them to make decisions,
    but they are handy in research (e.g. ``x_ahead`` is the true-in-the-model queue position)."""

    __slots__ = (
        "id", "inst", "i", "side", "price", "tick", "qty", "filled", "tif", "post_only", "status",
        "tag", "parent", "meta", "reason",
        "x_rem", "x_ahead", "x_credit", "x_live",
        "t_submit", "t_arrive", "t_ack", "t_cancel_req", "t_cancel_arrive", "t_done",
        "mid_decision", "mid_arrive", "ahead0", "level0",
    )

    def __init__(self, id, inst, i, side, price, tick, qty, tif, post_only, tag, parent, meta):
        self.id, self.inst, self.i, self.side = id, inst, i, side
        self.price, self.tick, self.qty = price, tick, qty
        self.filled = 0.0
        self.tif, self.post_only = tif, post_only
        self.status = "sent"        # sent -> live -> (cancelling) -> filled | cancelled | rejected
        self.tag, self.parent, self.meta, self.reason = tag, parent, meta, ""
        self.x_rem, self.x_ahead, self.x_credit, self.x_live = qty, 0.0, 0.0, False
        nan = math.nan
        self.t_submit = self.t_arrive = self.t_ack = nan
        self.t_cancel_req = self.t_cancel_arrive = self.t_done = nan
        self.mid_decision = self.mid_arrive = self.ahead0 = self.level0 = nan

    @property
    def rem(self) -> float:
        return self.qty - self.filled

    @property
    def is_open(self) -> bool:
        return self.status in ("sent", "live", "cancelling")

    def __repr__(self):
        s = "B" if self.side > 0 else "S"
        return f"<Order {self.id} {self.inst} {s} {self.qty:g}@{self.price:g} {self.status} filled={self.filled:g}>"


class Fill:
    __slots__ = ("id", "order", "price", "qty", "maker", "t")

    def __init__(self, id, order, price, qty, maker, t):
        self.id, self.order, self.price, self.qty, self.maker, self.t = id, order, price, qty, maker, t

    def __repr__(self):
        return f"<Fill {self.id} {self.order.inst} {'B' if self.order.side > 0 else 'S'} {self.qty:g}@{self.price:g} {'maker' if self.maker else 'taker'}>"
