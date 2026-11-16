"""The event loop.

Two clocks matter. The exchange sees a market event at its timestamp; our strategy sees it `feed`
seconds later, on a separate local copy of the book. An order we send reaches the exchange `entry`
seconds later, and we hear back `response` seconds after that. Everything a strategy can read
(`sim.book`, `sim.position`, order status) is the delayed local view.

    market event --> exchange book --+--> fills on our resting orders --(response)--> strategy
                                     +--(feed)--> local book --> strategy --(entry)--> exchange
"""
from __future__ import annotations

import heapq
import math
from typing import Callable, Iterable

from .book import Book
from .core import BUY, DEPTH, EPS, FUNDING, GTC, INDEX, IOC, SELL, TRADE, Fill, Instrument, Order
from .funding import FundingCalc, premium_index
from .models import Latency, ProbQueue, QueueModel
from .venue import Venue

_LOCAL, _ARRIVE, _CANCEL, _RESP, _TIMER, _FSAMPLE, _SAMPLE, _LFUND = range(8)


class Strategy:
    """Subclass and override what you need. `sim` is the Sim; see its public methods."""

    def on_start(self, sim): ...
    def on_book(self, sim, inst: str): ...                  # our (delayed) book for inst changed
    def on_trade(self, sim, inst: str, side: int, price: float, qty: float): ...
    def on_fill(self, sim, fill: Fill): ...                  # one of our orders traded
    def on_order(self, sim, order: Order, event: str): ...   # "ack" | "cancelled" | "rejected" | "cancel_reject"
    def on_funding(self, sim, inst: str, rate: float, payment: float): ...
    def on_timer(self, sim, tag): ...
    def on_end(self, sim): ...


class Sim:
    def __init__(self, instruments: Iterable[Instrument], events: Iterable[tuple], strategy: Strategy,
                 latency: Latency | None = None, queue: QueueModel | dict | None = None,
                 funding: dict[str, FundingCalc] | None = None, funding_from: str = "model",
                 index_fn: dict[str, Callable] | None = None, sample_dt: float = 1.0,
                 fill_on_clear: bool = True, truth: dict | None = None):
        """
        instruments   the Instrument specs; event tuples refer to them by position
        events        time-sorted market events (see core.py for the tuple layout)
        latency       a models.Latency (default: zero latency)
        queue         a models.QueueModel, or {instrument name: QueueModel} (default: ProbQueue(1))
        funding       {perp name: FundingCalc}. Tracks the premium so strategies can read predicted funding.
        funding_from  "model": the FundingCalc also settles and pays;  "events": FUNDING events in the
                      stream pay, and the FundingCalc only tracks
        index_fn      {perp name: f(sim) -> index price} for streams with no INDEX events
        sample_dt     how often equity and positions are recorded
        fill_on_clear see venue.py, rule 4
        truth         optional latent series from the synthetic generator, passed through to the Result
        """
        assert funding_from in ("model", "events")
        self.instruments = list(instruments)
        self.names = [x.name for x in self.instruments]
        self.idx = {n: i for i, n in enumerate(self.names)}
        n = len(self.instruments)
        self.events, self.strategy = events, strategy
        self.latency = latency or Latency()
        if isinstance(queue, dict):
            qs = [queue.get(nm, ProbQueue(1.0)) for nm in self.names]
        else:
            qs = [queue or ProbQueue(1.0)] * n
        self.venues = [Venue(x, qs[i], fill_on_clear) for i, x in enumerate(self.instruments)]
        self._xb = [v.book for v in self.venues]
        self._lb = [Book(x) for x in self.instruments]
        self._fcalc: list[FundingCalc | None] = [None] * n
        for nm, c in (funding or {}).items():
            self._fcalc[self.idx[nm]] = c
        self.funding_from = funding_from
        self._index_fn = [None] * n
        for nm, f in (index_fn or {}).items():
            self._index_fn[self.idx[nm]] = f
        self.sample_dt, self.truth = sample_dt, truth

        self.now = 0.0
        self.t0 = self.t_end = math.nan
        self._heap: list = []
        self._seq = 0
        self._oid = self._fid = 0
        self._pos = [0.0] * n           # true position at the exchange
        self._lpos = [0.0] * n          # position the strategy knows about
        self._cash = [0.0] * n          # quote currency, includes fees and funding
        self._fees = [0.0] * n
        self._fund = [0.0] * n
        self._index = [math.nan] * n
        self._lindex = [math.nan] * n
        self._lastmid = [math.nan] * n
        self._open: list[dict[int, Order]] = [dict() for _ in range(n)]
        self._all_orders: list[Order] = []
        self._last_resp = [-math.inf] * n
        self._last_local = [-math.inf] * n
        self._mr = [0.0] * n
        self._mr_t = [-math.inf] * n
        self._fxj = [self.idx[x.fx_inst] if x.fx_inst else -1 for x in self.instruments]
        self._wants_book = type(strategy).on_book is not Strategy.on_book
        self._wants_trade = type(strategy).on_trade is not Strategy.on_trade
        # recorder
        self._r_fills: list[list] = []
        self._r_bbo = [([], [], []) for _ in range(n)]
        self._r_trades = [([], [], [], []) for _ in range(n)]
        self._r_last_bbo: list = [None] * n
        self._r_equity: list[tuple] = []
        self._r_funding: list[tuple] = []
        self._r_logs: dict[str, tuple[list, list]] = {}
        self.cancel_rejects = 0

    # =============================================================== API for strategies
    def book(self, inst: str) -> Book:
        """Our delayed view of the book."""
        return self._lb[self.idx[inst]]

    def true_book(self, inst: str) -> Book:
        """The exchange's book right now. For research and analytics; a real strategy cannot see this."""
        return self._xb[self.idx[inst]]

    def spec(self, inst: str) -> Instrument:
        return self.instruments[self.idx[inst]]

    def position(self, inst: str) -> float:
        """Position implied by the fills we have been told about."""
        return self._lpos[self.idx[inst]]

    def true_position(self, inst: str) -> float:
        return self._pos[self.idx[inst]]

    def open_orders(self, inst: str | None = None) -> list[Order]:
        if inst is not None:
            return list(self._open[self.idx[inst]].values())
        return [o for d in self._open for o in d.values()]

    def submit(self, inst: str, side: int, price: float, qty: float, tif: str = GTC, post_only: bool = False,
               tag: str = "", parent=None, meta=None) -> Order | None:
        i = self.idx[inst]
        spec = self.instruments[i]
        qty = spec.round_qty(qty)
        if qty <= 0:
            return None
        tick = spec.to_tick(price, side)
        self._oid += 1
        o = Order(self._oid, inst, i, side, spec.px(tick), tick, qty, tif, post_only, tag, parent, meta)
        o.t_submit = self.now
        o.mid_decision = self._lb[i].mid
        o.t_arrive = self.now + max(0.0, self.latency.entry(self, inst))
        self._open[i][o.id] = o
        self._all_orders.append(o)
        self._push(o.t_arrive, _ARRIVE, o)
        return o

    def buy(self, inst, price, qty, **kw):
        return self.submit(inst, BUY, price, qty, **kw)

    def sell(self, inst, price, qty, **kw):
        return self.submit(inst, SELL, price, qty, **kw)

    def take(self, inst: str, side: int, qty: float, cross_ticks: int = 5, **kw) -> Order | None:
        """IOC that crosses the spread by up to `cross_ticks` beyond the far touch we can see."""
        b = self._lb[self.idx[inst]]
        ref = b.ba if side > 0 else b.bb
        if ref is None:
            return None
        return self.submit(inst, side, (ref + side * cross_ticks) * b.tick, qty, tif=IOC, **kw)

    def cancel(self, order: Order) -> bool:
        if order is None or order.status not in ("sent", "live"):
            return False
        order.status = "cancelling"
        order.t_cancel_req = self.now
        t = max(self.now + max(0.0, self.latency.entry(self, order.inst)), order.t_arrive)
        self._push(t, _CANCEL, order)
        return True

    def cancel_all(self, inst: str | None = None) -> int:
        return sum(self.cancel(o) for o in self.open_orders(inst))

    def timer(self, delay: float, tag=None) -> None:
        self._push(self.now + delay, _TIMER, tag)

    def funding(self, inst: str) -> FundingCalc | None:
        """The funding tracker for a perp: .predicted(), .avg(), .locked(), .time_to_settle(now), ..."""
        return self._fcalc[self.idx[inst]]

    def index(self, inst: str) -> float:
        return self._lindex[self.idx[inst]]

    def log(self, key: str, value: float) -> None:
        """Record a number against the current time (e.g. your fair value) for later analysis."""
        ts, vs = self._r_logs.setdefault(key, ([], []))
        ts.append(self.now)
        vs.append(value)

    def msg_rate(self, inst: str) -> float:
        """Recent market messages per second on this instrument (1-second exponential window)."""
        i = self.idx[inst]
        dt = self.now - self._mr_t[i]
        return self._mr[i] * math.exp(-dt) if dt < 50 else 0.0

    def equity(self) -> float:
        """Mark-to-mid value of everything, in base currency."""
        tot = 0.0
        for i, x in enumerate(self.instruments):
            m = self._xb[i].mid
            if m == m:
                self._lastmid[i] = m
        for i, x in enumerate(self.instruments):
            m = self._lastmid[i]
            v = self._cash[i] + (self._pos[i] * m * x.contract_size if self._pos[i] else 0.0)
            tot += v * self._fx(i)
        return tot

    # =============================================================== internals
    def _fx(self, i: int) -> float:
        j = self._fxj[i]
        if j < 0:
            return self.instruments[i].fx
        m = self._lastmid[j]
        if m != m:
            return 0.0
        return 1.0 / m if self.instruments[i].fx_invert else m

    def _push(self, t: float, code: int, payload) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, code, payload))

    def _apply_fills(self, fills: list, resp_t: float | None = None) -> None:
        for o, tick, qty, maker in fills:
            i = o.i
            x = self.instruments[i]
            px = x.px(tick)
            notional = qty * px * x.contract_size
            fee = notional * (x.maker_fee if maker else x.taker_fee)
            self._pos[i] += o.side * qty
            self._cash[i] -= o.side * notional + fee
            self._fees[i] += fee
            self._fid += 1
            b = self._xb[i]
            self._r_fills.append([self._fid, o.id, o.inst, o.side, px, qty, maker, fee, self.now, math.nan,
                                  b.mid, b.best_bid, b.best_ask, b.bid_qty, b.ask_qty, o.tag, o.parent])
            if o.x_rem <= EPS:
                o.t_done = self.now
            t = resp_t if resp_t is not None else self._resp_time(i)
            self._push(t, _RESP, ("fill", o, Fill(self._fid, o, px, qty, maker, self.now)))

    def _resp_time(self, i: int) -> float:
        t = self.now + max(0.0, self.latency.response(self, self.names[i]))
        if t < self._last_resp[i]:
            t = self._last_resp[i]
        self._last_resp[i] = t
        return t

    def _apply_funding(self, i: int, rate: float) -> None:
        x, c = self.instruments[i], self._fcalc[i]
        px = self._index[i] if (c is None or c.notional_price == "index") else math.nan
        if px != px:
            m = self._xb[i].mid
            px = m if m == m else self._lastmid[i]
        pay = -self._pos[i] * px * x.contract_size * rate
        self._cash[i] += pay
        self._fund[i] += pay
        self._r_funding.append((self.now, x.name, rate, self._pos[i], px, pay))
        self._push(self.now + max(0.0, self.latency.feed(self, x.name)), _LFUND, (x.name, rate, pay))

    def _market(self, ev: tuple) -> None:
        ts, kind, i, a, b, c = ev
        v = self.venues[i]
        if kind == DEPTH or kind == TRADE:
            dt = ts - self._mr_t[i]
            self._mr[i] = (self._mr[i] * math.exp(-dt) if dt < 50 else 0.0) + 1.0
            self._mr_t[i] = ts
            if kind == TRADE:
                rt, rs, rp, rq = self._r_trades[i]
                rt.append(ts)
                rs.append(a)
                rp.append(b)
                rq.append(c)
            if v.resting:
                fills: list = []
                if kind == DEPTH:
                    v.on_depth(a, b, c, fills)
                else:
                    v.on_trade(a, b, c, fills)
                if fills:
                    self._apply_fills(fills)
            elif kind == DEPTH:
                v.book.set(a, b, c)
                if v.taken:
                    v.taken.pop((a, b), None)
            if kind == DEPTH:
                bk = v.book
                key = (bk.bb, bk.ba)
                if key != self._r_last_bbo[i]:
                    self._r_last_bbo[i] = key
                    rt, rb, ra = self._r_bbo[i]
                    rt.append(ts)
                    rb.append(bk.bb if bk.bb is not None else -1)
                    ra.append(bk.ba if bk.ba is not None else -1)
        elif kind == INDEX:
            self._index[i] = a
        elif kind == FUNDING:
            if self.funding_from == "events":
                self._apply_funding(i, a)
            return
        t = ts + max(0.0, self.latency.feed(self, self.names[i]))
        if t < self._last_local[i]:
            t = self._last_local[i]
        self._last_local[i] = t
        self._push(t, _LOCAL, ev)

    def _dispatch(self, item: tuple) -> None:
        _, _, code, p = item
        if code == _LOCAL:
            _, kind, i, a, b, c = p
            if kind == DEPTH:
                self._lb[i].set(a, b, c)
                if self._wants_book:
                    self.strategy.on_book(self, self.names[i])
            elif kind == TRADE:
                if self._wants_trade:
                    self.strategy.on_trade(self, self.names[i], a, b * self.instruments[i].tick, c)
            elif kind == INDEX:
                self._lindex[i] = a
        elif code == _TIMER:
            self.strategy.on_timer(self, p)
        elif code == _RESP:
            kind, o, data = p
            if kind == "fill":
                o.filled += data.qty
                self._lpos[o.i] += o.side * data.qty
                if o.filled >= o.qty - EPS:
                    o.status = "filled"
                    self._open[o.i].pop(o.id, None)
                self._r_fills[data.id - 1][9] = self.now
                self.strategy.on_fill(self, data)
            elif kind == "ack":
                o.t_ack = self.now
                if o.status == "sent":
                    o.status = "live"
                self.strategy.on_order(self, o, "ack")
            elif kind == "cancelled":
                if o.status != "filled":
                    o.status = "cancelled"
                o.reason = data or o.reason
                if o.t_done != o.t_done:
                    o.t_done = self.now
                self._open[o.i].pop(o.id, None)
                self.strategy.on_order(self, o, "cancelled")
            elif kind == "rejected":
                o.status, o.reason, o.t_done = "rejected", data, self.now
                self._open[o.i].pop(o.id, None)
                self.strategy.on_order(self, o, "rejected")
            elif kind == "cancel_reject":
                if o.status == "cancelling":
                    o.status = "filled" if o.filled >= o.qty - EPS else "live"
                self.strategy.on_order(self, o, "cancel_reject")
        elif code == _ARRIVE:
            o = p
            v = self.venues[o.i]
            o.mid_arrive = v.book.mid
            fills: list = []
            status, reason = v.arrive(o, fills)
            rt = self._resp_time(o.i)
            if fills:
                self._apply_fills(fills, rt)
            if status == "live":
                o.ahead0, o.level0 = o.x_ahead, v.book.qty(o.side, o.tick)
                self._push(rt, _RESP, ("ack", o, None))
            elif status == "rejected":
                self._push(rt, _RESP, ("rejected", o, reason))
            elif status == "cancelled":
                self._push(rt, _RESP, ("cancelled", o, reason))
        elif code == _CANCEL:
            o = p
            o.t_cancel_arrive = self.now
            ok = self.venues[o.i].cancel(o)
            if not ok:
                self.cancel_rejects += 1
                o.meta = dict(o.meta or {}, cancel_reject=True)
            self._push(self._resp_time(o.i), _RESP, ("cancelled" if ok else "cancel_reject", o, "cancel" if ok else None))
        elif code == _SAMPLE:
            self._sample()
            self._push(self.now + self.sample_dt, _SAMPLE, None)
        elif code == _FSAMPLE:
            i = p
            c = self._fcalc[i]
            idx = self._index[i]
            if idx != idx and self._index_fn[i] is not None:
                idx = self._index_fn[i](self)
                self._index[i] = self._lindex[i] = idx
            c.add(premium_index(self._xb[i], idx, c.impact_notional))
            if c.k >= c.n_total:
                rate = c.settle()
                if self.funding_from == "model":
                    self._apply_funding(i, rate)
            self._push(c.next_sample_at, _FSAMPLE, i)
        elif code == _LFUND:
            self.strategy.on_funding(self, *p)

    def _sample(self) -> None:
        eq = self.equity()
        self._r_equity.append((self.now, eq, tuple(self._pos), tuple(self._lastmid), tuple(self._cash),
                               tuple(self._fees), tuple(self._fund)))

    # =============================================================== run
    def run(self, until: float | None = None):
        it = iter(self.events)
        nxt = next(it, None)
        if nxt is None:
            raise ValueError("no market events")
        self.now = self.t0 = nxt[0]
        for i, c in enumerate(self._fcalc):
            if c is not None:
                c.sync(self.t0)
                self._push(c.next_sample_at, _FSAMPLE, i)
        self._push(self.t0 + self.sample_dt, _SAMPLE, None)
        self.strategy.on_start(self)
        heap, pop = self._heap, heapq.heappop
        last = self.t0
        while nxt is not None:
            ts = nxt[0]
            if until is not None and ts > until:
                break
            while heap and heap[0][0] < ts:
                item = pop(heap)
                self.now = item[0]
                self._dispatch(item)
            self.now = last = ts
            self._market(nxt)
            nxt = next(it, None)
        while heap and heap[0][0] <= last:
            item = pop(heap)
            self.now = item[0]
            self._dispatch(item)
        self.now = self.t_end = last
        self._sample()
        self.strategy.on_end(self)
        from .analytics import Result
        return Result(self)
