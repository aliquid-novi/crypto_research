"""Rules that move a quoter's clips in or out while it runs.

A controller is any object with two methods:

    shift(sim, strategy) -> int      extra ticks to push every clip away from fair value, called each cycle
    on_fill(sim, strategy, fill)     told about every fill of the strategy

It may only use what a live strategy could know: `sim.book`, `sim.now`, the fills it is told about,
and the strategy's own state (`strategy.vol` is the spread's recent volatility, `strategy.fair`, ...).
Write a new hypothesis as a new controller, then compare it with `lab.Experiment` on the same seeds.
"""
from __future__ import annotations

import math
from collections import deque


class Controller:
    def shift(self, sim, strat) -> int:
        return 0

    def on_fill(self, sim, strat, fill) -> None:
        pass


class VolState:
    """Splits time into low (0) and high (1) volatility at the running median of the spread's volatility."""

    def __init__(self, lookback: float = 900.0, every: float = 1.0):
        self.buf: deque = deque(maxlen=int(lookback / every))
        self.every, self._next, self._n, self.median = every, -math.inf, 0, math.nan

    def state(self, now: float, vol: float) -> int:
        if vol != vol:
            return 0
        if now >= self._next:
            self._next = now + self.every
            self.buf.append(vol)
            self._n += 1
            if self._n % 10 == 1:
                b = sorted(self.buf)
                self.median = b[len(b) // 2]
        return int(vol > self.median) if self.median == self.median else 0


class MarkoutFeedback(Controller):
    """The rule under test: watch the markouts of the first clips; if they are consistently negative in
    the current volatility state, push the ladder out a tick; if they turn positive again, pull it back.

    horizon        seconds after a fill at which its markout is read
    spread         True: markout of the hedged fill (quote leg's move minus the hedge leg's move since the
                   fill). False: the quote leg against its own mid only
    clips          which ladder levels count as "the first clips" (0 = nearest fair value)
    window         markouts remembered per volatility state
    min_fills      markouts needed before the rule acts
    hurdle_ticks   what a fill has to earn before it counts as good: the per-unit cost of hedging and
                   fees, in ticks. Markouts are judged net of it (0 = judge the raw sign)
    trigger_ticks  act when the average net markout is beyond +/- this many ticks
    step, max_shift  ticks moved per action, and the cap
    by_vol         keep a separate shift for low and high volatility (False: one shift for all states)
    relax          allow the ladder to come back in when markouts turn positive
    """

    def __init__(self, horizon: float = 5.0, clips=(0, 1), window: int = 20, min_fills: int = 8,
                 trigger_ticks: float = 0.0, step: int = 1, max_shift: int = 3, by_vol: bool = True,
                 relax: bool = True, spread: bool = True, hurdle_ticks: float = 0.0):
        self.h, self.clips, self.window, self.min_fills = horizon, set(clips), window, min_fills
        self.trigger, self.step, self.max_shift, self.by_vol, self.relax = trigger_ticks, step, max_shift, by_vol, relax
        self.spread, self.hurdle = spread, hurdle_ticks
        self.vs = VolState()
        self.shifts = {0: 0, 1: 0}
        self.buf = {0: deque(maxlen=window), 1: deque(maxlen=window)}
        self.pending: deque = deque()
        self.history: list[tuple] = []      # (time, vol state, average markout, new shift)
        self._state = 0

    def on_fill(self, sim, strat, fill):
        o = fill.order
        if o.tag == "quote" and isinstance(o.meta, dict) and o.meta.get("k") in self.clips:
            self.pending.append((sim.now + self.h, o.side, fill.price, self._state, sim.book(strat.hedge_leg).mid))

    def shift(self, sim, strat):
        self._state = s = self.vs.state(sim.now, strat.vol) if self.by_vol else 0
        b = sim.book(strat.quote)
        mid, hmid = b.mid, sim.book(strat.hedge_leg).mid
        while self.pending and self.pending[0][0] <= sim.now and mid == mid and hmid == hmid:
            _, side, px, st, h0 = self.pending.popleft()
            buf = self.buf[st]
            m = side * (mid - px)
            if self.spread and h0 == h0:
                m -= side * px * (hmid / h0 - 1.0)
            buf.append(m / b.tick)
            if len(buf) >= self.min_fills:
                avg = sum(buf) / len(buf) - self.hurdle
                if avg < -self.trigger and self.shifts[st] < self.max_shift:
                    self.shifts[st] += self.step
                    buf.clear()
                    self.history.append((sim.now, st, avg, self.shifts[st]))
                elif self.relax and avg > self.trigger and self.shifts[st] > 0:
                    self.shifts[st] -= self.step
                    buf.clear()
                    self.history.append((sim.now, st, avg, self.shifts[st]))
        return self.shifts[s]


class VolShift(Controller):
    """A rule with no feedback at all: push out `high` ticks whenever volatility is above its running
    median, `low` ticks otherwise. The comparison that tells you whether markouts add anything to
    simply knowing the volatility state."""

    def __init__(self, high: int = 1, low: int = 0):
        self.high, self.low, self.vs = high, low, VolState()

    def shift(self, sim, strat):
        return self.high if self.vs.state(sim.now, strat.vol) else self.low
