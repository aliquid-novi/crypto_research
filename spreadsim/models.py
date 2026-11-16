"""Pluggable models: where we sit in the queue, and how long messages take.

These are the two places a backtest is most likely to lie to you, so both are small classes you are
meant to subclass and swap. Nothing else in the engine needs to change when you do.
"""
from __future__ import annotations

import math
import random


# ---------------------------------------------------------------------------------- queue models
class QueueModel:
    """Tracks the quantity ahead of one of our resting orders at its price level.

    Level-2 data shows how much rests at a price, not who is where in the line. When the level
    shrinks without a trade, someone cancelled, and the data cannot say whether they were ahead of
    us or behind. `on_cancel` is that assumption."""

    def join(self, level_qty: float) -> float:
        """Quantity ahead when we join a level currently showing level_qty."""
        return level_qty

    def on_cancel(self, ahead: float, behind: float, cancelled: float) -> float:
        raise NotImplementedError


class RiskAverseQueue(QueueModel):
    """Cancels come from behind us whenever possible: we only advance through trades. Pessimistic."""

    def on_cancel(self, ahead, behind, cancelled):
        spill = cancelled - behind
        return ahead - spill if spill > 0 else ahead


class OptimisticQueue(QueueModel):
    """Cancels come from ahead of us first. The other bound."""

    def on_cancel(self, ahead, behind, cancelled):
        return ahead - cancelled if cancelled < ahead else 0.0


class ProbQueue(QueueModel):
    """A cancel is ahead of us with probability f(ahead) / (f(ahead) + f(behind)), f(x) = x**power.

    power=1 says every resting unit is equally likely to cancel. Higher powers push cancels toward
    whichever side of us holds more size. Same family as hftbacktest's probability queue models."""

    def __init__(self, power: float = 1.0):
        self.power = power

    def on_cancel(self, ahead, behind, cancelled):
        fa, fb = ahead ** self.power, behind ** self.power
        p_ahead = fa / (fa + fb) if fa + fb > 0 else 0.0
        from_behind = cancelled * (1.0 - p_ahead)
        spill = from_behind - behind if from_behind > behind else 0.0
        new = ahead - cancelled * p_ahead - spill
        return new if new > 0 else 0.0


# -------------------------------------------------------------------------------- latency models
def _pick(v, inst):
    return v.get(inst, v.get("*", 0.0)) if isinstance(v, dict) else v


class Latency:
    """Three delays, each a function of the simulator state and the instrument name:
        feed      exchange publishes a market event -> our strategy sees it
        entry     we send an order or cancel        -> the exchange acts on it
        response  the exchange acts (ack, fill)     -> we hear about it
    """

    def feed(self, sim, inst: str) -> float:
        return 0.0

    def entry(self, sim, inst: str) -> float:
        return 0.0

    def response(self, sim, inst: str) -> float:
        return 0.0


class ConstantLatency(Latency):
    """Fixed delays in seconds. Each argument is a number or a dict {instrument: seconds, "*": default}."""

    def __init__(self, feed=0.0, entry=0.0, response=0.0):
        self._feed, self._entry, self._response = feed, entry, response

    def feed(self, sim, inst):
        return _pick(self._feed, inst)

    def entry(self, sim, inst):
        return _pick(self._entry, inst)

    def response(self, sim, inst):
        return _pick(self._response, inst)


class LoadLatency(ConstantLatency):
    """Base delay x lognormal jitter, plus a term that grows with market message rate, plus rare spikes.

        delay = base * exp(sigma * z) + load * msgs_per_second(inst)      [x spike_mult with prob spike_p]

    `load` is in seconds per (message/second): 2e-6 adds 2 ms when the venue is printing 1,000 msg/s.
    This is the shape to test "does latency get worse exactly when it matters?"."""

    def __init__(self, feed=0.0, entry=0.0, response=0.0, sigma=0.3, load=0.0, spike_p=0.0, spike_mult=10.0, seed=0):
        super().__init__(feed, entry, response)
        self.sigma, self.load, self.spike_p, self.spike_mult = sigma, load, spike_p, spike_mult
        self.rng = random.Random(seed)

    def _draw(self, base, sim, inst):
        d = base * math.exp(self.sigma * self.rng.gauss(0.0, 1.0)) if self.sigma else base
        if self.load:
            d += self.load * sim.msg_rate(inst)
        if self.spike_p and self.rng.random() < self.spike_p:
            d *= self.spike_mult
        return d

    def feed(self, sim, inst):
        return self._draw(_pick(self._feed, inst), sim, inst)

    def entry(self, sim, inst):
        return self._draw(_pick(self._entry, inst), sim, inst)

    def response(self, sim, inst):
        return self._draw(_pick(self._response, inst), sim, inst)
