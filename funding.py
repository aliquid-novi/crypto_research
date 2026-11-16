"""Perpetual funding: the venue formula, tracked the way the venue computes it.

    premium sample  p = [max(0, impact_bid - index) - max(0, index - impact_ask)] / index
    average         P = weighted mean of the interval's samples (equal weights, or linearly increasing)
    rate            F = [P + clamp(I - P, -c, +c)] * (interval / 8h) * multiplier      (then capped)
    payment         -position * price * contract_size * F        (longs pay when F > 0)

`weights="linear"` is the Binance rule for intervals longer than an hour (the newest sample weighs most);
`weights="equal"` is Hyperliquid's hourly rule. Check your venue's documentation for the variant you trade.
"""
from __future__ import annotations

import math

from .core import BUY, SELL


def premium_index(book, index: float, impact_notional: float) -> float:
    ib, ia = book.impact(BUY, impact_notional), book.impact(SELL, impact_notional)
    if not (index > 0) or ib != ib or ia != ia:
        return math.nan
    return (max(0.0, ib - index) - max(0.0, index - ia)) / index


class FundingCalc:
    def __init__(self, interval: float = 3600.0, sample: float = 5.0, weights: str = "equal",
                 interest_8h: float = 1e-4, clamp: float = 5e-4, cap: float | None = None,
                 multiplier: float = 1.0, start: float = 0.0, impact_notional: float = 10_000.0,
                 notional_price: str = "index"):
        assert weights in ("equal", "linear")
        self.interval, self.sample, self.weights = interval, sample, weights
        self.interest_8h, self.clamp, self.cap, self.multiplier = interest_8h, clamp, cap, multiplier
        self.impact_notional, self.notional_price = impact_notional, notional_price
        self.n_total = int(round(interval / sample))
        self.last_rate = math.nan
        self.reset(start)

    def copy(self) -> "FundingCalc":
        return FundingCalc(self.interval, self.sample, self.weights, self.interest_8h, self.clamp, self.cap,
                           self.multiplier, self.start, self.impact_notional, self.notional_price)

    def reset(self, start: float) -> None:
        self.start = start
        self.k = 0                  # index of the last sample slot that has passed
        self._wsum = self._wp = 0.0
        self.last_premium = math.nan

    def sync(self, now: float) -> None:
        """Skip sample slots that passed before `now` (use when a run starts mid-interval)."""
        while now >= self.settle_at:
            self.reset(self.start + self.interval)
        self.k = max(self.k, int((now - self.start) / self.sample + 1e-9))

    # ---- schedule ---------------------------------------------------------------------------
    @property
    def settle_at(self) -> float:
        return self.start + self.interval

    @property
    def next_sample_at(self) -> float:
        return self.start + (self.k + 1) * self.sample

    def time_to_settle(self, now: float) -> float:
        return self.settle_at - now

    # ---- accumulation -----------------------------------------------------------------------
    def add(self, premium: float) -> None:
        self.k += 1
        if premium == premium:
            w = float(self.k) if self.weights == "linear" else 1.0
            self._wsum += w
            self._wp += w * premium
            self.last_premium = premium

    def avg(self) -> float:
        """Weighted average premium over the samples seen so far this interval."""
        return self._wp / self._wsum if self._wsum > 0 else math.nan

    def rate(self, avg_premium: float) -> float:
        """The funding rate that an interval-average premium maps to."""
        p = avg_premium if avg_premium == avg_premium else 0.0
        d = self.interest_8h - p
        d = self.clamp if d > self.clamp else (-self.clamp if d < -self.clamp else d)
        f = (p + d) * self.multiplier * (self.interval / 28800.0)
        if self.cap is not None:
            f = max(-self.cap, min(self.cap, f))
        return f

    def predicted(self) -> float:
        """The rate if the average stayed where it is now (the naive 'so far' prediction)."""
        return self.rate(self.avg())

    def baseline(self) -> float:
        """The rate paid whenever the average premium ends inside the flat zone."""
        return self.rate(self.interest_8h)

    def locked(self) -> tuple[float, float]:
        """(contribution of past samples to the FINAL average, share of final weight still to come).
        final_average = locked + sum over remaining samples of weight_i * premium_i."""
        n = self.n_total
        if self.weights == "linear":
            total, seen = n * (n + 1) / 2.0, self.k * (self.k + 1) / 2.0
        else:
            total, seen = float(n), float(self.k)
        return self._wp / total, max(0.0, 1.0 - seen / total)

    def settle(self) -> float:
        self.last_rate = self.predicted()
        self.reset(self.start + self.interval)
        return self.last_rate


def expected_rate(calc: FundingCalc, mu: float, sigma: float) -> float:
    """E[rate] when the final average premium is Normal(mu, sigma): the kinked payoff priced exactly.

    The payoff is flat at the interest rate between L = I - clamp and U = I + clamp and linear outside,
    i.e. long a call on the average premium struck at U and short a put struck at L."""
    i, c = calc.interest_8h, calc.clamp
    lo, hi = i - c, i + c
    if sigma <= 1e-15:
        return calc.rate(mu)
    phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    zu, zl = (mu - hi) / sigma, (lo - mu) / sigma
    f8 = i + (sigma * phi(zu) + (mu - hi) * Phi(zu)) - (sigma * phi(zl) + (lo - mu) * Phi(zl))
    f = f8 * calc.multiplier * (calc.interval / 28800.0)
    if calc.cap is not None:
        f = max(-calc.cap, min(calc.cap, f))
    return f
