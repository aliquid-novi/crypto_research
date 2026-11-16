"""A synthetic multi-leg market where you know the truth.

What is simulated, top down:

  factors      latent log random walks (e.g. "U" = the underlying's USD value, "FX" = KRW per USDT),
               with a calm/stressed regime and occasional jumps
  basis        each leg may carry a mean-reverting premium over its factors (the tradable spread)
  reference    each leg's latent fair price = ratio x factors x (1 + basis + funding dip), reached with
               an optional lag, so one venue can lead another
  order flow   limit orders arrive around the reference, resting size cancels, market orders arrive
               (tilted by book imbalance). Market orders push that leg's reference in their direction
               and the push decays, so flow pressure on one leg opens a gap to the other leg that
               closes again. When the reference moves through resting quotes, those quotes are stale:
               each is either hit by a fast trader or pulled by its owner, after random delays. That
               race is where adverse selection and latency sensitivity come from.
  funding      perp legs run the venue formula on their own simulated book and settle on schedule.
               Optionally the perp dips before a settlement and rebounds after it.

The output is an ordinary event stream, the same format real data is replayed in, plus `truth`
(the latent series), which strategies never see and analytics can grade against.

Everything here is a toy with invented parameters. Its job is to give you a world with known
answers to test estimators and execution logic on, not to resemble any particular market.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field, replace

import numpy as np

from .book import impact_price
from .core import BUY, DEPTH, FUNDING, INDEX, SELL, TRADE, Instrument
from .funding import FundingCalc

LN2 = math.log(2.0)


@dataclass
class Factor:
    name: str
    start: float = 100.0
    vol: float = 1.0e-4             # return standard deviation per sqrt(second)
    jump_rate: float = 0.0          # jumps per second
    jump_sd: float = 0.0            # jump size (log return standard deviation)


@dataclass
class Regime:
    """Two-state Markov regime (0 = calm, 1 = stressed). Multipliers apply in the stressed state."""
    enter: float = 1 / 1200.0       # hazard per second, calm -> stressed
    exit: float = 1 / 300.0         # hazard per second, stressed -> calm
    vol_mult: float = 2.5
    depth_mult: float = 0.5         # resting depth target
    market_mult: float = 2.0        # market-order arrival rate
    basis_halflife_mult: float = 1.0


@dataclass
class Basis:
    mean: float = 0.0               # long-run premium (fraction, e.g. 5e-4 = 5 bps)
    sd: float = 5e-4                # stationary standard deviation
    half_life: float = 600.0        # seconds


@dataclass
class Flow:
    limit_rate: float = 30.0        # limit-order arrivals per second (both sides together)
    market_rate: float = 2.0        # market-order arrivals per second
    target_depth: float = 400.0     # resting units per side the book settles around
    size_mean: float = 5.0          # mean limit-order size, in lots
    size_sigma: float = 0.8         # lognormal shape of order sizes
    market_size_mean: float = 6.0   # mean market-order size, in lots
    depth_ticks: float = 3.0        # mean distance of a new limit order behind the touch, in ticks
    half_spread: float = 0.6        # closest a new quote sits to the reference, in ticks
    snipe_prob: float = 0.6         # chance a stale level is hit (otherwise its owner pulls it)
    snipe_delay: float = 0.03       # mean seconds before a fast trader hits a stale level
    pull_delay: float = 0.08        # mean seconds before the owner pulls a stale level
    imbalance_sens: float = 0.15    # tilt of market-order side toward the heavier side of the touch
    impact_ticks: float = 0.2       # reference move, in ticks, caused by one average-sized market order
    impact_half_life: float = 20.0  # seconds for that flow-driven dislocation to decay by half
    max_ticks: int = 40             # levels further than this from the reference are dropped


@dataclass
class Leg:
    inst: Instrument
    factors: tuple[str, ...] = ("U",)
    ratio: float = 1.0
    lag: float = 0.0                # seconds; time constant with which this leg follows its target
    basis: Basis | None = None
    flow: Flow = field(default_factory=Flow)
    funding: FundingCalc | None = None
    funding_passthrough: float = 0.0    # share of predicted funding the price gives up before settlement
    funding_anticipation: float = 600.0  # seconds before settlement over which that dip builds


@dataclass
class Market:
    legs: list[Leg]
    factors: list[Factor]
    regime: Regime | None = None
    dt: float = 0.05                # grid for the latent prices, seconds
    index_dt: float = 1.0           # how often a perp's index price is published


class SynthMarket:
    """Result of `simulate`: instruments, the event list, and the latent truth."""

    def __init__(self, market: Market, events: list, truth: dict, seed: int):
        self.market, self.events, self.truth, self.seed = market, events, truth, seed
        self.instruments = [l.inst for l in market.legs]
        self.names = [x.name for x in self.instruments]
        self.duration = float(truth["t"][-1])

    def funding(self) -> dict[str, FundingCalc]:
        return {l.inst.name: l.funding.copy() for l in self.market.legs if l.funding is not None}

    def sim(self, strategy, **kw):
        """A Sim wired to this market: funding paid from the stream's FUNDING events, truth attached."""
        from .sim import Sim
        kw.setdefault("funding", self.funding())
        kw.setdefault("funding_from", "events")
        kw.setdefault("truth", self.truth)
        return Sim(self.instruments, self.events, strategy, **kw)

    def run(self, strategy, **kw):
        return self.sim(strategy, **kw).run()


class _Leg:
    """Mutable book state for one leg during generation."""

    def __init__(self, i: int, leg: Leg, rng: random.Random):
        self.i, self.leg, self.inst, self.F, self.rng = i, leg, leg.inst, leg.flow, rng
        self.tick, self.lot = leg.inst.tick, leg.inst.lot
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self.bb: int | None = None
        self.ba: int | None = None
        self.dsum = {BUY: 0.0, SELL: 0.0}
        self.ref = math.nan
        self.core = math.nan            # the reference before flow pressure (follows its target with the leg's lag)
        self.imp = 0.0                  # flow-driven dislocation of the reference, in price units
        self.flag: set = set()
        self.mu_l = math.log(self.F.size_mean) - 0.5 * self.F.size_sigma ** 2
        self.mu_m = math.log(self.F.market_size_mean) - 0.5 * self.F.size_sigma ** 2
        self.set_regime(1.0, 1.0)

    def set_regime(self, depth_mult: float, market_mult: float) -> None:
        F = self.F
        self.lam_l = F.limit_rate
        self.lam_m = F.market_rate * market_mult
        self.target = F.target_depth * depth_mult
        # cancel attempts are accepted with probability depth/target, which pins average depth at target
        self.lam_c = max(F.limit_rate - self.lam_m * F.market_size_mean / F.size_mean, 0.2 * F.limit_rate)
        self.rate = self.lam_l + self.lam_m + self.lam_c

    def size(self, mu: float) -> float:
        return self.lot * max(1.0, round(self.rng.lognormvariate(mu, self.F.size_sigma)))


class _Gen:
    def __init__(self, market: Market, duration: float, seed: int):
        self.m, self.T = market, duration
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.out: list[tuple] = []
        self.legs = [_Leg(i, l, self.rng) for i, l in enumerate(market.legs)]
        self.calc = [l.funding.copy() if l.funding is not None else None for l in market.legs]
        self._latent()

    # ---------------------------------------------------------------- latent paths (vectorised)
    def _latent(self) -> None:
        m, dt = self.m, self.m.dt
        n = int(round(self.T / dt)) + 1
        self.n, self.t = n, np.arange(n) * dt
        rg = self.np_rng
        reg = np.zeros(n, dtype=np.int8)
        if m.regime is not None:
            u, s = rg.random(n), 0
            pe, px = 1 - math.exp(-m.regime.enter * dt), 1 - math.exp(-m.regime.exit * dt)
            for k in range(1, n):
                if s == 0 and u[k] < pe:
                    s = 1
                elif s == 1 and u[k] < px:
                    s = 0
                reg[k] = s
        self.regime = reg
        vmult = np.where(reg == 1, m.regime.vol_mult, 1.0) if m.regime is not None else np.ones(n)
        self.fpath: dict[str, np.ndarray] = {}
        for f in m.factors:
            r = f.vol * math.sqrt(dt) * vmult * rg.standard_normal(n)
            if f.jump_rate > 0:
                r += (rg.random(n) < f.jump_rate * dt) * rg.normal(0.0, f.jump_sd, n)
            r[0] = 0.0
            self.fpath[f.name] = f.start * np.exp(np.cumsum(r))
        self.bpath: list[np.ndarray | None] = []
        for l in m.legs:
            if l.basis is None:
                self.bpath.append(None)
                continue
            b, z = l.basis, rg.standard_normal(n)
            hl = np.full(n, b.half_life)
            if m.regime is not None:
                hl = np.where(reg == 1, b.half_life * m.regime.basis_halflife_mult, b.half_life)
            a = np.exp(-dt * LN2 / hl)
            x = np.empty(n)
            x[0] = b.mean + b.sd * z[0]
            for k in range(1, n):
                x[k] = b.mean + (x[k - 1] - b.mean) * a[k] + b.sd * math.sqrt(1 - a[k] * a[k]) * z[k]
            self.bpath.append(x)
        self.base = []                  # ratio x factors, per leg (the leg's value before any premium)
        for l in m.legs:
            v = np.full(n, l.ratio)
            for nm in l.factors:
                v = v * self.fpath[nm]
            self.base.append(v)
        self.refs = [np.empty(n) for _ in m.legs]
        self.shift = [np.zeros(n) for _ in m.legs]
        self.imp = [np.zeros(n) for _ in m.legs]

    # ---------------------------------------------------------------- book primitives
    def _set(self, L: _Leg, side: int, tk: int, qty: float, t: float) -> None:
        d = L.bids if side > 0 else L.asks
        prev = d.get(tk, 0.0)
        if qty > 1e-12:
            d[tk] = qty
            if side > 0:
                if L.bb is None or tk > L.bb:
                    L.bb = tk
            elif L.ba is None or tk < L.ba:
                L.ba = tk
        else:
            qty = 0.0
            if not prev:
                return
            del d[tk]
            if side > 0:
                if tk == L.bb:
                    L.bb = max(d) if d else None
            elif tk == L.ba:
                L.ba = min(d) if d else None
        L.dsum[side] += qty - prev
        self.out.append((t, DEPTH, L.i, side, tk, qty))

    def _take(self, L: _Leg, aggr: int, limit: int | None, q: float, t: float) -> float:
        """An aggressive order: consume the opposite side up to `limit`. Returns unfilled quantity."""
        book = L.asks if aggr > 0 else L.bids
        while q > 1e-12:
            best = L.ba if aggr > 0 else L.bb
            if best is None or (limit is not None and (best > limit if aggr > 0 else best < limit)):
                break
            lv = book[best]
            f = lv if lv < q else q
            q -= f
            self.out.append((t, TRADE, L.i, aggr, best, f))
            self._set(L, -aggr, best, lv - f, t)
        return q

    # ---------------------------------------------------------------- order flow
    def _flow(self, L: _Leg, t: float) -> None:
        F, rng = L.F, self.rng
        u = rng.random() * L.rate
        rt = L.ref / L.tick
        if u < L.lam_l:                                         # a limit order arrives
            side = BUY if rng.random() < 0.5 else SELL
            d = int(rng.expovariate(1.0 / F.depth_ticks)) if F.depth_ticks > 0 else 0
            q = L.size(L.mu_l)
            if side > 0:
                tk = math.floor(rt - F.half_spread) - d
                if L.ba is not None and tk >= L.ba:
                    q = self._take(L, BUY, tk, q, t)
            else:
                tk = math.ceil(rt + F.half_spread) + d
                if L.bb is not None and tk <= L.bb:
                    q = self._take(L, SELL, tk, q, t)
            if q > 1e-12:
                book = L.bids if side > 0 else L.asks
                self._set(L, side, tk, book.get(tk, 0.0) + q, t)
        elif u < L.lam_l + L.lam_c:                             # a cancel attempt
            side = BUY if rng.random() < 0.5 else SELL
            depth = L.dsum[side]
            if depth <= 0 or rng.random() > depth / L.target:
                return
            book = L.bids if side > 0 else L.asks
            if not book:
                return
            x, acc = rng.random() * depth, 0.0
            for tk, lv in book.items():
                acc += lv
                if acc >= x:
                    break
            self._set(L, side, tk, lv - min(lv, L.size(L.mu_l)), t)
        else:                                                   # a market order
            if L.bb is None or L.ba is None:
                return
            bq, aq = L.bids[L.bb], L.asks[L.ba]
            p_buy = 0.5 + F.imbalance_sens * (bq - aq) / (bq + aq)
            side, q = (BUY if rng.random() < p_buy else SELL), L.size(L.mu_m)
            done = q - self._take(L, side, None, q, t)
            if F.impact_ticks:
                L.imp += side * F.impact_ticks * L.tick * done / (L.lot * F.market_size_mean)

    def _stale(self, L: _Leg, t: float) -> None:
        """After the reference moves: schedule a hit or a pull for every quote now through it."""
        F, rng, rt = L.F, self.rng, L.ref / L.tick
        if L.bb is not None and L.bb > rt:
            for tk in [k for k in L.bids if k > rt]:
                if (BUY, tk) not in L.flag:
                    L.flag.add((BUY, tk))
                    hit = rng.random() < F.snipe_prob
                    self._push(t + rng.expovariate(1.0 / (F.snipe_delay if hit else F.pull_delay)), "A", (L, hit, BUY, tk))
        if L.ba is not None and L.ba < rt:
            for tk in [k for k in L.asks if k < rt]:
                if (SELL, tk) not in L.flag:
                    L.flag.add((SELL, tk))
                    hit = rng.random() < F.snipe_prob
                    self._push(t + rng.expovariate(1.0 / (F.snipe_delay if hit else F.pull_delay)), "A", (L, hit, SELL, tk))
        lo, hi = rt - F.max_ticks, rt + F.max_ticks
        if L.bids and min(L.bids) < lo:
            for tk in [k for k in L.bids if k < lo]:
                self._set(L, BUY, tk, 0.0, t)
        if L.asks and max(L.asks) > hi:
            for tk in [k for k in L.asks if k > hi]:
                self._set(L, SELL, tk, 0.0, t)

    def _act(self, L: _Leg, hit: bool, side: int, tk: int, t: float) -> None:
        L.flag.discard((side, tk))
        rt = L.ref / L.tick
        book = L.bids if side > 0 else L.asks
        if tk not in book or (tk <= rt if side > 0 else tk >= rt):
            return                                  # gone already, or no longer stale
        if hit:
            self._take(L, -side, tk, math.inf, t)   # sweeps this level and anything better
        else:
            self._set(L, side, tk, 0.0, t)

    def _seed_book(self, L: _Leg) -> None:
        F, rt = L.F, L.ref / L.tick
        w = [math.exp(-d / max(F.depth_ticks, 0.5)) for d in range(F.max_ticks // 2)]
        tot = sum(w)
        for d, wd in enumerate(w):
            q = L.lot * round(L.target * wd / tot / L.lot)
            if q > 0:
                self._set(L, BUY, math.floor(rt - F.half_spread) - d, q, 0.0)
                self._set(L, SELL, math.ceil(rt + F.half_spread) + d, q, 0.0)

    # ---------------------------------------------------------------- main loop
    def _push(self, t: float, kind: str, payload) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, kind, payload))

    def run(self) -> SynthMarket:
        m, dt, rng = self.m, self.m.dt, self.rng
        self._heap, self._seq = [], 0
        alpha = [1.0 if l.lag <= 0 else 1 - math.exp(-dt / l.lag) for l in m.legs]
        decay = [math.exp(-dt * LN2 / max(l.flow.impact_half_life, 1e-9)) for l in m.legs]
        for j, L in enumerate(self.legs):
            b0 = self.bpath[j][0] if self.bpath[j] is not None else 0.0
            L.ref = L.core = self.base[j][0] * (1 + b0)
            self.refs[j][0] = L.ref
            self._seed_book(L)
            self._push(rng.expovariate(L.rate), "F", L)
            c = self.calc[j]
            if c is not None:
                c.sync(0.0)
                self.out.append((0.0, INDEX, j, float(self.base[j][0]), 0, 0.0))
                self._push(c.next_sample_at, "S", j)
                self._push(m.index_dt, "I", j)
        self._push(dt, "R", 1)
        cur_reg = 0
        while self._heap:
            t, _, kind, p = heapq.heappop(self._heap)
            if t > self.T:
                break
            if kind == "F":
                self._flow(p, t)
                self._push(t + rng.expovariate(p.rate), "F", p)
            elif kind == "A":
                self._act(p[0], p[1], p[2], p[3], t)
            elif kind == "R":
                k = p
                if m.regime is not None and self.regime[k] != cur_reg:
                    cur_reg = int(self.regime[k])
                    for L in self.legs:
                        L.set_regime(m.regime.depth_mult if cur_reg else 1.0, m.regime.market_mult if cur_reg else 1.0)
                for j, L in enumerate(self.legs):
                    leg, c = L.leg, self.calc[j]
                    b = self.bpath[j][k] if self.bpath[j] is not None else 0.0
                    s = 0.0
                    if c is not None and leg.funding_passthrough:
                        tts = c.settle_at - t
                        if tts < leg.funding_anticipation:
                            s = -leg.funding_passthrough * c.predicted() * (1 - max(tts, 0.0) / leg.funding_anticipation)
                    self.shift[j][k] = s
                    L.imp *= decay[j]
                    self.imp[j][k] = L.imp
                    L.core += (self.base[j][k] * (1 + b + s) - L.core) * alpha[j]
                    L.ref = L.core + L.imp
                    self.refs[j][k] = L.ref
                    self._stale(L, t)
                if k + 1 < self.n:
                    self._push(self.t[k + 1], "R", k + 1)
            elif kind == "I":
                k = min(int(t / dt), self.n - 1)
                self.out.append((t, INDEX, p, float(self.base[p][k]), 0, 0.0))
                self._push(t + m.index_dt, "I", p)
            elif kind == "S":
                j, c, L = p, self.calc[p], self.legs[p]
                idx = float(self.base[j][min(int(t / dt), self.n - 1)])
                ib = impact_price(L.bids, True, L.tick, c.impact_notional, L.inst.contract_size)
                ia = impact_price(L.asks, False, L.tick, c.impact_notional, L.inst.contract_size)
                prem = (max(0.0, ib - idx) - max(0.0, idx - ia)) / idx if ib == ib and ia == ia else math.nan
                c.add(prem)
                if c.k >= c.n_total:
                    self.out.append((t, FUNDING, j, c.settle(), 0, 0.0))
                self._push(c.next_sample_at, "S", j)
        names = [l.inst.name for l in m.legs]
        truth = {
            "t": self.t,
            "ref": {nm: self.refs[j] for j, nm in enumerate(names)},
            "base": {nm: self.base[j] for j, nm in enumerate(names)},
            "basis": {nm: self.bpath[j] for j, nm in enumerate(names) if self.bpath[j] is not None},
            "shift": {nm: self.shift[j] for j, nm in enumerate(names) if self.calc[j] is not None},
            "impact": {nm: self.imp[j] for j, nm in enumerate(names)},
            "factor": self.fpath,
            "regime": self.regime,
        }
        return SynthMarket(m, self.out, truth, 0)


def simulate(market: Market, duration: float = 1800.0, seed: int = 0) -> SynthMarket:
    """Generate `duration` seconds of market events. Same seed, same market."""
    sm = _Gen(market, duration, seed).run()
    sm.seed = seed
    return sm


# ------------------------------------------------------------------------------------- presets
def usd_spread(funding_interval: float = 1800.0, passthrough: float = 0.5, lag: float = 0.25) -> Market:
    """Two legs in one currency: a large-tick, slower SPOT leg and a small-tick, faster PERP leg that
    leads it, carries a mean-reverting premium, and pays funding. The premium is set rich on purpose
    so funding is often away from its baseline within a short run."""
    spot = Instrument("SPOT", tick=0.05, lot=1.0, maker_fee=0.0, taker_fee=1.0e-4, kind="spot")
    perp = Instrument("PERP", tick=0.01, lot=1.0, maker_fee=0.2e-4, taker_fee=2.0e-4, kind="perp")
    return Market(
        factors=[Factor("U", start=100.0, vol=0.8e-4, jump_rate=1 / 240.0, jump_sd=5e-4)],
        regime=Regime(),
        legs=[
            Leg(spot, lag=lag,
                flow=Flow(limit_rate=14.0, market_rate=0.8, target_depth=1500.0, size_mean=12.0,
                          market_size_mean=15.0, depth_ticks=1.5, half_spread=0.55,
                          snipe_delay=0.06, pull_delay=0.15, impact_ticks=0.12)),
            Leg(perp, lag=0.0, basis=Basis(mean=10e-4, sd=5e-4, half_life=240.0),
                flow=Flow(limit_rate=45.0, market_rate=3.0, target_depth=500.0, size_mean=5.0,
                          market_size_mean=6.0, depth_ticks=3.0, half_spread=0.7,
                          snipe_delay=0.01, pull_delay=0.03),
                funding=FundingCalc(interval=funding_interval, sample=5.0, weights="equal",
                                    impact_notional=5_000.0, multiplier=8.0),
                funding_passthrough=passthrough, funding_anticipation=min(600.0, funding_interval / 3)),
        ],
    )


def krx_style(funding_interval: float = 1800.0, passthrough: float = 0.5, lag: float = 0.3) -> Market:
    """Three legs: a KRW-quoted, very-large-tick stock (KRX) that lags; a USDT perp on the same
    underlying that leads and pays funding; and USDTKRW to hedge the currency. Base currency is USDT.
    A long-stock / short-perp position is long the KRW-per-USDT rate; the FX leg is there to hedge it."""
    fx = Instrument("USDTKRW", tick=0.1, lot=1.0, maker_fee=0.0, taker_fee=0.5e-4, kind="spot",
                    fx_inst="USDTKRW", fx_invert=True)
    krx = Instrument("KRX", tick=500.0, lot=1.0, maker_fee=0.0, taker_fee=1.0e-4, kind="spot",
                     fx_inst="USDTKRW", fx_invert=True)
    perp = Instrument("PERP", tick=0.01, lot=1.0, maker_fee=0.2e-4, taker_fee=2.0e-4, kind="perp")
    return Market(
        factors=[Factor("U", start=250.0, vol=0.8e-4, jump_rate=1 / 240.0, jump_sd=5e-4),
                 Factor("FX", start=1400.0, vol=0.15e-4)],
        regime=Regime(),
        legs=[
            Leg(krx, factors=("U", "FX"), lag=lag,
                flow=Flow(limit_rate=12.0, market_rate=0.7, target_depth=3000.0, size_mean=20.0,
                          market_size_mean=20.0, depth_ticks=1.2, half_spread=0.55,
                          snipe_delay=0.08, pull_delay=0.2, impact_ticks=0.08)),
            Leg(perp, lag=0.0, basis=Basis(mean=10e-4, sd=5e-4, half_life=240.0),
                flow=Flow(limit_rate=45.0, market_rate=3.0, target_depth=400.0, size_mean=4.0,
                          market_size_mean=5.0, depth_ticks=4.0, half_spread=0.8,
                          snipe_delay=0.01, pull_delay=0.03),
                funding=FundingCalc(interval=funding_interval, sample=5.0, weights="equal",
                                    impact_notional=5_000.0, multiplier=8.0),
                funding_passthrough=passthrough, funding_anticipation=min(600.0, funding_interval / 3)),
            Leg(fx, factors=("FX",), lag=0.0,
                flow=Flow(limit_rate=20.0, market_rate=1.0, target_depth=200_000.0, size_mean=2000.0,
                          market_size_mean=2500.0, depth_ticks=2.0, half_spread=0.6,
                          snipe_delay=0.03, pull_delay=0.08, impact_ticks=0.1)),
        ],
    )


def single(tick: float = 0.01, start: float = 100.0, **flow_kw) -> Market:
    """One instrument, for plain market-making practice."""
    inst = Instrument("X", tick=tick, lot=1.0, maker_fee=0.0, taker_fee=2.0e-4)
    return Market(factors=[Factor("U", start=start, vol=1.2e-4, jump_rate=1 / 120.0, jump_sd=6e-4)],
                  regime=Regime(), legs=[Leg(inst, flow=Flow(**flow_kw))])


def tweak(market: Market, leg: str, **changes) -> Market:
    """Copy of a market with one leg's fields replaced; use flow__x=... to change a Flow field.
    Example: tweak(usd_spread(), "SPOT", lag=1.0, flow__snipe_delay=0.2)"""
    legs = []
    for l in market.legs:
        if l.inst.name == leg:
            flow_ch = {k[6:]: v for k, v in changes.items() if k.startswith("flow__")}
            rest = {k: v for k, v in changes.items() if not k.startswith("flow__")}
            l = replace(l, flow=replace(l.flow, **flow_ch), **rest)
        legs.append(l)
    return replace(market, legs=legs)
