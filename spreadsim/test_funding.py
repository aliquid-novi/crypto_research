"""Funding: the formula against the venues' own worked examples, and the payment plumbing."""
import math
import random

import pytest

import spreadsim as ss
from spreadsim import BUY, DEPTH, FUNDING, INDEX, SELL


def test_binance_documented_example():
    # Binance FAQ, example 2: 8h average premium 0.0429% -> funding 0.0100%
    c = ss.FundingCalc(interval=28800, weights="linear")
    assert c.rate(0.000429) == pytest.approx(0.0001)
    assert c.rate(0.0006) == pytest.approx(0.0001) and c.rate(-0.0004) == pytest.approx(0.0001)   # edges of the flat zone
    assert c.rate(0.0008) == pytest.approx(0.0003) and c.rate(-0.0010) == pytest.approx(-0.0005)


def test_hyperliquid_documented_example():
    # Hyperliquid docs: premium 1% -> 0.95% on an 8h basis, paid hourly at one eighth
    c = ss.FundingCalc(interval=3600, weights="equal")
    assert c.rate(0.01) == pytest.approx(0.0095 / 8)


def test_linear_weights_lock_in_late():
    c = ss.FundingCalc(interval=28800, sample=5, weights="linear")
    for _ in range(4 * 720):                # four hours of samples
        c.add(0.001)
    locked, remaining = c.locked()
    assert remaining == pytest.approx(0.75, abs=0.001)          # only a quarter is fixed with 4h to go
    e = ss.FundingCalc(interval=28800, sample=5, weights="equal")
    for _ in range(4 * 720):
        e.add(0.001)
    assert e.locked()[1] == pytest.approx(0.5)
    assert locked == pytest.approx(0.001 * 0.25, rel=0.01)


def test_weighted_average_and_settlement_reset():
    c = ss.FundingCalc(interval=20, sample=5, weights="linear", start=100.0)
    for p in (0.001, 0.002, 0.003, 0.004):
        c.add(p)
    assert c.avg() == pytest.approx((1 * 0.001 + 2 * 0.002 + 3 * 0.003 + 4 * 0.004) / 10)
    r = c.settle()
    assert r == pytest.approx(c.rate(0.003)) and c.start == 120.0 and c.k == 0 and math.isnan(c.avg())


def test_expected_rate_matches_monte_carlo_and_exceeds_plug_in_near_the_kink():
    c = ss.FundingCalc(interval=28800)
    rng = random.Random(1)
    mu, sd = 0.0006, 0.0003                # sitting exactly on the upper kink
    mc = sum(c.rate(rng.gauss(mu, sd)) for _ in range(200_000)) / 200_000
    assert ss.expected_rate(c, mu, sd) == pytest.approx(mc, rel=0.02)
    assert ss.expected_rate(c, mu, sd) > c.rate(mu)
    assert ss.expected_rate(c, 0.002, 1e-9) == pytest.approx(c.rate(0.002))


def test_premium_index_has_a_dead_zone_inside_the_impact_spread():
    b = ss.Book(ss.Instrument("P", tick=0.1))
    b.set(BUY, 1000, 50.0), b.set(SELL, 1002, 50.0)
    assert ss.premium_index(b, 100.1, 1000.0) == 0.0
    assert ss.premium_index(b, 99.0, 1000.0) == pytest.approx(1.0 / 99.0)
    assert ss.premium_index(b, 101.0, 1000.0) == pytest.approx(-0.8 / 101.0)


class Hold(ss.Strategy):
    def __init__(self):
        self.paid = []

    def on_start(self, sim):
        sim.timer(1.0, "go")

    def on_timer(self, sim, tag):
        sim.submit("P", BUY, 101.0, 5, tif=ss.IOC)

    def on_funding(self, sim, inst, rate, payment):
        self.paid.append((round(sim.now, 3), inst, rate, payment))


def _events(with_funding_print):
    P = ss.Instrument("P", tick=1.0, kind="perp")
    ev = [(0.0, DEPTH, 0, BUY, 100, 50.0), (0.0, DEPTH, 0, SELL, 101, 50.0), (0.0, INDEX, 0, 99.0, 0, 0.0)]
    if with_funding_print:
        ev.append((20.0, FUNDING, 0, 0.002, 0, 0.0))
    ev.append((30.0, DEPTH, 0, SELL, 105, 1.0))
    return P, ev


def test_model_settled_funding_pays_on_the_true_position():
    P, ev = _events(False)
    calc = ss.FundingCalc(interval=20, sample=5, weights="equal", impact_notional=100.0, multiplier=1440.0)
    s = Hold()
    res = ss.Sim([P], ev, s, funding={"P": calc}).run()
    # bid 100 is above the index 99: premium = 1/99 every sample; x1440 x 20/28800 = x1
    rate = (1 / 99 - 0.0005)
    assert res.funding.rate.tolist() == pytest.approx([rate])
    assert res.funding.payment.iloc[0] == pytest.approx(-5 * 99.0 * rate)       # long pays, on index notional
    assert s.paid[0][0] == 20.0 and s.paid[0][3] == pytest.approx(-5 * 99.0 * rate)
    att = ss.pnl_attribution(res)
    assert att.loc["P", "funding"] == pytest.approx(-5 * 99.0 * rate)
    assert att.loc["TOTAL", "total"] == pytest.approx(res.pnl)


def test_event_settled_funding_uses_the_printed_rate_and_tracker_only_predicts():
    P, ev = _events(True)
    calc = ss.FundingCalc(interval=20, sample=5, weights="equal", impact_notional=100.0)
    seen = {}

    class Peek(Hold):
        def on_timer(self, sim, tag):
            if tag == "go":
                super().on_timer(sim, tag)
                sim.timer(11.0, "peek")
            else:
                c = sim.funding("P")
                seen.update(avg=c.avg(), tts=c.time_to_settle(sim.now), pred=c.predicted())

    res = ss.Sim([P], ev, Peek(), funding={"P": calc}, funding_from="events").run()
    assert res.funding.rate.tolist() == [0.002]
    assert res.funding.payment.iloc[0] == pytest.approx(-5 * 99.0 * 0.002)
    assert seen["avg"] == pytest.approx(1 / 99) and seen["tts"] == pytest.approx(8.0)
    assert seen["pred"] == pytest.approx((1 / 99 - 0.0005) * 20 / 28800)
