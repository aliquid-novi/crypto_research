"""Diagnostics, controllers and the experiment harness."""
import numpy as np
import pandas as pd
import pytest

import spreadsim as ss
from spreadsim import BUY, DEPTH, SELL, TRADE

X = ss.Instrument("X", tick=1.0)


class OneBid(ss.Strategy):
    def on_start(self, sim):
        sim.timer(1.0)

    def on_timer(self, sim, tag):
        sim.submit("X", BUY, 100, 5, tag="quote", meta={"k": 0})


def _run(after):
    ev = [(0.0, DEPTH, 0, BUY, 100, 10.0), (0.0, DEPTH, 0, SELL, 101, 10.0),
          (2.0, TRADE, 0, SELL, 100, 15.0), (2.0, DEPTH, 0, BUY, 100, 0.0), (2.0, DEPTH, 0, BUY, 99, 10.0)]
    return ss.Sim([X], ev + after + [(200.0, DEPTH, 0, SELL, 500, 1.0)], OneBid()).run()


def test_fill_table_markouts_and_excursions_by_hand():
    # filled at 100 at t=2 (mid then 100.5); at t=4 the book drops to 97/98; at t=30 it is back at 100/101
    after = [(4.0, TRADE, 0, SELL, 99, 10.0), (4.0, DEPTH, 0, BUY, 99, 0.0), (4.0, DEPTH, 0, BUY, 97, 10.0),
             (4.0, DEPTH, 0, SELL, 101, 0.0), (4.0, DEPTH, 0, SELL, 98, 10.0),
             (30.0, DEPTH, 0, SELL, 98, 0.0), (30.0, DEPTH, 0, SELL, 101, 10.0), (30.0, DEPTH, 0, BUY, 100, 10.0)]
    ft = ss.fill_table(_run(after), horizons=(1.0, 5.0, 60.0), window=20.0)
    r = ft.iloc[0]
    assert (r.price, r.qty, r.k) == (100.0, 5.0, 0)
    assert r.m1 == pytest.approx(0.0)           # mid 100.0 one second later (99 bid / 101 ask)
    assert r.m5 == pytest.approx(-2.5)          # mid 97.5
    assert r.m60 == pytest.approx(0.5)          # mid 100.5
    assert r.mae == pytest.approx(-2.5) and r.mfe == pytest.approx(0.0)
    # a trade printed at 99 (one tick below us) and the ask came down to 98 (two ticks through us)
    assert r.touch == 2.0 and r.through == 2.0


def test_shift_counterfactual_arithmetic():
    ft = pd.DataFrame({"qty": [10.0] * 4, "through": [0, 1, 2, 3], "touch": [1, 2, 3, 4],
                       "m5": [2.0, -1.0, -2.0, -3.0], "m120": [4.0, 1.0, 0.0, -2.0]})
    cf = ss.shift_counterfactual(ft, shifts=(1, 2), tick_value=0.5, cost_ticks=1.0)
    one = cf.loc[1]
    assert one.share_kept == 0.75 and one.m5_lost == 2.0 and one.m5_kept == pytest.approx(-2.0)
    assert one.gain_kept == 0.75 and one.value_lost == pytest.approx((4.0 - 1.0) / 4)
    assert one.net_ticks_per_fill == pytest.approx(0.0)
    assert one.net_money == pytest.approx((3 * 1 - (4.0 - 1.0)) * 10 * 0.5)
    assert cf.loc[2, "net_ticks_per_fill"] == pytest.approx((2 * 2 - (3.0 + 0.0)) / 4)


def test_conditional_groups_and_counts():
    df = pd.DataFrame({"g": [0, 0, 1, 1, 1], "v": [1.0, 3.0, 2.0, 4.0, 6.0]})
    c = ss.conditional(df, "g", "v")
    assert c.n.tolist() == [2, 3] and c.v_mean.tolist() == [2.0, 4.0] and c.v_median.tolist() == [2.0, 4.0]


class _Book:
    def __init__(self, mid):
        self.mid, self.tick = mid, 1.0


class _Sim:
    def __init__(self):
        self.now, self.mids = 0.0, {"Q": 100.0, "H": 50.0}

    def book(self, inst):
        return _Book(self.mids[inst])


class _Strat:
    quote, hedge_leg, vol = "Q", "H", 1.0


class _O:
    def __init__(self, side, k):
        self.tag, self.side, self.meta = "quote", side, {"k": k}


class _F:
    def __init__(self, side, price, k=0):
        self.order, self.price = _O(side, k), price


def _feed(ctrl, sim, markout, n, k=0):
    """n buy fills at 100 whose markout after ctrl.h seconds is `markout` ticks."""
    for _ in range(n):
        sim.mids["Q"] = 100.0
        ctrl.on_fill(sim, _Strat, _F(BUY, 100.0, k))
        sim.now += ctrl.h
        sim.mids["Q"] = 100.0 + markout
        ctrl.shift(sim, _Strat)
    return ctrl.shift(sim, _Strat)


def test_markout_rule_pushes_out_on_negative_markouts_and_relaxes_on_positive():
    c, sim = ss.MarkoutFeedback(horizon=5.0, min_fills=4, max_shift=2, by_vol=False, spread=False), _Sim()
    assert _feed(c, sim, -1.0, 3) == 0                  # not enough fills yet
    assert _feed(c, sim, -1.0, 1) == 1                  # four negative: out one tick
    assert _feed(c, sim, -1.0, 8) == 2 and _feed(c, sim, -1.0, 8) == 2      # capped
    assert _feed(c, sim, +1.0, 10) == 2                 # 12 negatives remembered: 10 positives are not yet a majority
    assert _feed(c, sim, +1.0, 1) == 1                  # the window's average turns positive: back in one tick
    assert _feed(c, sim, -1.0, 4, k=5) == 1             # fills on other clips are ignored


def test_markout_rule_hurdle_judges_markouts_net_of_cost():
    plain, sim = ss.MarkoutFeedback(horizon=5.0, min_fills=4, by_vol=False, spread=False), _Sim()
    net, sim2 = ss.MarkoutFeedback(horizon=5.0, min_fills=4, by_vol=False, spread=False, hurdle_ticks=1.5), _Sim()
    assert _feed(plain, sim, +1.0, 4) == 0              # markouts of +1 tick look fine against zero
    assert _feed(net, sim2, +1.0, 4) == 1               # but do not cover a 1.5 tick cost: push out


def test_markout_rule_spread_markout_removes_the_hedge_leg_move():
    c, sim = ss.MarkoutFeedback(horizon=5.0, min_fills=2, by_vol=False, spread=True), _Sim()
    for _ in range(2):                                   # quote leg falls 1%, hedge leg falls 1% too: spread unchanged
        sim.mids.update(Q=100.0, H=50.0)
        c.on_fill(sim, _Strat, _F(BUY, 100.0))
        sim.now += 5.0
        sim.mids.update(Q=99.0, H=49.5)
        out = c.shift(sim, _Strat)
    assert out == 0 and c.history == []


def test_experiment_is_paired_and_reproducible_and_parallel_matches_serial():
    worlds = {"a": ss.single(), "b": ss.tweak(ss.single(), "X", flow__impact_ticks=0.6)}
    variants = {"tight": lambda: ss.SimpleMM("X", half_spread_ticks=1.0), "wide": lambda: ss.SimpleMM("X", half_spread_ticks=2.0)}
    exp = ss.Experiment(worlds, variants, seeds=range(2), duration=120,
                        metrics=lambda res, w, v, s: {"n_fills": len(res.fills), "_tag": (w, v, s)})
    r1, r2 = exp.run(verbose=False), exp.run(workers=2, verbose=False)
    assert len(r1.table) == 8 and r1.table.pnl.tolist() == r2.table.pnl.tolist()
    assert r1.extras[("b", "wide", 1)]["_tag"] == ("b", "wide", 1) and "_tag" not in r1.table.columns
    p = r1.paired("pnl", "tight")
    assert p.loc[("a", "tight"), "diff"] == 0.0
    wide = r1.table.pivot_table(index=["world", "seed"], columns="variant", values="pnl")
    assert p.loc[("b", "wide"), "diff"] == pytest.approx((wide.loc["b", "wide"] - wide.loc["b", "tight"]).mean())
    assert set(r1.best("pnl")) <= {"tight", "wide"}


def test_static_shift_moves_the_quotes_and_trades_are_recorded():
    mk = ss.simulate(ss.usd_spread(), 300, seed=1)
    a = mk.run(ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4,), warmup=20))
    b = mk.run(ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4,), warmup=20, shift_ticks=2))
    oa, ob = a.orders[a.orders.tag == "quote"], b.orders[b.orders.tag == "quote"]
    da = (oa.side * (oa.mid_decision - oa.price)).mean() / 0.05
    db = (ob.side * (ob.mid_decision - ob.price)).mean() / 0.05
    assert db - da == pytest.approx(2.0, abs=0.6)
    assert len(a.trades["SPOT"]) > 50 and set(a.trades["SPOT"].side.unique()) <= {1, -1}
    assert (b.logs["shift"].value == 2).all()
