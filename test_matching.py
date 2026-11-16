"""Matching, queue position and latency: small scripted markets with answers worked out by hand."""
import math

import pytest

import spreadsim as ss
from spreadsim import BUY, DEPTH, SELL, TRADE

X = ss.Instrument("X", tick=1.0, taker_fee=0.001, maker_fee=-0.0005)


class Script(ss.Strategy):
    """Runs `actions`: {time: f(sim)}; records what the strategy is told."""

    def __init__(self, actions):
        self.actions, self.fills, self.events, self.seen = actions, [], [], []

    def on_start(self, sim):
        for t in self.actions:
            sim.timer(t - sim.now, t)

    def on_timer(self, sim, tag):
        self.actions[tag](sim)

    def on_fill(self, sim, fill):
        self.fills.append((round(sim.now, 6), fill.order.side, fill.price, fill.qty, fill.maker))

    def on_order(self, sim, order, event):
        self.events.append((round(sim.now, 6), order.id, event))


def book0(t=0.0, bid=(100, 10.0), ask=(101, 10.0)):
    return [(t, DEPTH, 0, BUY, bid[0], bid[1]), (t, DEPTH, 0, SELL, ask[0], ask[1])]


def run(events, actions, **kw):
    s = Script(actions)
    sim = ss.Sim([X], events + [(100.0, DEPTH, 0, SELL, 500, 1.0)], s, **kw)
    return sim, s, sim.run()


def test_queue_advances_through_trades_then_fills_on_the_excess():
    ev = book0() + [
        (2.0, TRADE, 0, SELL, 100, 6.0), (2.0, DEPTH, 0, BUY, 100, 4.0),     # 6 of the 10 ahead trade
        (2.5, DEPTH, 0, BUY, 100, 12.0),                                      # 8 more join behind us
        (3.0, TRADE, 0, SELL, 100, 6.0), (3.0, DEPTH, 0, BUY, 100, 6.0),     # 4 ahead, then 2 reach us
    ]
    sim, s, res = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)})
    o = sim._all_orders[0]
    assert o.ahead0 == 10.0
    assert s.fills == [(3.0, BUY, 100.0, 2.0, True)]
    assert sim.true_position("X") == 2.0


def test_depth_drop_explained_by_a_trade_is_not_counted_as_a_cancel():
    ev = book0() + [(2.0, TRADE, 0, SELL, 100, 6.0), (2.0, DEPTH, 0, BUY, 100, 4.0)]
    sim, s, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)}, queue=ss.OptimisticQueue())
    assert sim._all_orders[0].x_ahead == 4.0


@pytest.mark.parametrize("model,expect", [(ss.RiskAverseQueue(), 10.0), (ss.OptimisticQueue(), 4.0), (ss.ProbQueue(1.0), 7.0)])
def test_cancel_attribution_by_queue_model(model, expect):
    # we join behind 10; another 10 joins behind us; then 6 cancel with no trade
    ev = book0() + [(2.0, DEPTH, 0, BUY, 100, 20.0), (3.0, DEPTH, 0, BUY, 100, 14.0)]
    sim, s, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)}, queue=model)
    assert sim._all_orders[0].x_ahead == pytest.approx(expect)


def test_risk_averse_spills_into_the_front_once_nothing_is_behind():
    ev = book0() + [(2.0, DEPTH, 0, BUY, 100, 7.0)]            # 3 cancel, nobody is behind us
    sim, s, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)}, queue=ss.RiskAverseQueue())
    assert sim._all_orders[0].x_ahead == 7.0


def test_trade_through_our_price_fills_up_to_the_print_size():
    ev = book0() + [(2.0, TRADE, 0, SELL, 100, 10.0), (2.0, DEPTH, 0, BUY, 100, 0.0),
                    (2.0, TRADE, 0, SELL, 99, 3.0)]
    sim, s, _ = run(ev + [(1.5, DEPTH, 0, BUY, 99, 5.0)], {1.0: lambda sim: sim.buy("X", 100, 5)}, fill_on_clear=False)
    assert s.fills == [(2.0, BUY, 100.0, 3.0, True)]


def test_level_cleared_by_a_trade_fills_us_when_fill_on_clear():
    ev = book0() + [(2.0, TRADE, 0, SELL, 100, 10.0), (2.0, DEPTH, 0, BUY, 100, 0.0)]
    _, s_on, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)}, fill_on_clear=True)
    _, s_off, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)}, fill_on_clear=False)
    assert s_on.fills == [(2.0, BUY, 100.0, 5.0, True)] and s_off.fills == []


def test_new_liquidity_crossing_our_quote_trades_with_us():
    ev = book0() + [(1.5, DEPTH, 0, BUY, 100, 0.0), (2.0, DEPTH, 0, SELL, 100, 3.0), (2.5, DEPTH, 0, SELL, 100, 4.0)]
    _, s, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 5)})
    assert s.fills == [(2.0, BUY, 100.0, 3.0, True), (2.5, BUY, 100.0, 1.0, True)]


def test_two_of_our_orders_cannot_be_filled_by_the_same_quantity():
    ev = book0() + [(3.0, TRADE, 0, SELL, 100, 14.0)]
    _, s, _ = run(ev, {1.0: lambda sim: sim.buy("X", 100, 3), 1.1: lambda sim: sim.buy("X", 100, 3)}, fill_on_clear=False)
    assert sorted(f[3] for f in s.fills) == [1.0, 3.0]          # 14 - 10 ahead = 4 units for us in total


def test_taker_walks_the_book_and_does_not_retake_what_it_took():
    ev = book0() + [(0.0, DEPTH, 0, SELL, 102, 10.0)]
    acts = {1.0: lambda sim: sim.submit("X", BUY, 102, 14, tif=ss.IOC), 2.0: lambda sim: sim.submit("X", BUY, 102, 10, tif=ss.IOC)}
    sim, s, res = run(ev, acts)
    assert [(f[2], f[3]) for f in s.fills] == [(101.0, 10.0), (102.0, 4.0), (102.0, 6.0)]
    assert res.orders.status.tolist() == ["filled", "cancelled"]
    fees = res.fills.fee.sum()
    assert fees == pytest.approx(0.001 * (101 * 10 + 102 * 10))


def test_post_only_rejects_and_ioc_without_liquidity_cancels():
    acts = {1.0: lambda sim: sim.submit("X", BUY, 101, 1, post_only=True), 1.1: lambda sim: sim.submit("X", BUY, 100, 1, tif=ss.IOC)}
    sim, s, res = run(book0(), acts)
    assert res.orders.status.tolist() == ["rejected", "cancelled"] and s.fills == []


def test_latency_delays_the_book_the_order_and_the_fill_report():
    lat = ss.ConstantLatency(feed=0.5, entry=0.2, response=0.3)
    seen = {}
    ev = book0() + [(1.0, DEPTH, 0, SELL, 101, 0.0), (1.0, DEPTH, 0, SELL, 103, 10.0)]

    def look(sim):
        seen[round(sim.now, 2)] = sim.book("X").best_ask

    def buy(sim):
        sim.submit("X", BUY, 105, 1, tif=ss.IOC)

    sim, s, res = run(ev, {1.2: look, 1.6: look, 1.3: buy}, latency=lat)
    assert seen == {1.2: 101.0, 1.6: 103.0}                    # the 1.0 update is visible from 1.5
    o = res.orders.iloc[0]
    assert o.t_arrive == pytest.approx(1.5) and o.mid_decision == 100.5 and o.mid_arrive == 101.5
    assert s.fills == [(1.8, BUY, 103.0, 1.0, False)]          # filled at the price that was really there
    sl = ss.slippage(res).iloc[0]
    assert sl.delay_cost == pytest.approx(1.0 / 100.5 * 1e4) and sl.cross_cost == pytest.approx(1.5 / 100.5 * 1e4)


def test_cancel_that_loses_the_race_is_rejected_and_counted():
    lat = ss.ConstantLatency(entry=0.5)
    ev = book0() + [(2.2, TRADE, 0, SELL, 100, 20.0)]
    acts = {1.0: lambda sim: sim.buy("X", 100, 5), 2.0: lambda sim: sim.cancel_all("X")}
    sim, s, res = run(ev, acts, latency=lat)
    assert s.fills == [(2.2, BUY, 100.0, 5.0, True)]
    assert sim.cancel_rejects == 1 and (2.5, 1, "cancel_reject") in s.events
    assert res.orders.cancel_reject.tolist() == [True]


def test_strategy_position_lags_the_true_position():
    lat = ss.ConstantLatency(response=1.0)
    pos = {}
    ev = book0() + [(2.0, TRADE, 0, SELL, 100, 20.0)]
    acts = {1.0: lambda sim: sim.buy("X", 100, 5), 2.5: lambda sim: pos.update(a=(sim.position("X"), sim.true_position("X"))),
            3.5: lambda sim: pos.update(b=(sim.position("X"), sim.true_position("X")))}
    run(ev, acts, latency=lat)
    assert pos == {"a": (0.0, 5.0), "b": (5.0, 5.0)}


def test_accounting_cash_position_fees_and_attribution_add_up():
    ev = book0() + [(2.0, TRADE, 0, SELL, 100, 20.0), (2.0, DEPTH, 0, BUY, 100, 0.0), (2.0, DEPTH, 0, BUY, 98, 10.0),
                    (2.0, DEPTH, 0, SELL, 101, 0.0), (2.0, DEPTH, 0, SELL, 99, 10.0)]
    acts = {1.0: lambda sim: sim.buy("X", 100, 5), 3.0: lambda sim: sim.submit("X", SELL, 98, 2, tif=ss.IOC)}
    sim, s, res = run(ev, acts)
    # bought 5 @100 as maker (rebate), sold 2 @98 as taker; 3 left marked at the final mid
    mid_end = res.bbo["X"].mid.dropna().iloc[-1]
    expect = -5 * 100 + 0.0005 * 500 + 2 * 98 - 0.001 * 196 + 3 * mid_end
    assert res.pnl == pytest.approx(expect)
    att = ss.pnl_attribution(res, horizon=0.5)
    assert att.loc["TOTAL", "total"] == pytest.approx(res.pnl)
    assert att.loc["X", ["spread_capture", "hedge_cost", "adverse_selection", "inventory", "fees", "funding"]].sum() == pytest.approx(res.pnl)
    assert att.loc["X", "spread_capture"] == pytest.approx(5 * 0.5)       # bought 0.5 below the 100.5 mid
    assert att.loc["X", "adverse_selection"] == pytest.approx(5 * (98.5 - 100.5))


def test_load_latency_grows_with_message_rate():
    lat = ss.LoadLatency(entry=0.001, sigma=0.0, load=1e-4)
    burst = [(5.0 + k * 0.001, DEPTH, 0, BUY, 99, 1.0 + k) for k in range(200)]
    acts = {2.0: lambda sim: sim.buy("X", 90, 1), 5.3: lambda sim: sim.buy("X", 90, 1)}
    sim, s, res = run(book0() + burst, acts, latency=lat)
    d = (res.orders.t_arrive - res.orders.t_submit).values
    assert d[0] == pytest.approx(0.001, abs=3e-4) and d[1] > 0.01
