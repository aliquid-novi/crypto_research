import numpy as np
import pandas as pd

import spreadsim as ss
from spreadsim import BUY, DEPTH, SELL, TRADE


def test_frames_merge_with_trades_before_the_depth_they_cause():
    X = ss.Instrument("X", tick=0.5)
    depth = pd.DataFrame({"t": [0.0, 0.0, 1.0], "inst": "X", "side": [1, -1, 1], "price": [100.0, 100.5, 100.0], "qty": [5.0, 5.0, 2.0]})
    trades = pd.DataFrame({"t": [1.0], "inst": "X", "side": [-1], "price": [100.0], "qty": [3.0]})
    ev = ss.events_from_frames([X], depth=depth, trades=trades)
    assert [e[1] for e in ev] == [DEPTH, DEPTH, TRADE, DEPTH]
    assert ev[2] == (1.0, TRADE, 0, SELL, 200, 3.0) and ev[3] == (1.0, DEPTH, 0, BUY, 200, 2.0)


def test_snapshots_diff_rebuilds_the_visible_book():
    X = ss.Instrument("X", tick=1.0)
    t = [0.0, 1.0, 2.0]
    bid_px = [[100, 99], [100, 99], [99, 98]]
    bid_q = [[5, 7], [3, 7], [7, 4]]
    ask_px = [[101, 102], [101, 102], [100, 101]]
    ask_q = [[6, 8], [6, 8], [2, 6]]
    df = ss.snapshots_to_frame("X", t, bid_px, bid_q, ask_px, ask_q)
    ev = ss.events_from_frames([X], depth=df)
    book = ss.Book(X)
    for e in ev:
        book.set(e[3], e[4], e[5])
    assert (book.best_bid, book.bid_qty, book.best_ask, book.ask_qty) == (99.0, 7.0, 100.0, 2.0)
    assert book.qty(BUY, 100) == 0.0 and book.qty(BUY, 98) == 4.0 and book.qty(SELL, 102) == 8.0   # 102 scrolled out: kept


def test_replay_from_frames_runs_and_fills():
    X = ss.Instrument("X", tick=1.0)
    depth = pd.DataFrame({"t": [0.0, 0.0, 5.0], "inst": "X", "side": [1, -1, -1], "price": [100.0, 101.0, 101.0], "qty": [5.0, 5.0, 5.0]})
    trades = pd.DataFrame({"t": [2.0], "inst": "X", "side": [-1], "price": [100.0], "qty": [9.0]})

    class S(ss.Strategy):
        def on_start(self, sim):
            sim.timer(1.0)

        def on_timer(self, sim, tag):
            sim.buy("X", 100.0, 2)

    res = ss.Sim([X], ss.events_from_frames([X], depth=depth, trades=trades), S()).run()
    assert res.fills.qty.sum() == 2.0 and res.fills.maker.all()
