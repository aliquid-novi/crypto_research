"""The whole strategy interface in one small class, and how to plug in your own models.

    python examples/03_write_your_own.py
"""
# %%
try:                                   # lets the examples run without installing the package
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
except NameError:
    pass
import pandas as pd

import spreadsim as ss
from spreadsim import BUY, SELL

pd.set_option("display.width", 200, "display.max_columns", 30)


class JoinTheBid(ss.Strategy):
    """Rest one bid at the best bid; if the best bid moves, cancel and rejoin; sell whatever we buy
    straight back at the market. Not a good idea: it is here to show every callback being used."""

    def __init__(self, inst, size=5):
        self.inst, self.size, self.order = inst, size, None
        self.queue_log = []

    def on_start(self, sim):
        sim.timer(5.0, "sample-queue")                       # timers are how you act on a clock

    def on_book(self, sim, inst):                            # our delayed copy of the book changed
        if inst != self.inst:
            return
        b = sim.book(inst)                                   # .best_bid .best_ask .mid .microprice .imbalance
        if b.bb is None or b.ba is None:                     #   .bid_qty .ask_qty .levels(side, n) .depth(side, n)
            return
        if self.order is not None and not self.order.is_open:
            self.order = None
        if self.order is None:
            self.order = sim.buy(inst, b.best_bid, self.size, post_only=True, tag="join")
        elif self.order.status == "live" and self.order.price != b.best_bid:
            sim.cancel(self.order)                           # takes `entry` latency to reach the exchange

    def on_fill(self, sim, fill):                            # fill.order, fill.price, fill.qty, fill.maker, fill.id
        if fill.order.tag == "join":
            sim.take(self.inst, SELL, fill.qty, cross_ticks=5, tag="exit", parent=fill.id)

    def on_order(self, sim, order, event):                   # "ack" | "cancelled" | "rejected" | "cancel_reject"
        if event == "cancel_reject":
            pass                                             # we were filled before the cancel got there

    def on_timer(self, sim, tag):
        sim.timer(5.0, tag)
        if self.order is not None and self.order.status == "live":
            # x_ahead is the model's true queue position. A live strategy cannot see it; a researcher can.
            self.queue_log.append((sim.now, self.order.x_ahead, self.order.level0))

    def on_funding(self, sim, inst, rate, payment): ...
    def on_end(self, sim): ...


# %% your own queue model: decide where a cancel came from
class FrontHeavyQueue(ss.QueueModel):
    """Orders near the front cancel less (they have waited longest): only 20% of cancels are ahead of us."""

    def on_cancel(self, ahead, behind, cancelled):
        from_ahead = min(ahead, 0.2 * cancelled) if behind > 0 else min(ahead, cancelled)
        return ahead - from_ahead


# %% your own latency model: any function of the simulator state
class SlowWhenBusy(ss.Latency):
    def entry(self, sim, inst):
        return 0.002 + 0.00002 * sim.msg_rate(inst)          # +2 ms per 100 messages/second on that venue

    def response(self, sim, inst):
        return 0.002

    def feed(self, sim, inst):
        return 0.001


# %%
mk = ss.simulate(ss.usd_spread(), duration=900, seed=2)
strat = JoinTheBid("SPOT")
res = mk.run(strat, queue={"SPOT": FrontHeavyQueue()}, latency=SlowWhenBusy())
print(pd.Series(ss.summary(res)).round(3).to_string())
print("\nlatency actually experienced, ms"); print(ss.latency_stats(res).round(2))
print("\nqueue position samples (time, ahead of us, level size when we joined):"); print(strat.queue_log[:5])
