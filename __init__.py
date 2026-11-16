"""spreadsim: a small, readable market simulator and backtester for spread trading research."""
from .analytics import (Result, compare, fair_value_error, latency_stats, maker_stats, markout_curve, markouts,
                        pnl_attribution, slippage, spread_pairs, summary, sweep)
from .book import Book
from .controllers import Controller, MarkoutFeedback, VolShift, VolState
from .core import BUY, DEPTH, FUNDING, GTC, INDEX, IOC, SELL, TRADE, Fill, Instrument, Order
from .diagnostics import conditional, fill_table, premium_series, shift_counterfactual, vol_series
from .data import events_from_frames, mid_index, snapshots_to_frame
from .lab import Experiment, Runs
from .funding import FundingCalc, expected_rate, premium_index
from .models import ConstantLatency, Latency, LoadLatency, OptimisticQueue, ProbQueue, QueueModel, RiskAverseQueue
from .sim import Sim, Strategy
from .strategies import SimpleMM, SpreadQuoter, naive_funding_adj
from .synth import (Basis, Factor, Flow, Leg, Market, Regime, SynthMarket, krx_style, simulate, single, tweak,
                    usd_spread)

__version__ = "0.2.0"
