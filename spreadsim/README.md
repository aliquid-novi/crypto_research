# spreadsim

A small market simulator and backtester for spread trading research: one leg quoted passively, the
other hedged by taking, with a perp that pays funding. It is about 3,000 lines of plain Python that you
are meant to read and change.

It does two jobs with one engine:

- **Synthetic markets with a known truth.** A generator produces order books for two or three linked
  instruments from latent prices you control. Plant an effect, then check whether your measurement
  finds it.
- **Replay of real data.** The same engine replays your own level-2 updates and trades.

## Start

```bash
pip install -e .            # needs numpy and pandas; matplotlib for the two plot helpers
python examples/01_quickstart.py
pytest                      # 44 tests, about 10 seconds
```

```python
import spreadsim as ss

mk = ss.simulate(ss.usd_spread(), duration=3600, seed=1)             # one hour, SPOT vs PERP
strat = ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9), clip=10)  # quote SPOT, hedge on PERP
res = mk.run(strat, latency=ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003))

ss.summary(res)             # headline numbers
ss.pnl_attribution(res)     # spread capture, hedge cost, adverse selection, inventory, fees, funding
ss.markout_curve(res, maker=True)
```

One simulated hour (about 425,000 market events) generates in 2 seconds and replays in 1 to 2.

## How it works

Two clocks. The exchange sees a market event at its timestamp. Your strategy sees it `feed` seconds
later, on its own copy of the book. An order you send arrives `entry` seconds later, and you hear the
result `response` seconds after that. Everything a strategy can read is the delayed view.

```
market event --> exchange book --+--> fills on our resting orders --(response)--> strategy
                                 +--(feed)--> our book --> strategy --(entry)--> exchange
```

A resting order of ours fills when (see `venue.py`):

1. a trade prints through our price;
2. a trade prints at our price and is bigger than the quantity ahead of us (the queue model decides
   what "ahead of us" is);
3. new opposite-side liquidity appears at or through our price;
4. a trade empties our level exactly (`fill_on_clear`, on by default).

A market event is a tuple `(ts, kind, instrument_index, a, b, c)`. The four kinds (`DEPTH`, `TRADE`,
`INDEX`, `FUNDING`) are described at the top of `core.py`.

## What you can change, and where

| To study | Change | How |
| --- | --- | --- |
| Queue position | `QueueModel.on_cancel` in `models.py` | Pass `queue=RiskAverseQueue()`, `ProbQueue(power)`, `OptimisticQueue()`, or your own subclass. A dict gives one per instrument. |
| Latency | `Latency.feed / entry / response` | `ConstantLatency`, `LoadLatency` (jitter, spikes, slower when the venue is busy), or your own function of the simulator state. |
| Liquidity and order flow | `Flow` fields in `synth.py` | `ss.tweak(market, "SPOT", flow__target_depth=300, flow__snipe_delay=0.01, ...)` |
| Which venue leads | `Leg.lag` | `ss.tweak(market, "SPOT", lag=1.0)` |
| The spread itself | `Basis` (mean, sd, half-life), `Flow.impact_ticks`, `Flow.impact_half_life` | Edit the preset or build your own `Market`. |
| Regimes | `Regime` multipliers for volatility, depth, flow and basis half-life | `Market(regime=Regime(...))` |
| Tick and lot size, fees | `Instrument` | `Instrument("X", tick=0.05, lot=1, maker_fee=..., taker_fee=...)` |
| Funding rule | `FundingCalc` | interval, sample spacing, `weights="equal"` or `"linear"`, clamp, cap, multiplier |
| Fill rules | `Venue` in `venue.py` | Four numbered rules, about 100 lines. |
| The strategy | subclass `Strategy` | `examples/03_write_your_own.py` uses every callback. |

Strategy API on `sim`: `book(inst)`, `position(inst)`, `open_orders(inst)`, `submit / buy / sell`,
`take` (IOC), `cancel`, `cancel_all`, `timer(delay, tag)`, `funding(inst)`, `index(inst)`,
`log(key, value)`, `msg_rate(inst)`, `now`. For research only: `true_book`, `true_position`, and the
`x_` fields on an order (`order.x_ahead` is its queue position in the model).

## The spread parts

**Legs and currencies.** Any number of instruments run in one simulation. Each has its own tick, lot,
fees, book, queue model and latency. A leg quoted in another currency names the instrument that
converts it (`fx_inst`), so PnL is in one base currency. `krx_style()` is a three-leg example: a
KRW stock with a very large tick that lags, a USDT perp that leads, and USDTKRW to hedge the currency.

**What makes the spread move** in the synthetic markets:

- a slow mean-reverting premium on the perp (`Basis`);
- flow pressure: market orders push one leg's price and the push decays (`impact_ticks`,
  `impact_half_life`), opening a gap to the other leg that closes again;
- lead-lag: one leg follows the shared price with a delay (`lag`);
- funding: optionally the perp dips before a settlement and rebounds after it
  (`funding_passthrough`, `funding_anticipation`).

**Funding.** `FundingCalc` runs the venue formula: premium samples from impact bid and ask against
the index, a weighted average, the clamp around the interest rate, payment on the position held at
settlement. A strategy reads it through `sim.funding("PERP")`:

| Call | Meaning |
| --- | --- |
| `.predicted()` | the rate if the average premium stayed where it is now |
| `.avg()` | average premium so far this interval |
| `.locked()` | (part of the final average already fixed, share of weight still to come) |
| `.time_to_settle(sim.now)` | seconds to the next settlement |
| `.baseline()` | the rate paid whenever the average ends in the flat zone |

`ss.expected_rate(calc, mu, sigma)` prices the kinked payoff for a normal final average.
`SpreadQuoter(funding_adj=f)` takes any function `f(sim, strategy)` returning a log shift of fair
value. `naive_funding_adj` shifts by the full predicted rate; it is the baseline to beat.
`examples/05_funding.py` tracks the running prediction against what settles, including an interval
where a prediction of about twice the baseline rate settles at baseline.

The preset markets use a 30-minute funding interval with a rate multiplier of 8 so that funding
matters inside a one-hour run. Set real values when you replay real data.

## Reports

All in `analytics.py`, all returning DataFrames or dicts. Positive is always good for us.

| Function | Answers |
| --- | --- |
| `pnl_attribution(res, horizon)` | Spread capture, hedge cost, adverse selection, inventory, fees, funding, per leg and in total. The components sum to the PnL exactly. |
| `markouts`, `markout_curve` | Per-fill and average markouts by horizon, against the mid or (synthetic only) the latent fair price. Mean, median, 10th and 90th percentile. |
| `slippage(res, tag="hedge")` | Per taker order: cost from the delay, cost from crossing, total. |
| `spread_pairs(res)` | Each passive fill with its hedge: edge on the quote, hedge cost measured from the moment of the quote fill, hedge delay, net. |
| `maker_stats(res)` | Fill share, queue position on joining, time to fill, cancel rejects. |
| `latency_stats(res)` | p50, p95, p99 of each hop as experienced. |
| `fair_value_error(res, key, inst)` | A fair value you logged against the latent truth. |
| `summary`, `compare`, `sweep` | One row per run; `sweep` runs a parameter grid over several seeds and reports standard errors. |

Raw records: `res.fills`, `res.orders`, `res.bbo[inst]`, `res.trades[inst]`, `res.equity`, `res.funding`,
`res.logs[key]`, `res.truth`.

## Testing a hypothesis

`experiments/markout_rule.py` is a worked test of one idea ("push the clips out when the first clips'
markouts are negative"), and `experiments/markout_rule_costs.py` tests its blind spot. Both write an
HTML report next to themselves. `experiments/template.py` is the same structure with the blanks left open.
The parts:

| Piece | What it is for |
| --- | --- |
| `lab.Experiment(worlds, variants, seeds, ...)` | Runs every variant on exactly the same markets in each world. `runs.paired("pnl", baseline)` gives the mean paired difference, its standard error, t and seeds won. `workers=2` runs in parallel on Linux and macOS. |
| `controllers.py` | Rules that move the ladder while the strategy runs: `MarkoutFeedback` (the rule above), `VolShift` (volatility only). A new placement idea is a ten-line `Controller` subclass. `SpreadQuoter(shift_ticks=k)` is the static version. |
| `diagnostics.fill_table(res, inst, hedge=...)` | One row per passive fill: markouts against its own mid (`m5`) and spread markouts with the hedge leg's move removed (`s5`), worst and best excursion, how many ticks further out it would still have filled, volatility state, clip, queue, book lean. |
| `diagnostics.shift_counterfactual(ft)` | "What if every clip had been k ticks further out?" from one run's fills: share of fills kept, their markouts, the value of the fills lost, net per fill. |
| `diagnostics.conditional(df, by, cols)` | Mean, standard error, median and count of any column within groups of any other. |
| `report.Report` and `report.plot_*` | A self-contained HTML report: signal-against-outcome scatter, value curves per world, markouts of kept and lost fills, the controller's path, paired-difference bars. |

Use spread markouts (`s...`) for a hedged trade. The leg's own markout (`m...`) also contains the market
move that the hedge cancels, so it is noisier and says less.

## Your own data

`examples/06_your_data.py` shows the exact table shapes. In short:

```python
events = ss.events_from_frames(instruments, depth=depth_df, trades=trades_df, index=index_df, funding=funding_df)
res = ss.Sim(instruments, events, strategy, latency=..., queue=..., funding={...}, funding_from="events").run()
```

`depth` rows are `t, inst, side, price, qty` (the new size at that price), starting with a full
snapshot. `trades` rows are `t, inst, side, price, qty` with the aggressor's side. If you only have
periodic top-N snapshots, `ss.snapshots_to_frame` diffs them into `depth` rows. With real data there
is no latent truth, so `vs="ref"` and `fair_value_error` are unavailable; everything else works.

## What it does not do

- **Your orders do not move the market.** Resting orders are overlaid on the data. Liquidity you take
  stays gone only until that level next updates. Results get less believable as your size grows
  relative to the book.
- **Queue position is an assumption.** Level-2 data cannot say where a cancel came from. Run the
  pessimistic and optimistic models as bounds; a conclusion that flips between them is not a conclusion.
- **The synthetic market is a toy.** Order flow is random apart from the mechanisms listed above, and
  every parameter is invented. A plain market maker makes money on the large-tick leg, which no real
  market allows for long. Absolute PnL means nothing. What you can trust is a difference between two
  configurations on the same seeds, and whether an estimator recovers what was planted.
- **Not modelled:** trading sessions, auctions and halts (so no KRX close), margin and liquidation,
  borrow and financing costs on a stock leg, order amendment, hidden or iceberg orders, gateway
  rejects and rate limits, self-trade prevention.
- **Funding predictions reach the strategy with no delay**, and the tracker assumes the venue's
  index is the one in the event stream.
- **Speed.** Pure Python, roughly 300,000 market events per second with a timer-driven strategy.
  Fine for hours of data; slow for full days of a busy book.
- **Not checked against a real exchange.** The matching rules are tested against hand-worked cases,
  not against live fills. Before trusting a number for a real decision, compare simulated fills with
  your own fill history on the same day.

## Relation to hftbacktest

[hftbacktest](https://github.com/nkaz001/hftbacktest) replays real tick data with feed and order
latency, queue-position models and multi-asset, multi-exchange support, at far higher speed (Rust
core, Numba strategies). It makes the same no-market-impact assumption. Its README and order-fill
documentation do not mention funding payments or a spread fair value, so you would add those yourself.

Use this package to learn the mechanics, change the internals and test ideas where the truth is known.
Move to hftbacktest when you need to replay days of real tick data. The concepts carry over directly:
feed and order latency, risk-averse and probability queue models, and the fill rules are the same ideas.

## Files

```
spreadsim/core.py        Instrument, Order, Fill, event layout
spreadsim/book.py        price-level book: mid, microprice, imbalance, depth, impact price
spreadsim/venue.py       matching: the four fill rules, taker walk, cancels
spreadsim/models.py      queue models and latency models
spreadsim/funding.py     FundingCalc, premium index, expected rate under uncertainty
spreadsim/sim.py         the event loop, Strategy base class, recorder
spreadsim/synth.py       synthetic market generator and presets (usd_spread, krx_style, single)
spreadsim/strategies.py  SimpleMM, SpreadQuoter, naive_funding_adj (plain baselines)
spreadsim/analytics.py   Result, attribution, markouts, slippage, pairs, sweep
spreadsim/diagnostics.py per-fill table, shift counterfactual, conditional tables
spreadsim/controllers.py rules that move the ladder while a strategy runs
spreadsim/lab.py         Experiment: worlds x variants x seeds with paired statistics
spreadsim/report.py      experiment figures and the HTML report
spreadsim/data.py        loaders for your own data
spreadsim/plots.py       plot_run, plot_markouts
examples/                six scripts, 01 to 06
experiments/             markout_rule.py, markout_rule_costs.py (worked hypothesis tests), template.py
EXERCISES.md             research drills that use the simulator
tests/                   hand-worked matching cases, venue funding examples, generator checks
```
