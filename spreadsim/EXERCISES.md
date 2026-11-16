# Drills

Each drill is a question, the knob that plants the answer, and the report that measures it. None of
them comes with the answer. The routine is the same every time:

1. **Predict.** Write the number you expect and why, before running anything.
2. **Measure** from the observable data only (`res.bbo`, `res.fills`, `res.orders`), over several seeds.
3. **Grade** against `mk.truth` or against the knob you set.
4. **Change the knob** and repeat. A method that works should track the change in size and direction.

Use `ss.sweep(..., seeds=range(4))` and read the `_se` columns, or copy `experiments/template.py` for
anything with more than one world. A difference under about two standard errors is noise.

## A. Trade optimisation (the quoter leg and the hedge)

| # | Question | Knob | Measure with |
| --- | --- | --- | --- |
| A1 | Where does latency damage show up: hedge cost, adverse selection or spread capture? Does the answer change with the benchmark you measure the hedge against? | `ConstantLatency`, each hop separately | `pnl_attribution`, `slippage`, `spread_pairs` |
| A2 | Which hop matters most: feed, order entry or response? | one hop at 100 ms, the others at 1 ms | `sweep` |
| A3 | If latency rises when the market is busy, how much worse is that than the same average latency spread evenly? | `LoadLatency(load=...)` against a constant with equal mean | `latency_stats`, `markout_curve` |
| A4 | Can you quote a tick better? How many fills do you gain or lose, and what do the extra fills cost? | `levels_bps` in `SpreadQuoter`, tick size in `Instrument` | `maker_stats`, `markouts(unit="ticks")` |
| A5 | How much of your result depends on the queue assumption? | `RiskAverseQueue` against `OptimisticQueue` | `sweep` |
| A6 | Fill against touch: are the times you got filled worse than the times your price was merely touched? | none; needs your own code on `res.bbo` and `res.orders` | your function |
| A7 | How often does a cancel lose the race to a fill, and what do those fills mark out at? | entry latency, `Flow.snipe_delay` | `res.orders.cancel_reject`, `markouts` on those fills |
| A8 | Where is the best single clip, and how flat is the top? | `levels_bps=(x,)` over a grid | `sweep` on `pnl`, with standard errors |
| A9 | Which leg should be quoted and which hedged? What makes the worse choice worse? | `SpreadQuoter("PERP", "SPOT")` against `("SPOT", "PERP")` | `pnl_attribution` |
| A10 | Define a "good fill" for a spread trade so that it separates a wrong fair value from a badly timed entry. Does your definition agree with the latent truth? | `halflife` in `SpreadQuoter` (makes fair value worse or better) | `fair_value_error`, `markouts(vs="ref")` against `vs="mid"` |

## B. Market making (one instrument: `ss.single()`)

| # | Question | Knob | Measure with |
| --- | --- | --- | --- |
| B1 | Does top-of-book imbalance predict the next mid move? By how much, and over what horizon? | `Flow.imbalance_sens`, `Flow.impact_ticks` (set both to 0 as a placebo) | your function on a strategy that logs `book.imbalance` |
| B2 | Is the microprice a better fair value than the mid? | `SimpleMM(fair="micro")` against `"mid"`, or log both | `fair_value_error`, `summary` |
| B3 | What is the fill probability within T seconds as a function of queue ahead on joining? | `Flow.target_depth`, tick size | `res.orders` (`ahead0`, `t_arrive`, `t_done`, `filled`) |
| B4 | At what market-order size does flow become adverse? | `Flow.impact_ticks`, `Flow.market_size_mean` | markouts bucketed by trade size (use `on_trade` to record the tape) |
| B5 | Split a market maker's PnL into spread capture, adverse selection and inventory. Which one does inventory skew change? | `SimpleMM(skew_ticks=...)` | `pnl_attribution` |
| B6 | Large tick against small tick: how do queue length, fill rate and adverse selection per fill differ? | `ss.single(tick=0.05)` against `tick=0.01` | `maker_stats`, `markout_curve` |
| B7 | How should the quoted half-spread scale with volatility? | `Regime.vol_mult`, `Factor.vol` | `sweep` over `half_spread_ticks` within each regime |

## C. Medium-frequency relative value

| # | Question | Knob | Measure with |
| --- | --- | --- | --- |
| C1 | Which leg leads, and by how long? | `Leg.lag` (see `examples/04_plant_and_recover.py`) | your estimator on gridded mids |
| C2 | What is the half-life of the spread, and how wide is the uncertainty from one hour of data? | `Basis.half_life`, `Flow.impact_half_life` | your estimator against `truth["basis"]` and `truth["impact"]` |
| C3 | The spread has a slow part and a fast part. Can you separate them from prices alone? | `Basis` against `Flow.impact_ticks` | compare with `truth` |
| C4 | Does thin liquidity change how fast the spread reverts? Plant it, then find it. | `Regime.basis_halflife_mult`, `Regime.depth_mult` | reversion estimates split by a liquidity measure you define, against `truth["regime"]` |
| C5 | How much of a deviation reverts, by size of deviation? Is the curve flat? | give one leg its own jump-only factor: `Factor("J", start=1.0, vol=0.0, jump_rate=1/300, jump_sd=1e-3)` and `factors=("U", "J")` on that leg | the size-binned reversion curve from the research note |
| C6 | For one signal, at what horizon does its predictive power die? | any of the above | correlation with forward returns at several horizons |

## F. Funding and fair value

| # | Question | Knob | Measure with |
| --- | --- | --- | --- |
| F1 | Replicate the settled rate from the premium samples. Does your number match `res.funding.rate`? | `FundingCalc(weights=...)` | a strategy that logs `calc.last_premium` each sample |
| F2 | How good is `.predicted()` as a forecast of the settled rate at 20, 10, 5 and 1 minutes to go? | `Basis.sd`, `Basis.half_life`, `weights` | log the prediction, compare with settlement, many seeds |
| F3 | How much of the funding payment is given back by the price move across the settlement? | `funding_passthrough` (0, 0.5, 1) | an event study of the perp mid around settlements |
| F4 | Beat `naive_funding_adj`. What weight does the prediction deserve at each time to go? | `funding_adj=` your function | `sweep` on `pnl` and `funding`, same seeds |
| F5 | When the average premium sits near the edge of the flat zone, what is the expected rate? | `Basis.mean` near 6 bps | `ss.expected_rate` against the plug-in `calc.predicted()` |

## Three legs (`ss.krx_style()`)

| # | Question | Knob | Measure with |
| --- | --- | --- | --- |
| K1 | How much PnL variance does the currency hedge remove? | `SpreadQuoter(fx=None)` against `fx="USDTKRW"` | `res.equity`, `pnl_attribution` |
| K2 | The hedge leg loses money on average. Is that a problem to fix or the price of the trade? What would you need to see to tell the two apart? | latency, `hedge_cross_ticks`, which leg leads | `spread_pairs`, `pnl_attribution` by leg |
| K3 | With a 14 bps tick on the stock, what decides whether a passive fill is good? | `Flow.target_depth`, `Leg.lag` | `maker_stats`, `markouts` |

## When a drill is done

Run the same code on real data. The simulator told you whether the method can find an effect that
exists; only the data can tell you whether the effect exists.
