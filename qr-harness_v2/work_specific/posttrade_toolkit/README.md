# posttrade: exchange-agnostic post-trade analytics

Markouts, adverse selection, spread-trade implementation shortfall, microstructure models, latency and
research statistics, for crypto perps and futures on any venue (Binance, Hyperliquid, CME, …). It covers both
single passive fills and two-leg round trips (decision → leg fills → exit fills).

Runs fully **offline**: numpy, pandas, scipy, statsmodels and matplotlib are required; lifelines and hmmlearn
are optional, with built-in fallbacks. There are no network calls. Every formula, unit and sign convention is
in the docstrings.

```
python run.py test        # 43 tests on synthetic data with known injected effects
python run.py demo        # builds and executes notebooks 00-07 on synthetic data -> notebooks/executed/
python run.py notebooks   # rebuild notebooks/*.ipynb from the notebooks/*.py sources
make test / make demo     # same, where make exists
```

The demo uses its own cache (`data_cache_demo/`) and always uses synthetic data, so it never overwrites the
cache built from your real data.

## Conventions (everywhere)

| | |
|---|---|
| time | int64 nanoseconds since epoch, **UTC**. `schemas.prepare` converts datetimes or s/ms/us integers |
| side `d` | +1 = **our** buy, −1 = our sell. Market trades: `aggressor_side` +1 = the taker bought |
| markout | `markout_h = d × (m_{t+h} − p)`, in **bps of p** (= bps of notional). **Positive = good for us** |
| fee | quote ccy, **positive = paid**, negative = rebate |
| funding | `rate > 0` ⇒ longs pay. `payment` = our cash, **positive = received** |
| spread trades | `intended_qty` signed in leg-A units (+ = buy A / sell B); `hedge_ratio` = units of B per unit of A |
| join rule | h > 0: first reference update **strictly after** t+h. h ≤ 0: last update **strictly before** t+h (so h = 0 is the book our fill hit). Horizons crossing a session closure are NaN |

## Layout

```
posttrade/
  config.py      COLUMN_MAP, SIDE_MAP/LIQUIDITY_MAP, VenueSpec + SessionSpec (CME Globex example), FEE_SCHEDULES, horizons, LEAD_MAP
  schemas.py     table schemas, prepare()/validate() (lists every problem), availability_report(), add_notional()
  synthetic.py   synthetic day with known effects + Hawkes/HMM/Kalman/Huang-Stoll simulators
  hygiene.py     reconciliation, fee/funding recompute, dedupe, Brownlees-Gallo, book sanity, feed gaps, MAD, CUSUM,
                 signature plot / noise variance, NTP offset + drift, staleness, forward-fill-only resampling
  reference.py   mid, weighted mid, Stoikov microprice, OFI, composite spread value, lead-venue mid, trailing vol
  markouts.py    wall-clock + event-time markouts, multi-reference, decomposition, fees, vol-scaling, aggressor-event
                 aggregation, taker-view spreads, conditional curves with cluster bootstrap, day-by-day processing
  adverse.py     AS ratio, pre-fill drift, late cancels, full vs partial, counterparty concentration, per-taker toxicity
  spreads_tca.py shortfall bridge per decision/leg (sums exactly), aggregates, hung legs, missed decisions, cost share,
                 decision- vs fill-anchored markouts, MAE/MFE
  models.py      markout regressions (OLS-NW / Huber / quantile), toxicity logit + avoided-loss backtest, fill intensity,
                 hazards (Cox or discrete-time logit), EV per quote, Heckman / IPW, 2-state HMM, Hawkes 1-d/2-d + GOF,
                 response function / propagator, Kyle λ, Roll, Huang-Stoll, Hasbrouck VAR, square-root impact
  dynamics.py    OU fit (with lag option), model vs empirical alpha decay, alpha-decay regression, causal Kalman β
  latency.py     per-hop latency, by hour / activity / vol, latency regressions, restart discontinuity
  stats.py       Newey-West, required n, effective n, E[max Sharpe], PSR / DSR, block + cluster bootstrap, BH, ResearchLog
  plots.py       matplotlib helpers (fixed colour order; diverging blue = good / orange = bad)
  store.py       cache hand-off between notebooks
notebooks/       00_plug_your_data … 07_research_stats (.py source + .ipynb)
tests/           pytest on synthetic data
run.py, Makefile
```

## What "tested" means

`synthetic.generate_day` builds a coherent session across three venues: our venue VA (leg A, where we also
quote), VB (leg B) and VL (the lead, showing VA's true price 2 s earlier). The spread is OU with a known θ.
It injects adverse selection of known size on our passive fills, toxic taker addresses, late cancels,
spread round trips with known delay/legging/partial/missed legs, funding, rising latency and hygiene
defects. The tests check that the code recovers them:

| check | result on the 8 h test day |
|---|---|
| markout curve: toxic − normal adverse move at 60 s ≈ −J (6 bps); h = 0 equals spread captured; no adverse move at h < 0 | ✓ |
| decomposition identity; effective = realised + impact | exact |
| shortfall components sum to realised − paper; fee component = −Σ fees; recovered legging gap = injected | exact (max error ~1e-13) |
| OU θ: lag-1 on the true spread within 3 SE; on the observed spread a 30 s lag removes most of the noise bias | ✓ (see weak points) |
| Kalman β RMSE < 0.03 and no look-ahead (changing future y leaves past priors unchanged) | ✓ |
| Hawkes μ, α, β (1-d, also on the lead-venue tape) and bivariate A matrix | ✓ |
| HMM: filtered state accuracy > 88% | ✓ |
| toxic addresses ranked top 5 by shrunk taker markout | ✓ |
| Newey–West SE > 3× OLS SE on overlapping 300 s returns | ✓ |
| E[max SR] for N = 100, σ = 0.5 ≈ 1.26 | 1.265 |
| dropped fill → position break; wrong fees, funding row, duplicates, bad ticks, crossed / zero-size / off-tick book, feed gap | all flagged |

## Plugging in real data

Everything happens in **`notebooks/00_plug_your_data.ipynb`**, in the cell marked **EDIT HERE**. Set
`USE_SYNTHETIC = False`, fill the `load_*` functions, then run the notebook. It validates every table, prints
**every** problem, shows which analyses are available, and caches the tables for 01–07.

### Which tables unlock which analyses

| table | required columns | unlocks |
|---|---|---|
| `fills` | fill_id, order_id, ts, venue, symbol, side, price, qty, notional, fee, liquidity | markouts (with book), fee recompute, dedupe, regressions, toxicity, HMM, Hawkes |
| + `counterparty` | | counterparty concentration / Lorenz |
| + `decision_id`, `leg`, `leg_role` | | spread round trips (with `decisions`); `spreads_tca.link_fills` can guess them by time, but check the result |
| `book_l1` | ts, venue, symbol, bid_px, bid_sz, ask_px, ask_sz | every reference price, markouts, microprice, OFI, book sanity, feed gaps, signature plot |
| `market_trades` | ts, venue, symbol, price, qty, aggressor_side | event-time markouts, bad ticks, taker spreads, Hawkes cross-excitation, response / propagator / Kyle / Roll / Huang-Stoll / VAR |
| + `buyer`, `seller` | | venue-wide per-taker toxicity (Hyperliquid publishes addresses; most venues do not) |
| `orders` | order_id, venue, symbol, side, price, qty, ts_sent, ts_ack, ts_cancel_sent, ts_done, final_status | AS ratio, late cancels, full/partial, fill intensity, hazards, EV per quote, Heckman/IPW, latency |
| + `ts_md_recv`, `ts_signal` | | per-hop latency decomposition |
| `decisions` | decision_id, ts, pair, signal, spread_fair_value, intended_qty, hedge_ratio, leg_A/B_symbol, leg_A/B_venue, exit_rule (+ optional ts_exit_decision) | implementation shortfall, alpha decay, OU / Kalman on the pair |
| `funding` | ts, venue, symbol, rate, mark, position_at_settlement, payment | funding recompute, carry in the shortfall bridge |
| `positions` | ts, venue, symbol, position | position reconciliation |
| `instruments` (optional) | venue, symbol (+ tick_size, multiplier, quote_ccy, kind) | real tick sizes (off-tick checks, microprice), contract multipliers (CME), notional |

### COLUMN_MAP edits

`COLUMN_MAP[table]` maps **your** column name → the schema name, listing only the columns that differ. Also set:

- `TS_UNITS`: how integer timestamps are encoded per table (`'s'`, `'ms'`, `'us'`, `'ns'`). Datetime columns are
  detected; naive datetimes are treated as `config.NAIVE_TZ` (UTC).
- `EXTRA_SIDES`: any side labels not already in `config.SIDE_MAP` (which covers buy/sell, b/s, B/A, bid/ask, ±1, BOT/SLD…).
- `EXTRA_VENUES`: a `VenueSpec` for each new venue: kind, trading `sessions` (None = 24/7), funding interval,
  whether it publishes addresses, and its fee model.
- `EXTRA_FEES`: your fee tier (`{'model': 'bps' | 'per_contract', 'maker': x, 'taker': y}`, negative = rebate).
  The numbers in `config.py` are placeholders.
- `LEAD_MAP`: `(lead_venue, lead_symbol) → (our_venue, our_symbol)` for lead-venue markouts. Use
  `reference.lead_mid(scale=…)` when the contracts differ in size or currency.

### Common pitfalls

- **Exchange vs local timestamps.** Use exchange time as `ts` for markouts (it orders events the way the venue
  saw them) and keep your local receive time as `ts_local`. Latency hops are only meaningful on one clock: a local
  send time paired with an exchange ack time measures clock offset plus latency. Estimate the offset and drift
  with `hygiene.ntp_offsets` first. Venues stamp different events (matching-engine time, gateway time, or
  publish time), so check which one yours uses.
- **Integer timestamp units.** Binance is in ms, many feeds in µs or ns, and CME MDP in ns. A wrong unit
  shows up as dates in 1970 or year 50000 in the 00 summary.
- **Fee sign.** The toolkit wants *positive = paid*. Some exports report rebates as positive or commissions
  as negative. After loading, check 00's `median_fee_bps_maker` and `_taker` against your tier. Futures fees
  are usually per contract (`model: 'per_contract'`), not bps.
- **Side encoding.** `side` is OUR direction for fills and the taker's for market trades. Binance's
  `isBuyerMaker = True` means the **taker sold**: convert it with
  `aggressor_side = np.where(is_buyer_maker, -1, 1)` before mapping. Hyperliquid uses `B`/`A` (bid side =
  buy, ask side = sell), which `SIDE_MAP` already handles.
- **Multi-fill sweeps.** One taker order sweeping several of your levels (or partially filling one order
  several times at the same timestamp) produces several fills sharing a ts. Averaging them as independent
  overweights sweeps, which are often the toxic events. Use `markouts.aggregate_events` (02 does) and cluster
  bootstrap CIs. Also deduplicate on (venue, trade_id, order_id, side) before anything else.
- **Funding sign.** `rate > 0` means longs pay shorts on most perps, but check your venue. The ledger's `payment`
  must be your cash (+ = received). 01 recomputes `−position × mark × rate × multiplier` and flags any
  disagreement, and a sign flip shows up as every row flagged. Interval lengths differ (1 h on Hyperliquid, 8 h
  on Binance by default). Dated futures have no funding: leave the table out and carry is 0.
- **CME sessions.** Add `VenueSpec("CME", "future", config.CME_GLOBEX, …)`. Markout horizons that cross the
  daily 16:00–17:00 CT halt or the weekend become NaN instead of carrying across the gap, and feed-gap
  detection ignores those closures. Add exchange holidays to `SessionSpec.holidays`. DST is handled through the
  `America/Chicago` time zone.
- **Contract multipliers.** CME qty is in contracts. Put `multiplier` (contract size) in `instruments`, then
  `schemas.add_notional` computes `price × qty × multiplier`. The shortfall bridge and funding recompute use the
  multiplier. Markouts in bps don't need it.
- **Different symbols across venues.** Pair your venue's symbol with the lead's in `LEAD_MAP`, and quote both in
  the same price units (`scale=` for contract size / inverse contracts / FX).
- **Big files.** For millions of fills, use `markouts.daily_markout_means` (one day at a time), and load one
  day or instrument at a time in your `load_*` functions.

## Known weak points (read before trusting a number)

- **Overlap.** Markouts of fills closer together than the horizon share price paths. CIs use a 60 s cluster
  bootstrap and regressions use Newey–West, but the HMM has no such fix: on the synthetic data, where toxicity
  is iid, it still reports persistent states (diagonal ≈ 0.9). Refit at non-overlapping horizons before
  believing regimes.
- **Random-side overlap.** For two-sided quoting, overlapping markouts of fills on *opposite* sides are
  negatively correlated, so overlap inflates SEs less than you might expect. Same-side streaks (inventory
  skew) bring it back. Newey–West handles both, so keep the default lags.
- **OU θ.** With few half-lives in the sample, θ has a large sampling SE (≈ √(2θ/T); ~17% for 8 h of a
  5 min half-life). Measurement noise (legs observed at slightly different times, bid/ask bounce) biases a
  lag-1 AR fit upward (+57% on the test day). Use `ou_fit(..., lag=k)` and check θ is stable across lags (04
  shows a table).
- **Bad-tick filter.** Brownlees–Gallo flags ~0.2% of good trades in fast markets with the defaults (k = 50,
  γ = 2 ticks). It is a flag, not a filter.
- **Full-sample fits.** The microprice, toxicity scores and HMM are fitted on the sample you pass. To act on
  them, fit on past days and evaluate on later days. The toxicity backtest uses one time split.
- **Selection corrections.** Heckman relies on distance-from-mid as an exclusion restriction, which is
  arguable. IPW can be dominated by a few rarely filled far quotes; 05 prints the Kish effective n.
- **Square-root impact.** It is only identifiable if your metaorders are a meaningful fraction of daily
  volume. Small clips give noise.
