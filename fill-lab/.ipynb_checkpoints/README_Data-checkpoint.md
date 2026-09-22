# fill-lab — data for testing passive-fill and flattening strategies

## Read this first

**This data is synthetic.** I could not reach `data.binance.vision`,
`api.hyperliquid.xyz` or `zenodo.org` from the sandbox this was built in — all
three are blocked by egress policy. So rather than hand you nothing, I built a
generator whose **output schemas match Binance Vision's real public files
exactly**. Analysis code you write against these CSVs runs unchanged on the real
files. `fetch_real_data.sh` pulls those real files on any machine with normal
internet.

What synthetic data is good for, and it is genuinely good for this:

- **Validating machinery against known ground truth.** Does your fill-intensity
  estimator recover the intensity that generated the data? Does your censoring
  handling remove the bias you know is there? You cannot ask that of real data,
  because you never know the answer.
- **Sample size.** ~100k hypothetical orders, versus the ~28 flatten episodes
  you will have at work. Build and debug here, run the validated code there.
- **Controlled contrasts.** Two instruments differing only in tick size, so you
  can see which conclusions are properties of the market and which are
  properties of the tick.

What it is **not** good for, and this matters:

- It cannot tell you anything about your desk, your venues, or whether passive
  flattening beats crossing at Star Beta. Every parameter here was chosen by
  hand to be plausible. Nothing is fitted to anything.
- Any number you get out of it is a property of my assumptions. If a result
  here surprises you, the first hypothesis should be that I mis-specified the
  generator, not that you have found something about markets.

Treat it as a test harness, not as evidence.

---

## Files

```
gen_market.py          generator: regimes, Hawkes flow, impact, L1 book
replay.py              fill simulator + policy harness
fetch_real_data.sh     pull the equivalent REAL files from Binance Vision
SCHEMA.md              column dictionary for every CSV

data/                  preset "btc_perp"   — small tick (0.10 on 78,000)
data_wide/             preset "wide_tick"  — large tick (20.0 on 78,000)
  l1_book.csv          top of book        (Binance bookTicker schema)
  trades.csv           trade prints       (Binance aggTrades schema)
  episodes.csv         35 flatten episodes with deadlines
  order_outcomes.csv   the survival dataset — ~100k / ~42k hypothetical orders
  ground_truth.csv     latent regime + efficient price (1s grid)
  params.json          every generating parameter
```

6 hours per instrument. `ground_truth.csv` has no real-world counterpart — that
is the whole reason to use synthetic data.

---

## The two instruments, and why there are two

One ratio decides what a passive order is actually doing:

```
time for the price to cross one tick   ~  (tick / sigma_1s)^2
time for the touch queue to clear      ~  queue / volume_rate
```

Whichever is shorter is the mechanism that fills you.

| | `btc_perp` | `wide_tick` |
|---|---|---|
| tick | 0.10 | 20.0 |
| mean spread | 0.022 bps | 4.6 bps |
| daily vol | 1.8% | 1.0% |
| fills at touch that were **queue**-driven | **0.7%** | **81.5%** |

On `btc_perp` the price sweeps through your level long before the queue clears,
so queue position is close to worthless and almost all of the queue-value
literature does not apply. On `wide_tick` the queue binds, and that literature
is exactly on point. Same code, same generator, opposite conclusions.

Work out which of these your desk's instruments look like **before** deciding
which papers to read. That is a two-line calculation on your own data.

---

## Running it

```bash
# regenerate a tape
python3 gen_market.py --hours 6 --seed 7 --preset btc_perp --outdir data

# sweep hypothetical passive orders across it -> order_outcomes.csv
python3 replay.py --datadir data --cadence 3 --horizon 120 \
                  --offsets 0,0.25,0.5,1,2,4,8

# how much do conclusions depend on the cancellation assumption?
python3 replay.py --datadir data --cancel-rate 0.02 \
                  --out data/order_outcomes_cancel02.csv
```

Episode-level policies:

```python
from replay import *
tape = Tape("data")
eps  = load_episodes("data/episodes.csv")

res = [run_episode(tape, e, policy_passive_then_cross,
                   offset_bps=0.5, n_slices=8) for e in eps]
```

`policy_cross_immediately`, `policy_twap_cross` and `policy_passive_then_cross`
are deliberately crude baselines, not recommendations. Write your own — the
interface is a function returning a list of child orders.

---

## The fill model, and the assumptions you are inheriting

A resting **sell** at price `p` placed at `t0` behind queue `Q0` fills at the
earlier of:

1. **Queue rule** — cumulative buy-initiated volume at prices `>= p` since `t0`
   exceeds `Q0`.
2. **Sweep rule** — any trade prints strictly above `p`. The market cannot print
   through your level while you are still resting on it; clearing you is what
   let it print.

Symmetrically for buys. Three assumptions worth arguing with:

- **Cancellations are ignored by default** (`cancel_rate=0.0`). In reality
  orders ahead of you cancel, pulling you up the queue and making fills
  *faster*. So the default is a conservative lower bound on fill probability.
  `--cancel-rate` exposes this. The gap between `0.0` and a plausible rate is
  real uncertainty in any passive-fill study, and measuring it on your own venue
  is more useful than picking a number.
- **Queue ahead is the size at your one price level**, not cumulative depth
  between the touch and you. Getting this wrong by using cumulative depth was an
  actual bug during construction: on a 0.1 tick, 4 bps out is 311 ticks, the
  model demanded a 52 BTC queue, and nothing ever filled.
- **No self-impact.** Your order is invisible to everyone else. Fine at small
  size, wrong at desk size, and it biases passive strategies to look better than
  they are.

**Markout sign convention:** positive = favourable to you, both sides.
```
sell:  (mid(t_fill) - mid(t_fill+h)) / mid(t_fill) * 1e4
buy:   (mid(t_fill+h) - mid(t_fill)) / mid(t_fill) * 1e4
```

---

## The column that most analyses leave out

`slippage_vs_place_mid_bps` is computed for **every** order, filled or not. If
an order did not fill by the horizon, it is priced at what crossing would have
cost at the horizon (`cross_price_at_horizon`), because you still had to get the
trade done.

`markout_*` columns exist only for fills.

Comparing a mean over `markout_*` with a mean over `slippage_vs_place_mid_bps`
is the whole Lo–MacKinlay–Zhang point in one line of pandas: conditioning on
fills is conditioning on a non-random subsample. Do that comparison first,
before anything else, on both instruments.

---

## Porting to real data

`fetch_real_data.sh` downloads real Binance USD-M futures daily files. They have
the same column names as these CSVs, so:

```python
tape = Tape("real_data/BTCUSDT-2026-09-18")   # unchanged
```

Two real-data differences to handle:
- Some Binance archive files ship without a header row. The script checks and
  prepends one.
- Real `bookTicker` is a throttled stream, so it under-samples the book. These
  files mimic that (published on every trade plus a 100ms heartbeat) rather than
  pretending you see every change.

There is no `ground_truth.csv` for real data. That asymmetry is the point:
validate the estimator here, then run it there.

---

## Questions this data can answer, and the ones it cannot

**Can:**
- How does fill probability vary with distance, book imbalance, trade intensity
  and volatility? (~100k orders, plenty of power.)
- Does your survival model recover the known generating intensity?
- How large is the bias from analysing only filled orders?
- Does queue position matter at this tick size? (`fill_reason`.)
- How sensitive is every conclusion above to `cancel_rate`?

**Cannot:**
- Whether any of this holds on Binance or Hyperliquid. Nothing here is fitted.
- Whether one episode-level policy beats another. With 35 episodes the
  standard error on mean slippage is around 1 bp, which swamps the gaps between
  the baseline policies. That is not a flaw in the data — it is the sample-size
  problem you will have at work, reproduced honestly. Note which questions
  survive it and which do not.
