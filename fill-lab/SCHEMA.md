# Column dictionary

All timestamps are **epoch milliseconds**. All prices in quote currency (USDT),
all quantities in base currency (BTC). Offsets and markouts in **basis points**.

---

## `l1_book.csv` — Binance USD-M futures `bookTicker` schema

Identical column names and order to the real Binance Vision daily files.

| column | notes |
|---|---|
| `update_id` | monotonically increasing |
| `best_bid_price` | |
| `best_bid_qty` | size resting at the touch |
| `best_ask_price` | |
| `best_ask_qty` | |
| `transaction_time` | when the book changed |
| `event_time` | when it was published (equal here; they differ in real data) |

A **sampled** view of the book: published on every trade plus a 100ms
heartbeat, which is what a throttled real stream gives you. Do not assume you
see every book change — you do not in production either.

---

## `trades.csv` — Binance USD-M futures `aggTrades` schema

| column | notes |
|---|---|
| `agg_trade_id` | |
| `price` | prints at the touch that was consumed |
| `quantity` | |
| `first_trade_id`, `last_trade_id` | one print per row here |
| `transact_time` | |
| `is_buyer_maker` | **`true` = the buyer was the maker = the aggressor SOLD** |

That last convention inverts more analyses than anything else in this dataset.
`is_buyer_maker == true` is a **sell**-initiated trade, and it is what fills a
resting **buy**.

---

## `episodes.csv` — flatten episodes

| column | notes |
|---|---|
| `episode_id` | |
| `start_time` | when you learn you hold the position |
| `deadline_time` | flat by here |
| `horizon_s` | `deadline - start`, seconds (480) |
| `side` | the side you must **trade** to get flat (`sell` = you are long) |
| `quantity` | base units to flatten |
| `arrival_mid` | mid at `start_time` — the natural cost benchmark |
| `arrival_bid`, `arrival_ask` | touch at `start_time` |

35 episodes per instrument. Deliberately few: this is the sample size you will
actually face, and treating it as adequate is the mistake to avoid.

---

## `order_outcomes.csv` — the survival dataset

One row per hypothetical resting limit order. This is the main file.

### Placement
| column | notes |
|---|---|
| `order_id` | |
| `place_time` | |
| `episode_id` | blank if the order falls outside any episode |
| `time_left_in_episode_s` | blank outside an episode |
| `side` | `sell` or `buy` |
| `offset_bps` | distance **outward from the same-side touch** |
| `limit_price` | rounded outward onto the instrument's tick grid |
| `distance_ticks` | `offset_bps` expressed in ticks |
| `placement` | `at_touch` / `behind` / `inside_spread` / `marketable` |
| `queue_ahead` | size modelled ahead of you at your level |
| `horizon_s` | how long it rests before being given up on |

### Market state at placement — your candidate features
| column | notes |
|---|---|
| `mid_at_place` | |
| `spread_ticks` | |
| `queue_imbalance` | `(bid_qty - ask_qty) / (bid_qty + ask_qty)`, so **positive = more size on the bid** |
| `trade_intensity_60s` | trades/sec over the trailing 60s |
| `realised_vol_60s_bps` | std of log mid returns over the trailing 60s |

### Outcome
| column | notes |
|---|---|
| `filled` | 1/0 |
| `censored` | `1 - filled`. **Right-censored at `horizon_s`** — the order may well have filled later. Survival methods exist for exactly this. |
| `fill_time` | blank if unfilled |
| `time_to_fill_s` | blank if unfilled |
| `fill_reason` | `queue` (volume cleared the queue ahead) or `sweep` (price traded through your level). The split tells you whether queue position matters on this instrument at all. |
| `mid_at_fill` | blank if unfilled |

### Markouts — **fills only**, positive = favourable
| column |
|---|
| `markout_1s_bps`, `markout_5s_bps`, `markout_30s_bps`, `markout_60s_bps` |

Blank for unfilled orders. Averaging these over fills only is the biased
statistic; that is deliberate, so you can measure the bias.

### Honest cost — **every order**
| column | notes |
|---|---|
| `cross_price_at_horizon` | what crossing at the horizon would have got you |
| `realised_price` | `limit_price` if filled, else `cross_price_at_horizon` |
| `slippage_vs_place_mid_bps` | signed, positive = better than the mid at placement. Defined for **all** orders. |

### Ground truth — has no real-data counterpart
| column | notes |
|---|---|
| `true_regime` | 0 calm / 1 normal / 2 stressed |

Use it to **score** a regime model, never as an input feature. If it appears on
the right-hand side of a fit, that fit is meaningless.

---

## `ground_truth.csv` — 1-second grid

| column | notes |
|---|---|
| `time` | |
| `regime`, `regime_name` | the latent Markov state |
| `efficient_price` | the unobservable price the book is pinned to |
| `observed_mid` | what a trader actually sees |
| `sigma_bps_per_sqrt_s` | true instantaneous volatility |
| `baseline_intensity_per_side` | true Hawkes baseline (excludes excitation) |
| `half_spread_ticks` | target half-spread for the regime |

`efficient_price` minus `observed_mid` is microstructure noise with the truth
known. Estimating volatility from `observed_mid` and comparing to
`sigma_bps_per_sqrt_s` shows you the bias directly.

---

## `params.json`

Every generating parameter, including the Hawkes branching ratio and the regime
transition matrix. If you fit a Hawkes model to `trades.csv`, this is what you
should recover.
