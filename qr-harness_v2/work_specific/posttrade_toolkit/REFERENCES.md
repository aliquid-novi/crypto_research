# Published reference points

What papers and practitioners have measured with the same methods this toolkit implements. Use them to
check that **your** numbers are in a plausible range, and to know where a method is known to mislead.

Read these first:

- **These are not targets.** Every figure below is from a different asset, venue, period and event
  definition. A number far from them is a reason to check your units, signs and event definitions,
  not proof that something is wrong.
- **Only numbers checked against the source text are quoted.** Where only a secondary description was
  available, the entry says so. Items found but not checked are listed at the end and quote no numbers.
- **No data or code from these sources is copied into the toolkit.** The sample data is still synthetic only.
- Links are for when you are back online. Everything you need to compare is written here.

| Toolkit piece | Reference points |
|---|---|
| Markout sign and plotting conventions (`markouts`, 02) | [Tread](#markouts-practitioner-conventions), [KX](#markouts-practitioner-conventions) |
| Order flow imbalance (`reference.ofi`, 05 features) | [Cont, Kukanov, Stoikov 2014](#order-flow-imbalance) |
| Microprice (`reference.fit_microprice`, 02) | [Stoikov 2018](#microprice) |
| Hawkes branching ratio (`models.fit_hawkes_*`, 05) | [Filimonov–Sornette 2012; Hardiman, Bercot, Bouchaud 2013; Bitcoin studies](#hawkes-branching-ratio) |
| Sign autocorrelation, response, propagator (05) | [Lillo–Farmer 2004; Bouchaud et al. 2004](#order-sign-memory-and-the-propagator) |
| Square-root impact (`models.sqrt_impact_fit`, 05) | [Donier–Bonart 2015](#square-root-impact) |
| Lead venue, cross-venue spreads (`LEAD_MAP`, 02, 04) | [Alexander et al. 2020; Alexander–Heck 2020; Makarov–Schoar 2020](#lead-lag-and-cross-venue-spreads) |
| Addresses, counterparty and taker toxicity (03) | [Barone–Lillo 2026 (Hyperliquid)](#hyperliquid-address-level-impact) |
| Deflated Sharpe, multiple trials (07) | [Bailey–López de Prado 2014](#multiple-trials) |

---

## Markouts: practitioner conventions

**Tread docs, "Markouts"** (execution vendor documentation).
<https://docs.tread.fi/real-time-algo-order-tracking/markouts>

- They plot log returns 1, 5 and 10 s before and after each fill, with ±1 SD bars and the spread in bps alongside.
- **Their sign is the opposite of ours.** For them a positive return means price drifted *away* (up after
  a buy), which they call generally bad. In this toolkit `markout = d × (m_{t+h} − p)` and **positive is good**.
  Flip the sign before comparing any vendor or broker markout chart.
- They state the caveat that maker markouts "are defined by sample bias": resting orders fill only when
  the market moves into them. This is the selection problem that notebook 05 corrects for.
- Reading guide: a good maker fill shows drift towards you *before* the fill and is flat after. Continued
  favourable drift *after* the fill suggests you could have quoted deeper. A constant slope through the
  fill points to an outside driver, such as trend, rather than your fills.

**KX, "What is markout, and how is it calculated?"** (data vendor explainer). <https://kx.com/?p=17802>

- A markout is meaningful only when the sign convention, reference price, horizon and market-data
  treatment are all stated.
- Report a curve over several horizons, not one chosen point. This is why `markout_curve` and the
  log-spaced `HORIZONS_S` are the defaults.

## Order flow imbalance

**Cont, Kukanov, Stoikov (2014), "The price impact of order book events", *J. Financial Econometrics* 12(1).**
arXiv 1011.6402.

- They study 50 US stocks (NYSE TAQ) over 10-second intervals.
- Regressing the mid change on OFI gives an **average R² of 65%**. Trade imbalance alone gives 32% (Table 2;
  Table 4 grand mean).
- The relation is linear, with a slope inversely proportional to depth.
- Regressing |ΔP| on traded volume gives 23%, against 58% for |OFI| in the same table.
- Comparing with ours: crypto L1 updates arrive in bursts (Hyperliquid once per block) and are sometimes
  sampled, which caps the attainable R². A much lower R² on Hyperliquid is not on its own a bug. If OFI does
  *no* better than trade imbalance, check the sign of your book-update handling first.

## Microprice

**Stoikov (2018), "The micro-price: a high-frequency estimator of future prices", *Quantitative Finance* 18(12).**

- The empirical claim is that the micro-price predicts short-horizon prices better than the mid or the
  weighted mid. The claim is qualitative; no number is quoted here.
- Comparing with ours: a direct check is whether the microprice's error at predicting the next mid
  change is smaller than the mid's. If it is not, check `n_imb`, `n_spread` and the tick size. On a market
  that is almost always one tick wide, the spread states collapse into one.

## Hawkes branching ratio

**Filimonov, Sornette (2012), "Quantifying reflexivity in financial markets", *Phys. Rev. E*.** arXiv 1201.3572.

- They fit an exponential kernel to E-mini S&P mid changes, 1998–2010, on short windows.
- The share of price changes *not* explained by self-excitation fell from about 70% in 1998 to below 30%
  from 2007. In other words, the branching ratio rose from roughly 0.3 to above 0.7.

**Hardiman, Bercot, Bouchaud (2013), "Critical reflexivity in financial markets", *EPJ B*.** arXiv 1302.1405.

- With a power-law kernel on two-month windows, the branching ratio **stays near 1 (critical)
  throughout 1998–2011**.
- They simulated critical (n = 1) power-law processes and fitted exponential kernels on 30-minute
  windows. The fits came back sub-critical, because most of a long-memory kernel lies outside a short
  window. What changed over time was the short-time cut-off, not the total reflexivity.

**Bitcoin studies:**

- Mark, Sila, Weber (2020), *European J. of Finance*, fitted power-law kernels to Bitcoin mid-price changes.
  They found long memory and criticality similar to fiat currency markets.
- A 2025 preprint (arXiv 2510.08085, not peer reviewed) calibrated Binance BTCUSDT order flow at n ≈ 0.81
  (SE 0.031) with a power law and ≈ 0.80 with an exponential kernel.

**What this means for the toolkit.** `fit_hawkes_1d`/`2d` use an exponential kernel, so
**their branching ratio is a lower bound on total reflexivity**, and a short-window one. Compare only
across fits made on the same window length. Do not read a value of 0.6 as "40% exogenous". If the QQ
plot in 05 bends above the diagonal in the tail, that is the missing long-memory part.

## Order-sign memory and the propagator

**Lillo, Farmer (2004), "The long memory of the efficient market", *Studies in Nonlinear Dynamics &
Econometrics*.** arXiv cond-mat/0311053.

- On London Stock Exchange data, the autocorrelation of market-order signs decays as a power law
  `C(τ) ~ τ^{-γ}` with **γ ≈ 0.6** (Hurst ≈ 0.7).
- Lillo, Mike and Farmer (2005) attribute it to traders splitting large orders. The exponent is linked to
  the tail of the order-size distribution, so it varies by market.

**Bouchaud, Gefen, Potters, Wyart (2004), "Fluctuations and response in financial markets", *Quantitative
Finance* 4(2).** arXiv cond-mat/0307332.

- This is the propagator model used in `models.propagator`, fitted on Paris Bourse data.
- Each trade's impact decays. The long memory in signs holds the price up, and the two balance so that
  prices stay close to diffusive.
- Comparing with ours: plot the sign autocorrelation on log-log axes. Lillo–Farmer's γ ≈ 0.6 is a
  slope of about −0.6. A curve near zero beyond a few lags is unusual for a liquid market. First check how
  one aggressive order that prints as several trades is handled: one event or several. Splitting one
  order into several trades inflates short-lag correlation, and merging trades changes the lag scale.

## Square-root impact

**Donier, Bonart (2015), "A million metaorder analysis of market impact on the Bitcoin", *Market
Microstructure and Liquidity*.** arXiv 1412.4503.

- They used MtGox BTC/USD, Aug 2011–Nov 2013, where each trader is identified.
- Their metaorder is a trader's run of aggressive orders, starting after about 1 hour of inactivity and
  ending at the next inactivity or reversal. This gives over 1 million metaorders.
- `I = Y σ (Q/V)^δ` holds over four decades of Q/V, with **δ ≈ 0.5**, and **Y ≈ 0.9** with a spread of
  about 0.35. σ and V are daily, which is the same normalisation as `models.sqrt_impact_fit`.
- Impact peaks at the end of the metaorder and then decays almost completely. Isolated (uninformed)
  metaorders keep far less than the 2/3 of peak impact that equilibrium models predict.
- Comparing with ours: the toolkit measures impact the same way, from the mid before the first fill to the
  mid after the last. Their law held over four decades of Q/V across a million metaorders. Clips that are
  small relative to daily volume put the impact below the noise, and δ then cannot be identified. That is
  what happens in the synthetic demo, and it may happen with your fills.

## Lead-lag and cross-venue spreads

**Alexander, Choi, Park, Sohn (2020), "BitMEX bitcoin derivatives: price discovery, informational
efficiency and hedging effectiveness", *J. Futures Markets* 40(1).**

- Using minute data, BitMEX derivatives led prices on major spot venues.
- What drove price discovery was the bid–ask spread, the spread between exchanges, and relative volume.

**Alexander, Heck (2020), "Price discovery in Bitcoin: the impact of unregulated markets", *J. Financial
Stability* 50.** Checked only through a secondary description, an author's 2026 comment letter.

- Perpetuals and futures on BitMEX, Huobi and OKEx led price discovery. CME futures and US spot mostly
  reacted.
- The same letter says later work (Alexander, Heck, Kaeck 2022) found Binance's USDT perpetual became the
  main source.
- **Leadership moves over time.** Re-estimate `LEAD_MAP` from your own data (cross-correlation or the
  Hasbrouck VAR in 05) rather than assuming it, especially for CME against crypto venues.

**Makarov, Schoar (2020), "Trading and arbitrage in cryptocurrency markets", *J. Financial Economics* 135(2).**

- They used trade data from 34 exchanges in 19 countries.
- Price gaps between exchanges are large and recur. They are much larger across countries than within one,
  and can persist for weeks. They widen when bitcoin rallies.
- Comparing with ours: the OU fit in 04 assumes one stationary mean. A spread can have long excursions
  driven by capital or settlement frictions. When the half-life is unstable across days, fit per regime
  or per day before trusting it.

## Hyperliquid: address-level impact

**Barone, Lillo (2026), "Trading in the sunshine or in the shade: market impact and adverse selection on
Hyperliquid", arXiv 2606.15715 (preprint).**

- From address-level data they rebuilt 4.3 million hidden metaorders and compared them with 465,000 public
  TWAP executions.
- Visible TWAPs paid less to execute and left smaller permanent impact. Hidden orders running alongside
  same-direction public TWAPs paid more, so the adverse-selection cost moved onto them.
- While a TWAP was live, displayed depth rose and leaned towards the side absorbing it.
- The abstract quotes no bps magnitudes, so none are given here.
- Comparing with ours: `publishes_addresses=True` venues let you rebuild metaorders the way Donier–Bonart
  did. Active public TWAPs are a candidate conditioning column for `conditional_curves` and for
  `taker_toxicity`.

## Multiple trials

**Bailey, López de Prado (2014), "The deflated Sharpe ratio", *J. Portfolio Management*.** SSRN 2460551.

- This is the formula `stats.deflated_sharpe` and `stats.expected_max_sharpe` implement. The inputs are the
  number of trials and the variance of the Sharpe ratios across them.
- No worked example from the paper was checked, so none is quoted. Notebook 07 shows the effect with your
  own trial count.

---

## Found but not checked (no numbers quoted)

- Easley, O'Hara, Yang, Zhang (2024), *Microstructure and market dynamics in crypto markets*. Roll, VPIN and
  Kyle λ on five major coins, reportedly predictive of later volatility. Only an unofficial upload was
  found; read the original before relying on it.
- Bieganowski, Ślepaczuk (2026), arXiv 2602.00776, *Explainable patterns in cryptocurrency microstructure*.
  The PDF could not be read here.
- A Hyperliquid order-book study of the 10 Oct 2025 liquidation cascade (J. of Financial Data Analytics,
  Lincoln University). It covers how fast spreads widen and recover, and whether quotes and sizes recover
  symmetrically.
