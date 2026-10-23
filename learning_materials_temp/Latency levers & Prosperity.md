# Latency levers and Prosperity

Yes, market making Binance–Hyperliquid at \~750 ms is possible in principle, but only by not competing on speed: you choose where you are slow and make the market pay for it. Prosperity is a good school for research hygiene and fair-value thinking, and a poor model of the venue.

## Getting paid for liquidity across venues

Cross-venue market making is wholesaling: you sell immediacy on the thinner venue and lay the risk off on the deeper one.

- **What you are paid for.** Someone on Hyperliquid wants to trade now and will cross your spread to do it. You take the other side, then offload the position on Binance, where liquidity is deeper. The spread you earn is the price of that service, minus what the hedge costs.&#32;
- **Who trades with you decides everything.** Hyperliquid-native flow (leverage traders, liquidations, people who simply prefer the venue) is mostly uninformed: that is the flow you want. Traders reacting to a Binance move you have not yet seen land are informed: every fill against them is a pick-off. Your P&L is the first group's spread minus the second group's toll.
- **Kearns' result, applied to the basis.** [Chakraborty and Kearns (2011)](https://www.cis.upenn.edu/~mkearns/papers/marketmaking.pdf) show that a simple laddered market maker earns exactly half of total local price movement minus the square of the net move:

```latex
\text{profit} = \tfrac{1}{2}\left(K - z^2\right), \qquad K = \sum_t |P_t - P_{t-1}|, \quad z = P_T - P_0
```

On an unbiased random walk every strategy earns zero in expectation; with even slight mean reversion the market maker's expected profit turns positive, and for an Ornstein–Uhlenbeck process it grows linearly with time. If you hedge on Binance, your residual exposure is to the **Hyperliquid–Binance basis**, not to price. That basis is held near zero by arbitrage, so the mean reversion the theorem needs is structurally present in the series you actually carry.

What the theorem leaves out is exactly what bites: it assumes no latency, no fees, and that every price crossing fills you, so there is no adverse selection. Its own frequency experiment, refreshing quotes only every L steps, kept over 80% of profit at L = 40, but in a world with no informed traders, staleness is free. In yours, staleness is precisely what faster traders charge you for.

## Levers against the latency gap

You cannot close a \~750 ms gap with speed, so every lever either prices the gap, shrinks it, or avoids the flow that exploits it.

| Lever | What it does | P&L term it moves | Cost |
| --- | --- | --- | --- |
| Quote where price *will be* | Fair value forecasts the Hyperliquid price \~750 ms ahead, using Binance's lead, not where Binance is now | Spread captured, adverse selection | Research: needs a lead-lag model with an economic reason |
| Width and depth ladder | Wider top level plus deeper levels (Kearns' ladder) so small moves fill you profitably and big ones fill you less | Adverse selection | Fewer fills |
| Selective quoting | Pull or widen in high-vol regimes, before events and funding times; quote only when flow is likely benign (as Albers et al. did with reversals) | Adverse selection | Fewer fills; needs a regime model |
| Coin selection | Wider-spread, less-contested coins with retail-heavy Hyperliquid flow | Spread captured, adverse selection | Less capacity |
| Clip size | Small quotes cap the loss per pick-off | Adverse selection, inventory | Less capacity |
| Venue roles | Passive on Hyperliquid, aggressive hedge on Binance; or passive hedge to save \~3 bp | Fees, hedge cost | Inventory risk if passive |
| Inventory band | Hedge only when inventory breaches a band; net across correlated coins | Hedge cost, inventory | Basis and directional risk inside the band |
| Fee tier | Firm's VIP tier, HYPE staking discounts, maker rebates | Fees | Volume or capital |
| Infrastructure | Tokyo VM, own non-validating node for faster reads and fill notifications | Adverse selection, hedge cost | Money and ops |
| Priority fees | ALO priority moves a new quote ahead of same-price quotes from the prior 400 ms | Adverse selection (queue) | Paid per action |

The first lever is the one most worth your research time: a quote that is right about where price will be when it goes live is not stale.

## Is it possible at 750 ± 100 ms and 2–5 ms?

Possible on the right coins, not at top of book on majors, and only the data can say which coins.

- **You may be at par, not behind.** Glassnode measured a 884 ms median for public-API orders from AWS Tokyo, so \~750 ms may be roughly what ordinary API users get. Your toll is paid to the minority who are faster (own nodes, priority fees), not to everyone.
- **Size for the tail, not the median.** ±100 ms of jitter means the stale window is a distribution. Spread and clip size should survive something like the 90th percentile window, because pick-offs cluster in fast markets, which is exactly when latency is worst.
- **The Binance side is not your problem.** At 2–5 ms, the hedge executes almost at the price that triggered it. The weak link on that leg is how fast you learn of the Hyperliquid fill.

The break-even, per round trip, for a given coin and spread setting:

```latex
\text{half-spread} > \text{fees} + \mathbb{E}[\text{pick-off loss} \mid \text{fill}] + \text{hedge slippage}
```

**Kill test.** Replay or small-size live quoting on two or three candidate coins. If Hyperliquid fill markouts against the Binance mid at \~1 s, net of fees, stay negative across every spread width you try, the setup does not work on those coins at your latency, and no amount of signal will rescue it.

## Prosperity in your context

Prosperity trains two of the five terms well and leaves out the three that dominate a Binance–Hyperliquid book.

**What the [rat\_hunters repo](https://github.com/rmtf1111/imc-prosperity-4) did** (finished #2 overall, #3 on algo):

- **Rounds 3–4, global mean reversion.** A fixed fair value per product, entering at a deviation threshold, justified by negative autocorrelation. Round 4 made thresholds asymmetric after finding downward excursions peaked around half the size of upward ones.
- **Round 5, structure hunting** across 50 products: a basket whose prices summed to a constant (market making there was near riskless because bots traded identical sizes at identical times), a pairs spread chosen by ADF rather than raw correlation, a price lattice where a 100-tick move reversed about 85% of the time, and a weak lead-lag (correlation about 0.05) that they overfit and lost money on. Everything else got a plain one-tick-inside quote.
- **Their overfitting notes are the best part.** Rolling-mean residuals mean-revert for almost any series; pick thresholds in stable PnL regions, not spikes; and in a simulation of 50 independent random walks, over 100 of 1,225 pairs looked cointegrated by chance.

**Mapped to the five terms:**

| Term | In Prosperity | In your Binance–Hyperliquid book |
| --- | --- | --- |
| Spread captured | Central: fair value is the whole game | Central, but fair value is a moving forecast, not a constant |
| Adverse selection | Mostly absent; bots are scripted and you act every tick | The dominant cost: informed takers hit stale quotes |
| Fees and rebates | Not part of their write-up; the spread is the only cost discussed | \~6.5 bp round trip at base tier, often larger than the edge |
| Hedge cost | Barely present; few cross-product hedges | Every fill needs a Binance hedge |
| Inventory | Position limits, not risk | Real basis and directional risk |

**Translating the ideas.** Prosperity's fixed fair value becomes your Binance-implied fair value. "Price minus fair" becomes the Hyperliquid–Binance basis, the one series that is structurally mean-reverting, which is Kearns' precondition. Thresholds become quote offsets and skew. The lattice reversal is a cousin of bid–ask bounce. Their failed lead-lag is the warning for your Binance-leads-Hyperliquid model: it only deserves trust if it has an economic reason (price discovery on the deeper venue does give one) and holds out of sample.

**Verdict.** Use Prosperity as a school for signal hygiene and fair-value thinking, not as a strategy template. Its mean reversion in price levels is largely there by design; crypto prices at seconds-to-minutes horizons sit close to a random walk, where Kearns' theorem says market making earns zero before costs. Look for mean reversion in the basis and in microstructure, not in price.

## Questions to work through

- [ ] Does the Hyperliquid–Binance basis on your candidate coins actually mean-revert at your quoting horizon? Estimate an OU half-life from your server data and compare it with \~750 ms.
- [ ] In Kearns' terms, what are K and z for the basis over a day? Is (K − z²)/2, in basis points, larger than your round-trip fees?
- [ ] What fraction of Hyperliquid volume on your coins is likely informed versus native? What would you measure to tell them apart?
- [ ] Which single lever in the table moves the largest term for your book, and how would you test it cheaply?
- [ ] Which Prosperity result would you trust out of sample, and what economic reason would you give for it?

## Sources

- Chakraborty, Kearns (2011). [Market Making and Mean Reversion](https://www.cis.upenn.edu/~mkearns/papers/marketmaking.pdf). EC '11.
- rat\_hunters. [IMC Prosperity 4 write-up and code](https://github.com/rmtf1111/imc-prosperity-4).
- Glassnode. [Hyperliquid order-to-fill latency](https://hyperlatency.glassnode.com/fill-latency).
- Blockworks Research. [Hyperliquid priority fees](https://app.blockworksresearch.com/research/hyper-liquid-priority-fees-internalizing-high-frequency-trading-revenue).
- Albers, Cucuringu, Howison, Shestopaloff (2025). [The Market Maker's Dilemma](https://www.alphaxiv.org/abs/2502.18625v1).
