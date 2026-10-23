# MFT market making: readiness exam

**Purpose.** Check that you understand enough to start researching Binance–Hyperliquid medium-frequency market making.
This is the end of the "learn the lay of the land" phase, not the end of learning.

**Covers:**
- `Market Making as One Machine.pdf` (**MM1**)
- `Holding time & exits.pdf` (**HT**)
- `Latency levers & Prosperity.md` (**LP**)
- `mft_mm_examples/mm_economics_vs_frequency_with_holding.ipynb` (**NB**)
- your own `hl_mm_baseline` and `hyperliquid_research` notebooks

**Rules**
- Closed book first. Write your answer under each question, then check it against the source.
- Each question names where to look. Don't read the source until you've written something.
- Part B calculations have an answer key at the end. Everything else is marked against the sources, not against a key.
- Mark each question ✅ (got it), 🟡 (partly), ❌ (didn't know). Re-do every 🟡/❌ a day later.

**Pass:**
- Parts A–D: at least 80% ✅, and no ❌ left after the re-do.
- Part E: every question has a written answer. These are judgement calls with no right answer, but you need *an* answer before building.

Expect about 3–4 hours in total. You don't have to do it in one sitting.

---

## Part A: The machine (concepts)

**A1.** Write the five-term net P&L equation from memory. In one line each, say what each term measures and its usual sign. *(MM1 p.2–3)*

> 

**A2.** Why should spread captured be measured against **fair value** rather than the mid? What goes wrong if you use the mid? *(MM1 p.3)*

> 

**A3.** Write the per-lot identity with the exit (quoted edge, staleness, holding drift, exit vs fair, fees). Why do the terms always add up exactly to realised P&L? *(HT p.1, NB §1)*

> 

**A4.** The five-term breakdown and the per-lot identity overlap. Map each identity term onto the five terms. Which identity term is *risk* rather than expected cost, and why? *(NB §7)*

> 

**A5.** Define a markout at horizon h, including the sign convention. Sketch (in words) what a markout curve looks like when you're being adversely selected, and when you're being paid to absorb noise. *(MM1 p.3)*

> 

**A6.** "You get filled most when you are wrong." Explain why fills are not a random sample of time. Name the paper with live evidence on Binance BTC, and its headline numbers. *(MM1 p.3)*

> 

**A7.** Signal mining vs market making. Why does progress compound in market making? And where does multiple testing *still* bite? *(MM1 p.1–4)*

> 

**A8.** What is LVR? Does it apply directly to Hyperliquid, or only by analogy? Why? *(MM1 p.4)*

> 

**A9.** State Kearns' result. Why is expected profit zero on a random walk, and positive on an OU process? *(LP, NB §4)*

> 

**A10.** Which series that you'd actually carry is plausibly mean-reverting, and why is it structurally so? Which series is close to a random walk at seconds-to-minutes horizons? *(LP, HT p.5)*

> 

**A11.** Name three things Kearns' model leaves out that matter at your latency. *(LP)*

> 

---

## Part B: Numbers (do these on paper; answer key at the end)

**B1.** A coin has 80% annualised volatility and trades 24/7. Roughly how many bp does it move (1 standard deviation) over 1 s, 0.75 s, 60 s and 5 minutes?

> 

**B2.** At base tier:
- Hyperliquid maker is 1.5 bp.
- Binance futures taker is 5 bp; Binance maker is 2 bp.

What is the round-trip fee for:
- (a) quote on HL, hedge as taker on Binance;
- (b) quote on HL, hedge passively on Binance;
- (c) quote on HL, exit against an offsetting HL fill?

> 

**B3.** A long lot, using the per-lot identity with prices in USD (1 bp = 0.01 at a price of 100):
- you quoted using fair F^q = 100.00 and were filled at p_in = 99.94;
- fair at the fill was 99.97;
- you hedged when fair was 99.99, selling at p_out = 99.985;
- fees are B2(a).

Compute each term in bp, and the net.

> 

**B4.** One Monte Carlo lot in HT p.2 has quoted edge +6.0, staleness −1.8, drift 0, exit −0.5 and fees −6.5 bp. Holding everything else fixed, what quoted edge (half-spread) breaks even? How does that compare with the "nothing below ~8 bp is profitable" result?

> 

**B5.** You buy at 100.00. The mid 1 s later is 99.98. What is the 1 s markout in bp? Is it adverse selection or noise? What would you need to see to tell?

> 

**B6.** Kearns ladder: the price path is 100, 101, 100, 101, 102, 101 (ticks). Compute K, z and the ladder's profit in ticks.

> 

**B7.** Little's law. Your inventory cap is 10 lots and your mean holding time is 60 s. What is the maximum sustainable lot rate per hour? What happens to it if you hold for 180 s instead?

> 

**B8.** HL blocks arrive about every 67 ms.
- How many blocks pass during a 750 ms order/cancel latency?
- How many during the 309 ms median feed lag you measured on the VM?

> 

**B9.** Break-even inequality. Fees are B2(a). The expected adverse move given a fill is 1.5 bp, and hedge slippage is 0.5 bp. What is the minimum half-spread? Now take HYPE: the spread is 1 tick (~0.13 bp) 70% of the time. What does that tell you about quoting HYPE at the touch?

> 

---

## Part C: Your latency and setup

**C1.** Name the four latencies in MM1 p.7, your number for each, and what each one controls. Which one is still unknown?

> 

**C2.** Draw (or describe) the stale-quote window after a Binance move: what happens at 0 ms, during the window, and at ~750 ms. Why does the 2–5 ms Binance hedge *not* shrink this window? *(MM1 p.7)*

> 

**C3.** Why does requoting faster than your latency not make quotes fresher? Why is requoting too slowly also bad? Which matters more: requote interval or width? *(HT p.4–5, NB §3)*

> 

**C4.** The cancel-first rule: within a block, cancels are processed before GTC/IOC orders. Why does this rarely help you at ~750 ms? *(MM1 p.6–8)*

> 

**C5.** Three ways out of an HL fill. Give the fee, the advantage and the trap of each. Why is a passive exit adversely selected, just like a passive entry? *(HT p.3)*

> 

**C6.** Holding longer:
- What gets better?
- What gets worse, and how fast?
- What falls?

Past a few minutes, what are you really doing? *(HT p.4)*

> 

**C7.** Why is the holding-time result "the most flattering number in the notebook"? Give both reasons. What's the measurement that would fix the first one? *(HT p.5–6, NB §3)*

> 

**C8.** Give the six design rules at these speeds (MM1 p.8) and the reason behind each.

> 

**C9.** "N(0.50, 0.05) ms" (your block-latency scenario in `hyperliquid_research`) and "750 ± 100 ms" are different quantities. What does each one measure, and where does each belong in the machine?

> 

---

## Part D: Reading your own results

**D1.** `hl_mm_baseline` uses `ORDER_DELAY_MS` = 0 on the VM clock. Explain precisely why this makes the quoting fills *infeasible*, not just optimistic. What minimum delay would be feasible? *(hl_mm_baseline Summary, §5)*

> 

**D2.** `through` vs `touch` fill bounds:
- What does each assume?
- Which is pessimistic?
- What does it mean when a conclusion survives only under `touch`?

*(hl_mm_baseline §5)*

> 

**D3.** The extended micro-price beats the mid by ~3% MSE at 250–500 ms. Is that an edge? Translate "3% MSE skill" into something about the five terms. Which term could it move, and why might that not show up in P&L? *(hl_mm_baseline §3.1, §8)*

> 

**D4.** What is a calibration slope? The textbook micro-price has slopes of 2.3–10.4. What does that say about it? *(hl_mm_baseline §3)*

> 

**D5.** Between the 43.6 h and 87.6 h runs, fair-value skill moved outside its CI. Why are the bootstrap CIs too narrow for "what will this do tomorrow"? What design would fix it?

> 

**D6.** `hl_mm_baseline` quotes HYPE around HL's own micro-price, with no Binance fair value and no hedge. Which parts of the machine in MM1 does it test, and which does it not test at all?

> 

**D7.** `hyperliquid_research`:
- What is the 67 ms block clock?
- What does single-block ρ(1) ≈ −0.20 mean?
- Why isn't it ms rounding?

In one sentence: where, if anywhere, does block timing enter the five-term equation? *(also MM1 p.9)*

> 

**D8.** Why did the tradfi results in `hyperliquid_research` change so much between the 7.2 h and 48 h samples? What does that tell you about choosing a sample for anything involving SP500/TSLA/CL?

> 

---

## Part E: Before you build (judgement; your answers become the first hypothesis cards)

There are no right answers here, but each needs a written answer with a reason.

**E1.** For a naive quoter on your pair, which of the five terms do you expect to be the biggest drag?
- What's the cheapest check that could prove you wrong?
- Which of your existing notebooks could already tell you, and which can't (because of the 0 ms delay)?

> 

**E2.** Which coins will you study?
- HYPE has a 1-tick spread 70% of the time. Does it fit "wider-spread coins the big makers don't crowd", or the opposite?
- What would you measure to pick two or three candidates? (MM1 p.8: quoted spread against annualised volatility.)

> 

**E3.** Of the "measure first" items, which one could kill the whole idea cheapest? What result would make you stop?
- markouts of HL fills against the Binance mid, 0–2 s;
- basis half-life;
- offset rate conditional on the Binance move;
- fill-notification latency.

> 

**E4.** Which venue leads price discovery for your pair, and at what horizon? What does that imply about which venue you should be passive on? (Your lead-lag results give a first answer for crypto.)

> 

**E5.** Rewrite one idea you already have as a *component change*. Use this format: "doing X changes term T at horizon h by at least Y bp, without cutting fill rate by more than Z%." State the kill result in that term.

> 

**E6.** Look back at your hypotheses and notebooks so far. Which were changes to one machine, and which were independent lottery tickets?

> 

**E7.** What is your current loop time, from "I changed X" to "I can see its effect on a markout"? Which single step dominates it?

> 

**E8.** What is the smallest live test that would check your fill model against reality? What would it cost, and what do you need to build first (order management, fill notifications, kill switch)?

> 

**E9.** Two topics the docs only touch: perp funding, and HL event risk (liquidation cascades, oracle/mark-price moves). For each, which term does it hit, and at what holding time does it start to matter?

> 

---

## Scorecard

| Part | Questions | ✅ | 🟡 | ❌ | Re-done |
|---|---|---|---|---|---|
| A: The machine | 11 | | | | |
| B: Numbers | 9 | | | | |
| C: Latency and setup | 9 | | | | |
| D: Your results | 8 | | | | |
| E: Before you build | 9 (answered?) | | | | |

Date started: ____  Date passed: ____

---

## Answer key (Part B only; read after attempting)

<details>
<summary>B1</summary>

σ per √s = 0.80 / √(365 × 24 × 3600) = 0.80 / 5,616 ≈ **1.42 bp/√s**.
- 1 s ≈ 1.4 bp
- 0.75 s ≈ 1.42 × 0.87 ≈ 1.2 bp
- 60 s ≈ 1.42 × 7.75 ≈ 11 bp
- 5 min ≈ 1.42 × 17.3 ≈ 25 bp
</details>

<details>
<summary>B2</summary>

- (a) 1.5 + 5 = **6.5 bp**
- (b) 1.5 + 2 = **3.5 bp**
- (c) 1.5 + 1.5 = **3 bp**
</details>

<details>
<summary>B3</summary>

- Quoted edge = (100.00 − 99.94) = +6 bp
- Staleness = (99.97 − 100.00) = −3 bp
- Holding drift = (99.99 − 99.97) = +2 bp
- Exit vs fair = (99.985 − 99.99) = −0.5 bp
- Fees = −6.5 bp
- **Net = −2 bp**

Check: p_out − p_in = 4.5 bp, and 4.5 − 6.5 = −2 ✓
</details>

<details>
<summary>B4</summary>

The net is −2.8 bp (shown as −2.7 after rounding), so break-even needs about 6.0 + 2.8 ≈ **8.8 bp** of quoted edge. That's consistent with "nothing below ~8 bp half-spread is profitable": fees alone are 6.5 bp.
</details>

<details>
<summary>B5</summary>

The markout is s·(P − p) = +1 × (99.98 − 100.00) = −0.02, which is **−2 bp**. One fill can't tell you which it is. You need the *average* markout curve across many fills: if it keeps falling after the fill, that's adverse selection; if it recovers, that's noise.
</details>

<details>
<summary>B6</summary>

- K = 1+1+1+1+1 = **5**
- z = 101 − 100 = **1**
- Profit = (5 − 1)/2 = **2 ticks**
</details>

<details>
<summary>B7</summary>

- Lot rate ≤ 10 / 60 s = 1 lot per 6 s = **600 lots/h**.
- At 180 s: 10 / 180 s ≈ **200 lots/h**, so throughput falls by 3×.
</details>

<details>
<summary>B8</summary>

- 750 / 67 ≈ **11 blocks**
- 309 / 67 ≈ **4.6 blocks**
</details>

<details>
<summary>B9</summary>

Minimum half-spread > 6.5 + 1.5 + 0.5 = **8.5 bp**. HYPE's touch is ~0.13 bp wide, about 65× too narrow, so quoting at the touch on HYPE can't cover fees at base tier. You'd have to quote far from the touch (and fill rarely), or pick a different coin.
</details>
