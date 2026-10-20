---
name: {{NAME}}
notebook: {{NOTEBOOK}}
card: {{CARD}}
created: {{DATE}}
method_for_part_b: ''      # the ONE method you're digging into this time
status: open               # open | done
---
# Post-implementation review: {{NAME}}

**Mark writes this, not an agent.** The point is to find out what you don't yet understand, so answer from memory
first and then check against the notebook. "Don't know" is a useful answer: it marks a gap. An agent may tell you
*where* something is (a cell or knob), never *what* the answer is.

**Scope:** Part A every time. Part B: **one** method only. Part C: **one** number. Skip what doesn't fit and write
`n/a: why` instead of forcing an answer.

## Part A — the result (every time)

**A1. The claim, in one sentence.** What, on which data, against which baseline.

**A2. The evidence that decides it.**
- Figure/table (cell):
- The number, with CI, n and n_trials:
- "If this number had been ___ instead, I would conclude ___."

**A3. My naive prediction vs what happened.** What I expected before seeing results, and how reality differed.
If I never wrote a prediction down, say so; don't reconstruct one after the fact.

**A4. Assumptions that would change the conclusion.**

| Assumption | Where it's encoded (knob / cell) | If it's wrong, the result moves… | Have I checked it? |
|---|---|---|---|
| | | | |

**A5. Assumptions encoded without my noticing.** Usual suspects: the clock and latency (which timestamp, what was knowable
when), the sample window and regime, units and normalisation, the fill model, anything computed on the full sample,
library defaults.

**A6. Another data-generating process that would produce this result even if my idea were false.** Describe it, then
name a test that would tell the two worlds apart.

**A7. What result would convince me I'm wrong?** Be concrete. If there's a card, compare with its `falsified_if`.
Has my standard drifted since I wrote it?

## Part B — one method, properly understood
Method: ___  (from the notebook, e.g. "Stoikov micro-price", "lattice mixture fit", "NUDFT scan")

Score each item **solid / shaky / gap**. The score matters less than finding the gaps.

| # | Item | Score | Notes |
|---|---|---|---|
| B1 | **Explain it** without code: purpose, inputs, outputs, what it estimates | | |
| B2 | **Reconstruct the maths:** derive the key formula or sketch its structure | | |
| B3 | **Where it breaks:** conditions that make it biased, inconsistent, aliased or misleading | | |
| B4 | **Known-answer check:** a simulation or analytic case with a known answer (can live in the rebuild notebook) | | |
| B5 | **Challenge it:** one null test, plus one alternative explanation investigated | | |
| B6 | **Trading relevance:** which quoting, hedging or risk decision it changes, and by how much it would need to move to matter | | |

## Part C — rebuild the key number
Notebook: `research/post/{{NAME}}_rebuild.py`. Write it yourself, without copying the original's code. Use the smallest
data slice that defines the number, and the same definition as the original.

| Number | Original notebook | My rebuild | Match? (tolerance) | If not, why |
|---|---|---|---|---|
| | | | | |

## Gaps found → next actions
- [ ]
