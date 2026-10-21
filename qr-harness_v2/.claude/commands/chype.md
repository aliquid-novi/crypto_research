---
description: Cheap hypothesis. One notebook in research/cheap with Mark's pre/post cell, an implementation and a results summary. No card, register, review or resolve.
argument-hint: <implementation request>  |  C### [extra instructions]
---

Input: $ARGUMENTS

A cheap hypothesis trades the full workflow for throughput. What stays: Mark writes the thinking, you write the code
and the numbers, and the data rules (CLAUDE.md rules 3, 4, 6, 7, 9, 10) still apply in full.

## Whose words go where
The notebook's **first markdown cell is Mark's**: a *Pre-implementation* section (hypothesis, why it should work,
assumptions, test statistic / method and why) and a *Post-implementation* section (what happened, what changed, why it
worked or didn't, whether the test was right, what changes to data or conditions could have changed the result).
- Never write, suggest or "tidy" anything in that cell. If Mark's message contains his pre-implementation answers,
  copy them in **verbatim** under the matching labels and tell him you did.
- He writes in the `.ipynb`. `python -m qrlib.cli sync` / `run` copy his cell back into the `.py` first, so always go
  through them; never regenerate the `.ipynb` another way.

## 1. New or existing?
- `$ARGUMENTS` starts with `C###`: that notebook (`research/cheap/C###_*.py`). Go to step 2.
- Otherwise it's a new request: `python -m qrlib.cli new-cheap "<short title from the request>" --request "<the request, verbatim>"`.
  This doesn't count towards the hypothesis-card limit.

## 2. Gate
`python -m qrlib.cli status C###`. If `stage` is `needs pre`, stop and tell Mark:
"Fill the Pre-implementation section of `research/cheap/C###_*.ipynb`, then run `/chype C###`."
Don't build anything on real data before that. You may ask mechanical questions (what data exists, what a metric means).

If the request or his pre section is ambiguous on something that changes the result (target, horizon, latency, fill
rule, data range, the test itself), ask him in chat and stop (rule 8). For anything minor, pick, mark it
`# ASSUMPTION:` and move on.

## 3. Implement (inline, no subagents)
- Edit the `.py` cells under "Implementation (agent)": KNOBS → data → method → leakage check → evaluation.
  Implement **his** test statistic or method from the pre section. If you think another one is better, say so in
  chat and in the Results "Questions for Mark", but implement his.
- Check coverage first (`data.coverage(...)`) and freeze the sample end in KNOBS.
- Rule 3: event data via `qrlib.pit` / `qrlib.fairvalue.l1_at`, known at `ts_recv + latency`. The lookahead hook
  scans this folder too.
- Rule 4: if anything is fitted or built from data, put it in one function and run `leakage.truncation_test` on it.
  If nothing is, say why under Results.
- Rule 6: `audit.log_run(CID, variant, params, metrics)` for every evaluated variant. Quoting results need both fill
  bounds (rule 5).
- Mark the core estimate cell (the number that supports or kills the idea) clearly, so Mark can find it and rebuild
  it if he wants.

## 4. Run
`python -m qrlib.cli run C###`, then `python tools/nb_dump.py research/cheap/C###_*.ipynb` and open every saved image.
If the leakage check fails, report it. Don't work around it.

## 5. Results (agent's cell only)
Fill "Results (agent)" in the `.py` and sync: weak points first, then numbers with n, CIs and `n_trials`, every
`# ASSUMPTION:`, and questions for Mark. No verdict on whether the hypothesis held, and never "edge".

In chat: 3–6 lines pointing at the Results cell. End with: "Your post-implementation section is waiting in the first
cell." `status` shows a cheap notebook as `awaiting post` until he fills it.
