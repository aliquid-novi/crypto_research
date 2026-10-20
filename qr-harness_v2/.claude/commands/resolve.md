---
description: Mark records the outcome and the lesson for a hypothesis; calibration updates
argument-hint: H### [held|failed] [what he thought, what happened, the lesson]
---

Input: $ARGUMENTS

This is Mark's verdict, not yours.

1. `python -m qrlib.cli status <H###>`. It must be registered with an intact lock and have a Results section. If
   there's no Results section, suggest `/review` first.
2. Show him, side by side: his locked prediction and falsification condition, the Results section, and `n_trials`.
3. If he hasn't said all three, ask:
   - **Held or failed?** Judged against *his* `falsified_if`, not against a new standard.
   - **What did you initially think, and what went wrong (or right)?**
   - **The lesson:** one or two sentences he'd want to read before the next similar idea.
   If the result is ambiguous against `falsified_if` (e.g. the CI touches the threshold), point that out and let him decide.
   Don't decide for him.
4. Record it: `python -m qrlib.cli resolve <H###> --outcome held|failed --lesson "<his words>"`. Then append his
   longer explanation, verbatim, under "## Resolution (Mark)" in the card body (the body stays editable).
5. Show `python -m qrlib.cli calibration` (Brier score; 0.25 = always saying 50%). If there are 5+ resolved cards,
   mention whether his confidence has tended to run high or low. Ask, don't lecture.
