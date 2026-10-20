---
description: Build, leakage-audit and execute the notebook for a registered hypothesis card
argument-hint: H###
---

Card: $ARGUMENTS

## 1. Gate
- `python -m qrlib.cli status $ARGUMENTS`. If it isn't registered, or the gate line says the lock is broken, stop and tell
  Mark (point him to `/hypothesis` or `python -m qrlib.cli register`). Don't build anything on real data until it is.
- Read the card in full. If the Setup is ambiguous (target, horizon, latency, fill rule, data range), write the
  question under "Open questions" on the card and stop (rule 8).
- Check data coverage first: `python -c "from qrlib import data; print(data.coverage('<VENUE>', '<SYMBOL>'))"`.
  The local copy can have holes or still be syncing. Report what's there before choosing a split.

## 2. Build: delegate to the `notebook-builder` subagent
Give it the card path and anything Mark said in this conversation. It creates or edits
`research/notebooks/H###_*.py` (via `python -m qrlib.cli new-notebook H###` if it doesn't exist) following the fixed
structure, with every unspecified choice marked `# ASSUMPTION:`.

## 3. Audit: delegate to the `leakage-auditor` subagent
It reviews the notebook source independently and writes `research/audits/H###_audit.md`. If it finds a blocking
issue, send that back to `notebook-builder` and re-audit. Don't execute until the audit has no blocking findings.

## 4. Execute
`python -m qrlib.cli run H###`, then `python tools/nb_dump.py research/notebooks/H###_*.ipynb` and open every saved
image. If the truncation test fails, report the failure. Don't loosen the test or move the cut.

## 5. Report back to Mark
- Weak points first. Then results with n, CIs, `n_trials`, both fill bounds (rule 5), and which conclusions survive only
  under `touch`. Never call anything an edge (rule 10).
- Put the same summary into the notebook: a **Summary** cell right after the title, and Findings + the full
  **Assumptions** list + **Open questions** at the end (update the .py, then `python -m qrlib.cli sync`).
- List every `# ASSUMPTION:` in your message too.
- Next step is `/review H###`.
