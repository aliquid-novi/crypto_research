---
description: Small notebook/code help (plumbing). Minimal diff; don't touch research logic.
argument-hint: <task>   e.g. "make the markout plot use a log x-axis" or "load BTC too"
---

Task: $ARGUMENTS

- Do exactly the task with the smallest diff. Edit the `.py` source, then `python -m qrlib.cli sync <file.py>`.
- Don't change: the model function's logic, targets, the split, KNOBS values, fill rules, or anything on the card.
  If the task needs one of those, say so and ask. Changing a KNOB of a registered hypothesis makes the next run a new
  trial, and it gets logged.
- Keep the guardrails: joins through `qrlib.pit`, nothing forward-looking (the scan hook will flag it).
- If you re-run the notebook, say so and say which outputs changed.
