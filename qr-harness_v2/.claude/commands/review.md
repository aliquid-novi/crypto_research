---
description: Results-critic reads the executed notebook and writes the card's Results section plus questions for Mark
argument-hint: H###
---

Card: $ARGUMENTS

1. Confirm the notebook has been executed since its last edit (`python tools/nb_dump.py research/notebooks/$ARGUMENTS_*.ipynb`
   warns on out-of-order or stale cells). If not, say so and stop.
2. Delegate to the `results-critic` subagent with the card path, the notebook path and the audit file
   (`research/audits/$ARGUMENTS_audit.md` if present).
3. Show Mark the Results section it wrote into the card body, and its questions. Don't add your own interpretation
   of whether the prediction held. That's Mark's call in `/resolve`.
4. If the critic found something that invalidates a result (leak, wrong target, fill bound missing, too few trials
   reported), say so plainly, first.
