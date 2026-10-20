---
description: Read a paper and write a paper card; Mark picks which claim (if any) becomes a hypothesis
argument-hint: <pdf path or link>
---

Paper: $ARGUMENTS

1. If it's a PDF, keep it under `research/papers/pdfs/` (gitignored). If it's a link, fetch it.
2. Delegate to the `paper-reader` subagent: it creates the card with `python -m qrlib.cli new-paper "<title>"` and
   fills it (summary, numbered testable claims with evidence and strength, reusable methods with equations, what would
   differ on Mark's data, weak points).
3. Show Mark the claims list and ask which, if any, he wants to turn into a hypothesis. If he picks one, run
   `/hypothesis` with `--source P###`, where the prediction and confidence are still his, not the paper's.
