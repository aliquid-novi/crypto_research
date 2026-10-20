---
description: Turn Mark's idea into a hypothesis card by interviewing him. You fill nothing he didn't say.
argument-hint: <idea in Mark's words>   e.g. "HYPE blocks are forecastable"
---

Idea: $ARGUMENTS

You are the interviewer, not the author. The card's content is Mark's; your job is to get it out of him precisely
enough that a notebook can test it and nobody can argue afterwards about what was predicted.

## 0. Outstanding vs resolved (before anything else)
Run `python -m qrlib.cli status`. Its last line is `hypotheses: N outstanding, M resolved`. Tell Mark both numbers.
If N ≥ 2, stop and ask with AskUserQuestion: **"Please confirm: N hypotheses are still unresolved (list the ids). Create a
new card anyway, or resolve one first?"** Don't interview until he answers. If he confirms, pass `--confirm` to
`new-card` in step 3. Without it, `new-card` refuses.

## 1. Check for overlap
Using the same `status` output: if an existing card (or a resolved one) covers this, say which, and ask whether this
is a new card, a variant of that card, or a re-test.

## 2. Interview (a few questions per turn, in this order, skipping what he already said)
1. **Mechanism:** why should this be true? Who is on the other side, and why haven't they already removed it?
2. **Setup:** instrument(s), venue(s), data and dates; the decision time and horizon; what's known when (latency).
3. **Evaluation target:** what exactly counts as "it works"? (A metric, a baseline it must beat, a threshold.)
4. **Prediction:** the number or direction he expects, in his words.
5. **Falsified if:** the result that would make him drop it. Push for something concrete (e.g. "95% CI of skill
   vs mid includes 0 at 500 ms").
6. **Confidence:** his probability (0–1) that the prediction holds.
7. **Variants** he wants tried (each is a trial) and anything he wants ruled out.

Never suggest a prediction, a falsification threshold or a confidence, even as an example of "something like". If he
asks what you think, say that's his call (CLAUDE.md rule 1) and offer to show what the data or past cards say instead.
Questions about mechanics (what data exists, what's measurable, what a metric means) you answer directly.

## 3. Write the card
- `python -m qrlib.cli new-card "<short title>" --family <fair-value|quoting|lead-lag|timing|spread|other> --source mark`
  (or `--source P###` if it came from a paper card; add `--confirm` only if Mark confirmed in step 0).
- Fill the body (Idea, Setup, Variants) with **his words**, lightly tidied, not reinterpreted.
- Put his prediction, falsified_if, confidence (a number) and `predicted_by: mark` in the front matter **only if he
  stated them**. Otherwise leave them empty and list what's missing under "Open questions".
- Show him the card and ask him to confirm or correct it.

## 4. Register (only when Mark says so)
When he confirms, run `python -m qrlib.cli register H###`. It refuses if a field is empty or `predicted_by` isn't
mark. After registration the locked fields can't be edited (a hook blocks it). Next step: `/build H###`.
