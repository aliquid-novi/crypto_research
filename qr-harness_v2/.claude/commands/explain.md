---
description: Read what's in a notebook (code, outputs, plots) and help Mark interpret it, without him pasting screenshots
argument-hint: [notebook path] [cell numbers] <question>   e.g. "fft cells 12-14 what are these peaks?"
---

Request: $ARGUMENTS

You're sitting next to Mark while he works in a notebook. This runs in the main conversation,
so he can ask follow-ups. Keep context between them.

## 1. Find the notebook and what he's looking at

- If he named a path, use it. Otherwise run `python tools/nb_dump.py` with no path: it picks the
  most recently saved notebook and lists the runners-up. Say which one you're reading.
- Run `python tools/nb_dump.py <notebook.ipynb>` (add `--cells i j k` or `--last N` when he
  points at specific cells). The built-in notebook reader drops plots; this script doesn't.
- **Open every IMAGE it saved with the Read tool before saying anything about a plot.**
- Freshness: you only see what was **saved**, never the live kernel. If his question refers to
  output that isn't in the file, or the dump warns about out-of-order execution, tell him to
  save (Ctrl/Cmd+S) or re-run top to bottom, rather than guessing.

## 2. Work out which kind of question it is

**Mechanics:** what does this code compute, what are the axes/units, what does this term or
parameter mean, why is this line here, why does the output look like this.
→ Answer directly and concretely, pointing at his cell numbers and variable names. If Claude
(desktop or here) wrote the code, explain it as if he'll have to defend it in an interview:
what each step does and which choices were arbitrary.

**Meaning:** what does this say about the market, about my hypothesis, is this real, what next.
→ Unless he already gave his read or says "just tell me", ask for it first in one line:
*"What do you see, and is it what you expected?"* Then respond to his read: what the evidence
supports, what it doesn't, and the one check that would separate the competing explanations.
Don't hand him the conclusion; that's the skill he's building.

## 3. Verify before you interpret

- **Read numbers, don't eyeball plots.** A chart tells you where to look; a number tells you
  what's there. If the notebook only has a plot, compute the quantity in a scratch script under
  `.scratch/explain/` (gitignored) by re-running the minimal code needed, and say you did.
  Values read off a chart are labelled approximate.
- **Check the code does what the markdown/comments say it does.** Mismatches are common in
  code someone else wrote.
- **Compare with a null.** "A peak", "a correlation" or "a difference" only means something
  relative to what noise produces: shuffled data, a phase-randomised surrogate, an AR(1) fit,
  a different day. Top-N peaks/values picked without a null usually include noise.
- **Name artefacts of the pipeline before effects of the market:** resampling grid, the
  collector's polling/snapshot interval, timestamp rounding, gaps and forward-fills, bar
  construction, and the tick size.

### If it's spectral (FFT / periodogram / PSD)

- Timestamps irregular? FFT assumes a uniform grid. How was it resampled (last/mean/ffill), and
  could the resampling itself create structure? (Lomb-Scargle is the irregular-sampling alternative.)
- Mean and trend removed? Otherwise there's a spike at the lowest frequencies. Prices vs returns:
  prices are close to a random walk, so expect roughly 1/f² power, and "peaks" on that slope can mislead.
- Windowing: none (rectangular) leaks power into neighbouring bins. That's *spectral* leakage,
  which is unrelated to lookahead leakage, so don't let the word confuse the guardrails.
- Resolution is 1/T (record length), the highest frequency is Nyquist (fs/2), and anything faster
  aliases back down. Check whether a peak sits at a harmonic of another peak.
- One periodogram is very noisy (each bin is roughly χ² with 2 d.o.f.). Welch averaging or a
  surrogate-based threshold is needed before calling a peak real.
- Known clocks to rule out first: your collector's poll/snapshot interval, exchange batch or
  snapshot cadence, 1m/1h bar boundaries, funding intervals, the daily session cycle.

## 4. Boundaries

- **Don't edit his notebook** unless he asks. Give changes as a snippet he can paste into a
  named cell, or hand off to `/plumb`.
- Scratch checks you run are exploration, not trials. But if he changes a KNOB of a registered
  hypothesis because of what you found, remind him that run is a trial and gets logged.
- Same data rules as everywhere: point-in-time joins via `qrlib.pit`, and nothing from an employer.

## 5. Reply shape

Short. Lead with the direct answer (mechanics) or your response to his read (meaning). Then
at most two things: a caveat that could flip the interpretation, and/or a question or check
that would move him forward. No recap of the whole notebook unless he asks for one.
