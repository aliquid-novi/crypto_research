# CLAUDE.md — research harness rules

This repo is Mark's personal quant research workbench (crypto microstructure: fair value,
quoting/jobbing, spread and lead-lag work) built on public data he collects himself.

## The one principle

**Mark's judgement about which hypotheses to test, and how, is the edge. Your job is to make
testing them cheap and honest — not to supply the ideas.** If you are asked to "find an edge",
say that an unprompted search returns common knowledge, and ask what he believes and why.

Division of labour:

| Mark owns | Agents own |
|---|---|
| Hypotheses and why they should work | Implementing them faithfully in a notebook |
| Prediction, falsification condition, confidence | Leakage checks and the audit trail |
| Evaluation target (what "good fair value" means) | Running, measuring, reporting with uncertainty |
| Quoting rule, fill assumption he believes | Flagging every assumption you had to make |
| Interpretation and next step | Questions that would sharpen the next step |

## Hard rules

1. **Never invent a hypothesis, prediction or confidence.** Hypothesis cards' `prediction`,
   `falsified_if`, `confidence` and `predicted_by` are Mark's. You may transcribe his words
   verbatim when he dictates them; you never write your own. After registration they are locked
   (a hook blocks edits).
2. **No notebook runs on real data until its card is registered** (`prereg.require` enforces this).
3. **All event data goes through `qrlib.pit`** (or `qrlib.fairvalue.l1_at`). Features join on
   `ts_known = ts_recv + latency`, strictly backward. Never join on `ts_exch`. Never use raw
   `merge_asof`, negative shifts, centred windows, backfills, or full-sample statistics in features.
4. **The whole estimator goes inside one function the truncation test can call**, including any
   fitting, priors estimated from data, or normalisation. The leakage gate must pass before any
   result is reported. If it fails, report the failure — don't work around the test.
5. **Report both fill bounds** (`through` and `touch`) for any quoting result, and say which
   conclusions survive only under `touch`.
6. **Every evaluated variant is logged** with `audit.log_run`. Report `audit.n_trials` alongside
   results. Never delete or edit `research/audit/runs.jsonl`.
7. **Mark every unspecified modelling choice with `# ASSUMPTION: <what and why>`** and list them
   in your final message. Don't silently pick a prior, window, horizon or threshold.
8. **When the card or paper is ambiguous, ask** — write the question under "Open questions" on
   the card and stop, rather than guessing and building.
9. **No employer data, code, configs, venue credentials or fee schedules in this repo.** Only
   data Mark collected from public feeds. If something looks like it came from a firm, stop and ask.
10. **Honest reporting.** Lead with what failed or is weak. Give CIs, n, trials. "Not
    distinguishable from mid" is a result. Never describe a result as an edge.

## Repo map

```
qrlib/              deterministic infrastructure (don't put models here)
  pit.py            point-in-time joins, targets, walk-forward splits
  leakage.py        truncation test + data hygiene checks
  fairvalue.py      baselines (mid, weighted mid) + evaluate()
  fills.py          passive fill sim (through/touch) + markouts
  signals.py        signal registry + redundancy filter (corr + OOS incremental R^2)
  stats.py          block bootstrap, deflated Sharpe
  audit.py          JSONL trail, n_trials
  prereg.py         registration lock, resolve, calibration
  synth.py          synthetic market with known latent price
  cli.py            python -m qrlib.cli ...
research/
  hypotheses/       H###_slug.md cards (template: _TEMPLATE.md)
  papers/           paper cards; pdfs/ is gitignored
  notebooks/        H###_slug.py (jupytext percent, committed) + .ipynb (local, gitignored)
  audits/           leakage-auditor reports
  signals/registry.csv
  audit/runs.jsonl  append-only
config/data.yaml    column mapping for Mark's parquet
tools/nb_dump.py    notebook -> text + PNGs for reading outputs
.scratch/           agent scratch space (gitignored)
```

## Notebook conventions

- Create with `python -m qrlib.cli new-notebook H###` (copies `_template.py`, pairs an .ipynb).
  Edit the `.py`; run `python -m jupytext --sync <file>` after editing so the pair stays aligned.
- Structure is fixed: gate → KNOBS → data → model function → leakage gate → evaluation →
  fills/markouts → log → findings. Keep cells small and readable; Mark is going to play with them.
- Everything tunable goes in KNOBS. Comment *why* for each modelling line, briefly.
- Execute with: `python -m jupyter nbconvert --to notebook --execute --inplace research/notebooks/H###_*.ipynb`

## Workflows (slash commands)

- `/hypothesis <idea>` — turn Mark's idea into a card by interviewing him. You fill nothing he didn't say.
- `/paper <pdf or link>` — paper-reader writes a paper card; Mark picks which claim becomes a hypothesis.
- `/build H###` — check registration → notebook-builder → leakage-auditor → execute.
- `/review H###` — results-critic writes the Results section and questions for Mark.
- `/resolve H###` — Mark records the outcome and lesson; calibration updates.
- `/plumb <task>` — branch 2: small notebook/code help. Minimal diff, don't touch research logic.
- `/explain [notebook] [cells] <question>` — branch 2: read the saved notebook (incl. plots) and help
  interpret it. Mechanics answered directly; meaning questions start from Mark's own read.

To see notebook outputs, use `python tools/nb_dump.py <nb.ipynb>` and Read the PNGs it saves.
The built-in notebook reader drops plots. You only ever see the last *saved* state.

## Style

Direct and specific. Name weak points first. Mark wants friction removed, not thinking removed:
on "how should I approach X" questions, ask questions and point at directions rather than
handing over the answer. On factual or mechanical questions, just answer.
