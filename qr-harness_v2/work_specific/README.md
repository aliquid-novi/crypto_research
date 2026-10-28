# work_specific

Standalone notebooks that Mark copies **one way** onto his work machine and runs there on firm market data.

- Nothing comes back: no firm data, outputs, paths, fee tiers or executed notebooks from the work machine.
- Notebooks here are built and validated on synthetic data only, and use only pandas / numpy / matplotlib / pyarrow
  (no `qrlib`). Copy the notebook **and its `_lib.py`** into the same folder.
- Notebooks stay lean (config + one call per step + the reasoning in markdown); the code lives in `<name>_lib.py`.
- Source of truth is the jupytext `.py`; `python -m qrlib.cli sync work_specific/<name>.py` regenerates the `.ipynb`.

| notebook | lib | what |
|---|---|---|
| `mm_feasibility` | `mm_feasibility_lib.py` | MM feasibility gates: quote Hyperliquid perps, hedge on Binance perps |
| `posttrade_toolkit/` (copy the whole folder) | `posttrade/` package | Exchange-agnostic post-trade analytics: markouts, adverse selection, spread shortfall, models, latency, research stats. Also needs scipy + statsmodels; own runner `python run.py test / demo / notebooks` |
