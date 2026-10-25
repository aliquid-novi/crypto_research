# work_specific

Standalone notebooks that Mark copies **one way** onto his work machine and runs there on firm market data.

- Nothing comes back: no firm data, outputs, paths, fee tiers or executed notebooks from the work machine.
- Notebooks here are built and validated on synthetic data only, and use only pandas / numpy / matplotlib / pyarrow
  (no `qrlib`), so a single file is all that needs copying.
- Source of truth is the jupytext `.py`; `python -m qrlib.cli sync work_specific/<name>.py` regenerates the `.ipynb`.

| notebook | what |
|---|---|
| `mm_feasibility` | MM feasibility gates: quote Hyperliquid perps, hedge on Binance perps |
