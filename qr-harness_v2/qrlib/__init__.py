"""qrlib: deterministic research infrastructure (no models live here).

pit        point-in-time joins, targets, walk-forward splits
leakage    truncation test + data hygiene checks
fairvalue  baselines (mid, weighted mid) + evaluate()
fills      passive fill sim (through/touch) + markouts
signals    signal registry + redundancy filter
stats      block bootstrap, deflated Sharpe
audit      append-only JSONL trail, n_trials
prereg     hypothesis cards: registration lock, resolve, calibration
synth      synthetic market with a known latent price
data       loader for Mark's market_data_v2 parquet (config/data.yaml)
cli        python -m qrlib.cli ...
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
