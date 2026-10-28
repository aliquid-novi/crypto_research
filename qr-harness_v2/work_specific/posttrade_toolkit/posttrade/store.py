"""Hand-off between notebooks: 00 validates and caches tables; 01-07 load them.

Cache = one pickle per table in ``<toolkit root>/data_cache/`` (gitignored). If the cache is
missing, ``load_tables`` generates the synthetic day instead and says so.
"""

from __future__ import annotations

import os
import pickle

import pandas as pd

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `python run.py demo` points this at a separate folder so it never overwrites your real cache.
CACHE = os.environ.get("POSTTRADE_CACHE", os.path.join(ROOT, config.CACHE_DIR_NAME))
FORCE_SYNTHETIC = os.environ.get("POSTTRADE_FORCE_SYNTHETIC", "") == "1"
LOG_PATH = os.path.join(CACHE, "research_log.csv")
TABLES = ["fills", "book_l1", "market_trades", "orders", "decisions", "funding", "positions", "instruments"]


def save_tables(tables: dict[str, pd.DataFrame | None], meta: dict | None = None, cache_dir: str = CACHE) -> str:
    """Pickle each non-empty table and a meta dict (source, truth for synthetic) to ``cache_dir``."""
    os.makedirs(cache_dir, exist_ok=True)
    for name in TABLES:
        path = os.path.join(cache_dir, f"{name}.pkl")
        df = tables.get(name)
        if df is not None and len(df):
            df.to_pickle(path)
        elif os.path.exists(path):
            os.remove(path)  # a table you no longer provide must not linger from an old run
    with open(os.path.join(cache_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta or {}, f)
    return cache_dir


def load_tables(cache_dir: str = CACHE, synthetic_hours: float = 24.0) -> tuple[dict[str, pd.DataFrame | None], dict]:
    """Return (tables, meta). Missing tables are None. Falls back to a synthetic day if no cache."""
    if not os.path.exists(os.path.join(cache_dir, "meta.pkl")):
        from . import synthetic

        print("No cache found - using a synthetic day. Run 00_plug_your_data.ipynb to cache your data.")
        tables, truth = synthetic.generate_day(synthetic.SynthConfig(hours=synthetic_hours))
        return tables, {"source": "synthetic", "truth": truth}
    tables = {}
    for name in TABLES:
        p = os.path.join(cache_dir, f"{name}.pkl")
        tables[name] = pd.read_pickle(p) if os.path.exists(p) else None
    with open(os.path.join(cache_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    print(f"Loaded cached tables from {cache_dir} (source: {meta.get('source', '?')}): "
          + ", ".join(f"{k}={len(v):,}" for k, v in tables.items() if v is not None))
    return tables, meta
