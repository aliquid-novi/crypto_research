"""Loader for Mark's market_data_v2 parquet, driven by config/data.yaml.

    from qrlib import data
    q  = data.load("l1", "HYPERLIQUID", "HYPE-USD-PERP", start="2026-10-18", end="2026-10-19 06:00")
    tr = data.load("trades", "BINANCE_FUTURES", "HYPE-USDT-PERP")
    data.coverage("HYPERLIQUID", "HYPE-USD-PERP")   # which date-hours exist locally (the sync can have holes)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pyarrow.dataset as ds
import yaml

from . import ROOT


@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))


def data_root():
    return (ROOT / config()["root"]).resolve()


def _path_symbol(symbol: str) -> str:
    for a, b in config().get("path_char_map", {}).items():
        symbol = symbol.replace(a, b)
    return symbol


def _sessions(venue: str) -> pd.Series:
    p = data_root() / "table=sessions"
    s = ds.dataset(p, format="parquet", partitioning="hive").to_table().to_pandas()
    s = s[(s.venue == venue) & (s.kind == "ws") & (~s.resumed)]
    return s.groupby("ordinal").opened_ns.min()


def load(table: str, venue: str, symbol: str, start=None, end=None) -> pd.DataFrame:
    """Standardised frame (see config/data.yaml), sorted by ts_recv. start/end filter on ts_exch."""
    spec = config()["tables"][table][venue]
    p = data_root() / f"table={table}" / f"venue={venue}" / f"symbol={_path_symbol(symbol)}"
    raw_cols = [spec["ts_exch"], spec["ts_recv"], *spec["cols"].values()]
    d = ds.dataset(p, format="parquet", partitioning="hive")
    have = set(d.schema.names)
    raw_cols = [c for c in raw_cols if c in have] + (["conn_ordinal"] if "conn_ordinal" in have else [])
    df = d.to_table(columns=raw_cols).to_pandas()
    if spec.get("drop_subscribe_replays") and "conn_ordinal" in df:
        opened = df.conn_ordinal.map(_sessions(venue))
        df = df[~(df[spec["ts_exch"]] < opened)]
    out = pd.DataFrame({"ts_exch": pd.to_datetime(df[spec["ts_exch"]].astype("int64"), unit="ns"),
                        "ts_recv": pd.to_datetime(df[spec["ts_recv"]].astype("int64"), unit="ns")})
    for std, raw in spec["cols"].items():
        if raw in df:
            out[std] = df[raw].to_numpy()
    if "conn_ordinal" in df:
        out["conn_ordinal"] = df["conn_ordinal"].to_numpy()
    if start is not None:
        out = out[out.ts_exch >= pd.Timestamp(start)]
    if end is not None:
        out = out[out.ts_exch < pd.Timestamp(end)]
    return out.sort_values("ts_recv", kind="stable").reset_index(drop=True)


def coverage(venue: str, symbol: str, tables=("l1", "trades")) -> pd.DataFrame:
    """Date-hours present locally per table (True/False), to spot holes before splitting train/test."""
    rows = {}
    for t in tables:
        base = data_root() / f"table={t}" / f"venue={venue}" / f"symbol={_path_symbol(symbol)}"
        for f in base.glob("date=*/hour=*"):
            rows.setdefault((f.parent.name[5:], f.name[5:]), {})[t] = any(f.glob("*.parquet"))
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().fillna(False)


def pair(hl_symbol: str) -> str:
    return config()["pairs"][hl_symbol]
