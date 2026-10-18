"""
load_market_data.py

Loader for normalized market data stored as:

normalized/
  table=l1/
  table=l2/
  table=trades/
  table=sessions/

Typical path:
table=<table>/venue=<venue>/symbol=<symbol>/date=YYYY-MM-DD/hour=HH/*.parquet

Examples:
    from load_market_data import get_md

    df = get_md("l1", "HYPERLIQUID", "HYPE-USD-PERP", "2026-10-18")
    df = get_md("trades", "BINANCE_FUTURES", "HYPE-USDT-PERP")
    df = get_md("sessions", "HYPERLIQUID", date="2026-10-18")
"""

from __future__ import annotations

import os
from datetime import date as Date, datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


VALID_TABLES = {"l1", "l2", "trades", "sessions"}

# Force the hive partition columns to stable string types.
PARTITION_SCHEMA = pa.schema(
    [
        ("venue", pa.string()),
        ("symbol", pa.string()),
        ("date", pa.string()),
        ("hour", pa.string()),
    ]
)
PARTITIONING = ds.partitioning(PARTITION_SCHEMA, flavor="hive")


def _candidate_roots() -> list[Path]:
    candidates = []

    env_root = os.getenv("MARKET_DATA_V2_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            Path.home() / "market_data_v2" / "normalized",
            Path("/mnt/c/Users/markp/OneDrive/Documents/00Github/market_data_v2/normalized"),
            Path("C:/Users/markp/OneDrive/Documents/00Github/market_data_v2/normalized"),
        ]
    )

    out = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_data_root(
    data_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve the local normalized-data directory.

    The path should contain folders such as:
        table=l1
        table=l2
        table=trades
        table=sessions
    """
    if data_root is not None:
        root = Path(data_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return root

    candidates = _candidate_roots()
    for root in candidates:
        if root.exists():
            return root

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the normalized market-data root.\n"
        f"Checked:\n{checked}\n\n"
        "Pass data_root=... or set MARKET_DATA_V2_ROOT."
    )


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize colon variants for comparison.

    Useful because Windows/rclone may replace ':' in HIP-3 symbols with a
    full-width colon-like character.
    """
    return (
        symbol.strip()
        .replace("：", ":")
        .replace("﹕", ":")
    )


def _resolve_symbol_partition(
    table_root: Path,
    venue: str,
    symbol: str,
) -> str:
    """
    Return the actual local symbol partition value.
    """
    venue_dir = table_root / f"venue={venue}"

    if not venue_dir.exists():
        available = sorted(
            p.name.removeprefix("venue=")
            for p in table_root.glob("venue=*")
            if p.is_dir()
        )
        raise FileNotFoundError(
            f"Venue not found: {venue}\n"
            f"Available venues: {available}"
        )

    wanted = _normalize_symbol(symbol)

    for p in venue_dir.glob("symbol=*"):
        if not p.is_dir():
            continue

        actual = p.name.removeprefix("symbol=")
        if _normalize_symbol(actual) == wanted:
            return actual

    available = sorted(
        _normalize_symbol(p.name.removeprefix("symbol="))
        for p in venue_dir.glob("symbol=*")
        if p.is_dir()
    )

    raise FileNotFoundError(
        f"Symbol not found: {symbol}\n"
        f"Venue: {venue}\n"
        f"Available symbols: {available}"
    )


def _normalize_date(value: Union[str, Date, datetime]) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, Date):
        return value.isoformat()

    value = str(value).strip()

    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(
            f"date must be YYYY-MM-DD, not {value!r}"
        ) from exc


def get_md(
    table: str,
    venue: str,
    symbol: Optional[str] = None,
    date: Optional[Union[str, Date, datetime]] = None,
    *,
    data_root: Optional[Union[str, Path]] = None,
    columns: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load market data into a pandas DataFrame.

    Parameters
    ----------
    table:
        "l1", "l2", "trades", or "sessions".

    venue:
        e.g. "HYPERLIQUID" or "BINANCE_FUTURES".

    symbol:
        Required for l1/l2/trades.
        Optional for sessions. If omitted, "_session" is used.

    date:
        Optional YYYY-MM-DD.
        If omitted, every available date for that table/venue/symbol is loaded.

    data_root:
        Optional path to the normalized directory.

    columns:
        Optional list of columns to return.

    sort_by:
        Optional column name to sort by after loading.

    Returns
    -------
    pandas.DataFrame
    """
    table = table.strip().lower()
    venue = venue.strip()

    if table not in VALID_TABLES:
        raise ValueError(
            f"Invalid table {table!r}. "
            f"Choose one of {sorted(VALID_TABLES)}."
        )

    if not venue:
        raise ValueError("venue cannot be blank.")

    if table == "sessions":
        symbol = symbol.strip() if symbol else "_session"
    else:
        if symbol is None or not symbol.strip():
            raise ValueError(
                f"symbol is required for table={table!r}."
            )
        symbol = symbol.strip()

    root = resolve_data_root(data_root)
    table_root = root / f"table={table}"

    if not table_root.exists():
        raise FileNotFoundError(
            f"Table directory does not exist: {table_root}"
        )

    actual_symbol = _resolve_symbol_partition(
        table_root=table_root,
        venue=venue,
        symbol=symbol,
    )

    dataset = ds.dataset(
        table_root,
        format="parquet",
        partitioning=PARTITIONING,
        exclude_invalid_files=True,
    )

    filt = (
        (ds.field("venue") == venue)
        & (ds.field("symbol") == actual_symbol)
    )

    normalized_date = None

    if date is not None:
        normalized_date = _normalize_date(date)

        date_dir = (
            table_root
            / f"venue={venue}"
            / f"symbol={actual_symbol}"
            / f"date={normalized_date}"
        )

        if not date_dir.exists():
            symbol_dir = (
                table_root
                / f"venue={venue}"
                / f"symbol={actual_symbol}"
            )

            available_dates = sorted(
                p.name.removeprefix("date=")
                for p in symbol_dir.glob("date=*")
                if p.is_dir()
            )

            raise FileNotFoundError(
                f"No data found for {normalized_date}.\n"
                f"Available dates: {available_dates}"
            )

        filt = filt & (ds.field("date") == normalized_date)

    arrow_table = dataset.to_table(
        filter=filt,
        columns=list(columns) if columns is not None else None,
    )

    df = arrow_table.to_pandas()

    if df.empty:
        raise FileNotFoundError(
            "No rows matched the request:\n"
            f"table={table}, venue={venue}, symbol={symbol}, "
            f"date={normalized_date or 'ALL'}"
        )

    if "table" not in df.columns:
        df.insert(0, "table", table)

    if sort_by is not None:
        if sort_by not in df.columns:
            raise KeyError(
                f"sort_by={sort_by!r} not found. "
                f"Available columns: {list(df.columns)}"
            )

        df = (
            df.sort_values(sort_by, kind="stable")
            .reset_index(drop=True)
        )

    return df


# Alias if you prefer a more generic function name.
load_data = get_md


def available_symbols(
    table: str,
    venue: str,
    *,
    data_root: Optional[Union[str, Path]] = None,
) -> list[str]:
    """List locally available symbols for a table/venue."""
    table = table.strip().lower()
    venue = venue.strip()

    if table not in VALID_TABLES:
        raise ValueError(
            f"Invalid table {table!r}. "
            f"Choose one of {sorted(VALID_TABLES)}."
        )

    root = resolve_data_root(data_root)
    venue_dir = root / f"table={table}" / f"venue={venue}"

    if not venue_dir.exists():
        return []

    return sorted(
        _normalize_symbol(p.name.removeprefix("symbol="))
        for p in venue_dir.glob("symbol=*")
        if p.is_dir()
    )


def available_dates(
    table: str,
    venue: str,
    symbol: Optional[str] = None,
    *,
    data_root: Optional[Union[str, Path]] = None,
) -> list[str]:
    """List locally available dates for a table/venue/symbol."""
    table = table.strip().lower()
    venue = venue.strip()

    if table == "sessions" and not symbol:
        symbol = "_session"

    if not symbol:
        raise ValueError(
            "symbol is required unless table='sessions'."
        )

    root = resolve_data_root(data_root)
    table_root = root / f"table={table}"

    actual_symbol = _resolve_symbol_partition(
        table_root,
        venue,
        symbol,
    )

    symbol_dir = (
        table_root
        / f"venue={venue}"
        / f"symbol={actual_symbol}"
    )

    return sorted(
        p.name.removeprefix("date=")
        for p in symbol_dir.glob("date=*")
        if p.is_dir()
    )
