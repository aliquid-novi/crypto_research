"""Table schemas, normalisation and validation, plus the "what can I run" availability report.

Column kinds:
  ts        -> int64 ns UTC (converted from datetimes / other integer units)
  float     -> float64
  int       -> int64
  str       -> object/str
  side      -> int8 in {+1, -1} (mapped through config.SIDE_MAP)
  liquidity -> 'maker' | 'taker'
  status    -> 'filled' | 'cancelled' | 'partial' | 'rejected'
  leg       -> 'A' | 'B'
  leg_role  -> 'entry' | 'exit'

``prepare(df, table)`` = rename via COLUMN_MAP -> coerce kinds -> validate. It raises
``SchemaError`` listing EVERY problem (missing columns, uncoercible values, bad categories)
rather than the first one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from ._util import to_ns


class SchemaError(ValueError):
    """Raised when a table does not satisfy its schema. Message lists every problem found."""


@dataclass(frozen=True)
class TableSchema:
    name: str
    required: dict[str, str]
    optional: dict[str, str] = field(default_factory=dict)
    sort_by: tuple[str, ...] = ("ts",)


SCHEMAS: dict[str, TableSchema] = {
    "fills": TableSchema(
        "fills",
        required={"fill_id": "str", "order_id": "str", "ts": "ts", "venue": "str", "symbol": "str",
                  "side": "side", "price": "float", "qty": "float", "notional": "float", "fee": "float",
                  "liquidity": "liquidity"},
        optional={"counterparty": "str", "decision_id": "str", "leg": "leg", "leg_role": "leg_role",
                  "trade_id": "str", "ts_local": "ts"},
    ),
    "book_l1": TableSchema(
        "book_l1",
        required={"ts": "ts", "venue": "str", "symbol": "str", "bid_px": "float", "bid_sz": "float",
                  "ask_px": "float", "ask_sz": "float"},
        optional={"ts_local": "ts"},
    ),
    "market_trades": TableSchema(
        "market_trades",
        required={"ts": "ts", "venue": "str", "symbol": "str", "price": "float", "qty": "float",
                  "aggressor_side": "side"},
        optional={"buyer": "str", "seller": "str", "trade_id": "str", "ts_local": "ts"},
    ),
    "orders": TableSchema(
        "orders",
        required={"order_id": "str", "venue": "str", "symbol": "str", "side": "side", "price": "float",
                  "qty": "float", "ts_sent": "ts", "ts_ack": "ts", "ts_cancel_sent": "ts", "ts_done": "ts",
                  "final_status": "status"},
        optional={"ts_md_recv": "ts", "ts_signal": "ts", "filled_qty": "float"},
        sort_by=("ts_sent",),
    ),
    "decisions": TableSchema(
        "decisions",
        required={"decision_id": "str", "ts": "ts", "pair": "str", "signal": "float",
                  "spread_fair_value": "float", "intended_qty": "float", "hedge_ratio": "float",
                  "leg_A_symbol": "str", "leg_A_venue": "str", "leg_B_symbol": "str", "leg_B_venue": "str",
                  "exit_rule": "str"},
        optional={"ts_exit_decision": "ts"},
    ),
    "funding": TableSchema(
        "funding",
        required={"ts": "ts", "venue": "str", "symbol": "str", "rate": "float", "mark": "float",
                  "position_at_settlement": "float", "payment": "float"},
    ),
    "positions": TableSchema(
        "positions", required={"ts": "ts", "venue": "str", "symbol": "str", "position": "float"},
    ),
    "instruments": TableSchema(
        "instruments",
        required={"venue": "str", "symbol": "str"},
        optional={"tick_size": "float", "multiplier": "float", "quote_ccy": "str", "kind": "str"},
        sort_by=(),
    ),
}
"""Every column in ``required`` must be present. Columns allowed to be all-NaN even though
required: orders.ts_cancel_sent (NaN = never cancelled), decisions.exit_rule (free text)."""

NULLABLE: dict[str, set[str]] = {
    "orders": {"ts_cancel_sent", "ts_done", "ts_ack"},
    "decisions": {"exit_rule", "spread_fair_value", "ts_exit_decision"},
    "fills": {"fee"},
}

_CATS = {
    "liquidity": ("maker", "taker"),
    "status": ("filled", "cancelled", "partial", "rejected"),
    "leg": ("A", "B"),
    "leg_role": ("entry", "exit"),
}


def apply_column_map(df: pd.DataFrame, table: str, column_map: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """Rename YOUR columns to schema names using ``column_map[table]`` (defaults to config.COLUMN_MAP)."""
    cmap = (column_map or config.COLUMN_MAP).get(table, {})
    return df.rename(columns=cmap)


def _map_side(s: pd.Series) -> tuple[pd.Series, int]:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        v = v.where(v.isin([1, -1]))
    else:
        v = s.astype(str).str.strip().str.lower().map({str(k).lower(): val for k, val in config.SIDE_MAP.items()})
    bad = int(v.isna().sum() - s.isna().sum())
    return v, bad


def _coerce(df: pd.DataFrame, col: str, kind: str, ts_unit: str, problems: list[str]) -> None:
    s = df[col]
    try:
        if kind == "ts":
            notna = s.notna().to_numpy()
            vals = np.zeros(len(s), dtype="int64")
            if notna.any():
                vals[notna] = to_ns(s[notna], ts_unit, config.NAIVE_TZ)
            # all present -> plain int64; some missing -> nullable Int64 (keeps ns precision)
            df[col] = vals if notna.all() else pd.arrays.IntegerArray(vals, ~notna)
        elif kind == "float":
            v = pd.to_numeric(s, errors="coerce")
            bad = int(v.isna().sum() - s.isna().sum())
            if bad:
                problems.append(f"{col}: {bad} values not numeric (e.g. {s[v.isna() & s.notna()].iloc[0]!r})")
            df[col] = v.astype("float64")
        elif kind == "int":
            df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif kind == "str":
            df[col] = s.astype(str).where(s.notna(), None)
        elif kind == "side":
            v, bad = _map_side(s)
            if bad:
                ex = s[v.isna() & s.notna()].iloc[0]
                problems.append(f"{col}: {bad} values not mappable to +1/-1 (e.g. {ex!r}); extend config.SIDE_MAP")
            df[col] = v.astype("float64").fillna(0).astype("int8") if not v.isna().any() else v
        elif kind in _CATS:
            mapping = {"liquidity": config.LIQUIDITY_MAP, "status": config.STATUS_MAP}.get(kind)
            v = s.astype(str).str.strip()
            v = v.str.lower().map(mapping) if mapping else v.str.upper() if kind == "leg" else v.str.lower()
            v = v.where(s.notna(), None)
            bad_mask = v.notna() & ~v.isin(_CATS[kind])
            bad_mask |= v.isna() & s.notna()
            if bad_mask.any():
                problems.append(f"{col}: {int(bad_mask.sum())} values outside {_CATS[kind]} "
                                f"(e.g. {s[bad_mask].iloc[0]!r})")
            df[col] = v
    except Exception as exc:  # report, never crash the validator
        problems.append(f"{col}: could not coerce to {kind}: {exc}")


def prepare(df: pd.DataFrame, table: str, column_map: dict[str, dict[str, str]] | None = None,
            ts_unit: str | None = None, strict: bool = True) -> pd.DataFrame:
    """Rename, coerce and validate one table; returns a sorted copy in schema form.

    Raises SchemaError (strict=True) listing every missing column and every coercion problem.
    With strict=False problems are attached as ``df.attrs['schema_problems']`` instead.
    """
    if table not in SCHEMAS:
        raise KeyError(f"unknown table {table!r}; known: {sorted(SCHEMAS)}")
    sch = SCHEMAS[table]
    out = apply_column_map(df.copy(), table, column_map)
    problems: list[str] = []
    missing = [c for c in sch.required if c not in out.columns]
    if missing:
        have = ", ".join(map(str, df.columns[:30]))
        problems.append(f"missing required columns {missing}. Columns present after COLUMN_MAP['{table}']: "
                        f"[{', '.join(map(str, out.columns[:30]))}] (raw: [{have}])")
    unit = ts_unit or config.TS_UNITS.get(table, config.DEFAULT_TS_UNIT)
    for col, kind in {**sch.required, **sch.optional}.items():
        if col in out.columns:
            _coerce(out, col, kind, unit, problems)
    nullable = NULLABLE.get(table, set())
    for col in sch.required:
        if col in out.columns and col not in nullable and out[col].isna().any():
            problems.append(f"{col}: {int(out[col].isna().sum())} null values in a required column")
    if problems:
        msg = f"[{table}] schema problems:\n  - " + "\n  - ".join(problems)
        if strict:
            raise SchemaError(msg)
        out.attrs["schema_problems"] = problems
    sort_cols = [c for c in sch.sort_by if c in out.columns]
    if sort_cols and not problems:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def validate(df: pd.DataFrame, table: str) -> list[str]:
    """Return a list of problems for an already-prepared table (empty list = OK). Never raises."""
    try:
        prepare(df, table, column_map={}, strict=True)
        return []
    except SchemaError as e:
        return str(e).splitlines()[1:]


# ---------------------------------------------------------------------------------------------
# Availability: which analyses can run with the tables provided
# ---------------------------------------------------------------------------------------------
# analysis -> (notebook, list of requirements). A requirement is "table" or "table.column".
ANALYSES: dict[str, tuple[str, list[str]]] = {
    "position reconciliation": ("01", ["fills", "positions"]),
    "fee recompute": ("01", ["fills"]),
    "funding recompute": ("01", ["funding"]),
    "fill dedupe": ("01", ["fills.trade_id"]),
    "bad-tick filter (trades)": ("01", ["market_trades"]),
    "book sanity + feed gaps": ("01", ["book_l1"]),
    "noise variance / signature plot": ("01", ["book_l1"]),
    "markouts (wall clock)": ("02", ["fills", "book_l1"]),
    "markouts vs lead venue": ("02", ["fills", "book_l1", "config.LEAD_MAP"]),
    "markouts (event time: trades/volume)": ("02", ["fills", "book_l1", "market_trades"]),
    "microprice reference": ("02", ["book_l1"]),
    "adverse-selection ratio": ("03", ["orders", "book_l1"]),
    "late cancels": ("03", ["orders", "fills"]),
    "counterparty concentration": ("03", ["fills.counterparty", "book_l1"]),
    "venue-wide taker toxicity": ("03", ["market_trades.buyer", "market_trades.seller", "book_l1"]),
    "spread round trips + shortfall": ("04", ["fills.decision_id", "fills.leg", "fills.leg_role", "decisions", "book_l1"]),
    "shortfall carry (funding)": ("04", ["decisions", "funding"]),
    "markout regressions / toxicity model": ("05", ["fills", "book_l1"]),
    "fill probability / hazards / selection": ("05", ["orders", "book_l1"]),
    "Hawkes (own fills x market)": ("05", ["fills", "market_trades"]),
    "response function / propagator / VAR": ("05", ["market_trades", "book_l1"]),
    "OU / Kalman / alpha decay": ("05", ["decisions", "book_l1"]),
    "latency per hop": ("06", ["orders.ts_md_recv", "orders.ts_signal"]),
    "latency vs markouts": ("06", ["orders", "fills", "book_l1"]),
    "research stats": ("07", []),
}


def availability_report(tables: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Which analyses can run with the tables provided, and why not when they can't.

    ``tables`` maps table name -> prepared DataFrame (or None / missing when unavailable).
    Returns DataFrame[analysis, notebook, status ('OK'|'UNAVAILABLE'), reason].
    """
    rows = []
    for name, (nb, reqs) in ANALYSES.items():
        reasons = []
        for r in reqs:
            if r.startswith("config."):
                if not getattr(config, r.split(".", 1)[1]):
                    reasons.append(f"{r} is empty")
                continue
            t, _, col = r.partition(".")
            df = tables.get(t)
            if df is None or len(df) == 0:
                reasons.append(f"no '{t}' table")
            elif col and (col not in df.columns or df[col].isna().all()):
                reasons.append(f"'{t}' has no '{col}' column (or it is all null)")
        rows.append({"analysis": name, "notebook": nb, "status": "OK" if not reasons else "UNAVAILABLE",
                     "reason": "; ".join(reasons)})
    return pd.DataFrame(rows)


def add_notional(fills: pd.DataFrame, instruments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return fills with ``notional = price * qty * multiplier`` (quote ccy, always positive).

    Use when your fills have no notional column (CME: qty in contracts, multiplier from the
    instruments table; perps usually multiplier 1).
    """
    from ._util import instrument_lookup

    out = fills.copy()
    mult = instrument_lookup(out, instruments, "multiplier", config.DEFAULT_MULTIPLIER)
    out["notional"] = np.abs(out["price"].astype(float) * out["qty"].astype(float) * mult)
    return out
