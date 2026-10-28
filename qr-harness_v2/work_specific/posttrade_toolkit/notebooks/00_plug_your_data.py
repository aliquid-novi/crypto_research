# %% [markdown]
# # 00 · Plug your data
#
# **Purpose.** Load your tables, map your column names onto the toolkit's schemas, validate them, and cache
# them for notebooks 01–07. This is the only notebook you should need to edit.
#
# **Tables.** `fills` (required for almost everything), `book_l1` (required for markouts), and optionally
# `market_trades`, `orders`, `decisions`, `funding`, `positions`, `instruments`. See README → *Plugging in real data*.
#
# **What to look at first.** The validation report at the bottom: every analysis is listed as OK or
# UNAVAILABLE, with the reason.
#
# **Exchange-agnostic.** Nothing here assumes a venue. Venue-specific facts (trading sessions such as the CME
# daily halt, contract multipliers, tick sizes, fee models, how sides and timestamps are encoded) go in the
# EDIT HERE cell or in `posttrade/config.py`.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import pandas as pd

from posttrade import config, schemas, store, synthetic
from posttrade.config import SessionSpec, VenueSpec

pd.set_option("display.width", 200)

# %% [markdown]
# ## ✏️ EDIT HERE
#
# 1. Set `USE_SYNTHETIC = False`.
# 2. Fill in each `load_*` function (return a DataFrame, or `None` if you don't have the table).
# 3. Map **your** column names to schema names in `COLUMN_MAP` (only the ones that differ).
# 4. Say how integer timestamps are encoded (`TS_UNITS`) and add any venue that isn't in `config.VENUES`.

# %%
USE_SYNTHETIC = True  # <- one switch: True = synthetic demo day, False = your loaders below
SYNTH_HOURS = float(os.environ.get("POSTTRADE_SYNTH_HOURS", 24))

DATA_DIR = r"C:\path\to\your\exports"  # <- edit


def load_fills():
    # e.g. return pd.read_csv(os.path.join(DATA_DIR, "fills.csv"))
    return None


def load_book_l1():
    # e.g. return pd.read_pickle(os.path.join(DATA_DIR, "bbo.pkl"))  (one row per top-of-book update)
    return None


def load_market_trades():
    return None


def load_orders():
    return None


def load_decisions():
    return None


def load_funding():
    return None


def load_positions():
    return None


def load_instruments():
    # one row per (venue, symbol): tick_size, multiplier (contract size, e.g. CME micro BTC = 0.1 BTC), quote_ccy
    return None


# YOUR column name -> schema column name, per table (only those that differ)
COLUMN_MAP = {
    "fills": {
        # "exec_id": "fill_id", "client_order_id": "order_id", "transact_time": "ts", "exchange": "venue",
        # "instrument": "symbol", "side": "side", "px": "price", "size": "qty", "commission": "fee",
        # "liquidity_flag": "liquidity",
    },
    "book_l1": {},
    "market_trades": {},
    "orders": {},
    "decisions": {},
    "funding": {},
    "positions": {},
    "instruments": {},
}

# How INTEGER timestamps are encoded per table: 's', 'ms', 'us' or 'ns' (datetimes are detected automatically)
TS_UNITS = {
    # "fills": "ms", "book_l1": "us",
}

# Extra side labels your data uses (case-insensitive) -> +1 buy / -1 sell
EXTRA_SIDES = {
    # "BOT": 1, "SLD": -1,
}

# Venues not already in config.VENUES. sessions=None means 24/7.
EXTRA_VENUES = {
    # "CME": VenueSpec("CME", "future", config.CME_GLOBEX, None, False, "EXAMPLE_PER_CONTRACT"),
    # "MYVENUE": VenueSpec("MYVENUE", "perp", None, 8.0, False, "EXAMPLE_BPS"),
}

# Your fee schedule(s): {'model': 'bps'|'per_contract', 'maker': x, 'taker': y}; negative = rebate
EXTRA_FEES = {
    # "MY_TIER": {"model": "bps", "maker": 0.0, "taker": 2.5},
}

# Lead venue for "lead-venue mid" markouts: (lead_venue, lead_symbol) -> (our_venue, our_symbol)
LEAD_MAP = {
    # ("BINANCE", "BTCUSDT"): ("HYPERLIQUID", "BTC"),
}

# %% [markdown]
# ## Apply the settings and load

# %%
config.COLUMN_MAP.update(COLUMN_MAP)
config.TS_UNITS.update(TS_UNITS)
config.SIDE_MAP.update({str(k).lower(): v for k, v in EXTRA_SIDES.items()})
config.VENUES.update(EXTRA_VENUES)
config.FEE_SCHEDULES.update(EXTRA_FEES)
if LEAD_MAP:
    config.LEAD_MAP.clear()
    config.LEAD_MAP.update(LEAD_MAP)

use_synth = USE_SYNTHETIC or store.FORCE_SYNTHETIC
if use_synth:
    raw, truth = synthetic.generate_day(synthetic.SynthConfig(hours=SYNTH_HOURS))
    column_map = {}  # synthetic data is already in schema form
    meta = {"source": "synthetic", "truth": truth}
    print(f"Synthetic day: {SYNTH_HOURS:g} hours, venues VA (ours, leg A), VB (leg B), VL (lead).")
else:
    loaders = {"fills": load_fills, "book_l1": load_book_l1, "market_trades": load_market_trades, "orders": load_orders,
               "decisions": load_decisions, "funding": load_funding, "positions": load_positions,
               "instruments": load_instruments}
    raw = {k: f() for k, f in loaders.items()}
    column_map = config.COLUMN_MAP
    meta = {"source": "your data", "truth": None}

# %% [markdown]
# ## Validate every table
#
# Each table is renamed with `COLUMN_MAP`, coerced (timestamps → int64 ns UTC, sides → ±1, liquidity →
# maker/taker) and checked. **All** problems in a table are listed at once. A table with problems is set aside
# (treated as missing) so the rest can still run.

# %%
tables, problems = {}, {}
for name in store.TABLES:
    df = raw.get(name)
    if df is None or len(df) == 0:
        tables[name] = None
        continue
    try:
        tables[name] = schemas.prepare(df, name, column_map=column_map)
    except schemas.SchemaError as e:
        problems[name] = str(e)
        tables[name] = None

for name, msg in problems.items():
    print(msg, "\n")
print("tables with schema problems:", list(problems) or "none")

# %% [markdown]
# **Missing `notional`?** Most exports have price and qty but no notional. For CME and other futures,
# `notional = price × qty × multiplier`; put multipliers in the `instruments` table and uncomment:

# %%
# raw["fills"] = schemas.add_notional(raw["fills"], tables.get("instruments"))   # then re-run the cell above

# %% [markdown]
# ## Validation report: what can run

# %%
summary = pd.DataFrame([
    {"table": k, "rows": (len(v) if v is not None else 0),
     "from": (pd.Timestamp(int(v["ts"].min()), tz="UTC") if v is not None and "ts" in v else None),
     "to": (pd.Timestamp(int(v["ts"].max()), tz="UTC") if v is not None and "ts" in v else None),
     "venues": (", ".join(sorted(map(str, v["venue"].unique()))) if v is not None and "venue" in v else "")}
    for k, v in tables.items()])
summary

# %%
report = schemas.availability_report(tables)
report

# %% [markdown]
# **How to read this.** Every analysis marked UNAVAILABLE names the table or column it needs. A common
# pattern: no `orders` table means no adverse-selection ratio, fill-probability models or latency hops. No
# `buyer`/`seller` in `market_trades` (most venues other than Hyperliquid) means no venue-wide taker toxicity.
# The other notebooks skip unavailable sections and say why.

# %% [markdown]
# ## Quick sanity: units and signs
#
# Look for buys and sells that are roughly balanced, fees in bps that match your tier (positive = paid,
# negative = rebate), and maker fills at or inside the touch.

# %%
f = tables["fills"]
if f is not None:
    fee_bps = 1e4 * f["fee"] / f["notional"]
    display(pd.DataFrame({
        "fills": len(f), "buy_share": (f["side"] > 0).mean(), "maker_share": (f["liquidity"] == "maker").mean(),
        "median_fee_bps_maker": fee_bps[f["liquidity"] == "maker"].median(),
        "median_fee_bps_taker": fee_bps[f["liquidity"] == "taker"].median(),
        "duplicate_fill_ids": f["fill_id"].duplicated().sum()}, index=["value"]).T)

# %% [markdown]
# ## Cache for notebooks 01–07

# %%
path = store.save_tables(tables, meta)
print("cached to", path)
