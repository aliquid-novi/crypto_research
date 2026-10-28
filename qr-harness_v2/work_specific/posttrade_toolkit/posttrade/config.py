"""Configuration: column maps, encodings, venues, sessions, fee models, horizon grids.

Everything venue-specific lives here so the analytics stay exchange-agnostic. The analysis
modules never branch on a venue name; they read a ``VenueSpec`` (sessions, whether the venue
publishes counterparty addresses, funding interval) and the ``instruments`` table (tick size,
contract multiplier).

EDIT THIS FILE (or override the dicts from 00_plug_your_data.ipynb) when you plug real data in.
All fee numbers below are ILLUSTRATIVE placeholders, not any venue's real schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------
# Column maps: YOUR column name -> schema column name, per table. Unmapped columns pass through.
# ---------------------------------------------------------------------------------------------
COLUMN_MAP: dict[str, dict[str, str]] = {
    "fills": {},
    "book_l1": {},
    "market_trades": {},
    "orders": {},
    "decisions": {},
    "funding": {},
    "positions": {},
    "instruments": {},
}

# ---------------------------------------------------------------------------------------------
# Encodings. Values are matched case-insensitively after str().strip().
# ---------------------------------------------------------------------------------------------
SIDE_MAP: dict[object, int] = {
    "1": 1, "+1": 1, "-1": -1, "1.0": 1, "-1.0": -1,
    "b": 1, "buy": 1, "bid": 1, "long": 1, "bot": 1, "bought": 1,
    "s": -1, "a": -1, "sell": -1, "ask": -1, "offer": -1, "short": -1, "sld": -1, "sold": -1,
}
"""How raw side labels map to +1 (buy) / -1 (sell).

Pitfall: some feeds publish ``is_buyer_maker`` (True means the TAKER SOLD). Convert that
yourself with ``aggressor_side = np.where(is_buyer_maker, -1, 1)`` before mapping.
Hyperliquid uses 'B' (bid side = buy) and 'A' (ask side = sell): both are covered above.
"""

LIQUIDITY_MAP: dict[str, str] = {
    "maker": "maker", "m": "maker", "passive": "maker", "add": "maker", "added": "maker",
    "taker": "taker", "t": "taker", "aggressive": "taker", "remove": "taker", "removed": "taker",
    "true": "maker", "false": "taker",  # e.g. an ``is_maker`` boolean column
}

STATUS_MAP: dict[str, str] = {
    "filled": "filled", "fill": "filled", "done": "filled",
    "cancelled": "cancelled", "canceled": "cancelled", "cxl": "cancelled", "expired": "cancelled",
    "partial": "partial", "partially_filled": "partial", "partial_fill": "partial",
    "rejected": "rejected", "reject": "rejected",
}

# Integer timestamps are interpreted in this unit unless a table overrides it in TS_UNITS.
DEFAULT_TS_UNIT = "ns"
TS_UNITS: dict[str, str] = {}  # e.g. {"fills": "ms", "book_l1": "us"}
# Naive datetime columns are interpreted in this tz then converted to UTC.
NAIVE_TZ = "UTC"


# ---------------------------------------------------------------------------------------------
# Sessions (trading hours). None = continuous 24/7 (crypto perps).
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionSpec:
    """Weekly trading-hours calendar in a local timezone.

    ``closed`` lists weekly closed windows as (weekday, "HH:MM", "HH:MM") in local time,
    weekday 0=Monday. End "24:00" means end of that day. ``holidays`` are local dates
    ("YYYY-MM-DD") closed all day. Markouts whose horizon straddles a closure are set to NaN
    and feed-gap detection ignores silence inside closures.
    """

    tz: str
    closed: tuple[tuple[int, str, str], ...]
    holidays: tuple[str, ...] = ()


# CME Globex style: Sun 17:00 - Fri 16:00 Chicago time, daily halt 16:00-17:00 Mon-Thu.
CME_GLOBEX = SessionSpec(
    tz="America/Chicago",
    closed=(
        (0, "16:00", "17:00"), (1, "16:00", "17:00"), (2, "16:00", "17:00"), (3, "16:00", "17:00"),
        (4, "16:00", "24:00"), (5, "00:00", "24:00"), (6, "00:00", "17:00"),
    ),
)


@dataclass(frozen=True)
class VenueSpec:
    """Venue properties the analytics need. Nothing else about a venue is assumed.

    kind: 'perp' | 'future' | 'spot' | 'other'.
    sessions: SessionSpec or None for 24/7.
    funding_interval_h: hours between funding settlements (None if no funding).
    publishes_addresses: market trades carry buyer/seller ids (e.g. Hyperliquid).
    fee_model: key into FEE_SCHEDULES.
    """

    name: str
    kind: str = "perp"
    sessions: SessionSpec | None = None
    funding_interval_h: float | None = None
    publishes_addresses: bool = False
    fee_model: str | None = None


VENUES: dict[str, VenueSpec] = {
    # examples of real venue *shapes* - fee_model points at placeholder schedules below
    "HYPERLIQUID": VenueSpec("HYPERLIQUID", "perp", None, 1.0, True, "EXAMPLE_BPS"),
    "BINANCE": VenueSpec("BINANCE", "perp", None, 8.0, False, "EXAMPLE_BPS"),
    "CME": VenueSpec("CME", "future", CME_GLOBEX, None, False, "EXAMPLE_PER_CONTRACT"),
    # synthetic venues used by synthetic.py, tests and `make demo`
    "VA": VenueSpec("VA", "perp", None, 1.0, True, "SYNTH"),
    "VB": VenueSpec("VB", "perp", None, 1.0, False, "SYNTH"),
    "VL": VenueSpec("VL", "perp", None, 8.0, False, "SYNTH"),
}


def venue_spec(venue: str) -> VenueSpec:
    """Return the VenueSpec for ``venue``; unknown venues get a 24/7, no-funding default."""
    return VENUES.get(venue, VenueSpec(venue, "other"))


# ---------------------------------------------------------------------------------------------
# Fee schedules. model 'bps': fee = notional * bps / 1e4.
# model 'per_contract': fee = qty * amount (qty in contracts, quote ccy per contract).
# Negative values are rebates. PLACEHOLDER NUMBERS - put your own tier in.
# ---------------------------------------------------------------------------------------------
FEE_SCHEDULES: dict[str, dict[str, float | str]] = {
    "EXAMPLE_BPS": {"model": "bps", "maker": 1.0, "taker": 4.0},
    "EXAMPLE_PER_CONTRACT": {"model": "per_contract", "maker": 1.0, "taker": 1.5},
    "SYNTH": {"model": "bps", "maker": -0.5, "taker": 3.0},
}

# Instrument defaults when no instruments table row exists.
DEFAULT_MULTIPLIER = 1.0
DEFAULT_TICK = None  # None = infer from data (smallest positive price increment)

# ---------------------------------------------------------------------------------------------
# Horizon grids
# ---------------------------------------------------------------------------------------------
def log_horizons(neg_max_s: float = 5.0, pos_min_s: float = 0.1, pos_max_s: float = 1800.0,
                 n_pos: int = 15, n_neg: int = 4) -> np.ndarray:
    """Log-spaced wall-clock horizon grid in seconds, including negative horizons and 0.

    Returns sorted float array: -neg_max_s .. -pos_min_s (n_neg points), 0, pos_min_s .. pos_max_s.
    """
    pos = np.geomspace(pos_min_s, pos_max_s, n_pos)
    neg = -np.geomspace(pos_min_s, neg_max_s, n_neg)[::-1] if n_neg > 0 else np.array([])
    return np.round(np.concatenate([neg, [0.0], pos]), 4)


HORIZONS_S: np.ndarray = log_horizons()
EVENT_HORIZONS: dict[str, list[float]] = {
    "trades": [1, 2, 5, 10, 20, 50, 100],         # next n market trades
    "mid_changes": [1, 2, 5, 10, 20, 50],         # next n mid changes
    "volume": [],                                 # next v units of volume (set per instrument)
}

# Lead venue mapping for "lead-venue mid aligned to our venue":
# (lead_venue, lead_symbol) -> (our_venue, our_symbol)
LEAD_MAP: dict[tuple[str, str], tuple[str, str]] = {("VL", "AAA"): ("VA", "AAA")}

# Where 00_plug_your_data.ipynb caches validated tables (pickle; gitignored).
CACHE_DIR_NAME = "data_cache"
