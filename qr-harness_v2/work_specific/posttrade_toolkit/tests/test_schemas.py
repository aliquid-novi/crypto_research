import numpy as np
import pandas as pd
import pytest

from posttrade import _util, config, schemas


def test_synthetic_tables_validate(day):
    tables, _ = day
    for name, df in tables.items():
        out = schemas.prepare(df, name, column_map={})
        assert len(out) == len(df)
    rep = schemas.availability_report(tables)
    assert (rep["status"] == "OK").all(), rep[rep["status"] != "OK"]


def test_missing_and_bad_columns_are_all_reported():
    df = pd.DataFrame({"ts": [1, 2], "venue": ["X", "X"], "symbol": ["S", "S"], "side": ["buy", "??"],
                       "price": [1.0, "abc"], "qty": [1, 1]})
    with pytest.raises(schemas.SchemaError) as e:
        schemas.prepare(df, "fills", column_map={})
    msg = str(e.value)
    for needle in ["fill_id", "order_id", "notional", "fee", "liquidity", "side: 1 values", "price: 1 values"]:
        assert needle in msg


def test_column_map_side_and_time_units():
    raw = pd.DataFrame({"time_ms": [1_700_000_000_000, 1_700_000_000_500], "exch": ["HYPERLIQUID"] * 2,
                        "coin": ["BTC"] * 2, "bid": [1.0, 1.0], "bsz": [1, 2], "ask": [1.1, 1.1], "asz": [3, 4]})
    cmap = {"book_l1": {"time_ms": "ts", "exch": "venue", "coin": "symbol", "bid": "bid_px", "bsz": "bid_sz",
                        "ask": "ask_px", "asz": "ask_sz"}}
    out = schemas.prepare(raw, "book_l1", column_map=cmap, ts_unit="ms")
    assert out["ts"].iloc[0] == 1_700_000_000_000 * 1_000_000
    t = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01T00:00:00Z"]), "venue": ["V"], "symbol": ["S"], "price": [1.0],
                      "qty": [1.0], "aggressor_side": ["A"]})
    assert schemas.prepare(t, "market_trades", column_map={})["aggressor_side"].iloc[0] == -1


def test_cme_session_masks_markouts_across_halt():
    # 2026-01-06 is a Tuesday. 15:59:50 Chicago = 21:59:50 UTC; halt 16:00-17:00 CT.
    t0 = pd.Timestamp("2026-01-06 21:59:50", tz="UTC").value
    venue = np.array(["CME", "CME", "BINANCE"])
    t = np.array([t0, t0, t0], dtype="int64")
    t1 = t + np.array([5, 30, 30], dtype="int64") * _util.NS
    assert list(_util.same_session_mask(venue, t, t1)) == [True, False, True]
    cl = _util.closure_intervals(config.CME_GLOBEX, t0, t0 + 6 * 86400 * _util.NS)
    # the Friday 16:00 -> Sunday 17:00 weekend is one merged closure of 49h
    assert np.any(np.isclose((cl[:, 1] - cl[:, 0]) / 3600e9, 49.0))
