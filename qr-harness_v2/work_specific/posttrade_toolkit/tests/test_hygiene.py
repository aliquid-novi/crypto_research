import numpy as np
import pandas as pd

from posttrade import hygiene, reference


def test_dedupe(day):
    tables, truth = day
    dd, removed = hygiene.dedupe_fills(tables["fills"])
    assert sorted(removed["fill_id"]) == truth["duplicate_fill_ids"]
    assert dd["fill_id"].is_unique


def test_reconcile_finds_dropped_fill(day, clean):
    tables, truth = day
    rp = hygiene.reconcile_positions(clean, tables["positions"])
    fl = rp[rp["flag"]]
    d = truth["dropped_fill"]
    assert (fl["venue"] == d["venue"]).all() and len(fl) > 0
    assert fl["ts"].min() > d["ts"] and np.isclose(fl["break"].iloc[0], d["side"] * d["qty"])


def test_fee_and_funding_recompute(day, clean):
    tables, truth = day
    fe = hygiene.recompute_fees(clean)
    assert sorted(fe.loc[fe["flag"], "fill_id"]) == truth["fee_error_fill_ids"]
    fu = hygiene.recompute_funding(tables["funding"])
    assert list(fu.index[fu["flag"]]) == [truth["funding_error_row"]]


def test_bad_ticks(day):
    tables, truth = day
    mt = tables["market_trades"]
    va = mt[mt["venue"] == "VA"]
    flags = hygiene.bad_ticks(va, k=50, gamma_ticks=2, instruments=tables["instruments"])
    flagged = set(va.loc[flags, "trade_id"])
    assert set(truth["bad_tick_trade_ids"]) <= flagged
    # false positives (bid/ask bounce + fast moves) stay rare
    assert (len(flagged) - len(truth["bad_tick_trade_ids"])) / len(va) < 0.005


def test_book_defects_and_gap(day):
    tables, truth = day
    b = tables["book_l1"]
    s = hygiene.book_sanity(b, tables["instruments"])
    rows = truth["book_defect_rows"]
    assert s.loc[rows["crossed"], "crossed"] and s.loc[rows["zero_size"], "nonpos_size"] and s.loc[rows["off_tick"], "off_tick"]
    assert s["any"].sum() == 3
    gaps = hygiene.feed_gaps(b)
    ven, sym, a, z = truth["feed_gap"]
    assert len(gaps) == 1 and gaps["venue"].iat[0] == ven and abs(gaps["gap_s"].iat[0] - (z - a) / 1e9) < 2


def test_mad_cusum_noise_ntp_resample(mid_ref):
    x = np.r_[np.random.default_rng(0).normal(0, 1, 500), 50.0]
    assert hygiene.mad_flags(x)["flag"].to_numpy()[-1]
    y = pd.Series(np.r_[np.random.default_rng(1).normal(0, 1, 60), np.random.default_rng(2).normal(-2, 1, 20)])
    c = hygiene.cusum(y, h=5.0, baseline_n=60)
    assert not c["alarm"].iloc[:60].any() and c["alarm"].iloc[60:].any()
    va = mid_ref[mid_ref["venue"] == "VA"]
    sig = hygiene.signature(va)
    assert sig["rv_per_day"].iloc[0] > sig["rv_per_day"].iloc[-1] * 0.5
    assert hygiene.noise_variance(va) > 0
    # NTP: server clock 5 ms ahead and drifting +2 us/s, symmetric 1 ms one-way delay
    t0 = np.arange(0, 600, 10, dtype="int64") * 1_000_000_000
    off = 5_000_000 + 2_000 * (t0 / 1e9)
    t1 = t0 + 1_000_000 + off
    t2 = t1 + 50_000
    t3 = t2 - off + 1_000_000
    r = hygiene.ntp_offsets(pd.DataFrame({"t0": t0, "t1": t1, "t2": t2, "t3": t3}), min_delay_quantile=1.0)
    assert abs(r["intercept"] - 5e6) < 1 and abs(r["slope_ns_per_s"] - 2000) < 1e-3
    # forward-fill only
    ref = pd.DataFrame({"ts": [0, 10_000_000_000], "venue": "V", "symbol": "S", "value": [1.0, 2.0]})
    rs = hygiene.resample_ffill(ref, 1.0)
    assert set(rs["value"]) == {1.0, 2.0} and rs["age_s"].max() < 10


def test_staleness(day, mid_ref):
    tables, _ = day
    q = tables["fills"].head(50)
    s = hygiene.staleness(q, mid_ref, ["value"])
    assert (s["age_s"] >= 0).all() and (s["age_s"] <= 1.0 + 1e-9).all()
