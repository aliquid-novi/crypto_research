import numpy as np
import pandas as pd

from posttrade import markouts, reference

H = [-5.0, 0.0, 1.0, 5.0, 30.0, 60.0, 120.0]


def test_markout_curve_recovers_injected_adverse_selection(day, quote_events, mid_ref):
    tables, truth = day
    f, ev = quote_events
    dec = markouts.decompose(ev, mid_ref, H)
    toxic_events = set(f.loc[f["fill_id"].isin(truth["toxic_fill_ids"]), "ts"])
    is_tox = ev["ts"].isin(toxic_events).to_numpy()
    J, lag, tau = truth["toxic_jump_bps"], truth["lead_lag_s"], truth["toxic_tau_s"]
    adv = dec["adverse"]
    # toxic minus normal adverse move at 60 s ~ -J (1 - exp(-(60 - lag)/tau)) ~ -J
    diff = adv.loc[is_tox, 60.0].mean() - adv.loc[~is_tox, 60.0].mean()
    assert abs(diff - (-J)) < 1.2, diff
    # before the lead-lag has elapsed there is nothing to see; negative horizons carry no adverse move
    assert abs(adv.loc[is_tox, 1.0].mean()) < 0.5
    assert abs(adv[-5.0].mean()) < 0.5
    # whole curve: spread captured at 0, then drifts down by ~ p_toxic * J
    c = markouts.markout_curve(dec["markout"], clusters=markouts.time_clusters(ev, 60))
    sc = dec["spread_captured"].mean()
    assert np.isclose(c.loc[0.0, "mean"], sc)
    expected_120 = sc - truth["p_toxic"] * J * (1 - np.exp(-(120 - lag) / tau))
    assert abs(c.loc[120.0, "mean"] - expected_120) < 3 * c.loc[120.0, "se"] + 0.2
    assert c.loc[120.0, "mean"] < c.loc[1.0, "mean"]


def test_decomposition_identity(quote_events, mid_ref):
    _, ev = quote_events
    d = markouts.decompose(ev, mid_ref, H)
    recon = d["adverse"].add(d["spread_captured"], axis=0)
    ok = np.isfinite(d["markout"].to_numpy())
    assert np.allclose(recon.to_numpy()[ok], d["markout"].to_numpy()[ok])


def test_effective_equals_realised_plus_impact(day, mid_ref):
    tables, _ = day
    mt = tables["market_trades"]
    mt = mt[mt["venue"] == "VA"]
    t = markouts.taker_spreads(mt, mid_ref, 30.0)
    ok = t.notna().all(axis=1)
    assert ok.mean() > 0.95
    assert np.allclose(t.loc[ok, "effective_bps"], t.loc[ok, "realised_bps"] + t.loc[ok, "impact_bps"])
    # takers cross the spread: effective spread is positive on average
    assert t["effective_bps"].mean() > 0


def test_fees_and_vol_scaling(day, quote_events, mid_ref):
    f, _ = quote_events
    f = f[~f["fill_id"].isin(day[1]["fee_error_fill_ids"])]
    mo = markouts.compute_markouts(f, mid_ref, [0.0, 10.0])
    net = markouts.net_of_fees(mo, f)
    # synthetic maker fee is a rebate (-0.5 bps): net = gross + 0.5
    assert np.allclose((net - mo).dropna().to_numpy(), 0.5)
    vol = reference.trailing_vol(mid_ref)
    vs = markouts.vol_scaled(mo, f, vol)
    assert vs[0.0].isna().all() and vs[10.0].notna().mean() > 0.9


def test_event_time_and_daily(day, quote_events, mid_ref):
    tables, _ = day
    f, ev = quote_events
    trades = tables["market_trades"]
    e1 = markouts.event_time_markouts(ev, mid_ref, trades, [1, 10])
    assert e1.notna().mean().min() > 0.95
    ev_mid = markouts.mid_change_events(mid_ref)
    e2 = markouts.event_time_markouts(ev, mid_ref, ev_mid, [1, 5])
    assert e2.notna().mean().min() > 0.95
    e3 = markouts.event_time_markouts(ev, mid_ref, trades, [5.0], size_col="qty")
    assert e3.notna().mean().min() > 0.95
    daily = markouts.daily_markout_means(f, mid_ref, [0.0, 60.0])
    full = markouts.markout_curve(markouts.compute_markouts(f, mid_ref, [0.0, 60.0]), weights=f["notional"].to_numpy(), n_boot=10)
    comb = markouts.combine_daily(daily)
    assert np.allclose(comb.to_numpy(), full["mean"].to_numpy())


def test_sweeps_aggregate(quote_events):
    f, ev = quote_events
    assert len(ev) < len(f)
    assert np.isclose(ev["qty"].sum(), f["qty"].sum())
    assert ev["n_fills"].max() >= 2


def test_conditional_curves(quote_events, mid_ref):
    _, ev = quote_events
    mo = markouts.compute_markouts(ev, mid_ref, [1.0, 60.0])
    b = markouts.quantile_buckets(ev["qty"], 3)
    cc = markouts.conditional_curves(mo, b, n_boot=50)
    assert set(cc["bucket"]) == {"Q1", "Q2", "Q3"} and (cc["lo"] <= cc["hi"]).all()
