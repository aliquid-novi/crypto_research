import numpy as np
import pandas as pd

from posttrade import _util, dynamics, markouts, models, reference, synthetic


def test_hawkes_1d_recovers_params_and_gof():
    rng = np.random.default_rng(3)
    t = synthetic.simulate_hawkes(0.2, 0.5, 2.0, 8 * 3600, rng)
    r = models.fit_hawkes_1d(t, T=8 * 3600)
    assert abs(r["mu"] - 0.2) < 0.03 and abs(r["alpha"] - 0.5) < 0.05 and abs(r["beta"] - 2.0) < 0.3
    tau = models.hawkes_rescaled_1d(t, r["mu"], r["alpha"], r["beta"])
    assert abs(tau.mean() - 1) < 0.05


def test_hawkes_on_lead_venue_trades(day):
    tables, truth = day
    mt = tables["market_trades"]
    lt = np.sort(mt.loc[mt["venue"] == "VL", "ts"].to_numpy())
    r = models.fit_hawkes_1d((lt - truth["t0"]) / 1e9, T=truth["T_s"])
    mu, a, b = truth["hawkes"]
    assert abs(r["mu"] - mu) < 0.04 and abs(r["alpha"] - a) < 0.07 and abs(r["beta"] - b) < 0.5


def test_hawkes_2d():
    rng = np.random.default_rng(4)
    A = np.array([[0.3, 0.4], [0.0, 0.3]])
    tt = synthetic.simulate_hawkes_2d([0.1, 0.2], A, 1.5, 8 * 3600, rng)
    r = models.fit_hawkes_2d(tt, 8 * 3600)
    assert np.abs(r["A"] - A).max() < 0.08
    assert np.allclose(r["mu"], [0.1, 0.2], atol=0.03)
    res = models.hawkes_rescaled_2d(tt, r)
    assert all(abs(x.mean() - 1) < 0.07 for x in res)


def test_hmm_states():
    rng = np.random.default_rng(5)
    x, s = synthetic.simulate_hmm(5000, rng=rng)
    h = models.fit_hmm2(x)
    assert abs(h["means"][h["toxic_state"]] - (-3)) < 0.4
    acc = ((h["p_toxic_filtered"] > 0.5) == (s == 1)).mean()
    assert acc > 0.88
    assert ((h["p_toxic_smoothed"] > 0.5) == (s == 1)).mean() >= acc - 0.01


def test_newey_west_wider_than_ols_on_overlapping(mid_ref):
    # 300 s forward returns sampled every 5 s overlap by 59/60: OLS SEs are far too small.
    va = mid_ref[mid_ref["venue"] == "VA"]
    grid = va.iloc[::5].assign(side=1, price=lambda d: d["value"])
    mo = markouts.compute_markouts(grid, mid_ref, [300.0], future_join="backward")
    X = pd.DataFrame({"noise": np.random.default_rng(0).normal(size=len(grid))})
    reg = models.markout_regression(mo, X, ts=grid["ts"].to_numpy(), quantiles=(), huber=False)
    c = reg[reg["feature"] == "const"].set_index("model")
    assert c.loc["ols_nw", "se"] > 3 * c.loc["ols_iid", "se"]


def test_markout_regression_models(quote_events, mid_ref, day):
    _, ev = quote_events
    mo = markouts.compute_markouts(ev, mid_ref, [5.0, 30.0])
    book = day[0]["book_l1"]
    imb = _util.asof_values(ev, book.assign(imb=reference.imbalance(book)), ["imb"], direction="backward", allow_exact=False)
    X = models.signed_features(pd.DataFrame({"imb": imb["imb"].to_numpy() - 0.5}), ["imb"], ev["side"].to_numpy())
    reg = models.markout_regression(mo, X, ts=ev["ts"].to_numpy())
    assert {"ols_nw", "huber", "q05", "q10"} <= set(reg["model"])
    # synthetic imbalance tracks where the true price sits in the spread -> positive signed coefficient
    assert reg.query("model == 'ols_nw' and feature == 'imb' and horizon_s == 5.0")["coef"].iat[0] > 0


def test_toxicity_model(day, quote_events, mid_ref):
    tables, truth = day
    f, ev = quote_events
    mo = markouts.compute_markouts(ev, mid_ref, [60.0])[60.0].to_numpy()
    lead = reference.lead_mid(tables["book_l1"])
    lm = _util.asof_values(ev, lead, ["value"], direction="backward", allow_exact=False)["value"].to_numpy()
    m0 = markouts.ref_at_fill(ev, mid_ref)
    X = pd.DataFrame({"lead_gap": 1e4 * ev["side"].to_numpy() * (lm - m0) / m0})
    r = models.toxicity_model(X, mo, ev["ts"].to_numpy(), threshold_bps=0)
    assert 0 < r["p_star"] < 1
    assert r["backtest"]["net_bps"].notna().all()


def test_fill_probability_and_hazard(day, mid_ref):
    tables, _ = day
    o = tables["orders"]
    o = o[o["order_id"].str.startswith("Q")]
    smp = models.quote_distance_samples(o, mid_ref)
    fi = models.fit_fill_intensity(smp)
    assert fi["k"] > 0  # fill intensity falls with distance from mid
    sv = models.survival_frame(o, mid_ref)
    hz = models.fit_hazard(sv, ["dist_bps"], use_lifelines=False)
    assert hz["method"] == "discrete_time_logit" and hz["params"].loc["dist_bps", "coef"] < 0


def test_selection_corrections(day, mid_ref):
    tables, _ = day
    o = tables["orders"]
    o = o[o["order_id"].str.startswith("Q")]
    sv = models.survival_frame(o, mid_ref)
    sel = sv["event"].to_numpy()
    filled = sv[sel == 1]
    y = np.random.default_rng(0).normal(0, 1, len(filled))
    hk = models.heckman_two_step(sv[["dist_bps"]], sel, filled[["dist_bps"]], y)
    assert "imr" in hk["outcome"].params.index
    ip = models.ipw_regression(sv[["dist_bps"]], sel, filled[["dist_bps"]], y)
    assert len(ip["weights"]) == len(filled)


def test_impact_models(day, mid_ref, clean):
    tables, _ = day
    mt = tables["market_trades"]
    va = mt[(mt["venue"] == "VA")]
    ev = models.trade_event_series(va, mid_ref)
    resp = models.response_function(ev, 20)
    assert resp["C"].iloc[0] == 1.0
    G = models.propagator(resp)
    assert len(G) == 20 and np.isfinite(G["G_bps"]).all()
    k = models.kyle_lambda(va, mid_ref, 60)
    assert np.isfinite(k["lambda"])
    assert np.isfinite(models.roll_spread(va["price"].to_numpy())["cov"])
    hv = models.hasbrouck_var(ev, p=3, horizon=20)
    assert 0 <= hv["trade_info_share"] <= 1
    meta = models.metaorders(clean[clean["decision_id"].notna()], mid_ref, ["decision_id", "leg", "leg_role"])
    out = models.sqrt_impact_fit(meta, daily_volume=1e4, daily_vol_bps=300)
    assert "table" in out


def test_huang_stoll():
    hs = synthetic.simulate_huang_stoll(rng=np.random.default_rng(6))
    r = models.huang_stoll(hs["price"], hs["sign"])
    assert abs(r["S"] - 0.1) < 0.005 and abs(r["lam"] - 0.4) < 0.02


def test_ev_per_quote(day, mid_ref):
    tables, _ = day
    o = tables["orders"]
    o = o[o["order_id"].str.startswith("Q")]
    sv = models.survival_frame(o, mid_ref)
    mf = pd.DataFrame({"dist_bps": sv.loc[sv["event"] == 1, "dist_bps"], "markout_bps": 1.0})
    t = models.ev_per_quote(sv, mf, fee_bps=-0.5)
    assert np.allclose(t["ev_bps"].dropna(), (t["p_fill"] * 1.5).dropna())
