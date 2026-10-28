import os

import numpy as np
import pandas as pd

from posttrade import adverse, dynamics, hygiene, latency, markouts, reference, stats, synthetic


def test_ou_theta_from_synthetic_spread(day):
    tables, truth = day
    sp = reference.composite_spread(tables["book_l1"], ("VA", "AAA"), ("VB", "BBB"), 1.0, log=True, on="A")
    g = hygiene.resample_ffill(sp.assign(venue="p", symbol="p"), 5.0, ["value"])
    # the true latent OU spread at 1 s: plain AR(1)
    pure = dynamics.ou_fit(truth["spread_s"], 1.0)
    se_theta = pure.se_b / (pure.b * pure.dt_s)  # delta method: theta = -ln(b)/dt
    assert abs(pure.theta - truth["theta"]) < 3 * se_theta, (pure.theta, truth["theta"], se_theta)
    # the observed spread is noisy: lag-1 AR overstates theta; a 30 s lag removes most of the bias
    # (compare with the pure fit on the same path so sampling error cancels)
    noisy = dynamics.ou_fit(g["value"], 5.0, lag=1)
    robust = dynamics.ou_fit(g["value"], 5.0, lag=6)
    assert noisy.theta > robust.theta
    assert abs(noisy.theta / pure.theta - 1) > abs(robust.theta / pure.theta - 1)
    assert abs(robust.theta / pure.theta - 1) < 0.3, (robust.theta, pure.theta)


def test_kalman_tracks_beta_without_lookahead():
    k = synthetic.make_kalman_example()
    kf = dynamics.kalman_hedge_ratio(k["y"], k["x"], q=(1e-4, 1e-6), r=0.25)
    err = kf["beta_prior"].iloc[500:] - k["beta"].iloc[500:]
    assert np.sqrt(np.mean(err ** 2)) < 0.03
    # causality: changing a future y does not change today's prior
    y2 = k["y"].copy()
    y2.iloc[3000:] += 100
    kf2 = dynamics.kalman_hedge_ratio(y2, k["x"], q=(1e-4, 1e-6), r=0.25)
    assert np.allclose(kf["beta_prior"].iloc[:3001], kf2["beta_prior"].iloc[:3001])


def test_alpha_decay(day):
    tables, truth = day
    sp = reference.composite_spread(tables["book_l1"], ("VA", "AAA"), ("VB", "BBB"), 1.0, log=True, on="A")
    g = hygiene.resample_ffill(sp.assign(venue="p", symbol="p"), 5.0, ["value"])
    g["z"] = (g["value"] - g["value"].mean()) / truth["ou_sd"]
    emp = dynamics.empirical_alpha_decay(g, [60, 300, 900], mu=float(g["value"].mean()))
    top = emp[emp["bucket"] == emp["bucket"].max()].set_index("horizon_s")
    assert top.loc[900.0, "mean_captured"] > top.loc[60.0, "mean_captured"] > 0
    reg = dynamics.alpha_decay_regression(g, [60, 300])
    assert (reg["coef"] > 0).all()  # positive z -> spread falls -> short pays
    m = dynamics.model_alpha_decay(1.0, truth["theta"], [0, 1e9])
    assert np.allclose(m, [0, 1])


def test_stats():
    assert abs(stats.expected_max_sharpe(100, 0.5) - 1.26) < 0.01
    assert np.isclose(stats.required_n(1, 10), 384.16)
    assert np.isclose(stats.sharpe_tstat(0.1, 400), 2.0)
    assert stats.effective_n_cluster(100, 10, 0.5) < 100
    bh = stats.benjamini_hochberg(np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]))
    assert bh["reject"].sum() == 2
    assert stats.probabilistic_sharpe(0.1, 0.0, 1000) > 0.99
    r = np.random.default_rng(0).normal(0.05, 1, 1000)
    d = stats.deflated_sharpe(r, 100, 0.05)
    assert d["dsr"] < stats.probabilistic_sharpe(d["sr"], 0, 1000)
    bb = stats.block_bootstrap(r, block=10, n_boot=200)
    assert bb["lo"] < bb["estimate"] < bb["hi"]
    x = np.convolve(np.random.default_rng(1).normal(size=5000), np.ones(20), "valid")
    assert stats.newey_west_se(x, 20) > x.std() / np.sqrt(len(x)) * 2
    assert stats.effective_n_autocorr(x) < len(x) / 5


def test_research_log(tmp_path):
    log = stats.ResearchLog(os.path.join(tmp_path, "log.csv"))
    log.log("H1", {"h": 60}, {"t": 1.2})
    assert log.log("H2", {"h": 30}, {"t": 2.5}) == 2 and log.n_trials("H1") == 1


def test_toxic_addresses_ranked_top(day, mid_ref):
    tables, truth = day
    mt = tables["market_trades"]
    tox = adverse.taker_toxicity(mt[mt["venue"] == "VA"], mid_ref, horizon_s=30)
    assert set(tox.index[: len(truth["toxic_addresses"])]) == set(truth["toxic_addresses"])


def test_adverse_selection_ratio_and_late_cancels(day, clean, mid_ref):
    tables, truth = day
    o = tables["orders"]
    o = o[o["order_id"].str.startswith("Q")]
    r = adverse.adverse_selection_ratio(o, mid_ref, horizon_s=30)
    assert r["ratio"] > 1.1
    f = clean[clean["fill_id"].str.startswith("F")]
    lc = adverse.late_cancels(f, o)
    assert set(lc.loc[lc["late_cancel"], "order_id"]) == set(truth["late_cancel_order_ids"]) - {truth["dropped_fill"]["order_id"]}
    # late-cancel fills are concentrated in toxic fills -> worse markouts
    mo = markouts.compute_markouts(lc, mid_ref, [60.0])[60.0]
    assert mo[lc["late_cancel"]].mean() < mo[~lc["late_cancel"]].mean()
    comp = adverse.fill_completeness(f, o)
    assert set(comp.dropna()) == {"full", "partial"}
    drift = adverse.pre_fill_drift(f, mid_ref)
    assert drift.shape[1] == 7


def test_counterparty_concentration(day, quote_events, mid_ref):
    tables, truth = day
    f, _ = quote_events
    mo = markouts.compute_markouts(f, mid_ref, [60.0])[60.0].to_numpy()
    cc = adverse.counterparty_concentration(f, mo)
    top = set(cc.index[:5])
    assert len(top & set(truth["toxic_addresses"])) >= 4


def test_latency_rises_through_day(day):
    tables, _ = day
    lat = latency.hop_latencies(tables["orders"])
    hr = latency.hour_of_day(lat["ts"])
    s = latency.summarise_by(lat, hr)
    q50 = s["q50"]["sent_to_ack_ms"]
    assert q50.iloc[-1] > 2 * q50.iloc[0]
    assert (lat["negative_hops"] == 0).all()
    rd = latency.restart_discontinuity(lat["ts"].to_numpy(), lat["sent_to_ack_ms"].to_numpy(), [int(lat["ts"].median())])
    assert np.isfinite(rd["diff"].iat[0])
