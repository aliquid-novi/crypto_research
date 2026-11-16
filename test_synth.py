"""The synthetic market: reproducible, well-formed, and containing the effects it claims to contain."""
import numpy as np
import pytest

import spreadsim as ss


@pytest.fixture(scope="module")
def mk():
    return ss.simulate(ss.usd_spread(), 1200, seed=3)


@pytest.fixture(scope="module")
def idle(mk):
    return mk.run(ss.Strategy())


def test_same_seed_same_market():
    a, b = ss.simulate(ss.usd_spread(), 120, seed=5), ss.simulate(ss.usd_spread(), 120, seed=5)
    c = ss.simulate(ss.usd_spread(), 120, seed=6)
    assert a.events == b.events and a.events != c.events


def test_events_are_time_sorted_and_books_never_cross(mk, idle):
    t = np.array([e[0] for e in mk.events])
    assert (np.diff(t) >= 0).all()
    for nm in mk.names:
        b = idle.bbo[nm].dropna()
        assert (b.ask > b.bid).all() and len(b) > 50


def test_mid_tracks_the_latent_reference(mk, idle):
    for leg in mk.market.legs:
        b = idle.bbo[leg.inst.name].dropna()
        err = np.abs(b.mid.values - idle.ref_at(leg.inst.name, b.t.values)) / leg.inst.tick
        assert np.median(err) < 1.5 and np.percentile(err, 99) < 6


def test_perp_leads_spot(mk, idle):
    grid = np.arange(60, 1190, 0.25)
    rp = np.diff(np.log(idle.mid_at("PERP", grid)))
    rs = np.diff(np.log(idle.mid_at("SPOT", grid)))
    lead = np.corrcoef(rp[:-1], rs[1:])[0, 1]       # perp now, spot next
    lag = np.corrcoef(rs[:-1], rp[1:])[0, 1]        # spot now, perp next
    assert lead > lag + 0.03


def test_premium_reverts_at_roughly_the_configured_half_life():
    mk = ss.simulate(ss.usd_spread(passthrough=0.0), 3600, seed=11)
    x, dt = mk.truth["basis"]["PERP"], mk.market.dt
    k = int(60 / dt)
    phi = np.corrcoef(x[:-k], x[k:])[0, 1]
    half_life = -np.log(2) * 60 / np.log(phi)
    assert 100 < half_life < 500                    # configured 240 s; one hour is a short sample


def test_funding_dip_appears_only_with_passthrough():
    a = ss.simulate(ss.usd_spread(passthrough=0.0), 1900, seed=2)
    b = ss.simulate(ss.usd_spread(passthrough=1.0), 1900, seed=2)
    assert np.abs(a.truth["shift"]["PERP"]).max() == 0.0
    sh = b.truth["shift"]["PERP"]
    t = b.truth["t"]
    assert sh[(t > 1200) & (t < 1800)].min() < 0 and abs(sh[(t > 1810)].min()) < 1e-12
    assert sum(1 for e in b.events if e[1] == ss.FUNDING) == 1


def test_replay_with_a_strategy_is_reproducible_and_books_balance(mk):
    lat = ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003)
    r1 = mk.run(ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9)), latency=lat)
    r2 = mk.run(ss.SpreadQuoter("SPOT", "PERP", levels_bps=(4, 9)), latency=lat)
    assert len(r1.fills) > 10 and r1.pnl == r2.pnl and r1.fills.equals(r2.fills)
    att = ss.pnl_attribution(r1)
    assert att.loc["TOTAL", "total"] == pytest.approx(r1.pnl, abs=1e-6)
    # every quote fill was hedged one-for-one
    assert abs(r1.equity.pos_SPOT.iloc[-1] + r1.equity.pos_PERP.iloc[-1]) <= 10


def test_three_leg_market_runs_and_the_fx_leg_offsets_the_perp_notional():
    mk = ss.simulate(ss.krx_style(), 900, seed=4)
    strat = ss.SpreadQuoter("KRX", "PERP", fx="USDTKRW", levels_bps=(8, 20), clip=10, max_clips=3)
    res = mk.run(strat, latency=ss.ConstantLatency(feed=0.002, entry=0.003, response=0.003))
    e = res.equity.iloc[-1]
    assert len(res.fills[res.fills.inst == "KRX"]) > 3
    assert e.pos_KRX == pytest.approx(-e.pos_PERP, abs=10)
    assert e.pos_USDTKRW == pytest.approx(e.pos_PERP * e.mid_PERP, rel=0.05, abs=300)
    att = ss.pnl_attribution(res)
    assert abs(att.loc["TOTAL", "fx_translation"]) < 0.05 * max(1.0, abs(att.loc["TOTAL", ["spread_capture", "hedge_cost", "fees"]]).sum())
