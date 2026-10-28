import numpy as np

from posttrade import reference, spreads_tca


def _sf(day, clean):
    tables, _ = day
    return spreads_tca.shortfall(tables["decisions"], clean, tables["book_l1"], tables["funding"], tables["instruments"])


def test_components_sum_exactly(day, clean):
    r = _sf(day, clean)
    t = r["trades"]
    lhs = t["realised_net"] - t["paper"]
    rhs = t[spreads_tca.COMPONENTS].sum(axis=1)
    assert np.allclose(lhs, rhs, atol=1e-9)
    assert (~t["open"]).all()


def test_known_components(day, clean):
    tables, truth = day
    r = _sf(day, clean)
    t = r["trades"].set_index("decision_id")
    sf = clean[clean["decision_id"].notna()]
    assert np.isclose(t["fees"].sum(), -sf["fee"].sum())
    dt = truth["decisions"].set_index("decision_id")
    missed = dt.index[dt["missed"]]
    assert (~t.loc[missed, "filled_any"]).all()
    # a missed decision's shortfall is all opportunity: realised 0, opportunity = -paper
    assert np.allclose(t.loc[missed, "opportunity"], -t.loc[missed, "paper"])
    # the leg gap recovered from fills equals the injected legging gap
    filled = dt.index[~dt["missed"]]
    assert np.allclose(t.loc[filled, "legging_s"], dt.loc[filled, "legging_gap_ns"] / 1e9, atol=1e-6)
    lo = t.loc[filled, "leg_order"].map({"A_first": "A", "B_first": "B"})
    assert (lo == dt.loc[filled, "first_leg"]).all()
    # taker entry costs half the spread on average -> negative
    assert t.loc[filled, "entry_cost"].mean() < 0
    hung = spreads_tca.hung_legs(r["trades"], r["legs"], hung_s=10)
    assert set(dt.index[dt["hung"].astype("boolean").fillna(False).to_numpy(dtype=bool)]) <= set(hung["decision_id"])


def test_reports(day, clean, mid_ref):
    tables, _ = day
    r = _sf(day, clean)
    t = r["trades"]
    cs = spreads_tca.cost_share(t)
    assert np.isclose(cs["sr_net_approx"], cs["sr_gross"] * (1 - cs["cost_bps"] / cs["alpha_bps"]))
    assert set(spreads_tca.missed_report(t).index) == {"filled", "missed"}
    agg = spreads_tca.aggregate(t, "leg_order")
    assert agg["n"].sum() == t["filled_any"].sum()
    mm = spreads_tca.mae_mfe(r["legs"], clean, tables["book_l1"])
    assert (mm["mae"] <= mm["final"] + 1e-9).all() and (mm["mfe"] >= mm["final"] - 1e-9).all()
    dv = spreads_tca.decision_vs_fill_markouts(clean, tables["decisions"], mid_ref, [0.0, 60.0])
    assert dv["fill_time"].shape == dv["decision_time"].shape


def test_link_heuristic(day, clean):
    tables, _ = day
    stripped = clean.drop(columns=["decision_id", "leg", "leg_role"])
    linked = spreads_tca.link_fills(stripped[stripped["fill_id"].str.startswith("S")], tables["decisions"])
    orig = clean[clean["fill_id"].str.startswith("S")]
    assert (linked["decision_id"].to_numpy() == orig["decision_id"].to_numpy()).mean() > 0.95
    assert (linked["leg_role"].to_numpy() == orig["leg_role"].to_numpy()).mean() > 0.95
