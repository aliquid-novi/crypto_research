"""The guardrails are only worth having if they catch planted leaks.
Each test plants one and checks it's caught (and that clean code passes)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrlib import audit, fairvalue, fills, leakage, pit, prereg, signals
from qrlib.synth import make_market

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mkt():
    return make_market(n=6000, seed=3)


@pytest.fixture(scope="module")
def times(mkt):
    q = mkt["quotes"]
    return pd.date_range(q["ts_recv"].iloc[0] + pd.Timedelta("1s"), q["ts_recv"].iloc[-1] - pd.Timedelta("10s"), freq="500ms")


# ---------- point-in-time ----------

def test_asof_is_strictly_backward_with_latency():
    ev = pd.DataFrame({"ts_recv": pd.to_datetime(["2026-01-01 00:00:00.000", "2026-01-01 00:00:00.010"]), "x": [1.0, 2.0]})
    t = pd.to_datetime(["2026-01-01 00:00:00.010", "2026-01-01 00:00:00.014", "2026-01-01 00:00:00.016"])
    out = pit.asof_features(t, ev, ["x"], latency="5ms")
    # event 2 is known at .015: not at .010 or .014, yes at .016
    assert out["x"].tolist() == [1.0, 1.0, 2.0]


def test_target_columns_refused_as_features():
    ev = pd.DataFrame({"ts_recv": pd.to_datetime(["2026-01-01"]), "target_mid_1s": [1.0]})
    with pytest.raises(ValueError):
        pit.asof_features(pd.to_datetime(["2026-01-02"]), ev, ["target_mid_1s"])


# ---------- truncation test ----------

def clean_fv(d, t):
    book = fairvalue.l1_at(d["quotes"], t, latency="5ms")
    return pd.DataFrame({"wm": fairvalue.weighted_mid(book), "ema": fairvalue.mid(book).ewm(span=20).mean()})


def leak_full_sample_zscore(d, t):
    book = fairvalue.l1_at(d["quotes"], t)
    m = fairvalue.mid(book)
    return pd.DataFrame({"z": (m - m.mean()) / m.std()})


def leak_centred_window(d, t):
    book = fairvalue.l1_at(d["quotes"], t)
    return pd.DataFrame({"c": fairvalue.mid(book).rolling(9, center=True, min_periods=1).mean()})


def leak_forward_join(d, t):
    q = d["quotes"].sort_values("ts_recv")
    m = pd.merge_asof(pd.DataFrame({"t": t}), q, left_on="t", right_on="ts_recv", direction="forward")
    return pd.DataFrame({"f": ((m["bid_px"] + m["ask_px"]) / 2).to_numpy()}, index=t)


def leak_fitted_on_everything(d, t):
    book = fairvalue.l1_at(d["quotes"], t)
    imb = (book["bid_sz"] / (book["bid_sz"] + book["ask_sz"])).fillna(0.5)
    m = fairvalue.mid(book)
    beta = np.polyfit(imb.to_numpy(), (m.shift(-1) - m).fillna(0).to_numpy(), 1)[0]  # sees the future
    return pd.DataFrame({"fit": m + beta * imb})


def test_clean_pipeline_passes(mkt, times):
    rep = leakage.truncation_test(clean_fv, {"quotes": mkt["quotes"]}, times, latency="5ms")
    assert rep.passed


def test_ignoring_latency_is_caught(mkt, times):
    def no_latency(d, t):
        return pd.DataFrame({"wm": fairvalue.weighted_mid(fairvalue.l1_at(d["quotes"], t, latency="0ms"))})

    with pytest.raises(leakage.LeakageError):
        leakage.truncation_test(no_latency, {"quotes": mkt["quotes"]}, times, latency="5ms")


@pytest.mark.parametrize("fn", [leak_full_sample_zscore, leak_centred_window, leak_forward_join, leak_fitted_on_everything])
def test_planted_leaks_are_caught(mkt, times, fn):
    with pytest.raises(leakage.LeakageError):
        leakage.truncation_test(fn, {"quotes": mkt["quotes"]}, times)


# ---------- fills ----------

def test_quote_not_live_before_delay_and_through_vs_touch():
    t0 = pd.Timestamp("2026-01-01")
    myq = pd.DataFrame({"my_bid": [99.9], "my_ask": [100.1]}, index=[t0])
    tr = pd.DataFrame({
        "ts_recv": [t0 + pd.Timedelta("1ms"), t0 + pd.Timedelta("20ms"), t0 + pd.Timedelta("30ms")],
        "price": [99.0, 99.9, 100.2],
        "size": [1.0, 1.0, 1.0],
        "side": ["sell", "sell", "buy"],
    })
    through = fills.simulate_fills(myq, tr, live_delay="10ms", mode="through")
    touch = fills.simulate_fills(myq, tr, live_delay="10ms", mode="touch")
    # trade at 1ms is before the quote is live; 99.9 sell only fills under touch; 100.2 buy fills both
    assert sorted(through["our_side"]) == ["sell"]
    assert sorted(touch["our_side"]) == ["buy", "sell"]


def test_quote_fills_once_per_side():
    t0 = pd.Timestamp("2026-01-01")
    myq = pd.DataFrame({"my_bid": [99.9], "my_ask": [np.nan]}, index=[t0])
    tr = pd.DataFrame({"ts_recv": [t0 + pd.Timedelta(f"{k}0ms") for k in (2, 3, 4)], "price": [99.0] * 3, "size": [1.0] * 3, "side": ["sell"] * 3})
    assert len(fills.simulate_fills(myq, tr, live_delay="10ms")) == 1


# ---------- signal redundancy ----------

def test_redundancy_filter():
    rng = np.random.default_rng(0)
    n = 3000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = 0.5 * a + 0.5 * b + rng.normal(size=n)
    idx = pd.RangeIndex(n)
    existing = pd.DataFrame({"a": a}, index=idx)
    dup = pd.Series(a + 0.05 * rng.normal(size=n), index=idx)
    assert signals.redundancy_report(dup, existing, pd.Series(y, index=idx))["verdict"] == "reject"
    assert signals.redundancy_report(pd.Series(b, index=idx), existing, pd.Series(y, index=idx))["verdict"] == "accept"
    noise = pd.Series(rng.normal(size=n), index=idx)
    assert signals.redundancy_report(noise, existing, pd.Series(y, index=idx), min_incremental_r2=0.001)["verdict"] == "reject"


# ---------- pre-registration ----------

@pytest.fixture
def cards(tmp_path, monkeypatch):
    hyp = tmp_path / "hypotheses"
    hyp.mkdir()
    monkeypatch.setattr(prereg, "HYP_DIR", hyp)
    monkeypatch.setattr(audit, "LOG", tmp_path / "runs.jsonl")
    tmpl = (ROOT / "research/hypotheses/_TEMPLATE.md").read_text()
    (hyp / "H001_x.md").write_text(tmpl.replace("{{ID}}", "H001").replace("{{TITLE}}", "x").replace("{{FAMILY}}", "fair-value").replace("{{SOURCE}}", "mark"))
    return hyp


def test_cannot_register_or_run_without_marks_prediction(cards):
    with pytest.raises(prereg.PreregistrationError):
        prereg.register("H001")
    with pytest.raises(prereg.PreregistrationError):
        prereg.require("H001")
    meta, body = prereg.read_card("H001")
    meta |= {"prediction": "p", "falsified_if": "f", "confidence": 0.6, "predicted_by": "claude"}
    prereg.write_card("H001", meta, body)
    with pytest.raises(prereg.PreregistrationError, match="agents"):
        prereg.register("H001")


def test_editing_prediction_after_registration_is_caught(cards):
    meta, body = prereg.read_card("H001")
    meta |= {"prediction": "beats mid at 1s", "falsified_if": "CI spans 0", "confidence": 0.6, "predicted_by": "mark"}
    prereg.write_card("H001", meta, body)
    prereg.register("H001")
    assert prereg.require("H001")["status"] == "registered"
    meta, body = prereg.read_card("H001")
    meta["confidence"] = 0.9
    prereg.write_card("H001", meta, body)
    with pytest.raises(prereg.PreregistrationError, match="changed"):
        prereg.require("H001")


def test_new_card_needs_confirmation_when_two_unresolved(cards):
    assert prereg.tally() == {"outstanding": 1, "resolved": 0, "outstanding_ids": ["H001"]}
    prereg.check_new_card_allowed()                      # 1 outstanding: fine
    (cards / "H002_y.md").write_text((cards / "H001_x.md").read_text().replace("H001", "H002"))
    with pytest.raises(prereg.PreregistrationError, match="PLEASE CONFIRM"):
        prereg.check_new_card_allowed()
    assert prereg.check_new_card_allowed(confirmed=True)["outstanding"] == 2
    meta, body = prereg.read_card("H001")                # resolving one lifts the block
    meta |= {"prediction": "p", "falsified_if": "f", "confidence": 0.5, "predicted_by": "mark"}
    prereg.write_card("H001", meta, body)
    prereg.register("H001")
    prereg.resolve("H001", outcome=True)
    assert prereg.check_new_card_allowed() == {"outstanding": 1, "resolved": 1, "outstanding_ids": ["H002"]}


def test_new_post_scaffolds_blank_review_and_rebuild(tmp_path, monkeypatch):
    from qrlib import cli
    for f in ("_TEMPLATE.md", "_rebuild_template.py"):
        (tmp_path / f).write_text((ROOT / "research/post" / f).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(cli, "POST_DIR", tmp_path)
    md, py = cli.new_post("HL MM baseline", "../hl_mm_baseline.py")
    text = md.read_text(encoding="utf-8")
    assert md.name == "hl_mm_baseline_post.md" and "{{" not in text and "notebook: ../hl_mm_baseline.py" in text
    assert py.with_suffix(".ipynb").exists() and "{{" not in py.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):                      # never overwrites Mark's answers
        cli.new_post("HL MM baseline", "x")


def test_cheap_gate_and_marks_cell_survive_sync(tmp_path, monkeypatch):
    import nbformat as nbf
    from qrlib import cheap, cli
    (tmp_path / "_template.py").write_text((ROOT / "research/cheap/_template.py").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(cheap, "CHEAP_DIR", tmp_path)
    monkeypatch.setattr(audit, "LOG", tmp_path / "runs.jsonl")
    cid, py = cli.new_cheap("spread mean reverts", request="test 1s spread AR(1)")
    assert cid == "C001" and "> test 1s spread AR(1)" in py.read_text(encoding="utf-8")
    assert cheap.status(cid)["stage"] == "needs pre"
    with pytest.raises(cheap.CheapError, match="pre-implementation"):
        cheap.require(cid)
    # Mark fills the pre section in the .ipynb, and the agent then edits a code cell in the .py and syncs
    ipynb = py.with_suffix(".ipynb")
    nb = nbf.read(ipynb, as_version=4)
    md = nb.cells[0]
    for label in cheap.PRE:
        md.source = md.source.replace(f"**{label}:** _(Mark)_", f"**{label}:** mark wrote {label}")
    nbf.write(nb, ipynb)
    py.write_text(py.read_text(encoding="utf-8").replace("# ---- data ----", "# ---- data ----\nx = 1"), encoding="utf-8")
    cli.sync(py)
    assert "mark wrote Hypothesis" in nbf.read(ipynb, as_version=4).cells[0].source      # not overwritten
    assert "mark wrote Hypothesis" in py.read_text(encoding="utf-8")                      # pulled into the .py
    assert cheap.require(cid)["Hypothesis"] == "mark wrote Hypothesis"
    assert cheap.status(cid)["stage"] == "awaiting post"
    nb = nbf.read(ipynb, as_version=4)                   # editing the pre afterwards is recorded, not hidden
    nb.cells[0].source = nb.cells[0].source.replace("mark wrote Hypothesis", "something else")
    nbf.write(nb, ipynb)
    cheap.require(cid)
    assert cheap.status(cid)["pre_changed"]


def test_calibration(cards):
    meta, body = prereg.read_card("H001")
    meta |= {"prediction": "p", "falsified_if": "f", "confidence": 0.8, "predicted_by": "mark"}
    prereg.write_card("H001", meta, body)
    prereg.register("H001")
    prereg.resolve("H001", outcome=False)
    cal = prereg.calibration()
    assert cal["n"] == 1 and abs(cal["brier"] - 0.64) < 1e-12


# ---------- hooks ----------

def _hook(script, payload):
    return subprocess.run([sys.executable, str(ROOT / ".claude/hooks" / script)], input=json.dumps(payload), capture_output=True, text=True)


def test_scan_hook_flags_leaky_code(tmp_path):
    d = tmp_path / "research" / "notebooks"
    d.mkdir(parents=True)
    f = d / "H009_bad.py"
    f.write_text("x = df['mid'].shift(-1)\ny = df.rolling(5, center=True).mean()\nz = s.fillna(method='bfill')  # lookahead-ok: not really\n")
    r = _hook("scan_lookahead.py", {"tool_name": "Write", "tool_input": {"file_path": str(f)}})
    assert r.returncode == 2
    assert "negative shift" in r.stderr and "centred" in r.stderr and "backfill" not in r.stderr


def test_protect_hook_blocks_locked_edit(cards):
    meta, body = prereg.read_card("H001")
    meta |= {"prediction": "p", "falsified_if": "f", "confidence": 0.6, "predicted_by": "mark"}
    prereg.write_card("H001", meta, body)
    prereg.register("H001")
    path = cards / "H001_x.md"
    r = _hook("protect_cards.py", {"tool_name": "Edit", "tool_input": {"file_path": str(path), "old_string": "confidence: 0.6", "new_string": "confidence: 0.9"}})
    assert r.returncode == 2
    ok = _hook("protect_cards.py", {"tool_name": "Edit", "tool_input": {"file_path": str(path), "old_string": "## Results (agent-written after the run)", "new_string": "## Results (agent-written after the run)\nfoo"}})
    assert ok.returncode == 0


def test_template_card_parses_all_fields(cards):
    meta, body = prereg.read_card("H001")
    for k in ("prediction", "falsified_if", "confidence", "predicted_by", "lock_hash"):
        assert k in meta, k
    assert body.lstrip().startswith("## Idea")
    prereg.write_card("H001", meta, body)  # round-trip keeps it parseable
    assert prereg.read_card("H001")[0]["id"] == "H001"


# ---------- notebook dump tool ----------

def test_nb_dump_extracts_plots_and_flags_out_of_order(tmp_path):
    import base64
    import nbformat

    png = base64.b64encode(bytes.fromhex("89504e470d0a1a0a")).decode()  # PNG header is enough
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("plot()", execution_count=2, outputs=[nbformat.v4.new_output("display_data", data={"image/png": png})]),
        nbformat.v4.new_code_cell("print(1)", execution_count=1, outputs=[nbformat.v4.new_output("stream", name="stdout", text="1\n")]),
    ]
    f = tmp_path / "zz_test_dump.ipynb"
    nbformat.write(nb, f)
    r = subprocess.run([sys.executable, str(ROOT / "tools/nb_dump.py"), str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "IMAGE saved" in r.stdout and "out of order" in r.stdout
    img = ROOT / ".scratch/nb_dump/zz_test_dump/cell000_out0.png"
    assert img.exists()
    import shutil
    shutil.rmtree(img.parent)
