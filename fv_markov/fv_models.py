"""
fv_models.py
------------
Fair-value candidates for a cross-venue perp spread, evaluated by whether they
predict the next move in the spread better than the incumbent rule does.

Key simplification: for a cross-venue spread the latent efficient price cancels.
If  p_HL = m + s_HL + e_HL  and  p_BIN = m + s_BIN + e_BIN,  then the spread is
(s_HL - s_BIN) + noise, with no random-walk component. A slow rolling mean
removes the funding/index basis; what is left ("y") is the fast distortion that
every model below tries to explain.

Ladder - run in order, stop at the first rung that fails to beat the previous:

    0. magnitude_check   is there anything here at all?
    1. mid               fair spread = slow basis                  (incumbent)
    2. microprice        + L1 microprice correction
    3. linear            + book-state features, OLS
    4. markov            + 2-regime switching, regime-interacted

Every model is nested in the next, so the fallback is always free.

Usage
-----
    import fv_models as fv

    f = fv.build_features(df)
    fv.magnitude_check(f, cost_bps=4.92)

    res = fv.run_ladder(f, horizon=5, cost_bps=4.92, n_splits=5)
    print(res["table"])
    print(res["overlap"])
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 0.  features
# ----------------------------------------------------------------------

BOOK_COLS = ["mp_adj", "imb_diff", "imb_hl", "imb_bin",
             "spr_hl", "spr_bin", "dep_hl", "dep_bin", "vol"]


def build_features(df, basis_window=120, vol_window=60):
    """
    df : the joined 1-minute frame (bin_* and hl_* columns from your notebook).

    Returns a frame whose 'y' column is the fast component of the spread in bps.
    Everything is known at bar t; nothing looks forward.
    """
    f = pd.DataFrame(index=df.index)
    ref = df["reference_mid"]

    def bps(x):
        return x / ref * 1e4

    spread = bps(df["hl_mid"] - df["bin_mid"])
    f["spread"] = spread
    f["basis"] = spread.rolling(basis_window, min_periods=basis_window).mean()
    f["y"] = spread - f["basis"]

    # L1 microprice correction, per venue and differenced
    f["mp_hl"] = bps(df["microprice_hl"] - df["hl_mid"])
    f["mp_bin"] = bps(df["microprice_bin"] - df["bin_mid"])
    f["mp_adj"] = f["mp_hl"] - f["mp_bin"]

    # book state
    f["imb_hl"] = df["imbalance_hl"]
    f["imb_bin"] = df["imbalance_bin"]
    f["imb_diff"] = f["imb_hl"] - f["imb_bin"]
    f["spr_hl"] = bps(2.0 * (df["hl_mid"] - df["hl_bid"]))
    f["spr_bin"] = bps(df["ask"] - df["bin_bid"])
    f["dep_hl"] = np.log1p(df["bid_size_hl"] + df["ask_size_hl"])
    f["dep_bin"] = np.log1p(df["bid_size_bin"] + df["ask_size_bin"])
    f["vol"] = spread.diff().rolling(vol_window,
                                     min_periods=vol_window // 2).std()

    return f.dropna()


def magnitude_check(f, cost_bps, verbose=True):
    """
    The arithmetic that decides whether to build anything. Run this first.

    If the microprice correction is small relative to the round-trip cost, no
    amount of modelling of the *level* will matter at this sampling frequency.
    """
    out = pd.Series({
        "|y| median": f["y"].abs().median(),
        "|y| 95th": f["y"].abs().quantile(0.95),
        "|mp_adj| median": f["mp_adj"].abs().median(),
        "|mp_adj| 95th": f["mp_adj"].abs().quantile(0.95),
        "quoted spread HL median": f["spr_hl"].median(),
        "quoted spread BIN median": f["spr_bin"].median(),
        "round-trip cost": float(cost_bps),
    }, name="bps")

    rho = f["y"].autocorr(1)
    n = len(f)
    n_eff = n * (1 - rho) / (1 + rho) if rho < 1 else np.nan

    if verbose:
        print(out.round(3).to_string())
        print()
        print(f"microprice adj (95th) as % of cost : "
              f"{100 * out['|mp_adj| 95th'] / cost_bps:.2f}%")
        print(f"bars                               : {n}")
        print(f"lag-1 autocorr of y                : {rho:.3f}")
        print(f"effective independent observations : {n_eff:.0f}")
    return out


def make_target(f, horizon):
    """Realised change in the fast spread over `horizon` bars."""
    return (f["y"].shift(-horizon) - f["y"]).rename("d")


# ----------------------------------------------------------------------
# 1.  linear models
# ----------------------------------------------------------------------

def _design(f, cols):
    Z = f[cols].to_numpy(float)
    return np.column_stack([np.ones(len(Z)), Z])


def fit_ols(Z, d, ridge=1e-6):
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ d)


# ----------------------------------------------------------------------
# 2.  minimal K-regime Markov switching regression (Hamilton filter)
# ----------------------------------------------------------------------
#
# Fitted to the ONE-STEP change in y, which is observable in real time.
# Its output is a filtered regime probability at each bar, conditioning only
# on information up to and including that bar. That probability is then used
# as an interaction term in the h-step trading model, so the regime machinery
# never has to forecast anything it cannot see.

_NORM = 1.0 / np.sqrt(2.0 * np.pi)

# Regressors for the regime equation. Deliberately short: the regime is meant
# to be a volatility/liquidity state, not a second return model. Adding the
# full book set here makes the likelihood flat and the fit slow.
MARKOV_COLS = ["vol", "imb_diff", "mp_adj"]


def _eta(dy, Z, b, sig):
    """(T, K) matrix of conditional normal densities."""
    resid = dy[:, None] - Z @ b.T
    return np.maximum(_NORM / sig * np.exp(-0.5 * (resid / sig) ** 2), 1e-300)


def _filter(eta, P, pi0):
    """Hamilton filter. filt[t] conditions on data <= t. pred[t] on data < t."""
    T, K = eta.shape
    filt = np.empty((T, K))
    pred = np.empty((T, K))
    xi = pi0
    ll = 0.0
    for t in range(T):
        xi = pi0 if t == 0 else P.T @ xi
        pred[t] = xi
        num = xi * eta[t]
        den = num.sum()
        ll += np.log(den)
        xi = num / den
        filt[t] = xi
    return ll, filt, pred


def _smooth(filt, pred, P):
    """Kim smoother. Returns smoothed probs and the summed joint counts."""
    T, K = filt.shape
    sm = np.empty((T, K))
    sm[-1] = filt[-1]
    counts = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        ratio = sm[t + 1] / pred[t + 1]
        sm[t] = filt[t] * (P @ ratio)
        sm[t] /= sm[t].sum()
        counts += filt[t][:, None] * P * ratio[None, :]
    return sm, counts


def _hamilton(dy, Z, b, sig, P, pi0=None):
    """Kept for out-of-sample filtering with frozen parameters."""
    K = len(sig)
    pi0 = np.full(K, 1.0 / K) if pi0 is None else pi0
    ll, filt, _ = _filter(_eta(dy, Z, b, sig), P, pi0)
    return ll, filt


def fit_markov(dy, Z, K=2, n_iter=150, tol=1e-7, n_starts=2, seed=0,
               ridge=1e-8):
    """
    EM (Hamilton filter + Kim smoother, closed-form M-step).

    Each M-step is K weighted least-squares solves, so this is roughly two
    orders of magnitude faster than searching the likelihood directly and it
    cannot wander into a region where the filter underflows.
    """
    T, p = Z.shape
    b0 = fit_ols(Z, dy, ridge=ridge)
    sd = float(np.std(dy - Z @ b0)) or 1.0
    rng = np.random.default_rng(seed)
    best = None

    for s in range(n_starts):
        b = b0[None, :] * np.ones((K, 1))
        if s:
            b = b * (1 + 0.2 * rng.standard_normal((K, p)))
        sig = sd * np.linspace(0.5, 1.8, K)
        P = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(P, 0.9)
        pi0 = np.full(K, 1.0 / K)

        ll_old = -np.inf
        ll = -np.inf
        for _ in range(n_iter):
            ll, filt, pred = _filter(_eta(dy, Z, b, sig), P, pi0)
            sm, counts = _smooth(filt, pred, P)

            for k in range(K):
                w = sm[:, k]
                Zw = Z * w[:, None]
                A = Z.T @ Zw + ridge * np.eye(p)
                b[k] = np.linalg.solve(A, Zw.T @ dy)
                r = dy - Z @ b[k]
                sig[k] = max(np.sqrt((w * r * r).sum() / w.sum()), 1e-8)

            P = counts / counts.sum(axis=1, keepdims=True)
            pi0 = sm[0]

            if ll - ll_old < tol * abs(ll_old):
                break
            ll_old = ll

        if np.isfinite(ll) and (best is None or ll > best["ll"]):
            best = {"b": b.copy(), "sig": sig.copy(), "P": P.copy(),
                    "pi0": pi0.copy(), "ll": float(ll)}

    if best is None:
        raise RuntimeError("Markov fit failed from every start")

    order = np.argsort(best["sig"])        # label regimes by vol, low first
    return {"b": best["b"][order], "sig": best["sig"][order],
            "P": best["P"][np.ix_(order, order)],
            "pi0": best["pi0"][order], "ll": best["ll"]}


def markov_probs(f, exog_cols=MARKOV_COLS, K=2, train_frac=0.5, **kw):
    """
    Fit on the first `train_frac` of the sample, then filter the whole sample
    with those frozen parameters. Returns (probs DataFrame, params).

    probs[t] uses only information up to and including bar t.
    """
    y = f["y"].to_numpy(float)
    dy = np.diff(y, prepend=np.nan)
    exog_cols = [c for c in exog_cols if c in f.columns]
    Z = np.column_stack([
        np.ones(len(f)),
        np.r_[np.nan, y[:-1]],                       # y_{t-1}
        f[exog_cols].shift(1).to_numpy(float),       # x_{t-1}
    ])
    ok = np.isfinite(dy) & np.isfinite(Z).all(axis=1)

    n_tr = int(ok.sum() * train_frac)
    idx = np.flatnonzero(ok)
    tr = idx[:n_tr]

    par = fit_markov(dy[tr], Z[tr], K=K, **kw)
    par["cols"] = ["const", "y_lag1"] + list(exog_cols)
    _, filt = _hamilton(dy[idx], Z[idx], par["b"], par["sig"], par["P"],
                        par["pi0"])

    probs = pd.DataFrame(np.nan, index=f.index,
                         columns=[f"p{k}" for k in range(K)])
    probs.iloc[idx] = filt
    return probs, par


# ----------------------------------------------------------------------
# 3.  evaluation
# ----------------------------------------------------------------------

def _oos_r2(d, dhat):
    ss_res = np.sum((d - dhat) ** 2)
    ss_tot = np.sum(d ** 2)
    return 1.0 - ss_res / ss_tot


def _trade_stats(d, dhat, cost_bps):
    """
    Entry rule: fade when the model expects a move bigger than the cost.
    Short the spread when dhat < -cost, long when dhat > +cost.
    Realised net P&L per trade in bps.
    """
    side = np.where(dhat > cost_bps, 1.0,
                    np.where(dhat < -cost_bps, -1.0, 0.0))
    took = side != 0
    if took.sum() == 0:
        return dict(n=0, hit=np.nan, mean_net=np.nan, total_net=0.0), side
    pnl = side[took] * d[took] - cost_bps
    return dict(n=int(took.sum()),
                hit=float((pnl > 0).mean()),
                mean_net=float(pnl.mean()),
                total_net=float(pnl.sum())), side


def _splits(n, n_splits, embargo):
    """Expanding-window walk-forward with an embargo between train and test."""
    fold = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        tr_end = fold * i
        te_beg = tr_end + embargo
        te_end = n if i == n_splits else min(fold * (i + 1), n)
        if te_beg >= te_end:
            continue
        yield np.arange(0, tr_end), np.arange(te_beg, te_end)


def run_ladder(f, horizon, cost_bps, n_splits=5, embargo=None,
               book_cols=BOOK_COLS, markov_cols=MARKOV_COLS,
               K=2, markov_train_frac=None):
    """
    Walk-forward comparison of the four rungs at one horizon.

    Returns dict with:
      table   : per-model out-of-sample R2 and trade statistics
      overlap : 2x2 of which trades the incumbent and the best model take
      probs   : the filtered regime probabilities (for plotting / naming)
      par     : fitted Markov parameters
    """
    embargo = horizon * 3 if embargo is None else embargo
    # Fit the regime parameters on the first training block only, so the
    # regime probabilities in every test fold are genuinely out-of-sample.
    if markov_train_frac is None:
        markov_train_frac = 1.0 / (n_splits + 1)

    probs, par = markov_probs(f, exog_cols=markov_cols, K=K,
                              train_frac=markov_train_frac)
    g = f.join(probs).copy()
    g["d"] = make_target(f, horizon)
    g = g.dropna()

    pcol = f"p{K - 1}"                       # prob of the high-volatility state

    specs = {
        "1_mid":        ["y"],
        "2_microprice": ["y", "mp_adj"],
        "3_linear":     ["y"] + list(book_cols),
    }

    d = g["d"].to_numpy(float)
    n = len(g)
    preds = {k: np.full(n, np.nan) for k in list(specs) + ["4_markov"]}

    for tr, te in _splits(n, n_splits, embargo):
        for name, cols in specs.items():
            Z = _design(g, cols)
            b = fit_ols(Z[tr], d[tr])
            preds[name][te] = Z[te] @ b

        base = _design(g, ["y"] + list(book_cols))
        p = g[pcol].to_numpy(float)[:, None]
        Zm = np.column_stack([base, base[:, 1:] * p])   # regime interactions
        bm = fit_ols(Zm[tr], d[tr], ridge=1e-3)
        preds["4_markov"][te] = Zm[te] @ bm

    rows, sides = {}, {}
    for name, dhat in preds.items():
        m = np.isfinite(dhat)
        stats, side = _trade_stats(d[m], dhat[m], cost_bps)
        stats["oos_r2"] = _oos_r2(d[m], dhat[m])
        rows[name] = stats
        s = np.full(n, np.nan)
        s[m] = side
        sides[name] = s

    table = pd.DataFrame(rows).T[["oos_r2", "n", "hit", "mean_net", "total_net"]]

    if table["n"].max() < 30:
        print(f"WARNING: at most {int(table['n'].max())} trades clear the "
              f"{cost_bps} bps threshold. Nothing below is measurable; either "
              f"the deviations are too small for this cost or the horizon is "
              f"too short. Read oos_r2 only.")

    # what does the model change relative to the incumbent?
    base_s = sides["1_mid"]
    best_name = (table["mean_net"].idxmax()
                 if table["mean_net"].notna().any()
                 else table["oos_r2"].idxmax())
    best_s = sides[best_name]
    m = np.isfinite(base_s) & np.isfinite(best_s)
    took_b, took_m = base_s[m] != 0, best_s[m] != 0
    dd, bb, mm = d[m], base_s[m], best_s[m]

    def _cell(mask, side):
        if mask.sum() == 0:
            return dict(n=0, mean_net=np.nan)
        pnl = side[mask] * dd[mask] - cost_bps
        return dict(n=int(mask.sum()), mean_net=float(pnl.mean()))

    overlap = pd.DataFrame({
        "both":            _cell(took_b & took_m, bb),
        "incumbent only":  _cell(took_b & ~took_m, bb),
        f"{best_name} only": _cell(~took_b & took_m, mm),
    }).T

    return {"table": table.round(3), "overlap": overlap.round(3),
            "probs": probs, "par": par, "best": best_name,
            "preds": preds, "horizon": horizon, "cost_bps": cost_bps,
            "sides": pd.DataFrame(sides, index=g.index), "data": g}


def name_the_regime(res, cols=("vol", "spr_hl", "spr_bin",
                               "dep_hl", "dep_bin", "y")):
    """
    The test that decides whether the regime is a mechanism or a story.

    Compare the mean of each observable in the high-volatility state against the
    low-volatility state. If one or two variables separate the regimes cleanly,
    use those variables directly and throw the Markov model away.
    """
    g = res["data"]
    K = res["par"]["b"].shape[0]
    hi = g[f"p{K - 1}"] > 0.5
    cols = [c for c in cols if c in g.columns]
    out = pd.DataFrame({
        "low state": g.loc[~hi, cols].mean(),
        "high state": g.loc[hi, cols].mean(),
    })
    out["ratio"] = out["high state"] / out["low state"]
    out["time in state"] = np.nan
    out.loc[out.index[0], "time in state"] = float(hi.mean())
    return out.round(3)


def regime_summary(par):
    """
    The regime in trader's language: how fast the spread pulls back, how noisy
    it is, and how long each state lasts. Nothing here needs the filter to
    interpret - if the two rows look the same, there is one regime.
    """
    b, sig, P = par["b"], par["sig"], par["P"]
    K = len(sig)
    i_lag = par.get("cols", ["const", "y_lag1"]).index("y_lag1")
    rho = 1.0 + b[:, i_lag]                      # fitted on dy, so rho = 1 + c
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where((rho > 0) & (rho < 1), -np.log(2) / np.log(rho), np.nan)
    stay = np.diag(P)
    return pd.DataFrame({
        "rho (pullback)": rho,
        "half-life (bars)": hl,
        "sigma (bps/bar)": sig,
        "stay prob": stay,
        "expected duration (bars)": 1.0 / np.maximum(1 - stay, 1e-9),
    }, index=[f"state {k}" for k in range(K)]).round(3)


def regime_health(res, verbose=True):
    """
    Three tells that separate a real regime from a fitted artefact. Calibrated
    on simulated data where the answer is known: with two true regimes all
    three pass; with one true regime at least one fails every time.

      1. sigma ratio between 1.5 and 6.  A ratio of 30 or 700 means one state
         has collapsed onto a handful of outliers, not a state of the world.
      2. both rho in (0, 1).  A rho above 1 or near 0 is a state with no
         mean-reversion behaviour to speak of - nothing to condition on.
      3. both states occupied at least 10% of the time, and expected duration
         above a few bars.  A state you are in 2% of the time cannot be traded.
    """
    rs = regime_summary(res["par"])
    K = len(rs)
    hi = res["data"][f"p{K - 1}"] > 0.5
    occ = np.array([1 - hi.mean(), hi.mean()]) if K == 2 else None

    sig = rs["sigma (bps/bar)"].to_numpy()
    rho = rs["rho (pullback)"].to_numpy()
    dur = rs["expected duration (bars)"].to_numpy()
    ratio = sig.max() / max(sig.min(), 1e-12)

    checks = {
        "sigma ratio in [1.5, 6]": (1.5 <= ratio <= 6.0, round(float(ratio), 2)),
        "all rho in (0, 1)": (bool(((rho > 0) & (rho < 1)).all()),
                              np.round(rho, 3).tolist()),
        "min occupancy >= 10%": (bool(occ is None or occ.min() >= 0.10),
                                 None if occ is None else np.round(occ, 3).tolist()),
        "min duration >= 5 bars": (bool(dur.min() >= 5.0),
                                   np.round(dur, 1).tolist()),
    }
    if verbose:
        for k, (ok, val) in checks.items():
            print(f"{'PASS' if ok else 'FAIL'}  {k:<26} {val}")
        if not all(ok for ok, _ in checks.values()):
            print("\nAt least one tell failed: treat the Markov rung's result "
                  "as an artefact and stop at rung 3.")
    return checks


def fv_frame(f, res=None, horizon=None):
    """
    The three fair values side by side, in bps, for one plot.

        fv_static : fixed - the spread is assumed to belong at zero
        fv_moving : rolling basis - the funding/index level the spread sits on
        fv_model  : basis plus the part of today's gap the model says is
                    permanent, i.e. the gap it does NOT expect to close

    Caveat worth saying out loud before anyone reads the chart: fv_static
    assumes the whole gap closes, fv_model only claims the conditional mean.
    So |spread - fv_model| is smaller by construction, and comparing the two
    at the same entry threshold flatters the static rule. The honest
    comparison is the net-of-cost column in run_ladder's table, not this plot.
    """
    out = pd.DataFrame({
        "spread": f["spread"],
        "fv_static": 0.0,
        "fv_moving": f["basis"],
    })
    if res is not None:
        g = res["data"]
        dhat = res["preds"][res["best"]]
        out["fv_model"] = pd.Series(g["basis"].to_numpy() +
                                    g["y"].to_numpy() + dhat, index=g.index)
    return out


def deviation_series(f, res=None):
    """
    Deviation-from-fair series in bps, shaped to drop straight into plot_pnl.

        static : f['spread']          - what you are feeding it now
        moving : f['y']               - spread less the rolling basis
        model  : -dhat                - the move the model expects, signed

    Read the caveat in fv_frame before putting the three on the same axes.
    """
    out = pd.DataFrame({"static": f["spread"], "moving": f["y"]})
    if res is not None:
        g = res["data"]
        out["model"] = pd.Series(-res["preds"][res["best"]], index=g.index)
    return out
