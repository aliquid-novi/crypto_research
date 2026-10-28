"""Models: markout regressions, toxicity classifier, fill probability and hazards, EV per quote,
selection correction, 2-state HMM, Hawkes, response/propagator, Kyle lambda, Roll, Huang-Stoll,
Hasbrouck VAR and square-root impact.

Optional dependencies: ``lifelines`` (Cox PH) and ``hmmlearn``. Every function has a pure
numpy/statsmodels path and uses that path when the optional package is missing.
Sign convention: d = +1 our buy; markouts in bps, positive = good for us.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

from ._util import NS, asof_values

try:  # optional
    import lifelines  # noqa: F401
    HAVE_LIFELINES = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_LIFELINES = False
try:  # optional
    import hmmlearn  # noqa: F401
    HAVE_HMMLEARN = True
except Exception:  # pragma: no cover
    HAVE_HMMLEARN = False


def _sm():
    import statsmodels.api as sm

    return sm


# =============================================================================================
# Markout regressions
# =============================================================================================
def signed_features(df: pd.DataFrame, features: list[str], side: np.ndarray) -> pd.DataFrame:
    """Multiply directional features by the fill side d so positive = favourable for our fill.

    Pass features oriented as "positive = pressure for the price to go UP" (e.g. imbalance - 0.5,
    OFI, lead-minus-our mid). Non-directional features (vol, spread, latency) should NOT be signed;
    pass them separately as ``unsigned`` in ``markout_regression``.
    """
    return df[features].mul(np.asarray(side, float), axis=0)


def markout_regression(mo: pd.DataFrame, X: pd.DataFrame, nw_lags: dict | int | None = None,
                       sample_spacing_s: float | None = None, ts: np.ndarray | None = None,
                       quantiles: tuple[float, ...] = (0.05, 0.10), huber: bool = True, ci: float = 0.95) -> pd.DataFrame:
    """Regress markout_h on features, for every horizon h (columns of ``mo``).

    Models: OLS with Newey-West SE (lags default = overlap = ceil(h / median spacing between fills),
    computed from ``ts`` or ``sample_spacing_s``), Huber robust regression (RLM), and quantile
    regressions at ``quantiles`` (tail risk: what predicts the worst fills). X should already be
    signed where directional (see ``signed_features``). Rows must be in time order for NW.
    Returns long DataFrame [horizon_s, model, feature, coef, se, lo, hi, n].
    """
    sm = _sm()
    from .stats import nw_lags_for_overlap

    z = sps.norm.ppf(0.5 + ci / 2)
    if sample_spacing_s is None:
        sample_spacing_s = float(np.median(np.diff(np.sort(ts))) / NS) if ts is not None and len(ts) > 1 else 1.0
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    rows = []
    for h in mo.columns:
        y = mo[h].to_numpy(dtype=float)
        m = np.isfinite(y) & np.all(np.isfinite(Xc.to_numpy()), axis=1)
        if m.sum() < Xc.shape[1] + 5:
            continue
        yy, XX = y[m], Xc[m]
        lags = nw_lags.get(h) if isinstance(nw_lags, dict) else nw_lags
        lags = lags if lags is not None else nw_lags_for_overlap(float(h), sample_spacing_s)
        fits = {"ols_nw": sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)}),
                "ols_iid": sm.OLS(yy, XX).fit()}
        if huber:
            fits["huber"] = sm.RLM(yy, XX, M=sm.robust.norms.HuberT()).fit()
        for q in quantiles:
            try:
                fits[f"q{int(round(q * 100)):02d}"] = sm.QuantReg(yy, XX).fit(q=q, max_iter=2000)
            except Exception:  # quantile regression can fail on degenerate data; skip it
                pass
        for name, r in fits.items():
            for feat in XX.columns:
                c, se = float(r.params[feat]), float(r.bse[feat])
                rows.append({"horizon_s": float(h), "model": name, "feature": feat, "coef": c, "se": se,
                             "lo": c - z * se, "hi": c + z * se, "n": int(m.sum())})
    return pd.DataFrame(rows)


# =============================================================================================
# Toxicity
# =============================================================================================
def toxicity_model(X: pd.DataFrame, markout_bps: np.ndarray, ts: np.ndarray, threshold_bps: float = 0.0,
                   train_frac: float = 0.6) -> dict:
    """Logistic toxicity classifier with a cost-based decision threshold and avoided-loss backtest.

    label = markout_h < threshold_bps. Fit on the first ``train_frac`` of fills by time.
    Threshold p* = g / (g + l), g = mean markout of non-toxic train fills (gain kept),
    l = -mean markout of toxic train fills (loss avoided): skipping a fill with P(toxic)=p is worth
    it when p*l > (1-p)*g. Backtest on the later fills: avoided_loss = -sum(markout of skipped
    toxic), forgone_gain = sum(markout of skipped non-toxic), net = avoided - forgone (bps summed;
    positive = filter would have helped). Evaluate with this, not AUC.
    Returns dict(model, p_star, g, l, backtest (DataFrame over cutoffs), test_summary).
    """
    sm = _sm()
    order = np.argsort(ts, kind="stable")
    Xo = X.iloc[order].reset_index(drop=True).astype(float)
    y = np.asarray(markout_bps, float)[order]
    ok = np.isfinite(y) & np.all(np.isfinite(Xo.to_numpy()), axis=1)
    Xo, y = Xo[ok].reset_index(drop=True), y[ok]
    lab = (y < threshold_bps).astype(int)
    ntr = int(len(y) * train_frac)
    Xc = sm.add_constant(Xo, has_constant="add")
    model = sm.Logit(lab[:ntr], Xc.iloc[:ntr]).fit(disp=False)
    g = float(y[:ntr][lab[:ntr] == 0].mean())
    l = float(-y[:ntr][lab[:ntr] == 1].mean())
    p_star = g / (g + l)
    p = np.asarray(model.predict(Xc.iloc[ntr:]))
    yt = y[ntr:]
    rows = []
    for cut in np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), [p_star]])):
        skip = p > cut
        avoided = float(-yt[skip & (yt < 0)].sum())
        forgone = float(yt[skip & (yt >= 0)].sum())
        rows.append({"cutoff": cut, "skipped_frac": float(skip.mean()), "avoided_loss_bps": avoided,
                     "forgone_gain_bps": forgone, "net_bps": avoided - forgone, "is_p_star": bool(np.isclose(cut, p_star))})
    bt = pd.DataFrame(rows)
    return {"model": model, "p_star": p_star, "g": g, "l": l, "backtest": bt,
            "test_summary": bt[bt["is_p_star"]].iloc[0].to_dict(), "n_train": ntr, "n_test": len(yt),
            "test_mean_markout": float(yt.mean())}


# =============================================================================================
# Fill probability
# =============================================================================================
def quote_distance_samples(orders: pd.DataFrame, ref: pd.DataFrame, step_s: float = 1.0) -> pd.DataFrame:
    """Sample every resting order on a ``step_s`` grid over [ts_ack, ts_done) and record its distance
    from the reference: dist_bps = 1e4 * d * (m - price) / m (positive = behind the mid, passive).

    Returns [order_id, ts, dist_bps, filled_here] where filled_here marks the last sample of an
    order that ended filled/partial (the fill happened within that step). Exposure time per row = step_s.
    """
    step = int(step_s * NS)
    o = orders[orders["ts_ack"].notna() & orders["ts_done"].notna()].copy()
    a = o["ts_ack"].to_numpy(dtype="int64")
    b = o["ts_done"].to_numpy(dtype="int64")
    nstep = np.maximum((b - a) // step, 0) + 1
    rep = np.repeat(np.arange(len(o)), nstep)
    within = np.arange(len(rep)) - np.repeat(np.cumsum(nstep) - nstep, nstep)
    ts = a[rep] + within * step
    q = pd.DataFrame({"ts": ts, "venue": o["venue"].to_numpy()[rep], "symbol": o["symbol"].to_numpy()[rep]})
    m = asof_values(q, ref, ["value"], direction="backward", allow_exact=True)["value"].to_numpy()
    d = o["side"].to_numpy(dtype=float)[rep]
    px = o["price"].to_numpy(dtype=float)[rep]
    filled = np.isin(o["final_status"].to_numpy(), ["filled", "partial"])
    if "filled_qty" in o.columns:
        filled &= o["filled_qty"].fillna(1).to_numpy(dtype=float) > 0
    last = within == (nstep[rep] - 1)
    return pd.DataFrame({"order_id": o["order_id"].to_numpy()[rep], "ts": ts, "dist_bps": 1e4 * d * (m - px) / m,
                         "filled_here": last & filled[rep]})


def fit_fill_intensity(samples: pd.DataFrame, step_s: float = 1.0, bins: np.ndarray | None = None) -> dict:
    """Fit lambda(delta) = A exp(-k delta) (fills per second at distance delta bps behind mid).

    Empirical lambda per distance bin = fills / exposure seconds. Fitted by Poisson GLM
    fills_bin ~ exp(log A - k delta_bin) with offset log(exposure). Returns dict(A, k, table).
    """
    sm = _sm()
    s = samples[np.isfinite(samples["dist_bps"])]
    if bins is None:
        bins = np.unique(np.quantile(s["dist_bps"], np.linspace(0, 1, 13)))
    c = pd.cut(s["dist_bps"], bins, include_lowest=True)
    g = s.groupby(c, observed=True)
    tab = pd.DataFrame({"delta_bps": g["dist_bps"].mean(), "exposure_s": g.size() * step_s,
                        "fills": g["filled_here"].sum()})
    tab["lambda_per_s"] = tab["fills"] / tab["exposure_s"]
    X = sm.add_constant(tab["delta_bps"].to_numpy())
    r = sm.GLM(tab["fills"].to_numpy(), X, family=sm.families.Poisson(), offset=np.log(tab["exposure_s"].to_numpy())).fit()
    tab["lambda_fit"] = np.exp(r.params[0] - (-r.params[1]) * tab["delta_bps"])
    return {"A": float(np.exp(r.params[0])), "k": float(-r.params[1]), "k_se": float(r.bse[1]), "table": tab.reset_index(drop=True),
            "glm": r}


def survival_frame(orders: pd.DataFrame, ref: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per order for hazard models: duration_s = ts_done - ts_ack, event = 1 if it filled
    (filled/partial with filled qty), dist_bps at ack (as quote_distance_samples), plus any
    ``extra`` covariate columns aligned to orders.index."""
    o = orders[orders["ts_ack"].notna() & orders["ts_done"].notna()].copy()
    m = asof_values(o.assign(ts=o["ts_ack"].astype("int64")), ref, ["value"], direction="backward")["value"].to_numpy()
    out = pd.DataFrame({"order_id": o["order_id"].to_numpy(),
                        "duration_s": (o["ts_done"].astype("int64") - o["ts_ack"].astype("int64")).to_numpy() / NS,
                        "event": np.isin(o["final_status"], ["filled", "partial"]).astype(int),
                        "dist_bps": 1e4 * o["side"].to_numpy(dtype=float) * (m - o["price"].to_numpy(dtype=float)) / m},
                       index=o.index)
    if "filled_qty" in o.columns:
        out["event"] = out["event"] * (o["filled_qty"].fillna(1).to_numpy(dtype=float) > 0)
    if extra is not None:
        out = out.join(extra)
    out["duration_s"] = out["duration_s"].clip(lower=1e-6)
    return out.dropna()


def fit_hazard(surv: pd.DataFrame, covariates: list[str], use_lifelines: bool | None = None,
               bin_edges_s: tuple[float, ...] = (0, 0.5, 1, 2, 4, 8, 16, 32, 64, np.inf)) -> dict:
    """Fill hazard model. Cox PH via lifelines when available (and ``use_lifelines`` not False),
    otherwise a discrete-time logistic hazard:

        logit P(fill in bin b | alive at start of b) = alpha_b + beta . covariates

    fitted on person-period data (one row per order per time bin it was alive in). Positive beta =
    covariate raises the fill hazard (log hazard ratio in Cox; log odds per bin in the fallback;
    they agree closely when per-bin probabilities are small). Returns dict(method, params DataFrame
    [coef, se, lo, hi], model).
    """
    use = HAVE_LIFELINES if use_lifelines is None else (use_lifelines and HAVE_LIFELINES)
    if use:
        from lifelines import CoxPHFitter

        cph = CoxPHFitter().fit(surv[["duration_s", "event", *covariates]], "duration_s", "event")
        s = cph.summary
        params = pd.DataFrame({"coef": s["coef"], "se": s["se(coef)"], "lo": s["coef lower 95%"], "hi": s["coef upper 95%"]})
        return {"method": "cox_ph(lifelines)", "params": params, "model": cph}
    sm = _sm()
    edges = np.asarray(bin_edges_s, float)
    dur = surv["duration_s"].to_numpy()
    nb = np.searchsorted(edges, dur, side="right")  # bins entered (1-based index of the bin containing dur)
    nb = np.clip(nb, 1, len(edges) - 1)
    rep = np.repeat(np.arange(len(surv)), nb)
    b = np.arange(len(rep)) - np.repeat(np.cumsum(nb) - nb, nb)
    ev = np.zeros(len(rep), int)
    last = b == (nb[rep] - 1)
    ev[last] = surv["event"].to_numpy()[rep[last]]
    pp = pd.DataFrame(surv[covariates].to_numpy()[rep], columns=covariates)
    dummies = pd.get_dummies(pd.Categorical(b), prefix="bin", dtype=float)
    Xd = pd.concat([dummies, pp.astype(float)], axis=1)
    r = sm.GLM(ev, Xd, family=sm.families.Binomial()).fit()
    params = pd.DataFrame({"coef": r.params, "se": r.bse, "lo": r.params - 1.96 * r.bse, "hi": r.params + 1.96 * r.bse})
    return {"method": "discrete_time_logit", "params": params.loc[covariates], "baseline": params.drop(index=covariates),
            "model": r, "bin_edges_s": edges}


def ev_per_quote(samples_or_orders: pd.DataFrame, markout_at_fill: pd.DataFrame, fee_bps: float,
                 bins: np.ndarray | None = None) -> pd.DataFrame:
    """Expected value per quote by distance bin:  EV = P(fill) * (E[markout | fill] - fee_bps).

    ``samples_or_orders``: survival_frame output (one row per order: dist_bps, event).
    ``markout_at_fill``: DataFrame [dist_bps, markout_bps] for filled orders (markout at your chosen
    horizon). fee_bps: maker fee in bps (negative = rebate). Units: bps of notional per quote.
    """
    s = samples_or_orders
    bins = bins if bins is not None else np.unique(np.quantile(s["dist_bps"].dropna(), np.linspace(0, 1, 9)))
    cs = pd.cut(s["dist_bps"], bins, include_lowest=True)
    cm = pd.cut(markout_at_fill["dist_bps"], bins, include_lowest=True)
    t = pd.DataFrame({"p_fill": s.groupby(cs, observed=False)["event"].mean(), "n_quotes": s.groupby(cs, observed=False).size(),
                      "mo_given_fill": markout_at_fill.groupby(cm, observed=False)["markout_bps"].mean(),
                      "n_fills": markout_at_fill.groupby(cm, observed=False).size()})
    t["ev_bps"] = t["p_fill"] * (t["mo_given_fill"] - fee_bps)
    return t


# =============================================================================================
# Selection correction
# =============================================================================================
def heckman_two_step(Z: pd.DataFrame, selected: np.ndarray, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Heckman two-step: probit selection (filled ~ Z) on ALL orders, then OLS of the outcome
    (markout) on X plus the inverse Mills ratio lambda = phi(Z g)/Phi(Z g) on the selected rows.

    The IMR coefficient estimates rho*sigma: significantly non-zero = fills are a selected sample
    and naive markout regressions are biased. Z should contain at least one variable that affects
    fill probability but not the markout (exclusion restriction), else identification rests on
    the probit's non-linearity only. SEs of step 2 are NOT corrected for the generated regressor
    (they are too small) - bootstrap by order if you need inference.
    """
    sm = _sm()
    Zc = sm.add_constant(Z.astype(float), has_constant="add")
    sel = np.asarray(selected, int)
    pr = sm.Probit(sel, Zc).fit(disp=False)
    xb = Zc.to_numpy() @ pr.params.to_numpy()
    imr = sps.norm.pdf(xb) / np.clip(sps.norm.cdf(xb), 1e-12, None)
    m = sel == 1
    if len(X) != m.sum():
        raise ValueError("X and y must be the selected rows only (len == selected.sum()), in the same order")
    Xo = sm.add_constant(X.astype(float), has_constant="add").reset_index(drop=True)
    Xo["imr"] = imr[m]
    ols = sm.OLS(np.asarray(y, float), Xo, missing="drop").fit()
    return {"probit": pr, "outcome": ols, "imr_coef": float(ols.params["imr"]), "imr_t": float(ols.tvalues["imr"])}


def ipw_regression(Z: pd.DataFrame, selected: np.ndarray, X: pd.DataFrame, y: np.ndarray, clip: float = 0.01) -> dict:
    """Inverse-probability weighting: logit P(fill | Z) on all orders, then WLS of the outcome on X
    for filled orders with weights 1 / p_hat (clipped at ``clip``). Estimates the markout
    relationship for the population of QUOTES, not just the quotes that happened to fill.
    X and y are for the selected rows (same order as selected == 1)."""
    sm = _sm()
    Zc = sm.add_constant(Z.astype(float), has_constant="add")
    sel = np.asarray(selected, int)
    lg = sm.Logit(sel, Zc).fit(disp=False)
    p = np.clip(np.asarray(lg.predict(Zc)), clip, 1)[sel == 1]
    yy = np.asarray(y, float)
    ok = np.isfinite(yy)  # selected rows without an outcome (e.g. markout past end of data) are dropped
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    wls = sm.WLS(yy[ok], Xc[ok], weights=1 / p[ok]).fit(cov_type="HC1")
    return {"selection": lg, "outcome": wls, "weights": 1 / p}


# =============================================================================================
# Two-state Gaussian HMM
# =============================================================================================
def _forward(x, pi, P, mu, sd):
    n = len(x)
    B = sps.norm.pdf(x[:, None], mu[None, :], sd[None, :]) + 1e-300
    alpha = np.zeros((n, 2))
    c = np.zeros(n)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ P) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, B


def fit_hmm2(x: np.ndarray, n_iter: int = 200, tol: float = 1e-6, use_hmmlearn: bool = False, seed: int = 0) -> dict:
    """Two-state Gaussian HMM on a markout series by Baum-Welch (scaled forward-backward).

    State 'toxic' = the state with the LOWER mean. Returns dict(means, sds, P, pi, loglik,
    p_toxic_filtered (forward filter: uses data up to t only - usable in real time),
    p_toxic_smoothed (uses all data - descriptive only), toxic_state).
    ``use_hmmlearn=True`` fits with hmmlearn when installed; the filter is always this module's.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if use_hmmlearn and HAVE_HMMLEARN:
        from hmmlearn.hmm import GaussianHMM

        m = GaussianHMM(2, covariance_type="diag", n_iter=n_iter, tol=tol, random_state=seed).fit(x[:, None])
        mu, sd, P, pi = m.means_.ravel(), np.sqrt(m.covars_.ravel()), m.transmat_, m.startprob_
    else:
        q = np.quantile(x, [0.25, 0.75])
        mu, sd = np.array([q[1], q[0]]), np.array([x.std(), x.std()])
        P = np.array([[0.95, 0.05], [0.05, 0.95]])
        pi = np.array([0.5, 0.5])
        prev = -np.inf
        for _ in range(n_iter):
            alpha, c, B = _forward(x, pi, P, mu, sd)
            n = len(x)
            beta = np.zeros((n, 2))
            beta[-1] = 1
            for t in range(n - 2, -1, -1):
                beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gam = alpha * beta
            gam /= gam.sum(1, keepdims=True)
            xi = (alpha[:-1, :, None] * P[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
            P = xi.sum(0) / xi.sum((0, 2))[:, None]
            pi = gam[0]
            w = gam.sum(0)
            mu = (gam * x[:, None]).sum(0) / w
            sd = np.sqrt((gam * (x[:, None] - mu) ** 2).sum(0) / w) + 1e-9
            ll = np.log(c).sum()
            if ll - prev < tol:
                break
            prev = ll
    alpha, c, B = _forward(x, pi, P, mu, sd)
    n = len(x)
    beta = np.zeros((n, 2))
    beta[-1] = 1
    for t in range(n - 2, -1, -1):
        beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gam = alpha * beta
    gam /= gam.sum(1, keepdims=True)
    k = int(np.argmin(mu))
    return {"means": mu, "sds": sd, "P": P, "pi": pi, "loglik": float(np.log(c).sum()), "toxic_state": k,
            "p_toxic_filtered": alpha[:, k], "p_toxic_smoothed": gam[:, k]}


# =============================================================================================
# Hawkes (exponential kernel phi(t) = alpha * beta * exp(-beta t); alpha = branching ratio)
# =============================================================================================
def _decay_sum(t: np.ndarray, marks: np.ndarray, beta: float, block: float = 300.0) -> np.ndarray:
    """S_n = sum_{m < n} marks_m * exp(-beta (t_n - t_m)) for sorted t, vectorised in blocks."""
    n = len(t)
    out = np.zeros(n)
    if n == 0:
        return out
    span = max(block / max(beta, 1e-12), 1e-9)  # keep beta*(t - t_ref) < ~block for exp safety
    carry, t_prev = 0.0, t[0]
    i = 0
    while i < n:
        j = int(np.searchsorted(t, t[i] + span, side="right"))
        j = max(j, i + 1)
        tb = t[i:j]
        tr = tb[0]
        e = np.exp(beta * (tb - tr))
        cs = np.concatenate([[0.0], np.cumsum(marks[i:j] * e)])[:-1]  # strictly before (index-wise)
        out[i:j] = cs / e + carry * np.exp(-beta * (tb - t_prev))
        # carry to the end of block = value just after the last event
        last = j - 1
        carry = out[last] + marks[last]
        t_prev = t[last]
        i = j
    return out


def hawkes_loglik_1d(params: np.ndarray, t: np.ndarray, T: float) -> float:
    """Log-likelihood of a univariate exp-kernel Hawkes on [0, T]; params = (mu, alpha, beta)."""
    mu, a, b = params
    S = _decay_sum(t, np.ones(len(t)), b)
    lam = mu + a * b * S
    return float(np.sum(np.log(lam)) - mu * T - a * np.sum(1 - np.exp(-b * (T - t))))


def fit_hawkes_1d(times_s: np.ndarray, T: float | None = None, x0: tuple | None = None) -> dict:
    """MLE of a univariate Hawkes with kernel alpha*beta*exp(-beta t) on event times in SECONDS.

    alpha = branching ratio (expected events triggered per event; < 1 for stationarity),
    beta = decay rate (1/s), mu = baseline rate (1/s). Ties (identical timestamps) should be
    jittered or merged first. Returns dict(mu, alpha, beta, se (from the inverse Hessian),
    loglik, n, T).
    """
    t = np.sort(np.asarray(times_s, float))
    t = t - t[0] if T is None else t
    T = float(T if T is not None else t[-1])
    n = len(t)
    x0 = x0 or (0.5 * n / T, 0.5, 10.0 * n / T)
    def nll(z):
        mu, b = np.exp(z[0]), np.exp(z[2])
        a = 1 / (1 + np.exp(-z[1]))
        return -hawkes_loglik_1d(np.array([mu, a, b]), t, T)
    z0 = np.array([np.log(x0[0]), np.log(x0[1] / (1 - x0[1])), np.log(x0[2])])
    r = optimize.minimize(nll, z0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    r = optimize.minimize(nll, r.x, method="L-BFGS-B")
    mu, a, b = np.exp(r.x[0]), 1 / (1 + np.exp(-r.x[1])), np.exp(r.x[2])
    est = np.array([mu, a, b])
    se = _numerical_se(lambda p: -hawkes_loglik_1d(p, t, T), est)
    return {"mu": float(mu), "alpha": float(a), "beta": float(b), "se": se, "loglik": float(-r.fun), "n": n, "T": T}


def _numerical_se(f, x: np.ndarray) -> np.ndarray:
    k = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1e-6)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def hawkes_rescaled_1d(times_s: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Time-rescaling residuals tau_k = Lambda(t_k) - Lambda(t_{k-1}); ~ iid Exp(1) if the model fits.

    Lambda(t) = mu t + alpha * (N(t-) - S(t)), S(t) = sum_{t_i < t} exp(-beta (t - t_i)).
    """
    t = np.sort(np.asarray(times_s, float))
    S = _decay_sum(t, np.ones(len(t)), beta)
    Lam = mu * t + alpha * (np.arange(len(t)) - S)
    return np.diff(Lam)


def _hawkes2_parts(times: list[np.ndarray], betas: np.ndarray):
    allt = np.concatenate(times)
    typ = np.concatenate([np.full(len(x), k) for k, x in enumerate(times)])
    o = np.argsort(allt, kind="stable")
    allt, typ = allt[o], typ[o]
    S = np.zeros((2, 2, len(allt)))  # S[i, j, n] with decay beta_i from source j
    for i in range(2):
        for j in range(2):
            S[i, j] = _decay_sum(allt, (typ == j).astype(float), betas[i])
    return allt, typ, S


def hawkes_loglik_2d(mu: np.ndarray, A: np.ndarray, betas: np.ndarray, times: list[np.ndarray], T: float,
                     parts=None) -> float:
    """Bivariate exp-kernel Hawkes log-likelihood. lambda_i(t) = mu_i + sum_j A[i,j] beta_i S_ij(t)."""
    allt, typ, S = parts if parts is not None else _hawkes2_parts(times, betas)
    ll = 0.0
    for i in range(2):
        m = typ == i
        lam = mu[i] + betas[i] * (A[i, 0] * S[i, 0, m] + A[i, 1] * S[i, 1, m])
        comp = mu[i] * T + sum(A[i, j] * np.sum(1 - np.exp(-betas[i] * (T - times[j]))) for j in range(2))
        ll += np.sum(np.log(lam)) - comp
    return float(ll)


def fit_hawkes_2d(times_s: list[np.ndarray], T: float) -> dict:
    """MLE of a bivariate Hawkes (e.g. [our fills, lead-venue trades]) on [0, T] seconds.

    Kernel of source j on target i: A[i, j] * beta_i * exp(-beta_i t) (one decay per target).
    A[i, j] = expected type-i events triggered by one type-j event. Spectral radius of A < 1 for
    stationarity. A[0, 1] > 0 with A[1, 0] ~ 0 means type 1 leads type 0.
    Returns dict(mu (2,), A (2,2), beta (2,), branching_spectral_radius, loglik).
    """
    times = [np.sort(np.asarray(x, float)) for x in times_s]
    rate = np.array([len(x) / T for x in times])

    def unpack(z):
        return np.exp(z[:2]), np.exp(z[2:6]).reshape(2, 2), np.exp(z[6:8])

    def nll(z):
        mu, A, b = unpack(z)
        return -hawkes_loglik_2d(mu, A, b, times, T, _hawkes2_parts(times, b))

    z0 = np.concatenate([np.log(0.5 * rate), np.log(np.full(4, 0.15)), np.log(10 * rate.mean() * np.ones(2))])
    r = optimize.minimize(nll, z0, method="L-BFGS-B")
    mu, A, b = unpack(r.x)
    return {"mu": mu, "A": A, "beta": b, "branching_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
            "loglik": float(-r.fun), "n": [len(x) for x in times], "T": T}


def hawkes_rescaled_2d(times_s: list[np.ndarray], fit: dict) -> list[np.ndarray]:
    """Time-rescaling residuals per dimension for a fitted bivariate Hawkes (~ Exp(1) if it fits)."""
    times = [np.sort(np.asarray(x, float)) for x in times_s]
    allt, typ, S = _hawkes2_parts(times, fit["beta"])
    out = []
    for i in range(2):
        m = typ == i
        cnt = np.stack([np.cumsum(typ == j) - (typ == j) for j in range(2)])  # events of j strictly before
        Lam = fit["mu"][i] * allt + sum(fit["A"][i, j] * (cnt[j] - S[i, j]) for j in range(2))
        out.append(np.diff(Lam[m]))
    return out


# =============================================================================================
# Response function, propagator, Kyle lambda, Roll, Huang-Stoll, Hasbrouck VAR, sqrt impact
# =============================================================================================
def trade_event_series(trades: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Trades of ONE instrument in time order with sign eps and the pre-trade reference m_n
    (strictly before the trade). Columns ts, eps, qty, price, m."""
    t = trades.sort_values("ts", kind="stable").reset_index(drop=True)
    m = asof_values(t, ref, ["value"], direction="backward", allow_exact=False)["value"].to_numpy()
    return pd.DataFrame({"ts": t["ts"].to_numpy(), "eps": t["aggressor_side"].to_numpy(dtype=float),
                         "qty": t["qty"].to_numpy(dtype=float), "price": t["price"].to_numpy(dtype=float), "m": m}).dropna()


def response_function(ev: pd.DataFrame, max_lag: int = 50) -> pd.DataFrame:
    """R(l) = E[eps_n (m_{n+l} - m_n) / m_n] * 1e4 (bps) and sign autocorrelation C(l) = E[eps_n eps_{n+l}].

    Event time = trade count. Returns DataFrame index lag 0..max_lag [R_bps, C].
    """
    e = ev["eps"].to_numpy()
    m = ev["m"].to_numpy()
    R, C = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
    for l in range(max_lag + 1):
        R[l] = np.mean(e[: len(e) - l] * (m[l:] - m[: len(m) - l]) / m[: len(m) - l]) * 1e4 if l else 0.0
        C[l] = np.mean(e[: len(e) - l] * e[l:])
    return pd.DataFrame({"R_bps": R, "C": C}, index=pd.Index(range(max_lag + 1), name="lag"))


def propagator(resp: pd.DataFrame, L: int | None = None) -> pd.DataFrame:
    """Solve for the propagator G from R and C (transient impact model m_t = sum_{t'<t} G(t-t') eps_t'):

        R(l) = sum_{j=1..L} G(j) [C(|l - j|) - C(j)],   l = 1..L

    Returns DataFrame index lag 1..L [G_bps]. G decaying while R rises = impact is transient and
    sign autocorrelation sustains the price move.
    """
    L = L or len(resp) - 1
    C = resp["C"].to_numpy()
    C = C / C[0]
    R = resp["R_bps"].to_numpy()[1: L + 1]
    M = np.array([[C[abs(l - j)] - C[j] for j in range(1, L + 1)] for l in range(1, L + 1)])
    G = np.linalg.lstsq(M, R, rcond=None)[0]
    return pd.DataFrame({"G_bps": G}, index=pd.Index(range(1, L + 1), name="lag"))


def kyle_lambda(trades: pd.DataFrame, ref: pd.DataFrame, interval_s: float = 60.0, nw_lags: int = 5,
                use_notional: bool = False) -> dict:
    """Kyle's lambda: regress the reference return over each interval (bps) on signed volume in it.

        dm_bps_k = c + lambda * sum_{trades in k} eps * qty  (+ e),  NW SE.

    Units: bps per unit of signed qty (or per unit of notional if ``use_notional``). One instrument.
    """
    t = trades.sort_values("ts", kind="stable")
    step = int(interval_s * NS)
    k = t["ts"].to_numpy(dtype="int64") // step
    sv = t["aggressor_side"].to_numpy(dtype=float) * t["qty"].to_numpy(dtype=float)
    if use_notional:
        sv = sv * t["price"].to_numpy(dtype=float)
    agg = pd.Series(sv).groupby(k).sum()
    edges = np.arange(k.min(), k.max() + 2, dtype="int64") * step
    q = pd.DataFrame({"ts": edges, "venue": t["venue"].iat[0], "symbol": t["symbol"].iat[0]})
    m = asof_values(q, ref, ["value"], direction="backward")["value"].to_numpy()
    dm = 1e4 * np.diff(m) / m[:-1]
    x = agg.reindex(np.arange(k.min(), k.max() + 1), fill_value=0.0).to_numpy()
    from .stats import ols_nw

    r = ols_nw(dm, x, nw_lags)
    return {"lambda": float(r.params[1]), "se": float(r.bse[1]), "t": float(r.tvalues[1]), "r2": float(r.rsquared),
            "n": int(r.nobs), "data": pd.DataFrame({"signed_volume": x, "dm_bps": dm})}


def roll_spread(prices: np.ndarray) -> dict:
    """Roll (1984): S = 2 sqrt(-cov(dp_t, dp_{t-1})) from trade prices. NaN when the covariance is
    positive (model assumption violated). Returns dict(spread (price units), spread_bps, cov)."""
    p = np.asarray(prices, float)
    dp = np.diff(p)
    cv = np.cov(dp[1:], dp[:-1])[0, 1]
    s = 2 * np.sqrt(-cv) if cv < 0 else np.nan
    return {"spread": float(s), "spread_bps": float(1e4 * s / np.mean(p)) if np.isfinite(s) else np.nan, "cov": float(cv)}


def huang_stoll(prices: np.ndarray, signs: np.ndarray) -> dict:
    """Huang-Stoll (1997) basic model by NLS:  dP_t = (S/2) dQ_t + lam (S/2) Q_{t-1} + e_t.

    S = traded spread (price units), lam = share of the half-spread due to adverse selection +
    inventory (1 - lam = order processing). With moments E[e dQ] = E[e Q_{t-1}] = 0 the GMM
    estimator is exactly identified and equals this NLS. Returns dict(S, lam, se_S, se_lam, n).
    """
    p = np.asarray(prices, float)
    q = np.asarray(signs, float)
    dp = np.diff(p)
    dq = np.diff(q)
    ql = q[:-1]

    def res(th):
        return dp - th[0] / 2 * dq - th[1] * th[0] / 2 * ql
    r = optimize.least_squares(res, x0=[2 * np.std(dp), 0.5])
    J = r.jac
    s2 = r.fun @ r.fun / (len(dp) - 2)
    cov = np.linalg.inv(J.T @ J) * s2
    return {"S": float(r.x[0]), "lam": float(r.x[1]), "se_S": float(np.sqrt(cov[0, 0])), "se_lam": float(np.sqrt(cov[1, 1])),
            "n": len(dp)}


def hasbrouck_var(ev: pd.DataFrame, p: int = 5, horizon: int = 50) -> dict:
    """Hasbrouck (1991) two-variable VAR in trade time with r_t = reference return (bps) between trades,
    x_t = trade sign:

        r_t = sum_{i=1..p} a_i r_{t-i} + sum_{i=0..p} b_i x_{t-i} + u_r
        x_t = sum_{i=1..p} c_i r_{t-i} + sum_{i=1..p} d_i x_{t-i} + u_x

    Cumulative impulse response of the price to a unit trade shock u_x (bps) over ``horizon`` trades;
    its limit is the permanent impact. ``trade_info_share`` = share of permanent price variance due
    to trades. Returns dict(irf_cum, permanent_bps, trade_info_share, coefs).
    """
    m = ev["m"].to_numpy()
    r = np.concatenate([[0.0], 1e4 * np.diff(m) / m[:-1]])
    r = np.concatenate([r[1:], [0.0]])  # r_t = move from before trade t to before trade t+1
    x = ev["eps"].to_numpy()
    n = len(x)
    rows = np.arange(p, n - 1)
    Xr = np.column_stack([r[rows - i] for i in range(1, p + 1)] + [x[rows - i] for i in range(0, p + 1)])
    Xx = np.column_stack([r[rows - i] for i in range(1, p + 1)] + [x[rows - i] for i in range(1, p + 1)])
    br, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(rows)), Xr]), r[rows], rcond=None)
    bx, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(rows)), Xx]), x[rows], rcond=None)
    ur = r[rows] - np.column_stack([np.ones(len(rows)), Xr]) @ br
    ux = x[rows] - np.column_stack([np.ones(len(rows)), Xx]) @ bx
    a, b = br[1: p + 1], br[p + 1:]
    c, d = bx[1: p + 1], bx[p + 1:]

    def irf(shock_x: float, shock_r: float) -> np.ndarray:
        R = np.zeros(horizon + p)
        X = np.zeros(horizon + p)
        for t in range(p, horizon + p):
            X[t] = (shock_x if t == p else 0.0) + sum(c[i - 1] * R[t - i] + d[i - 1] * X[t - i] for i in range(1, p + 1))
            R[t] = (shock_r if t == p else 0.0) + sum(a[i - 1] * R[t - i] for i in range(1, p + 1)) + \
                sum(b[i] * X[t - i] for i in range(0, p + 1))
        return np.cumsum(R[p:])
    ix = irf(1.0, 0.0)
    ir = irf(0.0, 1.0)
    vx, vr = np.var(ux), np.var(ur)
    share = ix[-1] ** 2 * vx / (ix[-1] ** 2 * vx + ir[-1] ** 2 * vr)
    return {"irf_cum": ix, "permanent_bps": float(ix[-1]), "trade_info_share": float(share),
            "coefs": {"a": a, "b": b, "c": c, "d": d}}


def metaorders(fills: pd.DataFrame, ref: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Build metaorders from our fills grouped by ``group_cols`` (e.g. ['decision_id', 'leg', 'leg_role']
    or ['order_id']). I_bps = 1e4 * d * (m_after_last - m_before_first) / m_before_first: the move
    in OUR trading direction (positive = impact against us). Q = total qty, duration_s."""
    f = fills.copy()
    f["m_before"] = asof_values(f, ref, ["value"], direction="backward", allow_exact=False)["value"].to_numpy()
    f["m_after"] = asof_values(f, ref, ["value"], direction="forward", allow_exact=False)["value"].to_numpy()
    g = f.sort_values("ts").groupby(group_cols)
    out = g.agg(venue=("venue", "first"), symbol=("symbol", "first"), side=("side", "first"), Q=("qty", "sum"),
                t_start=("ts", "min"), t_end=("ts", "max"), m0=("m_before", "first"), m1=("m_after", "last"))
    out["I_bps"] = 1e4 * out["side"] * (out["m1"] - out["m0"]) / out["m0"]
    out["duration_s"] = (out["t_end"] - out["t_start"]) / NS
    return out.reset_index()


def sqrt_impact_fit(meta: pd.DataFrame, daily_volume: float | np.ndarray, daily_vol_bps: float | np.ndarray,
                    n_bins: int = 8) -> dict:
    """Fit I = Y * sigma * (Q / V)^delta by log-log regression on binned averages.

    meta: metaorders() output (I_bps, Q). V = daily volume (same units as Q), sigma = daily vol (bps).
    Single metaorder impacts are dominated by noise and often negative, so we bin log(Q/V),
    average I/sigma per bin, and regress log(mean I/sigma) on log(mean Q/V) over bins with positive
    mean. delta ~ 0.5 is the square-root law. Returns dict(Y, delta, se_delta, table).
    """
    m = meta.copy()
    m["x"] = m["Q"] / np.asarray(daily_volume, float)
    m["y"] = m["I_bps"] / np.asarray(daily_vol_bps, float)
    m = m[(m["x"] > 0) & np.isfinite(m["y"])]
    m["bin"] = pd.qcut(np.log(m["x"]), n_bins, duplicates="drop")
    t = m.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"), y_se=("y", "sem"))
    tt = t[t["y"] > 0]
    if len(tt) < 2:
        return {"Y": np.nan, "delta": np.nan, "se_delta": np.nan, "table": t.reset_index()}
    sm = _sm()
    r = sm.WLS(np.log(tt["y"]), sm.add_constant(np.log(tt["x"])), weights=tt["n"]).fit()
    return {"Y": float(np.exp(r.params.iloc[0])), "delta": float(r.params.iloc[1]), "se_delta": float(r.bse.iloc[1]),
            "table": t.reset_index()}
