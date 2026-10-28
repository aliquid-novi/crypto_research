"""Synthetic market with KNOWN injected effects, used by the tests and by `make demo`.

One coherent session of data across three venues:

* VB:BBB  - leg B. Log price = log(price_b) + c(t), c = common random-walk factor.
* VA:AAA  - leg A and our quoting venue. True log price X_A = log(price_a) + beta*c(t) + s(t),
            s = OU spread (known theta). VA's book shows X_A delayed by ``lead_lag_s``.
* VL:AAA  - lead venue: shows X_A with no delay, so it leads VA by ``lead_lag_s``.

Injected effects (all recorded in the returned ``truth`` dict):

* Adverse selection on our passive fills: a fraction ``p_toxic`` of fill events is followed by a
  move of the common factor AGAINST us of size ``toxic_jump_bps`` with an exponential ramp
  (time constant ``toxic_tau_s``). Expected markout vs VA mid at horizon h (bps):
      E[mo_h] = E[spread captured] - p_toxic * J * (1 - exp(-(h - lag)/tau)) for h > lag.
* Toxic market-trade addresses: their aggressive trades on VA predict the 30 s move.
* Late cancels concentrated on toxic fills.
* Spread round trips with known decision->fill delays, legging gaps, partial and missed legs.
* Funding hourly on VA and VB; latency rising linearly through the session.
* Hygiene defects: duplicate fills, a dropped fill (position break), wrong fees, bad ticks,
  a crossed book, a zero-size quote, an off-tick price, a feed gap and a wrong funding payment.

Also: standalone simulators for Hawkes (uni/bivariate), a 2-state HMM, a Kalman test series and
the Huang-Stoll model, each with known parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

NS = 1_000_000_000


@dataclass
class SynthConfig:
    seed: int = 7
    start: str = "2026-01-05 00:00:00"   # UTC, a Monday
    hours: float = 24.0
    dt_s: float = 1.0                     # book update grid
    venue_a: str = "VA"
    venue_b: str = "VB"
    lead_venue: str = "VL"
    symbol_a: str = "AAA"
    symbol_b: str = "BBB"
    price_a: float = 100.0
    price_b: float = 50.0
    tick_a: float = 0.01
    tick_b: float = 0.005
    vol_bps_per_sqrt_s: float = 1.0       # common-factor vol, bps per sqrt(second)
    beta: float = 1.0                     # log hedge ratio of A on B
    lead_lag_s: float = 2.0
    ou_half_life_s: float = 300.0
    ou_sd_bps: float = 20.0               # stationary sd of the spread s
    # our passive quoting on VA
    fill_rate_per_s: float = 0.12
    p_toxic: float = 0.35
    toxic_jump_bps: float = 6.0
    toxic_tau_s: float = 10.0
    sweep_frac: float = 0.10
    late_cancel_p_toxic: float = 0.30
    late_cancel_p_normal: float = 0.02
    cancelled_orders_per_fill: float = 3.0
    # venue-wide market trades on VA
    mkt_trade_rate: float = 0.5
    n_addresses: int = 300
    n_toxic_addresses: int = 5
    toxic_trade_share: float = 0.08
    toxic_hit_rate: float = 0.95
    toxic_horizon_s: float = 30.0
    # lead-venue trades: Hawkes, kernel alpha*beta*exp(-beta t)
    hawkes_mu: float = 0.2
    hawkes_alpha: float = 0.5
    hawkes_beta: float = 2.0
    # spread book
    entry_z: float = 1.0
    hold_s: float = 300.0
    cooldown_s: float = 60.0
    clip_qty_a: float = 10.0
    p_missed: float = 0.10
    p_a_first: float = 0.6
    legging_mean_s: float = 1.5
    p_hung: float = 0.05
    p_partial_second_leg: float = 0.10
    p_partial_exit: float = 0.03
    # fees (bps of notional, negative = rebate)
    maker_bps: float = -0.5
    taker_bps: float = 3.0
    # funding
    funding_interval_s: float = 3600.0
    funding_rate_mean: float = 2e-5
    funding_rate_sd: float = 1e-5
    # latency (ms), rising linearly through the session
    latency_start_ms: float = 2.0
    latency_end_ms: float = 8.0


def _round_tick(x: np.ndarray, tick: float) -> np.ndarray:
    return np.round(np.round(x / tick) * tick, 10)


def _make_book(logp: np.ndarray, tick: float, rng: np.random.Generator, wide_p: float = 0.15) -> dict[str, np.ndarray]:
    """L1 book around exp(logp): 1-tick spread (2 ticks w.p. wide_p); imbalance tracks where the
    true price sits inside the spread, so the microprice carries information."""
    p = np.exp(logp)
    bid = np.round(np.floor(p / tick) * tick, 10)
    ask = np.round(bid + tick, 10)
    wide = rng.random(len(p)) < wide_p
    bid = np.where(wide, np.round(bid - tick, 10), bid)
    pos = np.clip((p - bid) / (ask - bid), 0, 1)
    imb = np.clip(pos + rng.normal(0, 0.15, len(p)), 0.02, 0.98)
    tot = rng.lognormal(np.log(20), 0.5, len(p))
    return {"bid_px": bid, "ask_px": ask, "bid_sz": np.round(imb * tot, 4), "ask_sz": np.round((1 - imb) * tot, 4)}


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """Univariate Hawkes by Ogata thinning. lambda(t) = mu + sum alpha*beta*exp(-beta(t-t_i)).

    alpha is the branching ratio (expected children per event). Times in seconds on [0, T].
    """
    t, excite, out = 0.0, 0.0, []
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t > T:
            break
        excite *= np.exp(-beta * w)
        if rng.random() * lam_bar <= mu + excite:
            out.append(t)
            excite += alpha * beta
    return np.array(out)


def simulate_hawkes_2d(mu: np.ndarray, alpha: np.ndarray, beta: float, T: float,
                       rng: np.random.Generator) -> list[np.ndarray]:
    """Bivariate Hawkes, kernel alpha[i, j]*beta*exp(-beta t): effect of a type-j event on type i.

    Returns [times_type0, times_type1] in seconds.
    """
    mu = np.asarray(mu, float)
    alpha = np.asarray(alpha, float)
    t, ex = 0.0, np.zeros(2)
    out: list[list[float]] = [[], []]
    while True:
        lam_bar = float(np.sum(mu + ex))
        w = rng.exponential(1.0 / lam_bar)
        t += w
        if t > T:
            break
        ex *= np.exp(-beta * w)
        lam = mu + ex
        u = rng.random() * lam_bar
        if u <= lam.sum():
            k = 0 if u <= lam[0] else 1
            out[k].append(t)
            ex += alpha[:, k] * beta
    return [np.array(o) for o in out]


def simulate_hmm(n: int, means: tuple[float, float] = (1.0, -3.0), sds: tuple[float, float] = (2.0, 3.0),
                 p_stay: tuple[float, float] = (0.98, 0.95), rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Gaussian HMM sample. Returns (observations, states) with state 1 = 'toxic'."""
    rng = rng or np.random.default_rng(0)
    s = np.zeros(n, dtype=int)
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < p_stay[s[i - 1]] else 1 - s[i - 1]
    x = np.where(s == 0, rng.normal(means[0], sds[0], n), rng.normal(means[1], sds[1], n))
    return x, s


def make_kalman_example(n: int = 5000, rng: np.random.Generator | None = None,
                        noise: float = 0.5) -> pd.DataFrame:
    """y_t = alpha + beta_t x_t + e_t with a slowly drifting sinusoidal beta_t. Columns x, y, beta."""
    rng = rng or np.random.default_rng(1)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    beta = 1.0 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, n))
    y = 5.0 + beta * x + rng.normal(0, noise, n)
    return pd.DataFrame({"x": x, "y": y, "beta": beta})


def simulate_huang_stoll(n: int = 20000, half_spread: float = 0.05, lam: float = 0.4,
                         rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Basic Huang-Stoll: dP_t = (S/2) dQ_t + lam (S/2) Q_{t-1} + e_t. Columns price, sign."""
    rng = rng or np.random.default_rng(2)
    q = rng.choice([-1, 1], n)
    dq = np.diff(q, prepend=q[0])
    qlag = np.concatenate([[0], q[:-1]])
    dp = half_spread * dq + lam * half_spread * qlag + rng.normal(0, 0.01, n)
    return pd.DataFrame({"price": 100 + np.cumsum(dp), "sign": q})


def generate_day(cfg: SynthConfig | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Generate all tables in schema form plus a ``truth`` dict of every injected value.

    Returns (tables, truth). Tables: fills, book_l1, market_trades, orders, decisions, funding,
    positions, instruments. Timestamps are int64 ns UTC.
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = pd.Timestamp(cfg.start, tz="UTC").value
    T = cfg.hours * 3600.0
    dt_ns = int(cfg.dt_s * NS)
    n = int(T / cfg.dt_s)
    grid = t0 + np.arange(n, dtype="int64") * dt_ns
    lag_steps = int(round(cfg.lead_lag_s / cfg.dt_s))
    va, vb, vl, sa, sb = cfg.venue_a, cfg.venue_b, cfg.lead_venue, cfg.symbol_a, cfg.symbol_b

    def gidx(ts: np.ndarray) -> np.ndarray:  # backward grid index (book as-of)
        return np.clip((np.asarray(ts) - t0) // dt_ns, 0, n - 1).astype(int)

    def lat_ms(ts: np.ndarray) -> np.ndarray:
        frac = np.clip((np.asarray(ts) - t0) / (T * NS), 0, 1)
        return cfg.latency_start_ms + (cfg.latency_end_ms - cfg.latency_start_ms) * frac

    # ---- our passive fill events (drawn first; toxic ones move the common factor) -------------
    n_ev = rng.poisson(cfg.fill_rate_per_s * T)
    ev_t = np.sort(t0 + (rng.random(n_ev) * (T - 5) * NS).astype("int64"))
    ev_d = rng.choice([-1, 1], n_ev)
    ev_tox = rng.random(n_ev) < cfg.p_toxic

    sig = cfg.vol_bps_per_sqrt_s * 1e-4 * np.sqrt(cfg.dt_s)
    c = np.concatenate([[0.0], np.cumsum(rng.normal(0, sig, n - 1))])
    J = cfg.toxic_jump_bps * 1e-4 / cfg.beta  # jump in factor units -> beta*J on AAA
    jumps = np.zeros(n)
    step = np.zeros(n + 1)
    w = int(10 * cfg.toxic_tau_s / cfg.dt_s)
    for ti, di in zip(ev_t[ev_tox], ev_d[ev_tox]):
        k0 = int((ti - t0) // dt_ns) + 1  # first grid point strictly after the fill
        if k0 >= n:
            continue
        k1 = min(k0 + w, n)
        tt = (grid[k0:k1] - ti) / NS
        jumps[k0:k1] += -di * J * (1 - np.exp(-tt / cfg.toxic_tau_s))
        if k0 + w < n:
            step[k0 + w] += -di * J
    jumps += np.cumsum(step[:n])
    c = c + jumps

    theta = np.log(2) / cfg.ou_half_life_s
    a = np.exp(-theta * cfg.dt_s)
    sd = cfg.ou_sd_bps * 1e-4
    s = np.zeros(n)
    s[0] = rng.normal(0, sd)
    eps = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    for k in range(1, n):
        s[k] = a * s[k - 1] + eps[k]

    xb = np.log(cfg.price_b) + c
    xa = np.log(cfg.price_a) + cfg.beta * c + s
    xa_va = np.concatenate([np.full(lag_steps, xa[0]), xa[: n - lag_steps]])

    books = {}
    for (ven, sym, lp, tick) in [(va, sa, xa_va, cfg.tick_a), (vl, sa, xa, cfg.tick_a), (vb, sb, xb, cfg.tick_b)]:
        b = _make_book(lp, tick, rng)
        books[(ven, sym)] = pd.DataFrame({"ts": grid, "venue": ven, "symbol": sym, **b})
    bva = books[(va, sa)]
    bid_va, ask_va = bva["bid_px"].to_numpy(), bva["ask_px"].to_numpy()
    mid_va = (bid_va + ask_va) / 2

    # ---- addresses -----------------------------------------------------------------------------
    normal_addr = np.array([f"0x{i:04x}n" for i in range(cfg.n_addresses)])
    toxic_addr = np.array([f"0x{i:04x}t" for i in range(cfg.n_toxic_addresses)])
    zipf = 1.0 / np.arange(1, cfg.n_addresses + 1) ** 0.6
    zipf /= zipf.sum()

    # ---- our passive fills ----------------------------------------------------------------------
    k_ev = gidx(ev_t)
    level = rng.choice([0, 1, 2], n_ev, p=[0.8, 0.15, 0.05])  # ticks behind the touch
    ev_px = np.where(ev_d > 0, bid_va[k_ev] - level * cfg.tick_a, ask_va[k_ev] + level * cfg.tick_a)
    ev_px = np.round(ev_px, 10)
    ev_qty = np.round(rng.lognormal(0, 0.5, n_ev), 4)
    ev_cp = np.where(ev_tox & (rng.random(n_ev) < 0.8), rng.choice(toxic_addr, n_ev),
                     rng.choice(normal_addr, n_ev, p=zipf))
    n_child = np.where(rng.random(n_ev) < cfg.sweep_frac, rng.choice([2, 3], n_ev), 1)
    rows = []
    fid = 0
    for i in range(n_ev):
        parts = rng.dirichlet(np.ones(n_child[i])) if n_child[i] > 1 else np.array([1.0])
        for p_ in parts:
            q = round(float(ev_qty[i] * p_), 6)
            rows.append((f"F{fid:07d}", f"QO{i:06d}", ev_t[i], ev_d[i], ev_px[i], q, ev_cp[i], i))
            fid += 1
    qf = pd.DataFrame(rows, columns=["fill_id", "order_id", "ts", "side", "price", "qty", "counterparty", "event"])
    qf["venue"], qf["symbol"], qf["liquidity"] = va, sa, "maker"
    qf["notional"] = qf["price"] * qf["qty"]
    qf["fee"] = qf["notional"] * cfg.maker_bps / 1e4
    qf["trade_id"] = "T" + qf["fill_id"].str[1:]
    qf["toxic"] = ev_tox[qf["event"].to_numpy()]

    # ---- orders for the quoting book ----------------------------------------------------------
    life = ((rng.exponential(15, n_ev) + 0.5) * NS).astype("int64")
    ts_sent = ev_t - life
    late = rng.random(n_ev) < np.where(ev_tox, cfg.late_cancel_p_toxic, cfg.late_cancel_p_normal)
    partial = (rng.random(n_ev) < 0.1) & ~late
    lat = (lat_ms(ts_sent) * 1e6 * (1 + rng.exponential(0.3, n_ev))).astype("int64")
    o_fill = pd.DataFrame({
        "order_id": [f"QO{i:06d}" for i in range(n_ev)], "venue": va, "symbol": sa, "side": ev_d,
        "price": ev_px, "qty": np.where(partial, ev_qty * 2, ev_qty), "ts_sent": ts_sent, "ts_ack": ts_sent + lat,
    })
    cxl = np.where(late, ev_t - rng.integers(200_000, 3_000_000, n_ev),
                   np.where(partial, ev_t + rng.integers(1, 5, n_ev) * NS, -1))
    o_fill["ts_cancel_sent"] = pd.array(np.where(cxl > 0, cxl, 0), dtype="Int64")
    o_fill.loc[cxl <= 0, "ts_cancel_sent"] = pd.NA
    o_fill["ts_done"] = np.where(partial, cxl + lat, ev_t)
    o_fill["final_status"] = np.where(partial, "partial", "filled")
    o_fill["filled_qty"] = ev_qty
    n_cx = int(cfg.cancelled_orders_per_fill * n_ev)
    cx_sent = np.sort(t0 + (rng.random(n_cx) * (T - 60) * NS).astype("int64"))
    cx_d = rng.choice([-1, 1], n_cx)
    kc = gidx(cx_sent)
    cx_lvl = rng.integers(0, 6, n_cx)
    cx_px = np.round(np.where(cx_d > 0, bid_va[kc] - cx_lvl * cfg.tick_a, ask_va[kc] + cx_lvl * cfg.tick_a), 10)
    cx_life = ((rng.exponential(15, n_cx) + 0.5) * NS).astype("int64")
    cx_lat = (lat_ms(cx_sent) * 1e6 * (1 + rng.exponential(0.3, n_cx))).astype("int64")
    o_cx = pd.DataFrame({
        "order_id": [f"QC{i:06d}" for i in range(n_cx)], "venue": va, "symbol": sa, "side": cx_d, "price": cx_px,
        "qty": np.round(rng.lognormal(0, 0.5, n_cx), 4), "ts_sent": cx_sent, "ts_ack": cx_sent + cx_lat,
        "ts_cancel_sent": pd.array(cx_sent + cx_life, dtype="Int64"), "ts_done": cx_sent + cx_life + cx_lat,
        "final_status": np.where(rng.random(n_cx) < 0.01, "rejected", "cancelled"), "filled_qty": 0.0,
    })

    # ---- venue-wide market trades on VA (with addresses) ---------------------------------------
    n_mt = rng.poisson(cfg.mkt_trade_rate * T)
    mt_t = np.sort(t0 + (rng.random(n_mt) * T * NS).astype("int64"))
    is_tox = rng.random(n_mt) < cfg.toxic_trade_share
    k_mt = gidx(mt_t)
    k_fut = np.clip(k_mt + int(cfg.toxic_horizon_s / cfg.dt_s), 0, n - 1)
    fut_sign = np.sign(mid_va[k_fut] - mid_va[k_mt])
    fut_sign[fut_sign == 0] = 1
    hit = rng.random(n_mt) < cfg.toxic_hit_rate
    mt_side = np.where(is_tox, np.where(hit, fut_sign, -fut_sign), rng.choice([-1, 1], n_mt)).astype(int)
    taker = np.where(is_tox, rng.choice(toxic_addr, n_mt), rng.choice(normal_addr, n_mt, p=zipf))
    maker = rng.choice(normal_addr, n_mt)
    mt = pd.DataFrame({
        "ts": mt_t, "venue": va, "symbol": sa,
        "price": np.where(mt_side > 0, ask_va[k_mt], bid_va[k_mt]),
        "qty": np.round(rng.lognormal(0, 0.8, n_mt), 4), "aggressor_side": mt_side,
        "buyer": np.where(mt_side > 0, taker, maker), "seller": np.where(mt_side > 0, maker, taker),
    })
    # our quoting fills appear in the public tape too (taker = counterparty, maker = us)
    ours = pd.DataFrame({
        "ts": qf["ts"], "venue": va, "symbol": sa, "price": qf["price"], "qty": qf["qty"],
        "aggressor_side": -qf["side"],
        "buyer": np.where(qf["side"] > 0, "OURS", qf["counterparty"]),
        "seller": np.where(qf["side"] > 0, qf["counterparty"], "OURS"),
    })
    mt = pd.concat([mt, ours], ignore_index=True)
    # lead venue trades: Hawkes
    ht = simulate_hawkes(cfg.hawkes_mu, cfg.hawkes_alpha, cfg.hawkes_beta, T, rng)
    lt = t0 + (ht * NS).astype("int64")
    bvl = books[(vl, sa)]
    kl = gidx(lt)
    lside = rng.choice([-1, 1], len(lt))
    mt_l = pd.DataFrame({"ts": lt, "venue": vl, "symbol": sa,
                         "price": np.where(lside > 0, bvl["ask_px"].to_numpy()[kl], bvl["bid_px"].to_numpy()[kl]),
                         "qty": np.round(rng.lognormal(0, 0.8, len(lt)), 4), "aggressor_side": lside})
    bvb = books[(vb, sb)]
    n_b = rng.poisson(0.3 * T)
    bt = np.sort(t0 + (rng.random(n_b) * T * NS).astype("int64"))
    kb = gidx(bt)
    bside = rng.choice([-1, 1], n_b)
    mt_b = pd.DataFrame({"ts": bt, "venue": vb, "symbol": sb,
                         "price": np.where(bside > 0, bvb["ask_px"].to_numpy()[kb], bvb["bid_px"].to_numpy()[kb]),
                         "qty": np.round(rng.lognormal(0, 0.8, n_b), 4), "aggressor_side": bside})
    market_trades = pd.concat([mt, mt_l, mt_b], ignore_index=True)
    market_trades["trade_id"] = [f"M{i:08d}" for i in range(len(market_trades))]

    # ---- spread decisions & round trips -------------------------------------------------------
    mid_vb = (bvb["bid_px"].to_numpy() + bvb["ask_px"].to_numpy()) / 2
    z = (np.log(mid_va) - np.log(cfg.price_a) - cfg.beta * (np.log(mid_vb) - np.log(cfg.price_b))) / sd
    dec_rows, sfills, sorders, dtruth = [], [], [], []
    k, nd, oid = 0, 0, 0
    hold = int(cfg.hold_s / cfg.dt_s)
    cool = int(cfg.cooldown_s / cfg.dt_s)
    books_leg = {"A": (va, sa, bva), "B": (vb, sb, bvb)}

    def taker_fill(leg, side, ts, qty, did, role):
        nonlocal oid
        ven, sym, bk = books_leg[leg]
        kk = gidx(np.array([ts]))[0]
        px = float(bk["ask_px"].iat[kk] if side > 0 else bk["bid_px"].iat[kk])
        lat_ = int(lat_ms(np.array([ts]))[0] * 1e6)
        o = f"SO{oid:06d}"
        oid += 1
        sorders.append((o, ven, sym, side, px, qty, ts - lat_, ts - lat_ // 2, pd.NA, ts, "filled", qty))
        sfills.append((o, ts, ven, sym, side, px, qty, did, leg, role))
        return px

    while k < n - hold - 600:
        if abs(z[k]) < cfg.entry_z:
            k += 1
            continue
        did = f"D{nd:05d}"
        nd += 1
        D = -int(np.sign(z[k]))
        td = int(grid[k] + rng.integers(1_000, 900_000_000))
        ma, mb = mid_va[k], mid_vb[k]
        hr = cfg.beta * ma / mb  # units of B per unit A for a log-beta hedge
        q_a = cfg.clip_qty_a
        t_exit = td + int(cfg.hold_s * NS)
        dec_rows.append((did, td, f"{sa}/{sb}", float(z[k]), 0.0, D * q_a, hr, sa, va, sb, vb,
                         f"time_{int(cfg.hold_s)}s", t_exit))
        tr = {"decision_id": did, "missed": False}
        if rng.random() < cfg.p_missed:
            tr["missed"] = True
        else:
            first = "A" if rng.random() < cfg.p_a_first else "B"
            second = "B" if first == "A" else "A"
            delay = int((rng.uniform(0.05, 0.4) + lat_ms(np.array([td]))[0] / 1e3) * NS)
            hung = rng.random() < cfg.p_hung
            gap = int((rng.uniform(20, 60) if hung else rng.exponential(cfg.legging_mean_s)) * NS)
            q = {"A": q_a, "B": round(q_a * hr, 6)}
            sgn = {"A": D, "B": -D}
            exec_q = dict(q)
            if rng.random() < cfg.p_partial_second_leg:
                exec_q[second] = round(q[second] * 0.5, 6)
            t1 = td + delay
            t2 = t1 + gap
            taker_fill(first, sgn[first], t1, exec_q[first], did, "entry")
            taker_fill(second, sgn[second], t2, exec_q[second], did, "entry")
            xq = dict(exec_q)
            if rng.random() < cfg.p_partial_exit:
                xq["B"] = round(exec_q["B"] * 0.5, 6)
            xd = int((rng.uniform(0.05, 0.4)) * NS)
            xg = int(rng.exponential(cfg.legging_mean_s) * NS)
            taker_fill(first, -sgn[first], t_exit + xd, xq[first], did, "exit")
            taker_fill(second, -sgn[second], t_exit + xd + xg, xq[second], did, "exit")
            tr.update(first_leg=first, delay_ns=delay, legging_gap_ns=gap, hung=hung,
                      partial_second=exec_q[second] < q[second], residual=bool(xq["B"] < exec_q["B"]))
        dtruth.append(tr)
        k += hold + cool

    decisions = pd.DataFrame(dec_rows, columns=["decision_id", "ts", "pair", "signal", "spread_fair_value",
                                                "intended_qty", "hedge_ratio", "leg_A_symbol", "leg_A_venue",
                                                "leg_B_symbol", "leg_B_venue", "exit_rule", "ts_exit_decision"])
    sf = pd.DataFrame(sfills, columns=["order_id", "ts", "venue", "symbol", "side", "price", "qty",
                                       "decision_id", "leg", "leg_role"])
    sf["fill_id"] = [f"S{i:07d}" for i in range(len(sf))]
    sf["liquidity"] = "taker"
    sf["notional"] = sf["price"] * sf["qty"]
    sf["fee"] = sf["notional"] * cfg.taker_bps / 1e4
    sf["trade_id"] = "T" + sf["fill_id"]
    so = pd.DataFrame(sorders, columns=["order_id", "venue", "symbol", "side", "price", "qty", "ts_sent", "ts_ack",
                                        "ts_cancel_sent", "ts_done", "final_status", "filled_qty"])
    so["ts_cancel_sent"] = pd.array([pd.NA] * len(so), dtype="Int64")

    fills_true = pd.concat([qf.drop(columns=["event", "toxic"]), sf], ignore_index=True).sort_values("ts", kind="stable")
    fills_true = fills_true.reset_index(drop=True)
    orders = pd.concat([o_fill, o_cx, so], ignore_index=True).sort_values("ts_sent", kind="stable").reset_index(drop=True)
    # latency hops: md_recv -> signal -> sent -> ack
    hop_scale = lat_ms(orders["ts_sent"].to_numpy()) / cfg.latency_start_ms
    compute = (rng.uniform(20e3, 80e3, len(orders)) * hop_scale).astype("int64")
    md = (rng.uniform(50e3, 200e3, len(orders)) * hop_scale).astype("int64")
    orders["ts_signal"] = orders["ts_sent"] - compute
    orders["ts_md_recv"] = orders["ts_signal"] - md

    # ---- funding (true positions) -------------------------------------------------------------
    fund_rows = []
    settle = t0 + (np.arange(1, int(T // cfg.funding_interval_s) + 1) * cfg.funding_interval_s * NS).astype("int64")
    for ven, sym, bk in [(va, sa, bva), (vb, sb, bvb)]:
        f_ = fills_true[(fills_true["venue"] == ven) & (fills_true["symbol"] == sym)]
        cum_t = f_["ts"].to_numpy()
        cum_p = np.cumsum((f_["side"] * f_["qty"]).to_numpy())
        for st in settle:
            j = np.searchsorted(cum_t, st, side="right") - 1
            pos = float(cum_p[j]) if j >= 0 else 0.0
            kk = gidx(np.array([st]))[0]
            mark = float((bk["bid_px"].iat[kk] + bk["ask_px"].iat[kk]) / 2)
            rate = float(rng.normal(cfg.funding_rate_mean, cfg.funding_rate_sd))
            fund_rows.append((st, ven, sym, rate, mark, pos, -pos * mark * rate))
    funding = pd.DataFrame(fund_rows, columns=["ts", "venue", "symbol", "rate", "mark", "position_at_settlement", "payment"])

    # ---- position snapshots (true) ------------------------------------------------------------
    snaps = t0 + np.arange(0, int(T // 600) + 1, dtype="int64") * (600 * NS)
    pos_rows = []
    for ven, sym in [(va, sa), (vb, sb)]:
        f_ = fills_true[(fills_true["venue"] == ven) & (fills_true["symbol"] == sym)]
        cum_t = f_["ts"].to_numpy()
        cum_p = np.cumsum((f_["side"] * f_["qty"]).to_numpy())
        j = np.searchsorted(cum_t, snaps, side="right") - 1
        pos_rows.append(pd.DataFrame({"ts": snaps, "venue": ven, "symbol": sym,
                                      "position": np.where(j >= 0, cum_p[np.maximum(j, 0)], 0.0)}))
    positions = pd.concat(pos_rows, ignore_index=True)

    # ---- hygiene defects ----------------------------------------------------------------------
    truth: dict = {"config": asdict(cfg), "t0": t0, "T_s": T}
    fills = fills_true.copy()
    qmask = fills["fill_id"].str.startswith("F")
    mid_i = int(np.flatnonzero(qmask.to_numpy())[len(np.flatnonzero(qmask.to_numpy())) // 2])
    dropped = fills.loc[mid_i].to_dict()
    fills = fills.drop(index=mid_i)
    truth["dropped_fill"] = dropped
    fee_err = fills[fills["fill_id"].str.startswith("F")].sample(4, random_state=cfg.seed).index
    fills.loc[fee_err, "fee"] = fills.loc[fee_err, "fee"] * 3 + 0.01
    truth["fee_error_fill_ids"] = sorted(fills.loc[fee_err, "fill_id"])
    dups = fills.sample(3, random_state=cfg.seed + 1)
    truth["duplicate_fill_ids"] = sorted(dups["fill_id"])
    fills = pd.concat([fills, dups], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    bad_ix = market_trades[market_trades["venue"] == va].sample(5, random_state=cfg.seed).index
    market_trades.loc[bad_ix, "price"] = market_trades.loc[bad_ix, "price"] * 1.02
    truth["bad_tick_trade_ids"] = sorted(market_trades.loc[bad_ix, "trade_id"])
    market_trades = market_trades.sort_values("ts", kind="stable").reset_index(drop=True)
    fund_err = int(len(funding) // 3)
    funding.loc[fund_err, "payment"] *= 2
    truth["funding_error_row"] = fund_err

    book = pd.concat(list(books.values()), ignore_index=True)
    gap_a = t0 + int(T * 0.4 * NS)
    gap_b = gap_a + 180 * NS
    gap_mask = (book["venue"] == vb) & (book["ts"] >= gap_a) & (book["ts"] < gap_b)
    book = book[~gap_mask].sort_values(["ts", "venue"], kind="stable").reset_index(drop=True)
    truth["feed_gap"] = (vb, sb, gap_a, gap_b)
    bb = book.index[book["venue"] == vb]
    crossed, zero, offtick = int(bb[1000]), int(bb[2000]), int(bb[3000])
    book.loc[crossed, "bid_px"] = book.loc[crossed, "ask_px"] + cfg.tick_b
    book.loc[zero, "ask_sz"] = 0.0
    book.loc[offtick, "bid_px"] = book.loc[offtick, "bid_px"] + cfg.tick_b / 3
    truth["book_defect_rows"] = {"crossed": crossed, "zero_size": zero, "off_tick": offtick}

    instruments = pd.DataFrame({"venue": [va, vl, vb], "symbol": [sa, sa, sb],
                                "tick_size": [cfg.tick_a, cfg.tick_a, cfg.tick_b],
                                "multiplier": [1.0, 1.0, 1.0], "quote_ccy": "USD", "kind": "perp"})

    truth.update(
        theta=theta, ou_sd=sd, lead_lag_s=cfg.lead_lag_s, toxic_addresses=list(toxic_addr),
        toxic_fill_ids=sorted(qf.loc[qf["toxic"], "fill_id"]), p_toxic=cfg.p_toxic,
        toxic_jump_bps=cfg.toxic_jump_bps, toxic_tau_s=cfg.toxic_tau_s,
        hawkes=(cfg.hawkes_mu, cfg.hawkes_alpha, cfg.hawkes_beta), decisions=pd.DataFrame(dtruth),
        late_cancel_order_ids=sorted(o_fill.loc[late, "order_id"]), grid_mid_va=mid_va, grid_ts=grid,
        spread_s=s,
    )
    tables = {"fills": fills, "book_l1": book, "market_trades": market_trades, "orders": orders,
              "decisions": decisions, "funding": funding, "positions": positions, "instruments": instruments}
    return tables, truth


def clean_fills(fills: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """Fills with the injected duplicates removed (keeps the first occurrence)."""
    return fills.drop_duplicates(subset=["fill_id"], keep="first").reset_index(drop=True)
