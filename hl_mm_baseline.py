# %% [markdown]
# # Market-making baselines on Hyperliquid HYPE: microprice + Avellaneda–Stoikov reservation quotes
#
# **What this is.** A baseline *analysis*, not a backtest you should trust for PnL. Two textbook methods,
# each measured the same way, so that any idea you have later can be dropped into the same slot and compared
# against them:
#
# 1. **Fair value: the microprice.** Two versions: the *weighted mid* (size-weighted touch) and Stoikov's
#    (2018) *micro-price* (expected mid in the long run, estimated from a Markov chain on imbalance and spread).
#    Scored on how well each predicts the mid **100–500 ms** ahead, against the plain mid.
# 2. **Quoting: Avellaneda–Stoikov (2008) reservation bid/ask.** Quotes are centred on a *reservation price*
#    that is skewed away from the current inventory, with a spread set by volatility, risk aversion and how fast fill
#    probability decays with distance. Requoted every **Δ ∈ {100, 250, 500} ms**. Scored on fills, markouts and
#    mark-to-mid PnL against a "join the touch" quoter.
#
# **Data.** `market_data_v2` (VM collector), HL `l1` = `bbo` channel (event-driven, has sizes) and HL `trades`
# (with aggressor side). The old `data/HYPERLIQUID` book is ~5.4 s snapshots. That's too coarse for 100–500 ms quoting, so it's unused.
#
# **Status:** exploratory and *unregistered*. There's no hypothesis card: this establishes baselines and
# makes no prediction. `qr-harness_v2` has no `qrlib`, so this notebook carries its own truncation test (§6),
# fill bounds (§7) and trial counter (§9).
#
# **Weak points up front**
# * Very little data: the covered segments printed in §2 (a weekend, with holes), and one 60/40 chronological split.
# * Fills are *simulated* from public trades. There's no queue position, our orders don't affect the book, and partial fills are ignored.
#   The truth is somewhere between the `through` (pessimistic) and `touch` (optimistic) bounds.
# * The HYPE spread is 1 tick (~0.13 bps) 78% of the time, so at the touch almost everything hinges on queue position, which we can't see.
# * Latency: by your convention the main run is in **exchange time + N(0.50, 0.05) ms**. A second run uses
#   **measured `ts_recv`** (HL data arrives ~280 ms after the block). Order-entry latency is unknown and not modelled
#   beyond these (see KNOBS).

# %% [markdown]
# ## Summary (run of 2026-10-20; sample frozen at 2026-10-19 06:00 UTC; 14.8 covered hours; n_trials = 54)
# **Bottom line:** on HYPE, no quoter here makes money under either fill bound, even before fees. The micro-price
# helps a little: it beats the mid at 250–500 ms, and centring quotes on it reduces adverse selection.
#
# **Data note.** The local `market_data_v2` copy was still syncing during the build. It has holes and grew between runs,
# so the notebook works only inside covered segments and freezes the sample end (`DATA_END_UTC`). Train (8.9 h) is almost
# all Sunday; test (5.9 h) is Monday's Asia session, which is about 3× busier. The old `data/HYPERLIQUID` book (~5.4 s
# snapshots) is too coarse for 100–500 ms quoting, so it's unused.
#
# **Fair value (exchange view, test): MSE skill vs the plain mid**
#
# | | 100 ms | 250 ms | 500 ms |
# |---|---|---|---|
# | Weighted mid | −9.0% | +3.5% | +3.9% |
# | Micro-price (textbook) | +1.0% | +1.1% | +1.0% (far too small) |
# | Micro-price (extended) | −10.5% | +2.1% | **+4.0% [3.6, 4.4]** |
#
# At 100 ms nothing beats the mid by more than 1%. The textbook micro-price only allows ±1-tick moves per step, which
# throws away the big moves the imbalance predicts, so it comes out 4–7× too small. The extended version (±4 ticks) is my
# modification, not the textbook method.
#
# **Quoting**
# * Every combination loses money, with CIs excluding zero. Quotes earn 0.1–0.3 bps of spread per fill, then lose
#   0.2–0.9 bps to the price moving against them within a second.
# * Centring AS quotes on the micro-price instead of the mid helps a little: at Δ = 250 ms the 1 s markout improves from −0.51 to
#   −0.42 bps. That's clear under `touch`; under `through` the intervals overlap.
# * Requoting faster loses less per fill: −0.34 / −0.42 / −0.51 bps at Δ = 100 / 250 / 500 ms.
# * Joining the touch depends entirely on the fill rule: −0.17 bps under `touch`, −0.83 under `through`. It's a bet on queue position.
# * The `measured ts_recv` results aren't credible yet. They assume zero order-entry latency on a book that's ~290 ms stale.
#
# The collector has also started writing `blocks` and `l2_fast` tables. `l2_fast` could support a queue-position fill model, and
# `blocks` could give real block timings for the block-timing notebook.
#
# Details are in §10 Findings; the full assumption list and open questions are at the end.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 4)

# dataviz reference palette, fixed order: blue, orange, aqua = the three quoting strategies / fair values
C1, C2, C3, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
# orange = touch quoter / weighted mid, blue = AS(mid), aqua = AS(micro-price) / extended micro-price, grey = textbook micro-price
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "lines.linewidth": 2, "font.size": 9,
})

# %% [markdown]
# ## 1. KNOBS
# Every tunable choice lives here. `# ASSUMPTION:` marks choices you didn't specify.

# %%
DATA_ROOT = Path("../market_data_v2/normalized")
SYMBOL = "HYPE-USD-PERP"
DATA_END_UTC = "2026-10-19 06:00"   # ASSUMPTION: sample frozen here so the findings match while the local sync is still adding files; None = use all
TRAIN_FRAC = 0.6            # ASSUMPTION: chronological 60/40 split of *covered* time; every parameter is fit on train only.
# Coverage: the local copy can have holes (missing hours, outages, a sync still in progress). We only work inside
# segments where both feeds are live.
MAX_L1_GAP_MS = 30_000      # ASSUMPTION: no HL bbo update for this symbol for 30 s = feed hole (normal median gap ~130 ms)
MAX_TR_GAP_MS = 120_000     # ASSUMPTION: no HL trade for 2 min = trade feed hole (normal rate ~1.6/s)
MIN_SEG_MS = 10 * 60_000    # ASSUMPTION: ignore covered segments shorter than 10 min

# --- latency scenarios (clock in which "known" and "live" are measured) ---
# exchange: an event at exchange time t is known at t + L_feed; a quote decided at tau is live from tau + L_order.
# recv:     everything on the VM clock; the book is known when it arrives (ts_recv); fills are judged on trade arrival.
SCENARIOS = {
    "exchange + N(0.50, 0.05) ms": dict(clock="exch", feed=(0.50, 0.05), order=(0.50, 0.05)),  # Mark's figure; ASSUMPTION: same for order entry
    "measured ts_recv": dict(clock="recv", order_delay_ms=0.0),
    # ASSUMPTION: in the recv scenario a quote can be hit by trades *received* after the decision, i.e. order-entry
    # latency = 0 on top of the feed delay. That's optimistic; raise order_delay_ms to test it.
}

# --- fair value (microprice) ---
FV_HORIZONS_MS = [100, 250, 500]   # the MFT window you asked for
FV_GRID_MS = 100                   # evaluate fair values on a 100 ms decision grid
MP_DT_MS = 100                     # ASSUMPTION: Stoikov chain sampled every 100 ms (one step = the shortest horizon)
N_IMB = 10                         # ASSUMPTION: imbalance in 10 equal-width buckets on [0,1] (equal width keeps the symmetrisation exact)
# Two versions of Stoikov's estimator:
#   textbook: as in the paper's code; mid moves of at most 1 tick per step, spreads of 1-2 ticks.
#   extended: HYPE moves ~1.5 ticks per 100 ms, so the textbook version throws away the big moves, which are the ones the
#             imbalance predicts. ASSUMPTION: moves up to 4 ticks per step and spreads up to 4 ticks (98.9% of train
#             transitions). Moves that land in a wider spread count as a final move (no continuation).
MP_VARIANTS = {"micro-price (textbook)": dict(n_spread=2, max_move=1),
               "micro-price (extended)": dict(n_spread=4, max_move=4)}
MP_FOR_QUOTES = "micro-price (extended)"   # ASSUMPTION: the variant that AS(micro-price) centres on

# --- quoting (Avellaneda–Stoikov) ---
QUOTE_DTS_MS = [100, 250, 500]     # requote interval Δ
QUOTE_SZ = 1.0                     # ASSUMPTION: 1 HYPE per quote (~$79; median touch size is ~44 HYPE)
Q_MAX = 5                          # ASSUMPTION: inventory limit in lots; at the limit the side that adds inventory is pulled
TAU_MS = 1000.0                    # ASSUMPTION: AS horizon (T - t) held fixed at 1 s (rolling, infinite-horizon use of the formula)
SKEW_AT_QMAX_TICKS = 2.0           # ASSUMPTION: gamma chosen so that at |q| = Q_MAX the reservation price moves 2 ticks
K_FIT_DELTAS = np.arange(0.5, 5.0, 1.0)   # distances from mid (ticks) used to fit fill intensity lambda(d) = A exp(-k d)
MAKER_FEE_BPS = 0.0                # ASSUMPTION: fees/rebates ignored; set your own tier here
MARKOUT_MS = [100, 250, 500, 1000, 5000]

# --- uncertainty ---
BOOT_REPS = 500
FV_BOOT_BLOCK = 600                # 600 x 100 ms = 1 min blocks for fair-value losses
FILL_BOOT_BLOCK = 50               # consecutive fills per block for markouts
PNL_BAR_MS = 60_000                # PnL increments per minute ...
PNL_BOOT_BLOCK = 10                # ... bootstrapped in 10-minute blocks
SEED = 0

TRIALS = []
def log_trial(**kw):
    TRIALS.append(kw)

# %% [markdown]
# ## 2. Data
# * HL trades and l1 replay a snapshot on every websocket (re)subscribe. Rows stamped before their connection opened are dropped.
# * Prices are converted to **integer ticks** so that "at the bid" vs "through the bid" comparisons are exact.
# * **Coverage.** The work happens only inside *segments* where both the bbo and the trade feed are live (no hole longer than
#   `MAX_L1_GAP_MS` / `MAX_TR_GAP_MS`). Inventory is flattened at mid at the end of each segment. The train/test split is
#   60% of *covered* time, not of wall-clock time.

# %%
def _path_sym(s):
    return s.replace(":", "")    # rclone writes ':' as U+F03A in Windows folder names

def load(table, venue, symbol, cols):
    p = DATA_ROOT / f"table={table}" / f"venue={venue}" / f"symbol={_path_sym(symbol)}"
    return ds.dataset(p, format="parquet", partitioning="hive").to_table(columns=cols).to_pandas()

sess = ds.dataset(DATA_ROOT / "table=sessions", format="parquet", partitioning="hive").to_table().to_pandas()
hl_open = (sess[(sess.venue == "HYPERLIQUID") & (sess.kind == "ws") & (~sess.resumed)]
           .groupby("ordinal").opened_ns.min())

l1 = load("l1", "HYPERLIQUID", SYMBOL, ["ts_exch_ns", "ts_recv_ns", "conn_ordinal", "bid_px", "bid_sz", "ask_px", "ask_sz"])
tr = load("trades", "HYPERLIQUID", SYMBOL, ["ts_exch_ns", "ts_recv_ns", "conn_ordinal", "price", "size", "aggressor"])
for name, df in (("l1", l1), ("trades", tr)):
    bad = df.ts_exch_ns < df.conn_ordinal.map(hl_open)
    if DATA_END_UTC:
        bad |= df.ts_exch_ns >= pd.Timestamp(DATA_END_UTC).value
    print(f"{name}: dropped {bad.sum()} rows (subscribe replays, or after DATA_END_UTC)")
    df.drop(index=df.index[bad], inplace=True)

sp = l1.ask_px - l1.bid_px
TICK = float(np.round(sp[sp > 0].min(), 10))
l1 = l1[(l1.bid_px > 0) & (l1.ask_px > l1.bid_px)].copy()   # drop crossed/empty books
l1["b"] = np.round(l1.bid_px / TICK).astype(np.int64)
l1["a"] = np.round(l1.ask_px / TICK).astype(np.int64)
tr["p"] = np.round(tr.price / TICK).astype(np.int64)
tr["side"] = np.where(tr.aggressor == "buy", 1, -1)          # +1 = buyer lifted the ask, -1 = seller hit the bid
for df in (l1, tr):
    df["t_exch"] = df.ts_exch_ns / 1e6
    df["t_recv"] = df.ts_recv_ns / 1e6
l1 = l1.sort_values(["t_exch", "t_recv"], kind="stable").reset_index(drop=True)
tr = tr.sort_values(["t_exch", "t_recv"], kind="stable").reset_index(drop=True)

def live_intervals(t, max_gap):
    """Maximal intervals where consecutive events are <= max_gap apart."""
    t = np.sort(t)
    br = np.where(np.diff(t) > max_gap)[0]
    return np.c_[np.r_[t[0], t[br + 1]], np.r_[t[br], t[-1]]]

def intersect(A, B):
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return np.array(out)

SEGS = intersect(live_intervals(l1.t_exch.values, MAX_L1_GAP_MS), live_intervals(tr.t_exch.values, MAX_TR_GAP_MS))
SEGS = SEGS[(SEGS[:, 1] - SEGS[:, 0]) >= MIN_SEG_MS]
cov = np.cumsum(SEGS[:, 1] - SEGS[:, 0])
j = int(np.searchsorted(cov, TRAIN_FRAC * cov[-1]))
T_SPLIT = SEGS[j, 1] - (cov[j] - TRAIN_FRAC * cov[-1])          # the time where 60% of covered time is reached
T0, T1 = SEGS[0, 0], SEGS[-1, 1]
PX_REF = float(l1.bid_px.median())
print(f"{SYMBOL}: tick={TICK} (~{TICK / PX_REF * 1e4:.3f} bps), {len(l1):,} l1 rows, {len(tr):,} trades")
print(pd.DataFrame({"from": pd.to_datetime(SEGS[:, 0], unit="ms"), "to": pd.to_datetime(SEGS[:, 1], unit="ms"),
                    "hours": (SEGS[:, 1] - SEGS[:, 0]) / 3.6e6}).round(2).to_string())
print(f"covered {cov[-1] / 3.6e6:.2f} h of {(T1 - T0) / 3.6e6:.2f} h wall-clock; "
      f"train/test split at {pd.to_datetime(T_SPLIT, unit='ms')} (train {TRAIN_FRAC * cov[-1] / 3.6e6:.2f} h, "
      f"test {(1 - TRAIN_FRAC) * cov[-1] / 3.6e6:.2f} h)")
print("spread (ticks) share:", ((l1.a - l1.b).value_counts(normalize=True).sort_index().head(5)).round(3).to_dict())

def grid(t_lo, t_hi, step, pad=0.0, segs=SEGS):
    """Decision times every `step` ms inside covered segments within [t_lo, t_hi), with t + pad still inside the
    segment. Returns (times, segment id). ASSUMPTION: coverage is treated as known (it is bookkeeping, not a signal)."""
    ts, sid = [], []
    for k, (s0, s1) in enumerate(segs):
        lo, hi = max(s0, t_lo), min(s1, t_hi) - pad
        if hi > lo:
            g = np.arange(lo, hi, step)
            ts.append(g); sid.append(np.full(len(g), k))
    return (np.concatenate(ts), np.concatenate(sid)) if ts else (np.array([]), np.array([], int))

# %% [markdown]
# ### 2.1 Clock views
# A *view* holds the book and the trades in one scenario's clock:
# * `known[i]` is when book row i becomes usable. It's a prefix max, so a row never becomes known before an earlier row.
# * `tt` is the time used to decide whether a trade hits a resting quote.
#
# In the exchange view, trades are judged on exchange time. In the recv view, on arrival time.

# %%
class View:
    def __init__(self, name, cfg, l1, tr, seed=SEED):
        rng = np.random.default_rng(seed)
        self.name, self.cfg = name, cfg
        if cfg["clock"] == "exch":
            mu, sd = cfg["feed"]
            lat = np.clip(rng.normal(mu, sd, len(l1)), 0, None)
            known = l1.t_exch.values + lat
            self.tt = tr.t_exch.values
            self.live_lo = cfg["order"][0]          # quote reaches the book L_order after the decision
            self.live_hi = -cfg["feed"][0]           # a fill must be known before the next decision
        else:
            known = l1.t_recv.values
            self.tt = tr.t_recv.values
            self.live_lo, self.live_hi = cfg["order_delay_ms"], 0.0
        o = np.argsort(known, kind="stable")
        self.known = np.maximum.accumulate(known[o])
        self.b, self.a = l1.b.values[o], l1.a.values[o]
        self.bs, self.as_ = l1.bid_sz.values[o], l1.ask_sz.values[o]
        ot = np.argsort(self.tt, kind="stable")
        self.tt, self.tp, self.ts, self.tsz = self.tt[ot], tr.p.values[ot], tr.side.values[ot], tr["size"].values[ot]

    def asof(self, q):
        """Index of the last book row known at time q (-1 if none)."""
        return np.searchsorted(self.known, q, "right") - 1

    def book_at(self, q):
        i = self.asof(q)
        ok = i >= 0
        i = np.clip(i, 0, None)
        out = dict(b=self.b[i], a=self.a[i], bs=self.bs[i], as_=self.as_[i])
        for k in out:
            out[k] = np.where(ok, out[k], np.nan) if out[k].dtype.kind == "f" else out[k]
        return out, ok

    def mid2_at(self, q):
        """2 x mid in ticks (an integer), as known at q."""
        i = np.clip(self.asof(q), 0, None)
        return (self.b[i] + self.a[i]).astype(float)

    def truncated(self, cut):
        """Same view with everything known/traded after `cut` deleted (for the leakage test)."""
        v = object.__new__(View)
        v.__dict__.update(self.__dict__)
        mb, mt = self.known <= cut, self.tt <= cut
        v.known, v.b, v.a, v.bs, v.as_ = self.known[mb], self.b[mb], self.a[mb], self.bs[mb], self.as_[mb]
        v.tt, v.tp, v.ts, v.tsz = self.tt[mt], self.tp[mt], self.ts[mt], self.tsz[mt]
        return v

VIEWS = {nm: View(nm, cfg, l1, tr) for nm, cfg in SCENARIOS.items()}
lag = (l1.t_recv - l1.t_exch)
print(f"HL ts_recv - ts_exch: min {lag.min():.0f} ms, median {lag.median():.0f} ms, p99 {lag.quantile(.99):.0f} ms")

# %% [markdown]
# ## 3. Method 1: the microprice (fair value)
#
# The mid $M=(P^b+P^a)/2$ ignores the queue sizes $Q^b, Q^a$ at the touch. Define the **imbalance**
# $$I=\frac{Q^b}{Q^b+Q^a}\in[0,1].$$
# A big bid queue against a thin ask queue ($I\to1$) suggests the next mid move is up.
#
# **(a) Weighted mid** (the "naive microprice"):
# $$P^{w} = I\,P^a + (1-I)\,P^b = M + \frac{S}{2}(2I-1),\qquad S=P^a-P^b .$$
# It jumps all the way to the ask when $I=1$. Stoikov shows it is *not* a martingale: it is noisy and over-reacts.
#
# **(b) Stoikov micro-price.** Define it as the mid's expected value in the long run, given today's state:
# $$P^{\text{micro}}=\lim_{n\to\infty}\mathbb E[M_{\tau_n}\mid I, S] = M + g(I,S),$$
# where $\tau_n$ are the times the mid changes. Discretise the state $x=(S,\ I\text{-bucket})$ and sample the book every
# `MP_DT_MS`. Between samples, either the mid stays put and the state moves ($x\to y$, matrix $Q$), or the mid moves by
# $k\in K=\{\pm\tfrac12,\pm1\}$ ticks. Then:
# * $R_1$ is the matrix of mid-move probabilities, and $G_1=(I-Q)^{-1}R_1K$ is the expected size of the first mid move;
# * $T$ is the matrix of state transitions *on* a move, and $B=(I-Q)^{-1}T$ is the state after the first move;
# * the adjustment is $g=\sum_{i\ge0}B^iG_1$.
#
# The data are **symmetrised** (each row is added again with $I\to1-I$, $\Delta M\to-\Delta M$), which forces $g$ to be
# antisymmetric so that the sum converges. Everything is estimated on **train**. States never seen in train, and spreads
# wider than the modelled ones, fall back to $g=0$ (the mid).
#
# **Two versions** (`MP_VARIANTS`):
# * *textbook*: as in Stoikov's code, $K=\{\pm\frac12,\pm1\}$ ticks and spreads of 1–2 ticks. Transitions outside that are dropped.
# * *extended*: HYPE's mid moves ~1.5 ticks per 100 ms, and the moves the textbook version drops are disproportionately the ones
#   the imbalance predicts (a thin queue gets eaten and the mid jumps). This version keeps moves up to ±4 ticks and spreads
#   up to 4 ticks. A move into a wider spread counts as a final move (it adds to $G_1$ but not to $B$).
#
# The plot below also shows the **realised** mean mid change 500 ms ahead by state, on train. A fair value that hugs it is
# anticipating the right amount.
#
# **How it's scored.** On a 100 ms grid over **test**: forecast error $e = M_{t+h}-\text{FV}_t$ (ticks) for
# $h\in\{100,250,500\}$ ms. *Skill* is $1-\text{MSE}(\text{FV})/\text{MSE}(\text{mid})$, with a 1-minute block-bootstrap CI.
# *Calibration slope* is the no-intercept OLS of $M_{t+h}-M_t$ on $\text{FV}_t-M_t$: 1 = right size, < 1 = over-reacts.

# %%
def state_index(b, a, bs, as_, n_spread):
    """(spread ticks, imbalance bucket) -> flat state index; -1 if outside the modelled states."""
    s = (a - b).astype(int)
    imb = bs / (bs + as_)
    bk = np.clip((imb * N_IMB).astype(int), 0, N_IMB - 1)
    ok = (s >= 1) & (s <= n_spread) & np.isfinite(imb)
    return np.where(ok, (s - 1) * N_IMB + bk, -1), imb

def fit_one_microprice(view, t_lo, t_hi, n_spread, max_move):
    """Stoikov (2018) micro-price adjustment g[state] in ticks, from book samples with t in [t_lo, t_hi]."""
    g_t, _ = grid(t_lo, t_hi, MP_DT_MS, pad=MP_DT_MS)
    bk0, ok0 = view.book_at(g_t)
    bk1, ok1 = view.book_at(g_t + MP_DT_MS)
    x0, _ = state_index(bk0["b"], bk0["a"], bk0["bs"], bk0["as_"], n_spread)
    x1, _ = state_index(bk1["b"], bk1["a"], bk1["bs"], bk1["as_"], n_spread)
    dm = ((bk1["b"] + bk1["a"]) - (bk0["b"] + bk0["a"])) / 2.0          # mid change in ticks (multiple of 1/2)
    keep = ok0 & ok1 & (x0 >= 0) & (np.abs(dm) <= max_move)
    if max_move <= 1:
        keep &= x1 >= 0                                                 # textbook: both states must be modelled
    x0, x1, dm = x0[keep], x1[keep], dm[keep]
    # symmetrise: mirror the imbalance bucket and flip the move
    mir = lambda x: np.where(x >= 0, (x // N_IMB) * N_IMB + (N_IMB - 1 - x % N_IMB), -1)
    x0, x1, dm = np.r_[x0, mir(x0)], np.r_[x1, mir(x1)], np.r_[dm, -dm]
    n = N_IMB * n_spread
    K = np.array([k / 2 for k in range(-2 * max_move, 2 * max_move + 1) if k != 0])   # possible mid moves (ticks)
    Qc = np.zeros((n, n)); Rc = np.zeros((n, len(K))); Tc = np.zeros((n, n))
    st = (dm == 0) & (x1 >= 0)                 # no mid move, state changes x0 -> x1
    mv = dm != 0                               # mid moved by dm
    np.add.at(Qc, (x0[st], x1[st]), 1)
    np.add.at(Rc, (x0[mv], np.searchsorted(K, dm[mv])), 1)
    mvin = mv & (x1 >= 0)                      # ... and landed in a modelled state, so the chain continues from x1
    np.add.at(Tc, (x0[mvin], x1[mvin]), 1)
    seen = (Qc.sum(1) + Rc.sum(1)) > 0
    row = np.where(seen, Qc.sum(1) + Rc.sum(1), 1.0)[:, None]
    Q, R1, T = Qc / row, Rc / row, Tc / row    # Q + R1 rows sum to 1; T has at most the row mass of R1
    inv = np.linalg.pinv(np.eye(n) - Q)
    G1 = inv @ R1 @ K                          # expected size of the next mid move, from each state
    B = inv @ T                                # state right after the next mid move
    g, term = G1.copy(), G1.copy()
    for _ in range(500):                       # g = G1 + B G1 + B^2 G1 + ...
        term = B @ term
        g += term
        if np.abs(term).max() < 1e-10:
            break
    g[~seen] = 0.0
    return dict(g=g, seen=seen, n_spread=n_spread, n_obs=int(keep.sum()))

def fit_microprice(view, t_lo, t_hi):
    return {nm: fit_one_microprice(view, t_lo, t_hi, **cfg) for nm, cfg in MP_VARIANTS.items()}

FV_NAMES = ["mid", "weighted mid"] + list(MP_VARIANTS)

def fair_values(view, t, mp):
    """Mid, weighted mid and the Stoikov micro-prices at decision times t, all in ticks, from the book known at t."""
    bk, ok = view.book_at(t)
    mid = (bk["b"] + bk["a"]) / 2.0
    _, imb = state_index(bk["b"], bk["a"], bk["bs"], bk["as_"], 1)
    out = {"mid": mid, "weighted mid": mid + (bk["a"] - bk["b"]) / 2.0 * (2 * imb - 1)}
    for nm, fit in mp.items():
        x, _ = state_index(bk["b"], bk["a"], bk["bs"], bk["as_"], fit["n_spread"])
        out[nm] = mid + np.where(x >= 0, fit["g"][np.clip(x, 0, None)], 0.0)   # outside the modelled states -> mid
    return pd.DataFrame({**out, "spread": bk["a"] - bk["b"], "imb": imb, "ok": ok}, index=t)

MP = {nm: fit_microprice(v, T0, T_SPLIT) for nm, v in VIEWS.items()}
for nm, mps in MP.items():
    for var, fit in mps.items():
        print(f"[{nm}] {var}: fit on {fit['n_obs']:,} train transitions; states seen {fit['seen'].sum()}/{len(fit['g'])}")

# %%
v0 = VIEWS["exchange + N(0.50, 0.05) ms"]
centres = (np.arange(N_IMB) + 0.5) / N_IMB
# realised mean mid change h ms ahead, by state, on TRAIN (what the fair values are trying to anticipate)
g_tr, _ = grid(T0, T_SPLIT, 100, pad=500)
bk_tr, _ = v0.book_at(g_tr)
x_tr, _ = state_index(bk_tr["b"], bk_tr["a"], bk_tr["bs"], bk_tr["as_"], 2)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for j in range(2):
    s = j + 1
    ax[j].plot(centres, s / 2 * (2 * centres - 1), color=C2, lw=1.6, ls="--", label="weighted mid − mid")
    for (var, fit), col, mk in zip(MP[v0.name].items(), [C_MUTED, C3], ["s", "o"]):
        ax[j].plot(centres, fit["g"][j * N_IMB:(j + 1) * N_IMB], marker=mk, color=col, ms=4, lw=1.6, label=f"{var} − mid")
    m = (x_tr >= j * N_IMB) & (x_tr < (j + 1) * N_IMB)
    real = pd.Series((v0.mid2_at(g_tr[m] + 500) - (bk_tr["b"][m] + bk_tr["a"][m])) / 2).groupby(x_tr[m] - j * N_IMB).mean()
    ax[j].plot(centres[real.index], real.values, "x", color=C_INK, ms=7, mew=2, label="realised mean ΔM over 500 ms")
    ax[j].axhline(0, color=C_MUTED, lw=0.8)
    ax[j].set(title=f"spread = {s} tick{'s' if s > 1 else ''}", xlabel="imbalance I = Qb / (Qb + Qa)")
ax[0].set_ylabel("adjustment to mid (ticks)")
ax[0].legend(frameon=False, fontsize=7)
fig.suptitle(f"{SYMBOL}: fair-value adjustment vs imbalance (train, exchange clock)", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.1 Fair-value scoring (test)

# %%
def block_boot_ci(x, block, reps=BOOT_REPS, seed=SEED, stat=np.mean):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return tuple(np.percentile([stat(x[i]) for i in idx], [2.5, 97.5]))

def score_fv(view, mp):
    t, _ = grid(T_SPLIT, T1, FV_GRID_MS, pad=max(FV_HORIZONS_MS))
    F = fair_values(view, t, mp)
    F = F[F.ok]
    rows = []
    for h in FV_HORIZONS_MS:
        fut = view.mid2_at(F.index.values + h) / 2.0
        e_mid = fut - F["mid"].values
        for fv in FV_NAMES:
            e = fut - F[fv].values
            d = e_mid ** 2 - e ** 2                                  # > 0 where FV beats mid
            sig = F[fv].values - F["mid"].values
            slope = np.sum(sig * e_mid) / np.sum(sig ** 2) if np.sum(sig ** 2) > 0 else np.nan
            lo, hi = block_boot_ci(d / np.mean(e_mid ** 2), FV_BOOT_BLOCK)
            rows.append({"scenario": view.name, "horizon ms": h, "fair value": fv, "n": len(e),
                         "RMSE ticks": np.sqrt(np.mean(e ** 2)), "skill vs mid": 1 - np.mean(e ** 2) / np.mean(e_mid ** 2),
                         "skill CI lo": lo, "skill CI hi": hi, "calib. slope": slope,
                         "share mid moved": np.mean(e_mid != 0)})
            if fv != "mid":
                log_trial(part="fair value", scenario=view.name, fv=fv, horizon=h)
    return pd.DataFrame(rows)

fv_tab = pd.concat([score_fv(VIEWS[nm], MP[nm]) for nm in VIEWS]).set_index(["scenario", "horizon ms", "fair value"])
fv_tab

# %%
fig, ax = plt.subplots(1, len(VIEWS), figsize=(10, 3.4), sharey=True)
for a_, nm in zip(ax, VIEWS):
    for j, (fv, col) in enumerate(zip(FV_NAMES[1:], [C2, C_MUTED, C3])):
        sub = fv_tab.xs((nm, fv), level=("scenario", "fair value")).sort_index()
        x = np.array(FV_HORIZONS_MS) + (j - 1) * 12
        a_.errorbar(x, sub["skill vs mid"], yerr=[sub["skill vs mid"] - sub["skill CI lo"], sub["skill CI hi"] - sub["skill vs mid"]],
                    fmt="o-", color=col, ms=4, lw=1.4, capsize=0, label=fv)
    a_.axhline(0, color=C_MUTED, lw=0.8)
    a_.set(title=nm, xlabel="horizon h (ms)", xticks=FV_HORIZONS_MS)
ax[0].set_ylabel("MSE skill vs mid (test, 95% CI)")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Does the fair value predict the mid h ms ahead better than the mid itself?", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Method 2: Avellaneda–Stoikov reservation bid/ask
#
# AS model a dealer with CARA risk aversion $\gamma$ and inventory $q$ around a mid that's a Brownian motion with
# volatility $\sigma$. Market orders hit a quote at distance $\delta$ from the mid at rate $\lambda(\delta)=A e^{-k\delta}$.
# The approximate optimal quotes are:
# $$r = s - q\,\gamma\sigma^2(T-t)\qquad\text{(reservation price)}$$
# $$\delta^a+\delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\!\Big(1+\frac{\gamma}{k}\Big)\qquad\text{(total spread)}$$
# $$\text{bid} = r - \tfrac12(\delta^a+\delta^b),\qquad \text{ask} = r + \tfrac12(\delta^a+\delta^b).$$
#
# * **Reservation price $r$**: the price at which the dealer is indifferent to one more unit. Long inventory ($q>0$)
#   pushes $r$ down, so the bid backs off and the ask leans in to sell. That skew is the whole inventory control.
# * **Spread**: the first term is the inventory-risk premium; the second, $\approx 2/k$ for small $\gamma$, is the
#   monopoly mark-up set by how fast fills fall off with distance.
# * **$s$** is the fair value the quotes centre on. This is where Method 1 plugs in: **AS(mid)** vs **AS(micro-price)**.
#
# **Parameters, all estimated on train in ticks and ms:**
# * $\sigma^2$: mean squared 1 s mid change / 1000 (ticks²/ms).
# * $k, A$: for each requote window Δ and each distance $d\in$ `K_FIT_DELTAS` from the mid at the window start, let
#   $p(d)$ = fraction of windows in which an aggressor traded at $\ge$ mid + d (asks) or $\le$ mid − d (bids). Then
#   $\lambda(d)\Delta = -\ln(1-p(d))$, and the OLS fit $\ln\lambda = \ln A - k d$ gives $k$. This is a *touch*-style fill intensity.
# * $\gamma$: no natural value exists in ticks. ASSUMPTION: chosen so that at $|q|=Q_{\max}$ the reservation price is skewed by
#   `SKEW_AT_QMAX_TICKS`: $\gamma = \text{skew}/(\sigma^2\tau Q_{\max})$.
# * $T-t$: held at $\tau$ = `TAU_MS` (rolling use of the finite-horizon formula).
#
# **On the tick grid.** bid = ⌊bid⌋ and ask = ⌈ask⌉ in ticks. Quotes are clamped to stay passive (bid ≤ best ask − 1,
# ask ≥ best bid + 1). At $|q|\ge Q_{\max}$ the side that would add inventory is pulled.
#
# **Baseline 0: join the touch.** bid = best bid, ask = best ask, with the same inventory limit and no skew.

# %%
def window_extremes(view, lo, hi):
    """For consecutive windows (lo_k, hi_k]: min price of seller-initiated trades and max of buyer-initiated ones (ticks)."""
    n = len(lo)
    smin = np.full(n, np.iinfo(np.int64).max); bmax = np.full(n, np.iinfo(np.int64).min)
    k = np.searchsorted(lo, view.tt, "left") - 1               # last window whose lo < t
    v = (k >= 0) & (view.tt <= hi[np.clip(k, 0, None)])
    sel, buy = v & (view.ts == -1), v & (view.ts == 1)
    np.minimum.at(smin, k[sel], view.tp[sel])
    np.maximum.at(bmax, k[buy], view.tp[buy])
    return smin, bmax

def fit_as_params(view, mp, dt, t_lo, t_hi):
    """sigma^2, k, A, gamma on [t_lo, t_hi] for requote interval dt."""
    g1, _ = grid(t_lo, t_hi, 1000.0, pad=1000.0)
    dm = (view.mid2_at(g1 + 1000) - view.mid2_at(g1)) / 2.0
    sig2 = np.mean(dm ** 2) / 1000.0                                   # ticks^2 per ms
    tau, _ = grid(t_lo, t_hi, dt, pad=dt)
    lo, hi = tau + view.live_lo, tau + dt + view.live_hi
    smin, bmax = window_extremes(view, lo, hi)
    mid = view.mid2_at(tau) / 2.0
    lam = []
    for d in K_FIT_DELTAS:
        p = 0.5 * (np.mean(bmax >= mid + d) + np.mean(smin <= mid - d))
        lam.append(-np.log(1 - p) / dt if 0 < p < 1 else np.nan)
    lam = np.array(lam)
    ok = np.isfinite(lam) & (lam > 0)
    slope, icpt = np.polyfit(K_FIT_DELTAS[ok], np.log(lam[ok]), 1)
    k, A = -slope, np.exp(icpt)
    gamma = SKEW_AT_QMAX_TICKS / (sig2 * TAU_MS * Q_MAX)
    half = 0.5 * (gamma * sig2 * TAU_MS + (2 / gamma) * np.log1p(gamma / k))
    return dict(dt=dt, sig2=sig2, k=k, A=A, gamma=gamma, half_spread=half, lam=lam)

AS = {(nm, dt): fit_as_params(VIEWS[nm], MP[nm], dt, T0, T_SPLIT) for nm in VIEWS for dt in QUOTE_DTS_MS}
pd.DataFrame({key: {"sigma (ticks/sqrt s)": np.sqrt(p["sig2"] * 1000), "k (1/tick)": p["k"], "A (1/ms)": p["A"],
                    "gamma": p["gamma"], "AS half-spread (ticks)": p["half_spread"],
                    "skew per lot (ticks)": p["gamma"] * p["sig2"] * TAU_MS}
              for key, p in AS.items()}).T

# %%
def run_quotes(view, mp, prm, strategy, fill, t_lo, t_hi):
    """Simulate one quoter on decision grid [t_lo, t_hi). Returns per-decision frame and fills.
    strategy: 'touch' | 'AS(mid)' | 'AS(micro-price)'.  fill: 'through' (trade strictly beyond our price) | 'touch' (at or beyond)."""
    dt = prm["dt"]
    tau, seg = grid(t_lo, t_hi, dt, pad=dt)
    F = fair_values(view, tau, mp)
    lo, hi = tau + view.live_lo, tau + dt + view.live_hi
    smin, bmax = window_extremes(view, lo, hi)
    bb, ba = view.book_at(tau)[0]["b"], view.book_at(tau)[0]["a"]
    s_fv = F[MP_FOR_QUOTES].values if strategy == "AS(micro-price)" else F["mid"].values
    skew = prm["gamma"] * prm["sig2"] * TAU_MS
    strict = fill == "through"
    q, cash = 0.0, 0.0
    n = len(tau)
    out = np.empty((n, 4))                    # bid, ask, q at decision, cash at decision
    fills = []
    # trades per window, for the first qualifying trade time
    k_tr = np.searchsorted(lo, view.tt, "left") - 1
    in_w = (k_tr >= 0) & (view.tt <= hi[np.clip(k_tr, 0, None)])
    first = np.searchsorted(k_tr[in_w], np.arange(n), "left"); last = np.searchsorted(k_tr[in_w], np.arange(n), "right")
    tt_w, tp_w, ts_w = view.tt[in_w], view.tp[in_w], view.ts[in_w]
    fee = MAKER_FEE_BPS * 1e-4 * PX_REF / TICK       # per lot, in ticks
    ok_arr, mid_arr = F.ok.values, F["mid"].values
    last_mid = np.nan
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1] and q != 0:
            # ASSUMPTION: flatten at the last known mid when the data has a hole (costless; keeps outage jumps out of PnL)
            cash += q * last_mid; q = 0.0
        if ok_arr[i]:
            last_mid = mid_arr[i]
        if not ok_arr[i]:
            out[i] = (np.nan, np.nan, q, cash); continue
        if strategy == "touch":
            b, a = bb[i], ba[i]
        else:
            r = s_fv[i] - q * skew
            b = np.floor(r - prm["half_spread"]); a = np.ceil(r + prm["half_spread"])
            b, a = min(b, ba[i] - 1), max(a, bb[i] + 1)
        b = np.nan if q >= Q_MAX else b
        a = np.nan if q <= -Q_MAX else a
        out[i] = (b, a, q, cash)
        if first[i] == last[i]:
            continue
        sl = slice(first[i], last[i])
        for side, px, hit in ((+1, b, smin[i] < b if strict else smin[i] <= b),
                              (-1, a, bmax[i] > a if strict else bmax[i] >= a)):
            if np.isnan(px) or not hit:
                continue
            tp_s = tp_w[sl]
            if side == 1:                                     # our bid is hit by sellers ...
                m = (ts_w[sl] == -1) & ((tp_s < px) if strict else (tp_s <= px))
            else:                                             # ... our ask is lifted by buyers
                m = (ts_w[sl] == 1) & ((tp_s > px) if strict else (tp_s >= px))
            tf = tt_w[sl][np.argmax(m)]
            q += side * QUOTE_SZ
            cash -= side * px * QUOTE_SZ + fee * QUOTE_SZ
            fills.append((tf, side, px, tau[i]))
    D = pd.DataFrame(out, columns=["bid", "ask", "q", "cash"], index=tau)
    D["mid"] = F["mid"].values
    D["equity"] = D.cash + D.q * D.mid                       # mark-to-mid, ticks x lots
    Fl = pd.DataFrame(fills, columns=["t", "side", "px", "tau"])
    return D, Fl

STRATS = ["touch", "AS(mid)", "AS(micro-price)"]
FILLS = ["through", "touch"]

# %% [markdown]
# ## 5. Fill model (both bounds, always reported)
# A resting bid at price $b$ during its window $(\tau+L_{\text{order}},\ \tau+\Delta-L_{\text{feed}}]$ is filled (whole `QUOTE_SZ`) if a
# **seller-initiated** trade printed
# * **through:** strictly below $b$. The level was swept, so wherever we sat in the queue we'd have been filled. *Pessimistic.*
# * **touch:** at or below $b$. Assumes we were at the front of the queue. *Optimistic.*
#
# The ask side mirrors this. At most one fill per side per window. Our orders don't change the book or anyone's behaviour.
# The fill time (for markouts) is the first qualifying trade's time.

# %% [markdown]
# ## 6. Leakage gate (truncation test)
# Every decision at time τ may use only data known by τ. The test re-runs the whole pipeline on data with
# **everything after a cut deleted**, including re-fitting the microprice and AS parameters, and checks that every decision
# at or before the cut is unchanged. The check covers bid, ask, inventory and cash.
# It also checks that the fair values on the FV grid are unchanged.

# %%
def pipeline(view, strategy, fill, dt):
    mp = fit_microprice(view, T0, T_SPLIT)
    prm = fit_as_params(view, mp, dt, T0, T_SPLIT)
    D, Fl = run_quotes(view, mp, prm, strategy, fill, T_SPLIT, T1)
    return D, mp

def truncation_test(view, strategy="AS(micro-price)", fill="touch", dt=250):
    D_full, mp_full = pipeline(view, strategy, fill, dt)
    for frac in (0.3, 0.7):
        cut = T_SPLIT + frac * (T1 - T_SPLIT)
        D_cut, mp_cut = pipeline(view.truncated(cut), strategy, fill, dt)
        keep = D_full.index.values <= cut
        a = D_full.loc[keep, ["bid", "ask", "q", "cash"]].values
        b = D_cut.reindex(D_full.index[keep])[["bid", "ask", "q", "cash"]].values
        assert np.allclose(a, b, equal_nan=True), f"LEAK in quotes ({view.name}, {strategy}, cut={cut})"
        tg, _ = grid(T_SPLIT, cut, FV_GRID_MS)
        fa = fair_values(view, tg, mp_full)[FV_NAMES].values
        fb = fair_values(view.truncated(cut), tg, mp_cut)[FV_NAMES].values
        assert np.allclose(fa, fb, equal_nan=True), f"LEAK in fair values ({view.name}, cut={cut})"
    return True

for nm, v in VIEWS.items():
    for st in STRATS:
        print(f"truncation test [{nm} | {st}]:", truncation_test(v, st))

# %% [markdown]
# ## 7. Quoting results (test)
# Per (scenario × requote Δ × strategy × fill bound):
# * **fills/h** and **share of decisions quoting each side**;
# * **markout at H** (bps, per fill) = side × (mid at fill + H − fill price) / price. It splits into
#   **spread capture** (mid at fill vs fill price, which we earn) and **adverse selection** (mid move after the fill, usually against us);
# * **PnL/h** (mark-to-mid, USD, no fees) with a block-bootstrap CI on per-minute increments, plus mean |inventory|.
#
# The `through` bound fills far less often. Conclusions that hold only under `touch` are queue-position bets.

# %%
def markouts(view, Fl):
    if len(Fl) == 0:
        return {}
    m0 = view.mid2_at(Fl.t.values) / 2.0
    out = {"capture": Fl.side.values * (m0 - Fl.px.values)}
    for H in MARKOUT_MS:
        out[H] = Fl.side.values * (view.mid2_at(Fl.t.values + H) / 2.0 - Fl.px.values)
    return {k: v * TICK / PX_REF * 1e4 for k, v in out.items()}       # bps

res_rows, RUNS = [], {}
for nm, view in VIEWS.items():
    for dt in QUOTE_DTS_MS:
        prm = AS[(nm, dt)]
        for st in STRATS:
            for fl in FILLS:
                D, Fl = run_quotes(view, MP[nm], prm, st, fl, T_SPLIT, T1)
                RUNS[(nm, dt, st, fl)] = (D, Fl)
                mk = markouts(view, Fl)
                hours = len(D) * dt / 3.6e6                        # covered hours (not wall-clock: the data has holes)
                eq = D.equity.ffill().values * TICK                    # USD
                bars = pd.Series(eq, index=D.index).groupby((D.index - D.index[0]) // PNL_BAR_MS).last().diff().dropna().values
                pnl_ci = np.array(block_boot_ci(bars, PNL_BOOT_BLOCK)) * 3.6e6 / PNL_BAR_MS
                row = {"scenario": nm, "Δ ms": dt, "strategy": st, "fill": fl, "fills": len(Fl), "fills/h": len(Fl) / hours,
                       "quoting bid %": 100 * D.bid.notna().mean(), "quoting ask %": 100 * D.ask.notna().mean(),
                       "mean |q| lots": D.q.abs().mean(), "PnL/h USD": (eq[-1] - eq[0]) / hours,
                       "PnL/h CI lo": pnl_ci[0], "PnL/h CI hi": pnl_ci[1]}
                if mk:
                    row["capture bps"] = mk["capture"].mean()
                    for H in (250, 1000, 5000):
                        row[f"markout {H}ms bps"] = mk[H].mean()
                    lo_, hi_ = block_boot_ci(mk[1000], FILL_BOOT_BLOCK)
                    row["mk1000 CI lo"], row["mk1000 CI hi"] = lo_, hi_
                res_rows.append(row)
                log_trial(part="quoting", scenario=nm, dt=dt, strategy=st, fill=fl)
res = pd.DataFrame(res_rows).set_index(["scenario", "Δ ms", "strategy", "fill"])
res

# %%
nm0 = "exchange + N(0.50, 0.05) ms"
fig, ax = plt.subplots(1, len(QUOTE_DTS_MS), figsize=(12, 3.6), sharey=True)
xlab = ["at fill"] + [f"{H} ms" if H < 1000 else f"{H // 1000} s" for H in MARKOUT_MS]
for a_, dt in zip(ax, QUOTE_DTS_MS):
    for st, col in zip(STRATS, [C2, C1, C3]):
        for fl, ls in (("touch", "-"), ("through", "--")):
            mk = markouts(VIEWS[nm0], RUNS[(nm0, dt, st, fl)][1])
            if not mk:
                continue
            y = [mk["capture"].mean()] + [mk[H].mean() for H in MARKOUT_MS]
            a_.plot(range(len(y)), y, ls, marker="o", ms=4, color=col, lw=1.6, label=f"{st} [{fl}]")
    a_.axhline(0, color=C_MUTED, lw=0.8)
    a_.set_xticks(range(len(xlab)), xlab, rotation=30)
    a_.set(xlabel="time after fill (points are the only horizons measured)", title=f"requote Δ = {dt} ms")
ax[0].set_ylabel("mean markout per fill (bps)")
ax[0].legend(frameon=False, fontsize=7)
fig.suptitle(f"Markout curves ({nm0}); 0 ms = spread captured at the fill, then adverse selection eats it", fontsize=10)
plt.tight_layout(); plt.show()

# %%
dt0 = 250
fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
for st, col in zip(STRATS, [C2, C1, C3]):
    for fl, ls in (("touch", "-"), ("through", "--")):
        D = RUNS[(nm0, dt0, st, fl)][0]
        x = pd.to_datetime(D.index, unit="ms")
        ax[0].plot(x, (D.equity - D.equity.iloc[0]) * TICK, ls, color=col, lw=1.2, label=f"{st} [{fl}]")
        if fl == "touch":
            ax[1].plot(x, D.q, color=col, lw=0.8, label=st)
ax[0].set(ylabel="cum. PnL (USD, mark-to-mid)", title=f"Test period, Δ = {dt0} ms, {nm0}, size {QUOTE_SZ:g} {SYMBOL.split('-')[0]}/quote")
ax[1].set(ylabel="inventory (lots) [touch]", xlabel="UTC")
ax[0].legend(frameon=False, fontsize=7, ncol=3); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. How to test an improvement against these baselines
# Everything downstream takes a fair value `s` in ticks at each decision time. To try an idea of your own:
# 1. Write it as a function of the **book and trades known at τ** only (use `view.book_at`, `view.asof`, or windows ending at τ).
# 2. Add it as a column in `fair_values`. §3.1 then scores it against mid, weighted mid and micro-price at 100–500 ms.
# 3. Add a strategy name in `run_quotes` that centres on it. §7 then gives fills, markouts and PnL against the same baselines, under both fill bounds and both latency scenarios.
# 4. Re-run §6. The truncation test must still pass, or nothing after it counts.
#
# A sharper fair value should show up first as **FV skill** (§3.1), then as **less adverse selection** at 250–1000 ms in the
# markout curves, and only last as PnL. PnL is the noisiest of the three by far. Each variant you add counts as a trial (§9).

# %% [markdown]
# ## 9. Trials

# %%
print(f"n_trials (evaluated variants in this notebook): {len(TRIALS)}")
pd.DataFrame(TRIALS).groupby("part").size()
# %% [markdown]
# ## 10. Findings (run of 2026-10-20; sample frozen at 2026-10-19 06:00 UTC; n_trials = 54)
# *Prose written to match the executed tables above. If they disagree after a re-run, trust the tables.*
#
# **Weak points first**
# * **Data.** 14.8 covered hours in two segments: Sun 10-18 03:14–12:00 and Mon 10-19 00:00–06:00 UTC. Train (8.9 h) is
#   almost all Sunday; test (5.9 h) is Monday's Asia session, which is busier. The mid moves in 14% of 100 ms intervals there, against ~4% on Sunday morning.
#   That's one regime change and one split.
# * **Fills are simulated:** no queue position, no fees, 1 HYPE per quote, and our orders don't move the book. `touch` assumes the front of the queue.
# * **The `measured ts_recv` scenario assumes zero order-entry latency** on a book that is ~290 ms stale. Its quoting numbers aren't
#   credible as they stand. Its *capture at fill* is negative because, on the VM clock, the book update for a fill's block can arrive before
#   the trade print.
# * **The "extended" micro-price is my modification of Stoikov's estimator** (ASSUMPTION in KNOBS), not the textbook method.
#
# **Fair value (test, 100 ms grid, n = 212,526)**
# * At **100 ms** almost nothing beats the mid. The weighted mid (−9.0%) and the extended micro-price (−10.5%) overshoot
#   (calibration slopes 0.40 and 0.25). The textbook micro-price is +1.0% [0.9, 1.1], but it is undersized (slope 1.5).
# * At **250 / 500 ms** the weighted mid reaches +3.5% / +3.9% and the extended micro-price +2.1% / **+4.0% [3.6, 4.4]**. The extended
#   micro-price is close to calibrated at 500 ms (slope 1.14). The textbook micro-price stays at about +1%, because its adjustment is 4–7× too small
#   (slopes 3.8 / 6.7): restricting moves to ±1 tick throws away the moves the imbalance predicts.
# * The recv clock gives the same picture (the weighted-mid CI at 250 ms crosses 0).
#
# **Quoting (exchange clock; mark-to-mid, no fees)**
# * **All 18 combinations lose money**, and every PnL/h CI excludes 0. The size is tiny (−$1.8 to −$5.0/h at 1 HYPE/quote), so read it per fill:
#   spread captured at the fill is −0.06 to 0.11 bps (touch quoter) or 0.12–0.34 bps (AS, which quotes ~2.3–2.5 ticks from mid).
#   Adverse selection by 1 s (0.2–0.9 bps) is larger than the capture in every case.
# * **AS(micro-price) vs AS(mid):** less adverse selection in every Δ and both fill bounds. At Δ = 250 ms the 1 s markout is
#   −0.42 [−0.47, −0.37] vs −0.51 [−0.56, −0.46] bps (touch), and −0.61 [−0.66, −0.56] vs −0.66 [−0.71, −0.60] (through). Under `through`
#   the (unpaired) CIs overlap, so the size of the improvement is only clear under `touch`. It also carries less inventory
#   (mean |q| 1.7 vs 2.3 lots).
# * **Shorter requote → less adverse selection.** AS(micro-price) [touch] 1 s markout: −0.34 / −0.42 / −0.51 bps at Δ = 100 / 250 / 500 ms.
# * **Join-the-touch is the bound-sensitive one.** It is the least adversely selected per fill under `touch` (−0.17 bps at 1 s) and the most under
#   `through` (−0.83). Anything that rests on joining the touch is a queue-position bet. It also sits pinned at ±5 lots (mean |q| 3.3).
# * AS's fitted fill decay is k ≈ 0.45 / tick, so its quotes sit ~2.4 ticks from mid in a market that's 1 tick wide 78% of the time. Most AS fills
#   come from trades that sweep through the touch.
#
# **What this baseline says to beat:** fair-value skill at 250–500 ms of about +4% MSE vs mid, and per-fill 1 s markouts of −0.34 to −0.75 bps
# before fees for AS. None of these quoters is profitable on this sample under either fill bound.
#
# **Open questions for Mark**
# 1. Which fill model do you believe for HL at the touch? The new `l2_fast` table could support a queue-position estimate between the two bounds.
# 2. What order-entry latency should the recv scenario use?
# 3. Quote size, inventory limit and skew (γ). What inventory risk do you actually want to carry?
# 4. Which fee tier should go into `MAKER_FEE_BPS`?
# 5. Keep the extended micro-price as the fair-value baseline, or only the textbook one?
# 6. Once more days sync, split by day (train on some days, test on others) instead of 60/40?

# %% [markdown]
# ## 11. Every assumption made (all marked `# ASSUMPTION:` in KNOBS / code)
# **Split and coverage**
# * The train/test split is 60/40 by *covered* time.
# * A feed hole is 30 s with no bbo update or 2 min with no trade. Segments shorter than 10 min are ignored.
# * Inventory is flattened at mid, for free, at every data hole. Coverage is treated as known: that's bookkeeping, not a signal.
# * The sample is frozen at `DATA_END_UTC` = 2026-10-19 06:00 UTC.
#
# **Micro-price**
# * 10 equal-width imbalance buckets; the Stoikov chain is sampled every 100 ms.
# * The extended variant keeps spreads ≤ 4 ticks and mid moves ≤ ±4 ticks per step. A move into a wider spread is a final move.
# * AS(micro-price) centres on the extended variant.
#
# **Quoting**
# * 1 HYPE per quote, inventory limit 5 lots.
# * AS horizon (T − t) held at 1 s. γ is set so that |q| = 5 skews the reservation price by 2 ticks (0.4 tick per lot).
# * Fill decay λ(d) = A·e^(−kd) is fitted on distances of 0.5–4.5 ticks from mid, using touch-style counting.
#
# **Latency**
# * N(0.50, 0.05) ms is applied to order entry as well as the feed.
# * Zero order-entry delay in the `measured ts_recv` scenario (optimistic).
#
# **Fills and costs**
# * Maker fees are 0.
# * One full fill per side per window. No queue position, no partial fills, and our orders have no effect on the book.
#
# **Open questions for Mark** (repeated from §10)
# 1. Which fill rule do you believe at the HL touch? Should we build a queue-position estimate from `l2_fast`?
# 2. What order-entry latency should the recv scenario use?
# 3. What quote size and inventory risk do you want?
# 4. Which fee tier should go into `MAKER_FEE_BPS`?
# 5. Should the extended or the textbook micro-price be the baseline?
# 6. Once more days sync, should we split by day instead of 60/40?
