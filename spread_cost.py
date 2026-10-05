"""
Spread-trade cost geometry for a two-venue perp spread.

Built for HYPE-side Hyperliquid PENDLE-USD vs Binance PENDLE-USDT, but nothing
here is venue-specific: swap the VenueFees objects.

The question it answers: given a maker/taker fill mix, how far does the spread
have to move before a round trip clears costs — and how often does it actually
move that far?

----------------------------------------------------------------------------
THE THING MOST COST MODELS GET WRONG
----------------------------------------------------------------------------
A spread round trip is FOUR fills, not two:

    open  leg A  (Hyperliquid)      close leg A
    open  leg B  (Binance)          close leg B

and a TAKER fill pays the venue's half-spread *on top of* the taker fee. On an
alt perp that half-spread is often larger than the fee. Both are handled below.

----------------------------------------------------------------------------
UNITS
----------------------------------------------------------------------------
Everything is in basis points of the reference mid. Convert your spread first:

    ref      = (df.hl_mid + df.bin_mid) / 2
    dev_bps  = (df.hl_mid - df.bin_mid) / ref * 1e4

Fee defaults below were checked on 2026-10-03 from secondary sources, not the
venues' own schedules. VERIFY THEM against your account before quoting any
number from this to anyone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NoEntries(ValueError):
    """The entry threshold is never reached in this sample."""


# ---------------------------------------------------------------- fee model

@dataclass(frozen=True)
class VenueFees:
    """Per-venue costs, all in bps of notional. Negative fee = rebate."""

    name: str
    maker_bps: float
    taker_bps: float
    half_spread_bps: float = 0.0   # paid only on taker legs
    extra_slip_bps: float = 0.0    # queue jumps, partial fills, latency

    def leg_cost_bps(self, maker_frac: float) -> float:
        """Expected cost of ONE fill on this venue at the given maker fraction."""
        m = float(np.clip(maker_frac, 0.0, 1.0))
        t = 1.0 - m
        return (m * self.maker_bps
                + t * (self.taker_bps + self.half_spread_bps)
                + self.extra_slip_bps)


# Hyperliquid perps, tier 6 ($7B+ 14-day volume): 0.024% taker, 0.000% maker.
# Separate market-maker rebate tiers exist (negative maker) — set maker_bps < 0
# if you qualify.
HYPERLIQUID_T6 = VenueFees(
    name="Hyperliquid PENDLE-USD (tier 6)",
    maker_bps=0.0,
    taker_bps=2.4,
    half_spread_bps=1.8,    # from your sample: (2.44145 - 2.441) / 2.4415 * 1e4
)

# Binance USDⓈ-M futures, VIP 0: 0.02% maker, 0.05% taker (no BNB discount).
BINANCE_VIP0 = VenueFees(
    name="Binance PENDLE-USDT (VIP 0)",
    maker_bps=2.0,
    taker_bps=5.0,
    half_spread_bps=2.0,    # from your sample: (2.441 - 2.440) / 2 / 2.4405 * 1e4
)


def round_trip_cost_bps(
    maker_frac: float | tuple[float, float],
    venue_a: VenueFees = HYPERLIQUID_T6,
    venue_b: VenueFees = BINANCE_VIP0,
    hold_hours: float = 0.0,
    funding_bps_per_hour: float = 0.0,
) -> float:
    """
    Total cost of one full round trip, in bps.

    maker_frac           scalar, or (frac_on_A, frac_on_B) if the venues differ.
    funding_bps_per_hour NET funding across both legs, signed:
                         positive = you receive, negative = you pay.
                         It is subtracted from cost, so a positive carry
                         lowers the bar and a negative one raises it.
    """
    if isinstance(maker_frac, (tuple, list, np.ndarray)):
        m_a, m_b = float(maker_frac[0]), float(maker_frac[1])
    else:
        m_a = m_b = float(maker_frac)

    fills = 2 * venue_a.leg_cost_bps(m_a) + 2 * venue_b.leg_cost_bps(m_b)
    carry = funding_bps_per_hour * hold_hours
    return fills - carry


def cost_breakdown(
    maker_frac: float | tuple[float, float],
    venue_a: VenueFees = HYPERLIQUID_T6,
    venue_b: VenueFees = BINANCE_VIP0,
    hold_hours: float = 0.0,
    funding_bps_per_hour: float = 0.0,
) -> pd.Series:
    """Same number, itemised — useful when a result looks too good."""
    if isinstance(maker_frac, (tuple, list, np.ndarray)):
        m_a, m_b = float(maker_frac[0]), float(maker_frac[1])
    else:
        m_a = m_b = float(maker_frac)

    rows = {}
    for v, m in ((venue_a, m_a), (venue_b, m_b)):
        t = 1.0 - m
        rows[f"{v.name}: fees"] = 2 * (m * v.maker_bps + t * v.taker_bps)
        rows[f"{v.name}: half-spread (taker legs)"] = 2 * t * v.half_spread_bps
        if v.extra_slip_bps:
            rows[f"{v.name}: extra slippage"] = 2 * v.extra_slip_bps
    if hold_hours:
        rows["funding carry"] = -funding_bps_per_hour * hold_hours
    s = pd.Series(rows, dtype=float)
    s["TOTAL round-trip cost"] = s.sum()
    return s


# ------------------------------------------------------------- spread prep

def deviation_bps(
    spread_bps: pd.Series,
    anchor: str | float = "rolling_median",
    window: int = 6000,
) -> pd.Series:
    """
    Deviation of the spread from its own level.

    The PENDLE basis sits persistently on one side of zero, so trading
    'spread > 0' is not the same as trading 'spread rich versus normal'.
    This subtracts a level so the thing you fade is a deviation.

        "rolling_median"  robust, moves with regime (window = bars)
        "mean" / "median" whole-sample constant
        float             a level you specify, in bps
    """
    s = pd.Series(spread_bps).astype(float)
    if isinstance(anchor, (int, float)):
        level = float(anchor)
    elif anchor == "mean":
        level = s.mean()
    elif anchor == "median":
        level = s.median()
    elif anchor == "rolling_median":
        level = s.rolling(window, min_periods=max(10, window // 10)).median()
    else:
        raise ValueError(f"unknown anchor: {anchor!r}")
    return (s - level).dropna()


# ------------------------------------------------------- forward behaviour

def first_touch_stats(
    dev_bps: pd.Series,
    take_profits_bps: Sequence[float],
    stop_bps: float | None = None,
    horizon_bars: int = 600,
    entry_bps: float = 0.0,
    cooldown_bars: int = 1,
    max_entries: int = 20_000,
) -> pd.DataFrame:
    """
    Fade the deviation and see what happens next.

    Entry at every bar where |dev| >= entry_bps (thinned by cooldown_bars),
    trading against the deviation. Then walk forward up to horizon_bars and
    record which barrier is touched first.

    Signed progress from entry i to bar j is

        p(j) = sign(dev[i]) * (dev[i] - dev[j])

    so p > 0 means the deviation has shrunk — the direction you want.

    Returns one row per take-profit with: hit rate, stop rate, timeout rate,
    mean GROSS bps per attempt (costs are added by plot_pnl, not here), and
    median bars to resolution.

    NOTE this is a geometry check on the spread path, not a backtest. It
    assumes you get filled at the price you see, on both legs, at entry and at
    exit. That assumption is doing a lot of work — see the module docstring in
    your spec's 'threats' field.
    """
    x = np.asarray(pd.Series(dev_bps).dropna(), dtype=float)
    n = x.size
    if n < horizon_bars + 2:
        raise ValueError(f"need > {horizon_bars + 2} bars, got {n}")

    cand = np.flatnonzero(np.abs(x[:-1]) >= entry_bps)
    cand = cand[cand < n - horizon_bars - 1]
    if cooldown_bars > 1:
        cand = cand[::cooldown_bars]
    if cand.size > max_entries:                       # keep it quick
        cand = cand[:: int(np.ceil(cand.size / max_entries))]
    if cand.size == 0:
        raise NoEntries(
            f"no bar has |deviation| >= {entry_bps:.2f} bps "
            f"(max seen: {np.abs(x).max():.2f} bps)"
        )

    out = []
    for tp in take_profits_bps:
        sl = np.inf if stop_bps is None else float(stop_bps)
        wins = stops = 0
        pnl = np.empty(cand.size)
        bars = np.empty(cand.size)

        for k, i in enumerate(cand):
            s = np.sign(x[i]) or 1.0
            seg = s * (x[i] - x[i + 1 : i + 1 + horizon_bars])
            w = np.flatnonzero(seg >= tp)
            l = np.flatnonzero(seg <= -sl)
            first_w = w[0] if w.size else np.inf
            first_l = l[0] if l.size else np.inf
            if first_w < first_l:
                wins += 1
                pnl[k], bars[k] = tp, first_w + 1
            elif first_l < np.inf:
                stops += 1
                pnl[k], bars[k] = -sl, first_l + 1
            else:
                pnl[k], bars[k] = seg[-1], horizon_bars   # timeout, exit at market

        out.append({
            "tp_bps": tp,
            "entries": cand.size,
            "hit_rate": wins / cand.size,
            "stop_rate": stops / cand.size,
            "timeout_rate": 1.0 - (wins + stops) / cand.size,
            "gross_bps_mean": pnl.mean(),
            "median_bars": float(np.median(bars)),
        })
    return pd.DataFrame(out)


# ------------------------------------------------------------------- plot

def plot_pnl(
    spread_bps: pd.Series,
    taker: float = 1.0,
    maker: float = 0.0,
    take_profits_bps: Sequence[float] = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30),
    *,
    venue_a: VenueFees = HYPERLIQUID_T6,
    venue_b: VenueFees = BINANCE_VIP0,
    stop_bps: float | None = None,
    horizon_bars: int = 600,
    entry_bps: float | None = None,
    cooldown_bars: int = 10,
    anchor: str | float = "rolling_median",
    anchor_window: int = 6000,
    hold_hours: float = 0.0,
    funding_bps_per_hour: float = 0.0,
    size_usd: float = 100_000.0,
    maker_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    plot_bars: int = 5_000,
    title: str = "PENDLE perp spread — cost geometry",
):
    """
    Four panels:

      1. spread deviation with the break-even band at your fill mix
      2. net bps per round trip vs take-profit, one line per maker fraction
      3. how often the spread actually travels that far (first-touch)
      4. expected value per attempt = hit-rate-weighted net, vs take-profit

    `taker` and `maker` should sum to 1; pass either (maker wins if they
    disagree and only one is given). Pass a (frac_a, frac_b) tuple as `maker`
    when the two venues have different fill mixes — which they will, since
    Hyperliquid maker is free at tier 6 and Binance maker is not.

    Returns (fig, results_dataframe).
    """
    if maker == 0.0 and taker != 1.0:        # user gave taker only
        maker = 1.0 - taker
    total = (np.mean(maker) if isinstance(maker, (tuple, list)) else maker) + taker
    if not np.isclose(total, 1.0, atol=0.05) and taker != 1.0:
        print(f"[warn] maker + taker = {total:.2f}, not 1.0 — using maker={maker}")

    cost = round_trip_cost_bps(maker, venue_a, venue_b, hold_hours,
                               funding_bps_per_hour)
    if entry_bps is None:
        entry_bps = cost

    dev = deviation_bps(spread_bps, anchor=anchor, window=anchor_window)

    try:
        res = first_touch_stats(
            dev, take_profits_bps, stop_bps=stop_bps, horizon_bars=horizon_bars,
            entry_bps=entry_bps, cooldown_bars=cooldown_bars,
        )
    except NoEntries as exc:
        fallback = float(np.quantile(dev.abs(), 0.95))
        print(f"\n*** {exc}\n*** At this fill mix the trade never even sets up: "
              f"one round trip costs {cost:.1f} bps and the deviation never gets "
              f"that big. Falling back to the 95th percentile of |deviation| "
              f"({fallback:.2f} bps) so the panels still show the shape — but the "
              f"answer for this fill mix is already no.\n")
        entry_bps = fallback
        res = first_touch_stats(
            dev, take_profits_bps, stop_bps=stop_bps, horizon_bars=horizon_bars,
            entry_bps=entry_bps, cooldown_bars=cooldown_bars,
        )
    res["cost_bps"] = cost
    res["net_bps_if_hit"] = res["tp_bps"] - cost
    res["ev_bps"] = res["gross_bps_mean"] - cost
    res["ev_usd"] = res["ev_bps"] / 1e4 * size_usd

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{title}\n{venue_a.name}  vs  {venue_b.name}", fontsize=12)

    # --- 1. the series and the band -------------------------------------
    ax = axes[0, 0]
    d = dev.iloc[:plot_bars]
    ax.plot(d.index, d.values, lw=0.7, color="#3b6ea5", label="deviation (bps)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhspan(-cost, cost, color="#c44", alpha=0.15,
               label=f"below cost (±{cost:.1f} bps)")
    ax.axhline(cost, color="#c44", lw=0.8, ls="--")
    ax.axhline(-cost, color="#c44", lw=0.8, ls="--")
    inside = float((dev.abs() < cost).mean())
    ax.set_title(f"Deviation vs break-even\n{inside:.0%} of bars sit entirely "
                 f"inside one round trip's cost", fontsize=10)
    ax.set_ylabel("bps")
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    # --- 2. net per round trip vs TP, by fill mix ------------------------
    ax = axes[0, 1]
    tps = np.asarray(take_profits_bps, dtype=float)
    for m in maker_grid:
        c = round_trip_cost_bps(m, venue_a, venue_b, hold_hours,
                                funding_bps_per_hour)
        ax.plot(tps, tps - c, marker="o", ms=3,
                label=f"{m:.0%} maker  (cost {c:.1f} bps)")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_title("Net bps per round trip (if the take-profit is reached)")
    ax.set_xlabel("take-profit, bps")
    ax.set_ylabel("net bps")
    ax.legend(fontsize=8)

    # --- 3. does it ever get there --------------------------------------
    ax = axes[1, 0]
    ax.bar(res["tp_bps"], res["hit_rate"], width=1.2, color="#4a8", alpha=0.85,
           label="reaches TP first")
    if stop_bps is not None:
        ax.bar(res["tp_bps"], res["stop_rate"], width=1.2, bottom=res["hit_rate"],
               color="#c55", alpha=0.7, label=f"stops out first ({stop_bps} bps)")
    ax.axvline(cost, color="#c44", ls="--", lw=1,
               label=f"break-even TP = {cost:.1f} bps")
    ax.set_ylim(0, 1)
    ax.set_title(f"Where it resolves within {horizon_bars} bars "
                 f"({res['entries'].iloc[0]:,} entries, |dev| ≥ {entry_bps:.1f} bps)")
    ax.set_xlabel("take-profit, bps")
    ax.set_ylabel("fraction of entries")
    ax.legend(fontsize=8)

    # --- 4. expected value ----------------------------------------------
    ax = axes[1, 1]
    for m in maker_grid:
        c = round_trip_cost_bps(m, venue_a, venue_b, hold_hours,
                                funding_bps_per_hour)
        ax.plot(res["tp_bps"], res["gross_bps_mean"] - c, marker="o", ms=3,
                label=f"{m:.0%} maker")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_title("Expected bps per attempt, net of cost\n"
                 f"(right axis: $ on {size_usd:,.0f} notional)")
    ax.set_xlabel("take-profit, bps")
    ax.set_ylabel("EV, bps")
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo / 1e4 * size_usd, hi / 1e4 * size_usd)
    ax2.set_ylabel("EV, $ per round trip")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    print(cost_breakdown(maker, venue_a, venue_b, hold_hours,
                         funding_bps_per_hour).round(2).to_string())
    print(f"\nbreak-even move  : {cost:.2f} bps  "
          f"(= {cost / 1e4 * size_usd:,.0f} USD per {size_usd:,.0f} notional)")
    print(f"deviation |x| >= cost : {float((dev.abs() >= cost).mean()):.1%} of bars")
    best = res.loc[res["ev_bps"].idxmax()]
    print(f"best TP in grid  : {best.tp_bps:.0f} bps  "
          f"EV {best.ev_bps:+.2f} bps  hit {best.hit_rate:.0%}")

    return fig, res


# ------------------------------------------------------------------ demo

def _synthetic(n: int = 40_000, seed: int = 0) -> pd.Series:
    """OU spread with a persistent positive level, roughly like the sample."""
    rng = np.random.default_rng(seed)
    x, theta, sigma, level = np.empty(n), 0.004, 0.55, 6.0
    x[0] = level
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (level - x[i - 1]) + sigma * rng.standard_normal()
    idx = pd.date_range("2026-09-28 07:39:28.5", periods=n, freq="100ms", tz="UTC")
    return pd.Series(x, index=idx, name="spread_bps")


if __name__ == "__main__":
    s = _synthetic()

    print("=" * 70, "\nALL TAKER\n", "=" * 70, sep="")
    fig, res = plot_pnl(s, taker=1.0, maker=0.0, stop_bps=12, horizon_bars=600)
    fig.savefig("spread_cost_all_taker.png", dpi=110)

    print("\n" + "=" * 70, "\nMAKER ON BOTH LEGS\n", "=" * 70, sep="")
    fig, res = plot_pnl(s, taker=0.0, maker=1.0, stop_bps=12, horizon_bars=600,
                        title="PENDLE perp spread — maker on both legs")
    fig.savefig("spread_cost_all_maker.png", dpi=110)
    print(res.round(3).to_string(index=False))
