
"""
mm_holding_time_model.py

Toy market-making economics across holding horizons.

The five economic levers used here are a repackaging of the attached notebook:

    1) spread capture
    2) adverse selection
    3) fees / rebates
    4) hedge cost
    5) inventory cost

Mapping to the original notebook:
    quoted edge          -> spread capture
    staleness            -> adverse selection
    holding drift        -> realised inventory P&L (zero mean under martingale FV)
    exit vs fair         -> passive offset spread OR Binance hedge cost
    fees                 -> fees / rebates

Important distinction:
- Accounting expected inventory P&L is zero if fair value is a martingale.
- "Inventory cost" below is therefore a risk penalty, not an assumed negative drift.
- Monte Carlo paths separately show realised inventory P&L dispersion.

This is synthetic research scaffolding, not a production-calibrated strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEC_PER_YEAR = 365 * 24 * 3600

FIVE_LEVERS = [
    "spread capture",
    "adverse selection",
    "fees / rebates",
    "hedge cost",
    "inventory cost",
]


def vol_bp_per_sqrt_s(annual_vol: float) -> float:
    """Annualised vol (0.80 = 80%) -> bp per sqrt(second)."""
    return annual_vol / np.sqrt(SEC_PER_YEAR) * 1e4


@dataclass(frozen=True)
class HoldingTimeParams:
    # Market / volatility
    annual_vol: float = 0.80

    # Entry quote economics on Hyperliquid
    entry_edge_bp: float = 6.0
    hl_quote_latency_s: float = 0.75

    # Adverse selection
    adverse_stale_bp: float = 3.0
    adverse_drift_bp: float = 1.5
    adverse_tau_s: float = 2.0

    # Fees
    fee_hl_maker_bp: float = 1.5
    rebate_hl_maker_bp: float = 0.0
    fee_bn_taker_bp: float = 5.0

    # Binance hedge
    bn_half_spread_bp: float = 0.5
    bn_slippage_bp: float = 0.15
    bn_hedge_latency_s: float = 0.01  # "a few ms" approximation

    # Chance of receiving an offsetting passive fill on HL
    hl_offset_rate_per_s: float = 0.03
    hl_offset_edge_bp: float = 6.0
    hl_offset_adverse_bp: float = 2.0

    # Strategy throughput / risk capacity
    fill_rate_per_s: float = 0.05
    max_inventory: float = 5.0

    # CARA-like inventory risk penalty.
    # Units are chosen for a toy certainty-equivalent in bp.
    risk_aversion_per_bp: float = 0.03


def _offset_probability(hold_max_s: np.ndarray, p: HoldingTimeParams) -> np.ndarray:
    lam = p.hl_offset_rate_per_s
    if lam <= 0:
        return np.zeros_like(hold_max_s, dtype=float)
    return 1.0 - np.exp(-lam * hold_max_s)


def _mean_wait_until_exit(
    hold_max_s: np.ndarray,
    p_offset: np.ndarray,
    p: HoldingTimeParams,
) -> np.ndarray:
    """
    E[min(T_offset, tau)] plus Binance hedge execution latency if no HL offset.
    """
    lam = p.hl_offset_rate_per_s

    if lam > 0:
        wait = p_offset / lam
    else:
        wait = hold_max_s.copy()

    return wait + (1.0 - p_offset) * p.bn_hedge_latency_s


def holding_sweep(
    holds_s: Sequence[float],
    p: HoldingTimeParams = HoldingTimeParams(),
) -> pd.DataFrame:
    """
    Expected per-lot and per-hour economics across max holding times.

    Sign convention:
        positive contribution = helps P&L
        negative contribution = hurts P&L

    The five-lever "risk-adjusted value" includes an inventory risk penalty.
    The accounting expected P&L excludes that penalty, because a martingale
    inventory move has zero mean.
    """
    tau = np.asarray(holds_s, dtype=float)

    if np.any(tau <= 0):
        raise ValueError("All holding times must be > 0.")

    p_offset = _offset_probability(tau, p)
    mean_hold = _mean_wait_until_exit(tau, p_offset, p)

    sigma = vol_bp_per_sqrt_s(p.annual_vol)

    # 1) Spread capture:
    # entry edge is earned on every fill; a passive HL offset can earn another edge.
    spread_capture = (
        p.entry_edge_bp
        + p_offset * p.hl_offset_edge_bp
    )

    # 2) Adverse selection:
    # initial stale-fill loss + short-lived drift + adverse selection on passive exit.
    post_fill_adverse = (
        p.adverse_drift_bp
        * (1.0 - np.exp(-mean_hold / p.adverse_tau_s))
    )

    adverse_selection = -(
        p.adverse_stale_bp
        + post_fill_adverse
        + p_offset * p.hl_offset_adverse_bp
    )

    # 3) Fees / rebates:
    # every trade pays the HL maker fee on entry.
    # Exit pays another HL maker fee if offset passively, otherwise Binance taker.
    entry_net_fee = p.fee_hl_maker_bp - p.rebate_hl_maker_bp
    exit_hl_net_fee = p.fee_hl_maker_bp - p.rebate_hl_maker_bp

    fees_rebates = -(
        entry_net_fee
        + p_offset * exit_hl_net_fee
        + (1.0 - p_offset) * p.fee_bn_taker_bp
    )

    # 4) Hedge cost:
    # only paid if passive offset never arrives before tau.
    hedge_cost = -(
        (1.0 - p_offset)
        * (p.bn_half_spread_bp + p.bn_slippage_bp)
    )

    # 5) Inventory cost:
    # martingale inventory P&L has zero mean, so use a risk penalty.
    # CE penalty ~ 0.5 * gamma * variance.
    inventory_variance_bp2 = (sigma ** 2) * mean_hold
    inventory_cost = -(
        0.5
        * p.risk_aversion_per_bp
        * inventory_variance_bp2
    )

    accounting_mean = (
        spread_capture
        + adverse_selection
        + fees_rebates
        + hedge_cost
    )

    risk_adjusted_mean = accounting_mean + inventory_cost

    inventory_std_bp = np.sqrt(inventory_variance_bp2)

    # Approximate throughput cap via Little's law.
    # This is deliberately simple: open lots ~= rate * mean holding time.
    lots_per_s = np.minimum(
        p.fill_rate_per_s,
        p.max_inventory / np.maximum(mean_hold, 1e-12),
    )
    lots_per_hour = lots_per_s * 3600.0

    accounting_pnl_hr = accounting_mean * lots_per_hour
    risk_adjusted_value_hr = risk_adjusted_mean * lots_per_hour

    out = pd.DataFrame({
        "hold_max_s": tau,
        "hl_offset_probability": p_offset,
        "mean_holding_s": mean_hold,

        "spread capture": spread_capture,
        "adverse selection": adverse_selection,
        "fees / rebates": fees_rebates,
        "hedge cost": hedge_cost,
        "inventory cost": inventory_cost,

        "accounting_expected_pnl_bp_per_lot": accounting_mean,
        "risk_adjusted_value_bp_per_lot": risk_adjusted_mean,
        "inventory_std_bp_per_lot": inventory_std_bp,

        "lots_per_hour": lots_per_hour,
        "accounting_expected_pnl_per_hour": accounting_pnl_hr,
        "risk_adjusted_value_per_hour": risk_adjusted_value_hr,
    })

    return out


def lever_table(
    holds_s: Sequence[float],
    p: HoldingTimeParams = HoldingTimeParams(),
    decimals: int = 3,
) -> pd.DataFrame:
    """Compact table for display in a notebook."""
    df = holding_sweep(holds_s, p).copy()
    cols = [
        "hold_max_s",
        "hl_offset_probability",
        "mean_holding_s",
        *FIVE_LEVERS,
        "accounting_expected_pnl_bp_per_lot",
        "risk_adjusted_value_bp_per_lot",
        "inventory_std_bp_per_lot",
        "lots_per_hour",
    ]
    return df[cols].round(decimals)


def plot_lever_decomposition(
    sweep: pd.DataFrame,
    figsize=(13, 5),
):
    """
    Grouped five-lever contributions + net lines.
    """
    x = np.arange(len(sweep))
    w = 0.14

    fig, ax = plt.subplots(figsize=figsize)

    for i, lever in enumerate(FIVE_LEVERS):
        ax.bar(
            x + (i - 2) * w,
            sweep[lever],
            width=w,
            label=lever,
        )

    ax.plot(
        x,
        sweep["accounting_expected_pnl_bp_per_lot"],
        "ko-",
        linewidth=2,
        label="accounting expected P&L",
    )

    ax.plot(
        x,
        sweep["risk_adjusted_value_bp_per_lot"],
        "k--",
        linewidth=1.7,
        label="risk-adjusted value",
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(
        x,
        [f"{v:g}s" for v in sweep["hold_max_s"]],
    )
    ax.set_xlabel("maximum holding time")
    ax.set_ylabel("bp per lot")
    ax.set_title("Same five levers, different weights as holding time increases")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    return fig, ax


def expected_equity_curve(
    sweep: pd.DataFrame,
    minutes: float = 30.0,
    step_s: float = 1.0,
) -> dict[float, pd.DataFrame]:
    """
    Stationary expected equity curves.

    The accounting mean is the expected P&L.
    The risk-adjusted curve subtracts the inventory certainty-equivalent penalty.
    Bands are a rough +/-1 sigma scaling assuming independent lot P&L.
    """
    t_s = np.arange(0, minutes * 60 + step_s, step_s)

    curves = {}

    for _, r in sweep.iterrows():
        hold = float(r["hold_max_s"])
        hours = t_s / 3600.0

        n_lots = r["lots_per_hour"] * hours

        accounting_mean = (
            r["accounting_expected_pnl_per_hour"] * hours
        )

        risk_adjusted_mean = (
            r["risk_adjusted_value_per_hour"] * hours
        )

        # Dominant illustrative dispersion = inventory diffusion risk.
        std = (
            r["inventory_std_bp_per_lot"]
            * np.sqrt(np.maximum(n_lots, 0.0))
        )

        curves[hold] = pd.DataFrame({
            "time_s": t_s,
            "expected_accounting_equity": accounting_mean,
            "risk_adjusted_equity": risk_adjusted_mean,
            "equity_std": std,
            "expected_lots": n_lots,
        })

    return curves


def plot_expected_equity_curves(
    sweep: pd.DataFrame,
    selected_holds: Sequence[float] | None = None,
    minutes: float = 30.0,
    show_bands: bool = True,
    figsize=(13, 6),
):
    """
    Plot expected accounting equity versus time by holding horizon.
    """
    if selected_holds is None:
        selected_holds = sweep["hold_max_s"].tolist()

    curves = expected_equity_curve(sweep, minutes=minutes)

    fig, ax = plt.subplots(figsize=figsize)

    for hold in selected_holds:
        key = min(curves, key=lambda x: abs(x - hold))
        c = curves[key]
        x = c["time_s"] / 60.0
        y = c["expected_accounting_equity"]

        line, = ax.plot(
            x,
            y,
            linewidth=1.8,
            label=f"{key:g}s hold",
        )

        if show_bands:
            sd = c["equity_std"]
            ax.fill_between(
                x,
                y - sd,
                y + sd,
                alpha=0.10,
            )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("elapsed time (minutes)")
    ax.set_ylabel("expected cumulative P&L (bp × lots)")
    ax.set_title(
        "Expected accounting equity by holding horizon\n"
        "Shading = illustrative ±1σ inventory-risk dispersion"
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    return fig, ax


def plot_risk_adjusted_equity_curves(
    sweep: pd.DataFrame,
    selected_holds: Sequence[float] | None = None,
    minutes: float = 30.0,
    figsize=(13, 5),
):
    if selected_holds is None:
        selected_holds = sweep["hold_max_s"].tolist()

    curves = expected_equity_curve(sweep, minutes=minutes)

    fig, ax = plt.subplots(figsize=figsize)

    for hold in selected_holds:
        key = min(curves, key=lambda x: abs(x - hold))
        c = curves[key]

        ax.plot(
            c["time_s"] / 60.0,
            c["risk_adjusted_equity"],
            linewidth=1.8,
            label=f"{key:g}s hold",
        )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("elapsed time (minutes)")
    ax.set_ylabel("risk-adjusted cumulative value")
    ax.set_title("Expected equity after charging for inventory risk")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    return fig, ax


def _lot_exit(
    rng: np.random.Generator,
    hold_max_s: float,
    p: HoldingTimeParams,
):
    """
    Sample whether the lot exits passively on HL or is forced to Binance.
    """
    lam = p.hl_offset_rate_per_s

    if lam > 0:
        offset_wait = rng.exponential(1.0 / lam)
    else:
        offset_wait = np.inf

    if offset_wait <= hold_max_s:
        return float(offset_wait), "HL offset"

    return float(hold_max_s + p.bn_hedge_latency_s), "Binance taker"


def simulate_inventory_equity_path(
    hold_max_s: float,
    p: HoldingTimeParams = HoldingTimeParams(),
    minutes: float = 10.0,
    dt: float = 0.05,
    seed: int = 1,
) -> pd.DataFrame:
    """
    One illustrative realised path.

    - Fair value follows a martingale random walk.
    - Candidate passive fills arrive as a Poisson process.
    - Each fill has random side (+1 long, -1 short).
    - Opposite HL exit time is exponential, capped by hold_max.
    - Inventory is marked to fair value continuously.
    - Entry/exit economics are charged using the five-lever decomposition.

    This is intentionally a teaching simulation, not a queue-level engine.
    """
    if hold_max_s <= 0:
        raise ValueError("hold_max_s must be > 0.")

    rng = np.random.default_rng(seed)

    n = int(minutes * 60 / dt) + 1
    t = np.arange(n) * dt

    sigma = vol_bp_per_sqrt_s(p.annual_vol)
    fair = np.zeros(n)
    fair[1:] = np.cumsum(
        rng.normal(
            0.0,
            sigma * np.sqrt(dt),
            n - 1,
        )
    )

    # Min-heap of (exit_step, unique_id, lot_dict)
    open_heap = []
    open_lots = {}
    uid = 0

    inventory = 0
    realised_fixed = 0.0

    equity = np.zeros(n)
    inventory_path = np.zeros(n, dtype=float)
    spread_path = np.zeros(n)
    adverse_path = np.zeros(n)
    fee_path = np.zeros(n)
    hedge_path = np.zeros(n)

    realised_spread = 0.0
    realised_adverse = 0.0
    realised_fees = 0.0
    realised_hedge = 0.0

    entry_fee = p.fee_hl_maker_bp - p.rebate_hl_maker_bp

    for i in range(n):
        # Close all lots due by this time.
        while open_heap and open_heap[0][0] <= i:
            _, lot_id, lot = heapq.heappop(open_heap)

            if lot_id not in open_lots:
                continue

            open_lots.pop(lot_id)

            side = lot["side"]
            inventory -= side

            hold_s = lot["hold_s"]
            exit_type = lot["exit_type"]

            # Post-fill adverse drift is charged on exit.
            adv_drift = (
                p.adverse_drift_bp
                * (1.0 - np.exp(-hold_s / p.adverse_tau_s))
            )

            realised_adverse -= adv_drift

            if exit_type == "HL offset":
                realised_spread += p.hl_offset_edge_bp
                realised_adverse -= p.hl_offset_adverse_bp
                realised_fees -= (
                    p.fee_hl_maker_bp - p.rebate_hl_maker_bp
                )
            else:
                realised_fees -= p.fee_bn_taker_bp
                realised_hedge -= (
                    p.bn_half_spread_bp + p.bn_slippage_bp
                )

        # Candidate new passive fill.
        if rng.random() < p.fill_rate_per_s * dt:
            side = 1 if rng.random() < 0.5 else -1

            # Do not worsen inventory beyond the cap.
            if abs(inventory + side) <= p.max_inventory:
                hold_s, exit_type = _lot_exit(
                    rng,
                    hold_max_s,
                    p,
                )

                exit_step = min(
                    n - 1,
                    i + max(1, int(round(hold_s / dt))),
                )

                lot = {
                    "side": side,
                    "entry_step": i,
                    "entry_fair": fair[i],
                    "hold_s": hold_s,
                    "exit_type": exit_type,
                }

                open_lots[uid] = lot
                heapq.heappush(
                    open_heap,
                    (exit_step, uid, lot),
                )
                uid += 1

                inventory += side

                # Entry economics realised immediately.
                realised_spread += p.entry_edge_bp
                realised_adverse -= p.adverse_stale_bp
                realised_fees -= entry_fee

        # Mark open inventory to current fair.
        inventory_mtm = 0.0

        for lot in open_lots.values():
            inventory_mtm += (
                lot["side"]
                * (fair[i] - lot["entry_fair"])
            )

        realised_fixed = (
            realised_spread
            + realised_adverse
            + realised_fees
            + realised_hedge
        )

        equity[i] = realised_fixed + inventory_mtm
        inventory_path[i] = inventory

        spread_path[i] = realised_spread
        adverse_path[i] = realised_adverse
        fee_path[i] = realised_fees
        hedge_path[i] = realised_hedge

    return pd.DataFrame({
        "time_s": t,
        "fair_bp": fair,
        "equity_bp": equity,
        "inventory": inventory_path,
        "cum_spread_capture": spread_path,
        "cum_adverse_selection": adverse_path,
        "cum_fees_rebates": fee_path,
        "cum_hedge_cost": hedge_path,
    })


def plot_sample_equity_inventory(
    holds_s: Sequence[float],
    p: HoldingTimeParams = HoldingTimeParams(),
    minutes: float = 10.0,
    dt: float = 0.05,
    seed: int = 1,
    figsize=(13, 4.5),
):
    """
    One separate figure per holding horizon.
    Equity and inventory share the time axis.
    """
    outputs = {}

    for hold in holds_s:
        path = simulate_inventory_equity_path(
            hold_max_s=hold,
            p=p,
            minutes=minutes,
            dt=dt,
            seed=seed,
        )
        outputs[float(hold)] = path

        fig, ax1 = plt.subplots(figsize=figsize)

        x = path["time_s"] / 60.0

        ax1.plot(
            x,
            path["equity_bp"],
            linewidth=1.4,
            label="mark-to-market equity",
        )
        ax1.set_xlabel("elapsed time (minutes)")
        ax1.set_ylabel("equity (bp × lots)")
        ax1.axhline(0, linewidth=0.8)

        ax2 = ax1.twinx()
        ax2.step(
            x,
            path["inventory"],
            where="post",
            alpha=0.45,
            label="inventory",
        )
        ax2.set_ylabel("inventory (lots)")
        ax2.axhline(0, linewidth=0.6, alpha=0.5)

        ax1.set_title(
            f"Realised path: max hold {hold:g}s "
            f"(HL quote latency {p.hl_quote_latency_s*1000:.0f} ms, "
            f"Binance hedge latency {p.bn_hedge_latency_s*1000:.0f} ms)"
        )

        fig.tight_layout()
        plt.show()

    return outputs


def relationship_summary(
    sweep: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact diagnostic view useful for Markdown discussion.
    """
    return sweep[[
        "hold_max_s",
        "hl_offset_probability",
        "mean_holding_s",
        "accounting_expected_pnl_bp_per_lot",
        "risk_adjusted_value_bp_per_lot",
        "inventory_std_bp_per_lot",
        "lots_per_hour",
    ]].copy()
