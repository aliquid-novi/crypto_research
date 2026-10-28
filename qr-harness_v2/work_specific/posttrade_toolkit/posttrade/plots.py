"""Matplotlib plots used by the notebooks. Every function returns the Axes (or Figure) it drew on.

Style: thin lines, recessive grid, a fixed categorical colour order (never cycled by rank), a
diverging blue (good) / orange (bad) map with a neutral midpoint for signed quantities like
markouts. Colours are colour-blind-checked as a set.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, BAD = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("bad_neutral_good", [BAD, "#f0efec", GOOD])


def style() -> None:
    """Apply the toolkit's matplotlib style (call once per notebook)."""
    plt.rcParams.update({
        "figure.figsize": (8, 4.2), "figure.dpi": 100, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 2.0, "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def _ax(ax):
    return ax if ax is not None else plt.subplots()[1]


def _xscale(ax, xs):
    xs = np.asarray(xs, float)
    if np.nanmax(np.abs(xs)) / max(np.nanmin(np.abs(xs[xs != 0])) if np.any(xs != 0) else 1, 1e-9) > 50:
        ax.set_xscale("symlog", linthresh=max(np.nanmin(np.abs(xs[xs != 0])), 1e-3))


def markout_curves(curves: dict[str, pd.DataFrame], ax=None, title: str = "Markout curve", ylabel: str = "bps",
                   xlabel: str = "horizon (s)") -> plt.Axes:
    """Plot one or more curves (index = horizon, columns mean/lo/hi) with CI bands; zero line; t=0 line."""
    ax = _ax(ax)
    for i, (name, c) in enumerate(curves.items()):
        col = SERIES[i % len(SERIES)] if i < len(SERIES) else MUTED
        x = np.asarray(c.index, float)
        ax.plot(x, c["mean"], color=col, label=name, marker="o", ms=3)
        if {"lo", "hi"} <= set(c.columns):
            ax.fill_between(x, c["lo"], c["hi"], color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=BASE, lw=1)
    ax.axvline(0, color=BASE, lw=1, ls=":")
    _xscale(ax, np.concatenate([np.asarray(c.index, float) for c in curves.values()]))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(curves) > 1:
        ax.legend()
    return ax


def heatmap(table: pd.DataFrame, ax=None, title: str = "", fmt: str = "{:.1f}", cbar_label: str = "bps",
            annotate: bool = True) -> plt.Axes:
    """Diverging heatmap of a (rows x columns) table centred at 0 (blue = good, orange = bad)."""
    ax = _ax(ax)
    v = table.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
    im = ax.imshow(v, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim - 1e-12, lim + 1e-12))
    ax.set_xticks(range(v.shape[1]), [f"{c:g}" if isinstance(c, (int, float, np.floating)) else str(c) for c in table.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(v.shape[0]), [str(i) for i in table.index])
    ax.grid(False)
    if annotate and v.size <= 400:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7, color=INK)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.04)
    ax.set_title(title)
    return ax


def decomposition(spread_captured: float, adverse: pd.DataFrame, ax=None, title: str = "Markout decomposition") -> plt.Axes:
    """Stacked view: constant spread captured + adverse curve (index horizon, mean/lo/hi) = markout."""
    ax = _ax(ax)
    x = np.asarray(adverse.index, float)
    ax.axhline(spread_captured, color=SERIES[2], lw=2, label=f"spread captured ({spread_captured:.2f})")
    ax.plot(x, adverse["mean"], color=SERIES[1], marker="o", ms=3, label="adverse move")
    ax.fill_between(x, adverse["lo"], adverse["hi"], color=SERIES[1], alpha=0.15, lw=0)
    ax.plot(x, adverse["mean"] + spread_captured, color=SERIES[0], marker="o", ms=3, label="markout = sum")
    ax.axhline(0, color=BASE, lw=1)
    _xscale(ax, x)
    ax.set(title=title, xlabel="horizon (s)", ylabel="bps")
    ax.legend()
    return ax


def coef_vs_horizon(reg: pd.DataFrame, feature: str, models: list[str] | None = None, ax=None) -> plt.Axes:
    """Coefficient of ``feature`` against horizon for each model in a markout_regression table, with CIs."""
    ax = _ax(ax)
    d = reg[reg["feature"] == feature]
    models = models or list(pd.unique(d["model"]))
    for i, m in enumerate(models):
        r = d[d["model"] == m].sort_values("horizon_s")
        ax.plot(r["horizon_s"], r["coef"], color=SERIES[i % 8], marker="o", ms=3, label=m)
        ax.fill_between(r["horizon_s"], r["lo"], r["hi"], color=SERIES[i % 8], alpha=0.12, lw=0)
    ax.axhline(0, color=BASE, lw=1)
    _xscale(ax, d["horizon_s"].to_numpy())
    ax.set(title=f"{feature}: coefficient vs horizon", xlabel="horizon (s)", ylabel="bps per unit")
    ax.legend()
    return ax


def lorenz(values: np.ndarray, ax=None, title: str = "Concentration (Lorenz curve)", xlabel: str = "share of counterparties",
           ylabel: str = "share of total") -> plt.Axes:
    """Lorenz curve of non-negative ``values`` sorted descending (largest contributors first) + Gini."""
    ax = _ax(ax)
    v = np.sort(np.clip(np.asarray(values, float), 0, None))[::-1]
    cum = np.concatenate([[0], np.cumsum(v) / v.sum()])
    x = np.linspace(0, 1, len(cum))
    gini = 2 * np.trapz(cum, x) - 1 if hasattr(np, "trapz") else 2 * np.trapezoid(cum, x) - 1
    ax.plot(x, cum, color=SERIES[0], label=f"observed (Gini {gini:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="equal")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    return ax


def qq_exponential(residuals: dict[str, np.ndarray], ax=None, title: str = "Time-rescaling QQ (Exp(1))") -> plt.Axes:
    """QQ plot of rescaled inter-event times against Exp(1) quantiles; on the diagonal = good fit."""
    ax = _ax(ax)
    mx = 0
    for i, (name, r) in enumerate(residuals.items()):
        r = np.sort(np.asarray(r, float))
        q = -np.log(1 - (np.arange(1, len(r) + 1) - 0.5) / len(r))
        ax.plot(q, r, ".", ms=2, color=SERIES[i % 8], label=name)
        mx = max(mx, q.max())
    ax.plot([0, mx], [0, mx], color=MUTED, ls="--", lw=1)
    ax.set(title=title, xlabel="Exp(1) quantile", ylabel="rescaled time")
    ax.legend()
    return ax


def signature_plot(sig: pd.DataFrame, ax=None) -> plt.Axes:
    """Realised variance per day vs sampling interval (hygiene.signature output)."""
    ax = _ax(ax)
    ax.plot(sig["interval_s"], sig["rv_per_day"], marker="o", ms=4, color=SERIES[0])
    ax.set_xscale("log")
    ax.set(title="Signature plot", xlabel="sampling interval (s)", ylabel="RV per day (bps²)")
    return ax


def waterfall(components: pd.Series, start_label: str = "paper", end_label: str = "realised", start: float = 0.0,
              ax=None, title: str = "Shortfall bridge", ylabel: str = "bps") -> plt.Axes:
    """Waterfall from ``start`` (paper PnL) through each component to the end (realised)."""
    ax = _ax(ax)
    labels = [start_label, *components.index, end_label]
    level = start
    ax.bar(0, start, color=SERIES[0], width=0.6)
    for i, (k, v) in enumerate(components.items(), start=1):
        ax.bar(i, v, bottom=level, color=GOOD if v >= 0 else BAD, width=0.6)
        level += v
    ax.bar(len(labels) - 1, level, color=SERIES[0], width=0.6)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set(title=title, ylabel=ylabel)
    return ax


def scatter(x, y, ax=None, title: str = "", xlabel: str = "", ylabel: str = "", logx: bool = False,
            trend_bins: int | None = 8) -> plt.Axes:
    """Scatter with an optional binned-mean trend line."""
    ax = _ax(ax)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ax.scatter(x, y, s=10, color=SERIES[0], alpha=0.5, lw=0)
    if trend_bins:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > trend_bins * 3:
            b = pd.qcut(x[m], trend_bins, duplicates="drop")
            g = pd.DataFrame({"x": x[m], "y": y[m], "b": b}).groupby("b", observed=True).mean()
            ax.plot(g["x"], g["y"], color=SERIES[1], marker="o", ms=4, label="binned mean")
            ax.legend()
    if logx:
        ax.set_xscale("log")
    ax.axhline(0, color=BASE, lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def cusum_plot(c: pd.DataFrame, h: float = 4.0, ax=None, title: str = "CUSUM of daily mean markout") -> plt.Axes:
    """Plot S+ and S- from hygiene.cusum with the alarm threshold h."""
    ax = _ax(ax)
    x = np.arange(len(c))
    ax.plot(x, c["s_pos"], color=SERIES[0], label="S+ (drift up)")
    ax.plot(x, c["s_neg"], color=SERIES[1], label="S- (drift down)")
    ax.axhline(h, color=MUTED, ls="--", lw=1, label=f"alarm h={h}")
    ax.set_xticks(x[:: max(len(x) // 10, 1)], [str(i) for i in c.index[:: max(len(x) // 10, 1)]], rotation=40, ha="right")
    ax.set(title=title, ylabel="cumulative z")
    ax.legend()
    return ax


def lines(df: pd.DataFrame, ax=None, title: str = "", xlabel: str = "", ylabel: str = "", logx: bool = False) -> plt.Axes:
    """Plot each column of df against its index (fixed colour order)."""
    ax = _ax(ax)
    for i, c in enumerate(df.columns):
        ax.plot(df.index, df[c], color=SERIES[i % 8], label=str(c), marker="o", ms=3)
    if logx:
        ax.set_xscale("log")
    ax.axhline(0, color=BASE, lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if df.shape[1] > 1:
        ax.legend()
    return ax


def hist(x, ax=None, bins: int = 60, title: str = "", xlabel: str = "", vline: float | None = 0.0) -> plt.Axes:
    """Histogram with an optional reference line."""
    ax = _ax(ax)
    x = np.asarray(x, float)
    ax.hist(x[np.isfinite(x)], bins=bins, color=SERIES[0], alpha=0.85, edgecolor="#fcfcfb", linewidth=0.5)
    if vline is not None:
        ax.axvline(vline, color=INK2, lw=1)
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    return ax


def time_series(ts_ns, values: dict[str, np.ndarray], ax=None, title: str = "", ylabel: str = "") -> plt.Axes:
    """Plot series against UTC time (ts in int64 ns)."""
    ax = _ax(ax)
    t = pd.to_datetime(np.asarray(ts_ns, dtype="int64"), utc=True)
    for i, (k, v) in enumerate(values.items()):
        ax.plot(t, v, color=SERIES[i % 8], lw=1.2, label=k)
    ax.set(title=title, ylabel=ylabel)
    if len(values) > 1:
        ax.legend()
    return ax
