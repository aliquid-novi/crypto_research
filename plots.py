"""Two quick looks at a run. Needs matplotlib; nothing else in the package does."""
from __future__ import annotations

import numpy as np

from .analytics import Result, markout_curve


def plot_run(res: Result, inst: str, fair_key: str | None = "fair", t0: float | None = None, t1: float | None = None):
    """Three stacked panels: price (mid, latent reference if known, logged fair value, our fills),
    position on every instrument, and equity."""
    import matplotlib.pyplot as plt

    t0 = res.t0 if t0 is None else t0
    t1 = res.t_end if t1 is None else t1
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    b = res.bbo[inst]
    b = b[(b.t >= t0) & (b.t <= t1)]
    ax[0].step(b.t, b.mid, where="post", color="0.45", lw=0.9, label=f"{inst} mid")
    if res.truth is not None:
        m = (res.truth["t"] >= t0) & (res.truth["t"] <= t1)
        ax[0].plot(res.truth["t"][m], res.truth["ref"][inst][m], color="tab:blue", lw=0.9, label="latent reference")
    if fair_key and fair_key in res.logs:
        lg = res.logs[fair_key]
        lg = lg[(lg.t >= t0) & (lg.t <= t1)]
        ax[0].plot(lg.t, lg.value, color="tab:orange", lw=0.8, label="strategy fair value")
    f = res.fills[(res.fills.inst == inst) & (res.fills.t >= t0) & (res.fills.t <= t1)]
    ax[0].scatter(f.t[f.side > 0], f.price[f.side > 0], marker="^", s=28, color="tab:green", zorder=3, label="our buys")
    ax[0].scatter(f.t[f.side < 0], f.price[f.side < 0], marker="v", s=28, color="tab:red", zorder=3, label="our sells")
    ax[0].set_ylabel("price")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=5, frameon=False)
    e = res.equity[(res.equity.t >= t0) & (res.equity.t <= t1)]
    for nm in res.names:
        scale = np.nanmax(np.abs(res.equity[f"pos_{nm}"].values)) or 1.0
        ax[1].step(e.t, e[f"pos_{nm}"] / scale, where="post", lw=1.0, label=f"{nm} (max {scale:g})")
    ax[1].set_ylabel("position / its max")
    ax[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=4, frameon=False)
    ax[2].plot(e.t, e.equity, color="0.2", lw=1.0)
    for r in res.funding.itertuples():
        if t0 <= r.t <= t1:
            for a in ax:
                a.axvline(r.t, color="tab:purple", lw=0.8, ls=":")
    ax[2].set_ylabel("equity (base ccy)")
    ax[2].set_xlabel("seconds   (dotted lines: funding settlements)")
    fig.tight_layout()
    return fig


def plot_markouts(runs: dict[str, Result], horizons=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0), **kw):
    """Mean markout by horizon for one or more runs (log time axis; horizon 0 drawn at the left edge)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    hs = [h for h in horizons if h > 0]
    for name, res in runs.items():
        c = markout_curve(res, horizons, **kw)
        line, = ax.plot(hs, [c.loc[h, "mean"] for h in hs], marker="o", ms=3.5, lw=1.2, label=name)
        if 0.0 in c.index:
            ax.scatter([hs[0] / 2.5], [c.loc[0.0, "mean"]], marker="s", s=22, color=line.get_color())
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("seconds after the fill   (square: edge at the fill)")
    ax.set_ylabel("mean markout, bps   (positive = good for us)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
