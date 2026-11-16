"""Figures and a self-contained HTML report for experiments. Needs matplotlib.

Figures follow one convention: grey for context, blue for the thing the title is about, orange for
the contrast; every panel says in its title what it shows; error bars are +/- one standard error.
"""
from __future__ import annotations

import base64
import html
import io

import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e88"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#bdbcb6", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.size": 9,
    })
    return plt


def plot_signal_vs_effect(df: pd.DataFrame, x: str, y: str, yerr: str | None = None, label: str | None = None,
                          xlabel: str = "", ylabel: str = "", title: str = "", agree: str = "opposite", ax=None,
                          clear_t: float = 2.0):
    """One point per world: a diagnostic signal (x) against what actually happened when acted on (y).

    agree="opposite": the signal is right when x and y have opposite signs (e.g. negative markouts and
    a gain from pushing out). agree="same": right when they share a sign (e.g. a predicted gain).
    Points whose outcome is within `clear_t` standard errors of zero are drawn grey: there the data
    cannot say which way the truth lies, so the signal can be neither right nor wrong."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.4))
    xs, ys = df[x].values, df[y].values
    es = df[yerr].values if yerr else np.zeros(len(df))
    clear = np.abs(ys) >= clear_t * es
    ok = (np.sign(xs) != np.sign(ys)) if agree == "opposite" else (np.sign(xs) == np.sign(ys))
    ax.axhline(0, color="#bdbcb6", lw=1)
    ax.axvline(0, color="#bdbcb6", lw=1)
    for m, col, lab in ((clear & ok, BLUE, "signal pointed the right way"), (clear & ~ok, ORANGE, "signal pointed the wrong way"),
                        (~clear, GREY, "outcome too close to zero to call")):
        if m.any():
            ax.errorbar(xs[m], ys[m], yerr=es[m] if yerr else None, fmt="o", color=col, ecolor=col,
                        elinewidth=1.2, capsize=2.5, ms=7, mec=SURFACE, mew=1.5, label=f"{lab} ({int(m.sum())})")
    if label:
        for xi, yi, t in zip(xs, ys, df[label].values):
            ax.annotate(str(t), (xi, yi), xytext=(6, 5), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel), ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    return ax


def plot_value_curves(runs, statics: dict[str, float], others: dict[str, str] | None = None, metric: str = "pnl",
                      xlabel: str = "", ylabel: str = "", ncols: int = 4, title: str = ""):
    """Small multiples, one per world: the metric across a family of static variants (x from `statics`),
    with other variants drawn as labelled horizontal lines at their own mean."""
    plt = _plt()
    worlds = runs.worlds
    nrows = int(np.ceil(len(worlds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows), squeeze=False)
    mean, se = runs.mean(metric), runs.se(metric)
    cols = [ORANGE, AQUA, "#6250d6"]
    for i, w in enumerate(worlds):
        ax = axes[i // ncols][i % ncols]
        xs = list(statics.values())
        ys = [mean.loc[w, v] for v in statics]
        es = [se.loc[w, v] for v in statics]
        ax.axhline(0, color="#bdbcb6", lw=1)
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color=GREY, ecolor=GREY, elinewidth=1.2, capsize=2.5, mec=SURFACE, mew=1.2, label="static ladder")
        b = int(np.argmax(ys))
        ax.plot([xs[b]], [ys[b]], "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, label="best static", zorder=4)
        for j, (v, lab) in enumerate((others or {}).items()):
            ax.axhline(mean.loc[w, v], color=cols[j % 3], lw=1.6, ls=(0, (4, 2)), label=lab)
        ax.set_title(w, fontsize=9.5)
        ax.set_xticks(xs)
        if i // ncols == nrows - 1:
            ax.set_xlabel(xlabel)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(len(worlds), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_markout_split(ft: pd.DataFrame, k: int = 1, rule: str = "through", title: str = "", ax=None, prefix: str = "m"):
    """Average markout by horizon for fills that would still fill k ticks further out (kept) and those
    that would not (lost). Horizons are the m<h> columns of a fill table (prefix="s": spread markouts)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.0))
    hs = sorted(float(c[1:]) for c in ft.columns if c.startswith(prefix) and c[1:].replace(".", "").isdigit())
    d = ft.dropna(subset=[rule])
    kept = d[rule] >= k
    ax.axhline(0, color="#bdbcb6", lw=1)
    for mask, col, lab in ((kept, ORANGE, f"kept: would still fill {k} tick{'s' if k > 1 else ''} out"),
                           (~kept, BLUE, "lost: would not have filled"), (kept | ~kept, GREY, "all fills")):
        g = d[mask]
        if not len(g):
            continue
        m = [g[f"{prefix}{h:g}"].mean() for h in hs]
        e = [g[f"{prefix}{h:g}"].std(ddof=1) / np.sqrt(g[f"{prefix}{h:g}"].count()) for h in hs]
        ax.errorbar(hs, m, yerr=e, fmt="-o", color=col, ecolor=col, elinewidth=1.1, capsize=2.5, mec=SURFACE, mew=1.2,
                    lw=1.4 if col == GREY else 2.0, label=f"{lab} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_xlabel("seconds after the fill"), ax.set_ylabel("average markout, ticks (positive = good)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    return ax


def plot_controller(res, title: str = "", t0: float | None = None, t1: float | None = None):
    """What a level controller did during one run: the spread's volatility, and the shift it applied."""
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    v, s = res.logs.get("vol"), res.logs.get("shift")
    t0 = res.t0 if t0 is None else t0
    t1 = res.t_end if t1 is None else t1
    if v is not None:
        v = v[(v.t >= t0) & (v.t <= t1)]
        ax[0].plot(v.t, v.value, color=GREY, lw=1.2)
        ax[0].axhline(v.value.median(), color=INK2, lw=1, ls=(0, (4, 2)))
        ax[0].set_ylabel("spread volatility\nbps per sqrt(s)")
        ax[0].set_title(title or "Volatility of the spread (dashed: its median) and the shift the rule applied")
    if s is not None:
        s = s[(s.t >= t0) & (s.t <= t1)]
        ax[1].step(s.t, s.value, where="post", color=BLUE, lw=2)
        ax[1].set_ylabel("ticks pushed out")
        ax[1].set_yticks(range(0, int(max(1, s.value.max())) + 1))
    f = res.fills[(res.fills.tag == "quote") & (res.fills.t >= t0) & (res.fills.t <= t1)]
    ax[1].plot(f.t, np.full(len(f), -0.25), "|", color=INK2, ms=7, mew=1)
    ax[1].set_xlabel("seconds   (ticks along the bottom: our passive fills)")
    fig.tight_layout()
    return fig


def plot_bars(df: pd.DataFrame, value: str, err: str | None = None, title: str = "", xlabel: str = "", ax=None, highlight=None):
    """Horizontal bars with error bars; index labels down the side. Bars about zero."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 0.42 * len(df) + 1.2))
    y = np.arange(len(df))[::-1]
    cols = [BLUE if (highlight is None or i in highlight) else GREY for i in df.index]
    ax.barh(y, df[value].values, xerr=df[err].values if err else None, color=cols, height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 1.1, "capsize": 2.5})
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([str(i) for i in df.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel), ax.set_title(title)
    return ax


class Report:
    """Collects text, tables and figures into one HTML file with the images embedded."""

    CSS = """
    :root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e0dfda;--card:#ffffff}
    @media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}}
    body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
    main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
    h1{font-size:25px;margin:0 0 4px;letter-spacing:-.01em} h2{font-size:18px;margin:34px 0 8px}
    p{margin:8px 0;max-width:78ch} .sub{color:var(--ink2);margin-bottom:18px}
    figure{margin:14px 0;padding:10px;background:var(--card);border:1px solid var(--line);border-radius:8px}
    figure img{display:block;width:100%;height:auto;border-radius:4px} figcaption{color:var(--ink2);font-size:13px;margin-top:6px}
    .tw{overflow-x:auto;margin:10px 0} table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
    th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
    th{color:var(--ink2);font-weight:600} th:first-child,td:first-child{text-align:left}
    tbody th{text-align:left;font-weight:500;color:var(--ink)} code{font-size:13px}
    """

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.parts = [f"<h1>{html.escape(title)}</h1>", f'<p class="sub">{html.escape(subtitle)}</p>' if subtitle else ""]

    def heading(self, text: str):
        self.parts.append(f"<h2>{html.escape(text)}</h2>")
        return self

    def text(self, *paragraphs: str):
        self.parts.extend(f"<p>{html.escape(p)}</p>" for p in paragraphs)
        return self

    def table(self, df: pd.DataFrame, caption: str = "", fmt: str = "{:.2f}"):
        body = df.to_html(float_format=lambda v: fmt.format(v), border=0, na_rep="")
        cap = f'<p class="sub">{html.escape(caption)}</p>' if caption else ""
        self.parts.append(f'<div class="tw">{body}</div>{cap}')
        return self

    def figure(self, fig, caption: str = "", dpi: int = 140):
        import matplotlib.pyplot as plt
        fig = fig if hasattr(fig, "savefig") else fig.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">'
                          f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        return self

    def save(self, path: str) -> str:
        doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f"<title>{html.escape(self.title)}</title><style>{self.CSS}</style></head><body><main>{''.join(self.parts)}</main></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return path
