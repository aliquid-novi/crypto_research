"""Controlled experiments: every variant faces exactly the same markets, and differences come with
standard errors.

    exp = Experiment(worlds={"base": market, ...},
                     variants={"baseline": lambda: SpreadQuoter(...), "idea": lambda: SpreadQuoter(..., controller=...)},
                     seeds=range(6), duration=1800, sim_kw={"latency": ...})
    runs = exp.run(workers=2)
    runs.paired("pnl", baseline="baseline")      # mean difference vs baseline, its standard error and t

A "world" is a synthetic Market; a "variant" is a zero-argument function returning a fresh Strategy.
The unit of replication is the seed: each seed is one independent market per world, shared by all
variants, so differences are paired and much less noisy than the raw numbers.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .analytics import Result, summary
from .synth import Market, simulate

_ACTIVE = None      # the experiment being run, visible to forked workers


def _unit(job):
    world, seed = job
    return _ACTIVE._run_unit(world, seed)


class Experiment:
    def __init__(self, worlds: dict[str, Market], variants: dict[str, Callable], seeds=range(4),
                 duration: float = 1800.0, sim_kw: dict | Callable[[], dict] | None = None,
                 metrics: Callable[[Result, str, str, int], dict] | None = None,
                 keep: Callable[[str, str, int], bool] | None = None, horizon: float = 5.0):
        """
        metrics   optional f(result, world, variant, seed) -> dict of extra numbers to record per run.
                  Keys starting with "_" may hold any object (a DataFrame, say); they are kept in
                  `runs.extras[(world, variant, seed)]` instead of the table
        keep      optional f(world, variant, seed) -> bool: which full Results to keep in memory
                  (default: none; a Result holds every quote change and trade of the run)
        sim_kw    keyword arguments for the Sim (latency, queue, ...). Pass a function returning a dict
                  when the models hold state or random generators, so every run starts fresh.
        """
        self.worlds, self.variants, self.seeds = worlds, variants, list(seeds)
        self.duration, self.sim_kw, self.metrics, self.keep, self.horizon = duration, sim_kw, metrics, keep, horizon

    def _run_unit(self, world: str, seed: int):
        mk = simulate(self.worlds[world], self.duration, seed)
        rows, kept, extras = [], {}, {}
        for name, make in self.variants.items():
            kw = self.sim_kw() if callable(self.sim_kw) else dict(self.sim_kw or {})
            res = mk.run(make(), **kw)
            row = {"world": world, "variant": name, "seed": seed, **summary(res, self.horizon)}
            if self.metrics is not None:
                extra = self.metrics(res, world, name, seed)
                aside = {k: v for k, v in extra.items() if k.startswith("_")}
                if aside:
                    extras[(world, name, seed)] = aside
                row.update({k: v for k, v in extra.items() if not k.startswith("_")})
            rows.append(row)
            if self.keep is not None and self.keep(world, name, seed):
                kept[(world, name, seed)] = res
        return rows, kept, extras

    def run(self, workers: int = 1, verbose: bool = True) -> "Runs":
        global _ACTIVE
        jobs = [(w, s) for w in self.worlds for s in self.seeds]
        rows, kept, extras = [], {}, {}

        def collect(out, job):
            rows.extend(out[0])
            kept.update(out[1])
            extras.update(out[2])
            if verbose:
                print(f"  done {job[0]!r} seed {job[1]}  ({len(rows)} runs)", flush=True)

        pool = None
        if workers > 1:
            _ACTIVE = self                                         # set before forking so workers inherit it
            try:
                import multiprocessing as mp
                pool = mp.get_context("fork").Pool(workers)        # fork: workers also inherit the lambdas
            except (ValueError, ImportError):
                print("  parallel runs need a platform with fork (Linux, macOS); running one at a time")
        if pool is not None:
            try:
                for job, out in zip(jobs, pool.imap(_unit, jobs)):
                    collect(out, job)
            finally:
                pool.close()
                pool.join()
        else:
            for job in jobs:
                collect(self._run_unit(*job), job)
        _ACTIVE = None
        return Runs(pd.DataFrame(rows), kept, list(self.worlds), list(self.variants), extras)


class Runs:
    def __init__(self, table: pd.DataFrame, results: dict, worlds: list[str], variants: list[str], extras: dict | None = None):
        self.table, self.results, self.worlds, self.variants = table, results, worlds, variants
        self.extras = extras or {}      # {(world, variant, seed): {"_name": object}} from metrics keys starting "_"

    def mean(self, metric: str = "pnl") -> pd.DataFrame:
        """Mean of a metric: rows are worlds, columns are variants."""
        return self.table.pivot_table(index="world", columns="variant", values=metric, aggfunc="mean").loc[self.worlds, self.variants]

    def se(self, metric: str = "pnl") -> pd.DataFrame:
        g = self.table.pivot_table(index="world", columns="variant", values=metric, aggfunc=lambda v: v.std(ddof=1) / math.sqrt(len(v)))
        return g.loc[self.worlds, self.variants]

    def paired(self, metric: str = "pnl", baseline: str | None = None) -> pd.DataFrame:
        """Each variant against the baseline on the same seeds.

        mean, se    the variant's own average and standard error across seeds
        diff        average of (variant - baseline) over seeds
        diff_se     standard error of that paired difference
        t           diff / diff_se. Beyond about +/-2 is worth believing; with few seeds, be stricter.
        wins        seeds in which the variant beat the baseline
        """
        baseline = baseline or self.variants[0]
        wide = self.table.pivot_table(index=["world", "seed"], columns="variant", values=metric)
        rows = []
        for w in self.worlds:
            g = wide.loc[w]
            for v in self.variants:
                d = (g[v] - g[baseline]).dropna()
                n = len(d)
                se = d.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
                rows.append({"world": w, "variant": v, "mean": g[v].mean(), "se": g[v].std(ddof=1) / math.sqrt(max(n, 1)),
                             "diff": d.mean(), "diff_se": se,
                             "t": d.mean() / se if se and se > 0 else math.nan,
                             "wins": f"{int((d > 0).sum())}/{n}"})
        return pd.DataFrame(rows).set_index(["world", "variant"])

    def best(self, metric: str = "pnl", among: list[str] | None = None) -> pd.Series:
        """The variant with the highest mean in each world (optionally restricted to `among`)."""
        m = self.mean(metric)
        return (m[among] if among else m).idxmax(axis=1)
