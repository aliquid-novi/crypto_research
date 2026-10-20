"""Signal registry + redundancy filter.

A new signal earns a place only if it is (a) not a near-copy of an existing one and (b) adds out-of-sample
explanatory power for the target on top of the existing set. Both are checked chronologically: fit on the first
part, score on the rest.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import ROOT

REGISTRY = ROOT / "research" / "signals" / "registry.csv"
FIELDS = ["name", "hypothesis", "description", "added", "verdict", "max_abs_corr", "incremental_r2_oos", "notebook"]


def _ols_r2_oos(X_tr, y_tr, X_te, y_te):
    A = np.c_[np.ones(len(X_tr)), X_tr]
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    pred = np.c_[np.ones(len(X_te)), X_te] @ beta
    return 1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_tr.mean()) ** 2)


def redundancy_report(new: pd.Series, existing: pd.DataFrame, y: pd.Series, max_corr: float = 0.9,
                      min_incremental_r2: float = 0.0005, train_frac: float = 0.6) -> dict:
    """verdict 'reject' if |corr| with any existing signal >= max_corr, or the OOS R^2 gain from adding `new` to the
    existing set is < min_incremental_r2. Rows with any NaN are dropped; order is taken as time order."""
    d = pd.concat([existing, new.rename("__new__"), y.rename("__y__")], axis=1).dropna()
    corr = d[existing.columns].corrwith(d["__new__"]).abs() if len(existing.columns) else pd.Series(dtype=float)
    mc = float(corr.max()) if len(corr) else 0.0
    n_tr = int(len(d) * train_frac)
    tr, te = d.iloc[:n_tr], d.iloc[n_tr:]
    base_cols = list(existing.columns)
    r2_base = _ols_r2_oos(tr[base_cols].to_numpy(), tr["__y__"].to_numpy(), te[base_cols].to_numpy(), te["__y__"].to_numpy()) \
        if base_cols else 0.0
    cols = base_cols + ["__new__"]
    r2_new = _ols_r2_oos(tr[cols].to_numpy(), tr["__y__"].to_numpy(), te[cols].to_numpy(), te["__y__"].to_numpy())
    inc = float(r2_new - r2_base)
    reasons = []
    if mc >= max_corr:
        reasons.append(f"|corr| {mc:.3f} with '{corr.idxmax()}' >= {max_corr}")
    if inc < min_incremental_r2:
        reasons.append(f"incremental OOS R^2 {inc:.5f} < {min_incremental_r2}")
    return {"verdict": "reject" if reasons else "accept", "max_abs_corr": mc,
            "most_correlated": (corr.idxmax() if len(corr) else None), "r2_oos_existing": float(r2_base),
            "r2_oos_with_new": float(r2_new), "incremental_r2_oos": inc, "n": len(d), "reasons": reasons}


def register_signal(name: str, hypothesis: str, description: str, report: dict, notebook: str = "") -> None:
    """Append a row to research/signals/registry.csv (accepted or rejected: both are worth remembering)."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new_file = not REGISTRY.exists() or REGISTRY.stat().st_size == 0
    with REGISTRY.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({"name": name, "hypothesis": hypothesis, "description": description,
                    "added": datetime.now(timezone.utc).isoformat(timespec="seconds"), "verdict": report["verdict"],
                    "max_abs_corr": round(report["max_abs_corr"], 4),
                    "incremental_r2_oos": round(report["incremental_r2_oos"], 6), "notebook": notebook})


def registry() -> pd.DataFrame:
    return pd.read_csv(REGISTRY) if REGISTRY.exists() and REGISTRY.stat().st_size else pd.DataFrame(columns=FIELDS)
