"""Append-only JSONL audit trail. Every evaluated variant is a trial and gets a line (CLAUDE.md rule 6).

Never edit or delete research/audit/runs.jsonl. n_trials counts 'run' records only.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from . import ROOT

LOG = ROOT / "research" / "audit" / "runs.jsonl"


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _append(rec: dict) -> dict:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": _git_rev(), **rec}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def log_run(hypothesis: str, variant: str, params: dict | None = None, metrics: dict | None = None,
            notebook: str = "", note: str = "") -> dict:
    """One evaluated variant (a model, a parameter setting, a fill bound...). Log it whether it worked or not."""
    return _append({"kind": "run", "hypothesis": hypothesis, "variant": variant, "params": params or {},
                    "metrics": metrics or {}, "notebook": notebook, "note": note})


def log_event(kind: str, hypothesis: str, **fields) -> dict:
    """Non-trial events (registration, resolution, leakage results)."""
    return _append({"kind": kind, "hypothesis": hypothesis, **fields})


def records(hypothesis: str | None = None, kind: str | None = None) -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (hypothesis is None or r.get("hypothesis") == hypothesis) and (kind is None or r.get("kind") == kind):
            out.append(r)
    return out


def n_trials(hypothesis: str | None = None) -> int:
    """Trials evaluated so far (for one hypothesis, or across everything)."""
    return len(records(hypothesis, kind="run"))
