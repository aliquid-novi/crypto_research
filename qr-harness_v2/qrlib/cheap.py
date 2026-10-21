"""Cheap hypotheses: research/cheap/C###_slug.py (+ paired .ipynb). No card, no registration, no review loop.

The notebook's FIRST markdown cell is Mark's: a pre-implementation section (filled before any code runs on real data)
and a post-implementation section (filled after reading the results). Mark writes it in the .ipynb, so the .ipynb is the
source of truth for that cell: sync() copies it back into the .py before regenerating the .ipynb, so it is never
overwritten.

require() is the gate: it refuses until the pre fields are filled, and fingerprints them on first use. If they change
afterwards it doesn't block, but warns and records the change in the audit trail.
"""
from __future__ import annotations

import hashlib
import re

from . import ROOT, audit

CHEAP_DIR = ROOT / "research" / "cheap"
PLACEHOLDER = "_(Mark)_"
PRE = ("Hypothesis", "Why I think it works", "Assumptions", "Test statistic / method, and why")
POST = ("What actually happened", "What changed vs my expectation", "Why it worked / didn't",
        "Was the test the right one? Why", "What changes to the data or initial conditions could have changed the result")


class CheapError(RuntimeError):
    pass


def path(cid: str):
    hits = sorted(CHEAP_DIR.glob(f"{cid}_*.py"))
    if not hits:
        raise CheapError(f"no cheap notebook for {cid} in {CHEAP_DIR}")
    return hits[0]


def _first_md_of_ipynb(py):
    import nbformat as nbf
    ipynb = py.with_suffix(".ipynb")
    if not ipynb.exists():
        return None
    for c in nbf.read(ipynb, as_version=4).cells:
        if c.cell_type == "markdown":
            return c.source
    return None


def _first_md_of_py(py):
    from .cli import py_to_cells
    for kind, src in py_to_cells(py.read_text(encoding="utf-8")):
        if kind == "markdown":
            return src
    return None


def marks_cell(cid: str) -> str:
    py = path(cid)
    return _first_md_of_ipynb(py) or _first_md_of_py(py) or ""


def fields(text: str, names) -> dict:
    """Text after each '**Name:**' label, up to the next label or heading. Placeholder counts as empty."""
    out = {}
    for n in names:
        m = re.search(rf"\*\*{re.escape(n)}:\*\*(.*?)(?=\n\s*\*\*[^*\n]+:\*\*|\n#|\Z)", text, re.S)
        out[n] = (m.group(1).replace(PLACEHOLDER, "").strip() if m else "")
    return out


def pre_hash(text: str) -> str:
    return hashlib.sha256(repr(sorted(fields(text, PRE).items())).encode()).hexdigest()


def require(cid: str) -> dict:
    """Notebook gate: refuse until Mark's pre-implementation fields are filled. Call it in the first code cell."""
    text = marks_cell(cid)
    pre = fields(text, PRE)
    missing = [k for k, v in pre.items() if not v]
    if missing:
        raise CheapError(f"{cid}: Mark hasn't filled the pre-implementation fields {missing} (first markdown cell).")
    h = pre_hash(text)
    seen = audit.records(cid, kind="cheap_pre")
    if not seen:
        audit.log_event("cheap_pre", cid, pre_hash=h)
    elif seen[-1].get("pre_hash") != h:
        audit.log_event("cheap_pre_changed", cid, old=seen[-1].get("pre_hash"), new=h)
        audit.log_event("cheap_pre", cid, pre_hash=h)
        print(f"WARNING {cid}: the pre-implementation section changed after it was first used. "
              f"The change is recorded in the audit trail.")
    return pre


def status(cid: str) -> dict:
    text = marks_cell(cid)
    pre, post = fields(text, PRE), fields(text, POST)
    stage = ("needs pre" if not all(pre.values()) else
             "awaiting post" if not all(post.values()) else "done")
    return {"id": cid, "stage": stage, "n_trials": audit.n_trials(cid),
            "pre_changed": bool(audit.records(cid, kind="cheap_pre_changed"))}


def all_ids() -> list[str]:
    return sorted({m.group(0) for p in CHEAP_DIR.glob("C[0-9][0-9][0-9]_*.py") if (m := re.match(r"C\d{3}", p.name))})


def pull_marks_cell(py) -> bool:
    """Copy the first markdown cell from the .ipynb (where Mark edits it) into the .py. True if the .py changed."""
    md = _first_md_of_ipynb(py)
    if md is None:
        return False
    text = py.read_text(encoding="utf-8")
    parts = re.split(r"(^# %%.*$)", text, flags=re.M)        # ['', '# %% [markdown]', body, '# %%', body, ...]
    for i in range(1, len(parts), 2):
        if "[markdown]" in parts[i]:
            new_body = "\n" + "\n".join(("# " + l) if l else "#" for l in md.splitlines()) + "\n\n"
            if parts[i + 1].strip("\n") == new_body.strip("\n"):
                return False
            parts[i + 1] = new_body
            py.write_text("".join(parts), encoding="utf-8")
            return True
    return False
