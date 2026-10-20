"""python -m qrlib.cli <command> ...

  new-card "<title>" --family F [--source mark|P###] [--confirm]
                                                      create research/hypotheses/H###_slug.md (draft). Refused while 2+
                                                      cards are unresolved, unless Mark confirmed (--confirm)
  register H###                                       lock Mark's prediction (refuses if fields are empty or not Mark's)
  status [H###]                                       list cards / show one
  resolve H### --outcome held|failed --lesson "..."   Mark's verdict; feeds calibration
  calibration                                         Brier score of Mark's confidence on resolved cards
  n-trials [H###]                                     trials logged in research/audit/runs.jsonl
  new-paper "<title>"                                 create research/papers/P###_slug.md
  new-notebook H###                                   research/notebooks/H###_slug.py from _template.py, paired .ipynb
  sync <notebook.py>                                  .py (percent format) -> .ipynb, keeping outputs of unchanged cells
  run H###                                            sync + execute the notebook in place
  new-post <name> --notebook <path> [--card H###]     research/post/<name>_post.md + <name>_rebuild.py (Mark fills both)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone

from . import ROOT, audit, prereg

NB_DIR = ROOT / "research" / "notebooks"
PAPER_DIR = ROOT / "research" / "papers"
POST_DIR = ROOT / "research" / "post"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "untitled"


def _next_id(folder, prefix: str) -> str:
    nums = [int(m.group(1)) for p in folder.glob(f"{prefix}[0-9][0-9][0-9]*.md") if (m := re.match(rf"{prefix}(\d{{3}})", p.name))]
    return f"{prefix}{(max(nums) + 1) if nums else 1:03d}"


def new_card(title: str, family: str, source: str = "mark", confirmed: bool = False):
    prereg.HYP_DIR.mkdir(parents=True, exist_ok=True)
    prereg.check_new_card_allowed(confirmed)
    hid = _next_id(prereg.HYP_DIR, "H")
    tmpl = (prereg.HYP_DIR / "_TEMPLATE.md").read_text(encoding="utf-8")
    tmpl = tmpl.replace("{{ID}}", hid).replace("{{TITLE}}", "x").replace("{{FAMILY}}", "x").replace("{{SOURCE}}", "x")
    meta, body = prereg.parse(tmpl)
    meta |= {"id": hid, "title": title, "family": family, "source": source,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = prereg.HYP_DIR / f"{hid}_{_slug(title)}.md"
    path.write_text(prereg.dump(meta, body), encoding="utf-8")
    return hid, path


def new_paper(title: str):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pid = _next_id(PAPER_DIR, "P")
    tmpl = (PAPER_DIR / "_TEMPLATE.md").read_text(encoding="utf-8")
    meta, body = prereg.parse(tmpl.replace("{{ID}}", pid).replace("{{TITLE}}", "x"))
    meta |= {"id": pid, "title": title, "read_on": datetime.now(timezone.utc).date().isoformat()}
    path = PAPER_DIR / f"{pid}_{_slug(title)}.md"
    path.write_text(prereg.dump(meta, body), encoding="utf-8")
    return pid, path


# ------------------------------------------------------------ notebooks

def py_to_cells(text: str):
    """Split a jupytext 'percent' file into (cell_type, source) pairs."""
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(("markdown", md))
        else:
            cells.append(("code", body))
    return cells


def sync(py_path) -> str:
    """Write the paired .ipynb. Uses jupytext if installed; otherwise a built-in converter that keeps the outputs
    of code cells whose source didn't change."""
    from pathlib import Path

    import nbformat as nbf

    py_path = Path(py_path)
    ipynb = py_path.with_suffix(".ipynb")
    try:
        import jupytext  # noqa: F401
        subprocess.run([sys.executable, "-m", "jupytext", "--set-formats", "ipynb,py:percent", str(py_path)], check=True)
        subprocess.run([sys.executable, "-m", "jupytext", "--sync", str(py_path)], check=True)
        return str(ipynb)
    except ImportError:
        pass
    old = {}
    if ipynb.exists():
        for c in nbf.read(ipynb, as_version=4).cells:
            if c.cell_type == "code":
                old.setdefault(c.source, c)
    cells = []
    for kind, src in py_to_cells(py_path.read_text(encoding="utf-8")):
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            c = nbf.v4.new_code_cell(src)
            if src in old:
                c.outputs, c.execution_count = old[src].outputs, old[src].execution_count
            cells.append(c)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ipynb)
    return str(ipynb)


def new_notebook(hid: str):
    meta, _ = prereg.read_card(hid)
    card = prereg.card_path(hid)
    NB_DIR.mkdir(parents=True, exist_ok=True)
    py = NB_DIR / f"{card.stem}.py"
    if py.exists():
        raise SystemExit(f"{py} already exists")
    tmpl = (NB_DIR / "_template.py").read_text(encoding="utf-8")
    py.write_text(tmpl.replace("{{ID}}", hid).replace("{{TITLE}}", str(meta.get("title", ""))).replace("{{CARD}}", card.name),
                  encoding="utf-8")
    sync(py)
    meta, body = prereg.read_card(hid)
    if not meta.get("notebook"):
        meta["notebook"] = f"research/notebooks/{py.name}"
        # 'notebook' is not a locked field, so this doesn't break the lock
        prereg.write_card(hid, meta, body)
    return py


def new_post(name: str, notebook: str, card: str = ""):
    """Post-implementation review: a markdown report and a rebuild notebook, both for Mark to fill."""
    POST_DIR.mkdir(parents=True, exist_ok=True)
    name = _slug(name) if not re.fullmatch(r"H\d{3}", name) else name
    md, py = POST_DIR / f"{name}_post.md", POST_DIR / f"{name}_rebuild.py"
    for p in (md, py):
        if p.exists():
            raise SystemExit(f"{p} already exists")
    subs = {"{{NAME}}": name, "{{NOTEBOOK}}": notebook, "{{CARD}}": card or "none",
            "{{DATE}}": datetime.now(timezone.utc).date().isoformat()}
    for src, dst in (("_TEMPLATE.md", md), ("_rebuild_template.py", py)):
        text = (POST_DIR / src).read_text(encoding="utf-8")
        for k, v in subs.items():
            text = text.replace(k, v)
        dst.write_text(text, encoding="utf-8")
    sync(py)
    return md, py


def run(hid: str):
    py = NB_DIR / f"{prereg.card_path(hid).stem}.py"
    prereg.require(hid)
    ipynb = sync(py)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", ipynb], check=True, cwd=NB_DIR)
    return ipynb


# ------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m qrlib.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new-card"); p.add_argument("title"); p.add_argument("--family", required=True); p.add_argument("--source", default="mark")
    p.add_argument("--confirm", action="store_true", help="Mark confirmed a new card despite unresolved ones")
    p = sub.add_parser("register"); p.add_argument("id")
    p = sub.add_parser("status"); p.add_argument("id", nargs="?")
    p = sub.add_parser("resolve"); p.add_argument("id"); p.add_argument("--outcome", required=True, choices=["held", "failed"])
    p.add_argument("--lesson", default="")
    sub.add_parser("calibration")
    p = sub.add_parser("n-trials"); p.add_argument("id", nargs="?")
    p = sub.add_parser("new-paper"); p.add_argument("title")
    p = sub.add_parser("new-notebook"); p.add_argument("id")
    p = sub.add_parser("sync"); p.add_argument("path")
    p = sub.add_parser("run"); p.add_argument("id")
    p = sub.add_parser("new-post"); p.add_argument("name"); p.add_argument("--notebook", required=True); p.add_argument("--card", default="")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "new-card":
            hid, path = new_card(a.title, a.family, a.source, confirmed=a.confirm)
            print(f"{hid} -> {path}")
            t = prereg.tally()
            print(f"hypotheses: {t['outstanding']} outstanding, {t['resolved']} resolved")
        elif a.cmd == "register":
            m = prereg.register(a.id)
            print(f"{a.id} registered at {m['registered_at']} (lock {m['lock_hash'][:12]})")
        elif a.cmd == "status":
            if a.id:
                meta, _ = prereg.read_card(a.id)
                try:
                    prereg.require(a.id); lock = "lock OK"
                except prereg.PreregistrationError as e:
                    lock = str(e)
                for k, v in meta.items():
                    print(f"{k:>14}: {v}")
                print(f"{'gate':>14}: {lock}\n{'n_trials':>14}: {audit.n_trials(a.id)}")
            else:
                for c in prereg.cards():
                    print(f"{c.get('id')}  {c.get('status'):<10}  trials={audit.n_trials(c.get('id')):<4} {c.get('title')}")
                t = prereg.tally()
                warn = (f"  <- {prereg.OUTSTANDING_LIMIT}+ unresolved: a new card needs Mark's confirmation"
                        if t["outstanding"] >= prereg.OUTSTANDING_LIMIT else "")
                print(f"hypotheses: {t['outstanding']} outstanding, {t['resolved']} resolved{warn}")
        elif a.cmd == "resolve":
            m = prereg.resolve(a.id, outcome=(a.outcome == "held"), lesson=a.lesson)
            print(f"{a.id} resolved: {'held' if m['outcome'] else 'failed'} (n_trials {m['n_trials_at_resolution']})")
        elif a.cmd == "calibration":
            c = prereg.calibration()
            print(c if c["n"] else "no resolved cards yet")
        elif a.cmd == "n-trials":
            print(audit.n_trials(a.id))
        elif a.cmd == "new-paper":
            pid, path = new_paper(a.title)
            print(f"{pid} -> {path}")
        elif a.cmd == "new-notebook":
            print(new_notebook(a.id))
        elif a.cmd == "sync":
            print(sync(a.path))
        elif a.cmd == "run":
            print(run(a.id))
        elif a.cmd == "new-post":
            md, py = new_post(a.name, a.notebook, a.card)
            print(f"{md}\n{py}")
    except prereg.PreregistrationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
