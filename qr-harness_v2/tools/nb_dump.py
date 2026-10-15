#!/usr/bin/env python3
"""Dump a saved notebook's code, outputs and plots so Claude Code can read them.

Claude Code's built-in notebook reader omits large outputs, which includes
every inline plot. This writes a compact text view and saves each image
output as a PNG that can then be opened with the Read tool.

    python tools/nb_dump.py                                             # most recently saved notebook in the repo
    python tools/nb_dump.py research/notebooks/H001_x.ipynb            # all cells
    python tools/nb_dump.py NB.ipynb --cells 7 8 9                      # some cells
    python tools/nb_dump.py NB.ipynb --last 3                           # last 3 code cells that produced output

Output goes to .scratch/nb_dump/<notebook>/ (gitignored). Reads only; never
modifies the notebook. Only shows what was SAVED - unsaved runs are invisible.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _text(x) -> str:
    return "".join(x) if isinstance(x, list) else (x or "")


def _clip(s: str, n: int) -> str:
    s = ANSI.sub("", s).rstrip()
    if len(s) <= n:
        return s
    half = n // 2
    return f"{s[:half]}\n... [{len(s) - n} chars clipped] ...\n{s[-half:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", nargs="?", help="omit to use the most recently saved .ipynb in the repo")
    ap.add_argument("--cells", type=int, nargs="*", help="cell indices (0-based, as printed)")
    ap.add_argument("--last", type=int, help="only the last N code cells that have outputs")
    ap.add_argument("--max-chars", type=int, default=4000, help="per text output")
    args = ap.parse_args()

    if not args.notebook:
        skip = {".scratch", ".ipynb_checkpoints", ".venv", "node_modules", ".git"}
        nbs = [p for p in ROOT.rglob("*.ipynb") if not skip & set(p.relative_to(ROOT).parts)]
        if not nbs:
            print("ERROR: no .ipynb found in the repo")
            return 1
        nbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print("Most recently saved: " + ", ".join(str(p.relative_to(ROOT)) for p in nbs[:3]) + "  (using the first)")
        args.notebook = str(nbs[0])
    nb_path = Path(args.notebook)
    if nb_path.suffix != ".ipynb":
        alt = nb_path.with_suffix(".ipynb")
        print(f"NOTE: {nb_path.name} has no outputs (jupytext source). Using {alt.name}." if alt.exists() else f"ERROR: {nb_path} is not an .ipynb and no paired .ipynb exists")
        if not alt.exists():
            return 1
        nb_path = alt
    nb = json.loads(nb_path.read_text(encoding="utf-8"))  # notebooks are UTF-8; Windows default is cp1252
    cells = nb.get("cells", [])

    out_dir = ROOT / ".scratch" / "nb_dump" / nb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    idx = list(range(len(cells)))
    if args.cells:
        idx = [i for i in args.cells if 0 <= i < len(cells)]
    elif args.last:
        with_out = [i for i, c in enumerate(cells) if c.get("cell_type") == "code" and c.get("outputs")]
        idx = with_out[-args.last :]

    saved = dt.datetime.fromtimestamp(nb_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    ran = [c.get("execution_count") for c in cells if c.get("cell_type") == "code"]
    print(f"# {nb_path}  (last saved {saved})")
    if ran and all(r is None for r in ran):
        print("WARNING: no code cell has an execution count - notebook was saved without running, or outputs were cleared.")
    seq = [r for r in ran if r is not None]
    if seq and seq != sorted(seq):
        print(f"WARNING: cells were run out of order (execution counts {seq}). Outputs may not match the code above them.")

    images = []
    for i in idx:
        c = cells[i]
        ct = c.get("cell_type")
        src = _text(c.get("source"))
        if ct == "markdown":
            print(f"\n--- [{i}] markdown\n{_clip(src, 1500)}")
            continue
        if ct != "code":
            continue
        print(f"\n--- [{i}] code (exec {c.get('execution_count')})\n{src.rstrip()}")
        for j, o in enumerate(c.get("outputs", [])):
            ot = o.get("output_type")
            if ot == "stream":
                print(f"  >> {o.get('name')}:\n{_clip(_text(o.get('text')), args.max_chars)}")
            elif ot == "error":
                tb = ANSI.sub("", "\n".join(o.get("traceback", [])))
                print(f"  >> ERROR {o.get('ename')}: {o.get('evalue')}\n{_clip(tb, args.max_chars)}")
            elif ot in ("execute_result", "display_data"):
                data = o.get("data", {})
                if "image/png" in data:
                    p = out_dir / f"cell{i:03d}_out{j}.png"
                    p.write_bytes(base64.b64decode(_text(data["image/png"])))
                    images.append(p)
                    print(f"  >> IMAGE saved: {p.relative_to(ROOT)}")
                elif "image/svg+xml" in data:
                    p = out_dir / f"cell{i:03d}_out{j}.svg"
                    p.write_text(_text(data["image/svg+xml"]))
                    print(f"  >> SVG saved (convert to PNG to view): {p.relative_to(ROOT)}")
                if "application/vnd.plotly.v1+json" in data or "application/vnd.bokehjs_exec.v0+json" in data:
                    print("  >> INTERACTIVE PLOT (plotly/bokeh): not viewable as an image. Re-plot with matplotlib, or fig.write_image().")
                if "text/plain" in data and "image/png" not in data:
                    print(f"  >> result:\n{_clip(_text(data['text/plain']), args.max_chars)}")
                elif "text/html" in data and "text/plain" not in data:
                    print(f"  >> html:\n{_clip(_text(data['text/html']), args.max_chars)}")

    if images:
        print(f"\n{len(images)} image(s) in {out_dir.relative_to(ROOT)} - open each with the Read tool to see it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
