"""Task runner (works without `make`, e.g. on Windows).

    python run.py test        # pytest on synthetic data
    python run.py notebooks   # rebuild notebooks/*.ipynb from the notebooks/*.py sources
    python run.py demo        # rebuild + execute every notebook on synthetic data -> notebooks/executed/

The demo uses its own cache folder (data_cache_demo/) and forces synthetic data, so it never
touches the cache that 00_plug_your_data.ipynb writes from your real data.
Notebook sources are "percent" scripts: `# %%` starts a code cell, `# %% [markdown]` a markdown
cell whose lines start with `# `. Edit either the .py (then rebuild) or the .ipynb directly.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")


def py_to_ipynb(src: str, dst: str) -> None:
    """Convert a percent-format script into an (unexecuted) .ipynb file."""
    with open(src, encoding="utf-8") as f:
        lines = f.read().splitlines()
    cells, cur, kind = [], [], None

    def flush():
        if kind is None:
            return
        text = list(cur)
        while text and not text[-1].strip():
            text.pop()
        while text and not text[0].strip():
            text.pop(0)
        if kind == "markdown":
            text = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in text]
        src_lines = [ln + "\n" for ln in text]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": {}, "source": src_lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for ln in lines:
        if ln.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "[markdown]" in ln else "code")
        elif kind is not None:
            cur.append(ln)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def build_notebooks() -> list[str]:
    out = []
    for src in sorted(glob.glob(os.path.join(NB_DIR, "[0-9][0-9]_*.py"))):
        dst = src[:-3] + ".ipynb"
        py_to_ipynb(src, dst)
        out.append(dst)
    print(f"built {len(out)} notebooks")
    return out


def _kernel_env() -> str:
    """Register a throwaway kernelspec that runs THIS interpreter, so execution never picks up a
    different Python that happens to own the default 'python3' kernel."""
    d = tempfile.mkdtemp(prefix="posttrade_kernel_")
    k = os.path.join(d, "kernels", "posttrade-py")
    os.makedirs(k)
    with open(os.path.join(k, "kernel.json"), "w") as f:
        json.dump({"argv": [sys.executable, "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": "posttrade", "language": "python"}, f)
    os.environ["JUPYTER_PATH"] = d + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    return "posttrade-py"


def demo() -> int:
    import nbformat
    from nbclient import NotebookClient

    paths = build_notebooks()
    os.environ["POSTTRADE_CACHE"] = os.path.join(ROOT, "data_cache_demo")
    os.environ["POSTTRADE_FORCE_SYNTHETIC"] = "1"
    os.environ.pop("MPLBACKEND", None)  # let the kernel use the inline backend so figures land in the notebook
    if sys.platform == "win32":  # silences a pyzmq/tornado event-loop warning on Windows
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    kernel = _kernel_env()
    out_dir = os.path.join(NB_DIR, "executed")
    os.makedirs(out_dir, exist_ok=True)
    failed = []
    for p in paths:
        t = time.time()
        nb = nbformat.read(p, as_version=4)
        try:
            NotebookClient(nb, kernel_name=kernel, timeout=1800, resources={"metadata": {"path": NB_DIR}}).execute()
            status = "ok"
        except Exception as e:  # keep going so every failure is reported in one run
            status = f"FAILED: {type(e).__name__}: {str(e)[-1500:]}"
            failed.append(os.path.basename(p))
        nbformat.write(nb, os.path.join(out_dir, os.path.basename(p)))
        print(f"{os.path.basename(p):32s} {time.time() - t:6.1f}s  {status}")
    print("demo:", "all notebooks ran" if not failed else f"{len(failed)} failed: {failed}")
    return 1 if failed else 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(ROOT, "tests")],
                               cwd=ROOT)
    if cmd == "notebooks":
        build_notebooks()
        return 0
    if cmd == "demo":
        return demo()
    print(__doc__)
    return 0 if cmd == "help" else 2


if __name__ == "__main__":
    sys.exit(main())
