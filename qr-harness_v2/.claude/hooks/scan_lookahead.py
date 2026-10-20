"""PostToolUse hook: scan research notebooks for lookahead patterns after every write.

Flags (CLAUDE.md rule 3): negative shifts, centred windows, backfills, raw/forward/nearest merge_asof, joins on
ts_exch, full-sample normalisation of a column, and negative time offsets in reindex/asof. A line ending in
`# lookahead-ok: <reason>` is skipped (targets belong in qrlib.pit.make_target, so that should be rare).

Exit 2 sends the findings to Claude, which must fix them or justify each with lookahead-ok.
Also usable by hand:  python .claude/hooks/scan_lookahead.py <file.py|file.ipynb> ...
"""
import json
import re
import sys
from pathlib import Path

RULES = [
    (re.compile(r"\.shift\(\s*-\s*\d"), "negative shift (future value): build targets with qrlib.pit.make_target"),
    (re.compile(r"\.shift\(\s*periods\s*=\s*-"), "negative shift (future value)"),
    (re.compile(r"center\s*=\s*True"), "centred window uses future rows"),
    (re.compile(r"bfill|backfill|method\s*=\s*['\"]b"), "backfill pulls future values backwards"),
    (re.compile(r"merge_asof\("), "raw merge_asof: use qrlib.pit.asof_features (strictly backward, with latency)"),
    (re.compile(r"direction\s*=\s*['\"](forward|nearest)"), "forward/nearest join"),
    (re.compile(r"on\s*=\s*['\"]ts_exch|ts_col\s*=\s*['\"]ts_exch"), "join on ts_exch (rule 3: join on ts_recv + latency)"),
    (re.compile(r"\(\s*(\w+(?:\[[^\]]+\])?)\s*-\s*\1\.mean\(\)\s*\)\s*/\s*\1\.std\(\)"), "full-sample z-score"),
]
SCOPE = re.compile(r"research/notebooks/.+\.(py|ipynb)$")


def lines_of(path: Path):
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for ci, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
            for li, line in enumerate(src.splitlines(), 1):
                yield f"cell {ci} line {li}", line
    else:
        for li, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            yield f"line {li}", line


def scan(path: Path):
    out = []
    for where, line in lines_of(path):
        code = line.split("#", 1)[0] if "lookahead-ok" not in line else ""
        if not code.strip() or code.lstrip().startswith("#"):
            continue
        for rx, msg in RULES:
            if rx.search(code):
                out.append(f"{path.name} {where}: {msg}\n    {line.strip()}")
    return out


def main(argv):
    if len(argv) > 1:
        paths = [Path(a) for a in argv[1:]]
    else:
        try:
            p = json.load(sys.stdin)
        except Exception:
            return 0
        fp = (p.get("tool_input") or {}).get("file_path") or (p.get("tool_input") or {}).get("notebook_path") or ""
        if not SCOPE.search(fp.replace("\\", "/")):
            return 0
        paths = [Path(fp)]
    found = [f for pth in paths if pth.exists() and pth.name != "_template.py" for f in scan(pth)]
    if found:
        print("Possible lookahead (CLAUDE.md rule 3). Fix, or justify each line with `# lookahead-ok: <reason>`:\n"
              + "\n".join(found), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
