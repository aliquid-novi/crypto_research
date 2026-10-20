"""PreToolUse hook (Write|Edit): block agent edits to locked fields of a registered hypothesis card.

Locked once status != draft: prediction, falsified_if, confidence, predicted_by, lock_hash, registered_at,
status, outcome, resolved_at, lesson. Those change only through `python -m qrlib.cli register/resolve`, with Mark's words.
The body (Results, Open questions...) stays editable. Exit 2 = block, message on stderr goes back to Claude.
"""
import json
import re
import sys
from pathlib import Path

import yaml

PROTECTED = ("prediction", "falsified_if", "confidence", "predicted_by", "lock_hash", "registered_at",
             "status", "outcome", "resolved_at", "lesson")
FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


def front(text):
    m = FM.match(text or "")
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None


def main():
    try:
        p = json.load(sys.stdin)
    except Exception:
        return 0
    ti = p.get("tool_input") or {}
    path = Path(ti.get("file_path") or "")
    if path.parent.name != "hypotheses" or not re.match(r"H\d{3}.*\.md$", path.name) or not path.exists():
        return 0
    cur = path.read_text(encoding="utf-8")
    before = front(cur)
    if not before or before.get("status") in (None, "", "draft"):
        return 0
    tool = p.get("tool_name")
    if tool == "Write":
        new = ti.get("content", "")
    elif tool == "Edit":
        old_s, new_s = ti.get("old_string", ""), ti.get("new_string", "")
        new = cur.replace(old_s, new_s) if ti.get("replace_all") else cur.replace(old_s, new_s, 1)
    else:
        return 0
    after = front(new)
    if after is None:
        print(f"Blocked: this edit would break the front matter of registered card {path.name}.", file=sys.stderr)
        return 2
    changed = [k for k in PROTECTED if before.get(k) != after.get(k)]
    if changed:
        print(f"Blocked: {path.name} is {before.get('status')}; locked fields {changed} can't be edited. "
              f"They are Mark's and change only via `python -m qrlib.cli register/resolve` (CLAUDE.md rule 1). "
              f"Edit the card body instead, or ask Mark.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
