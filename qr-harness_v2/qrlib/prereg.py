"""Hypothesis cards: registration lock, gate, resolution, calibration.

A card is research/hypotheses/H###_slug.md: YAML front matter + markdown body. The fields
prediction, falsified_if, confidence and predicted_by are Mark's (CLAUDE.md rule 1). register() refuses unless they
are filled and predicted_by == "mark", then stores a hash of them. require() is the notebook gate: it refuses an
unregistered card, or one whose locked fields changed after registration.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import yaml

from . import ROOT, audit

HYP_DIR = ROOT / "research" / "hypotheses"
LOCKED = ("prediction", "falsified_if", "confidence", "predicted_by")
OWNER = "mark"
OUTSTANDING_LIMIT = 2   # Mark's rule: with this many unresolved cards, a new card needs his explicit confirmation
_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


class PreregistrationError(RuntimeError):
    pass


def card_path(hid: str):
    hits = sorted(HYP_DIR.glob(f"{hid}_*.md")) + sorted(HYP_DIR.glob(f"{hid}.md"))
    if not hits:
        raise PreregistrationError(f"no card for {hid} in {HYP_DIR}")
    return hits[0]


def parse(text: str):
    m = _FM.match(text)
    if not m:
        raise PreregistrationError("card has no YAML front matter")
    return (yaml.safe_load(m.group(1)) or {}), m.group(2)


def dump(meta: dict, body: str) -> str:
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).rstrip("\n") + "\n---\n" + body


def read_card(hid: str):
    return parse(card_path(hid).read_text(encoding="utf-8"))


def write_card(hid: str, meta: dict, body: str) -> None:
    card_path(hid).write_text(dump(meta, body), encoding="utf-8")


def lock_hash(meta: dict) -> str:
    payload = json.dumps({k: meta.get(k) for k in LOCKED}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _missing(meta: dict) -> list[str]:
    miss = [k for k in ("prediction", "falsified_if", "predicted_by") if not str(meta.get(k) or "").strip()]
    c = meta.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0 <= c <= 1:
        miss.append("confidence (a number in [0, 1])")
    return miss


def register(hid: str) -> dict:
    meta, body = read_card(hid)
    if meta.get("status") not in (None, "", "draft"):
        raise PreregistrationError(f"{hid} is already {meta.get('status')}")
    miss = _missing(meta)
    if miss:
        raise PreregistrationError(f"{hid} can't be registered: Mark hasn't filled {miss}")
    if str(meta.get("predicted_by")).strip().lower() != OWNER:
        raise PreregistrationError(f"{hid}: predictions must come from Mark, not agents (predicted_by="
                                   f"{meta.get('predicted_by')!r})")
    meta["status"] = "registered"
    meta["registered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["lock_hash"] = lock_hash(meta)
    write_card(hid, meta, body)
    audit.log_event("register", hid, lock_hash=meta["lock_hash"])
    return meta


def require(hid: str) -> dict:
    """Notebook gate. Call it in the first cell; it raises unless the card is registered and unchanged."""
    meta, _ = read_card(hid)
    if meta.get("status") not in ("registered", "resolved"):
        raise PreregistrationError(f"{hid} is not registered (status={meta.get('status')!r}). "
                                   f"Mark fills prediction/falsified_if/confidence, then `python -m qrlib.cli register {hid}`.")
    if lock_hash(meta) != meta.get("lock_hash"):
        raise PreregistrationError(f"{hid}: locked fields changed after registration ({', '.join(LOCKED)})")
    return meta


def resolve(hid: str, outcome: bool, lesson: str = "", note: str = "") -> dict:
    """Mark's call: did the prediction hold? Recorded with the lesson; feeds calibration."""
    meta, body = read_card(hid)
    require(hid)
    if meta.get("status") == "resolved":
        raise PreregistrationError(f"{hid} is already resolved")
    meta["status"] = "resolved"
    meta["outcome"] = bool(outcome)
    meta["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["lesson"] = lesson
    meta["n_trials_at_resolution"] = audit.n_trials(hid)
    if note:
        body = body.rstrip("\n") + f"\n\n{note}\n"
    write_card(hid, meta, body)
    audit.log_event("resolve", hid, outcome=bool(outcome), lesson=lesson)
    return meta


def cards() -> list[dict]:
    out = []
    for p in sorted(HYP_DIR.glob("H[0-9][0-9][0-9]*.md")):
        try:
            meta, _ = parse(p.read_text(encoding="utf-8"))
            out.append(meta | {"_path": str(p)})
        except PreregistrationError:
            pass
    return out


def tally() -> dict:
    """Outstanding (draft or registered, i.e. not yet resolved) vs resolved cards."""
    cs = cards()
    out = [c.get("id") for c in cs if c.get("status") != "resolved"]
    return {"outstanding": len(out), "resolved": len(cs) - len(out), "outstanding_ids": out}


def check_new_card_allowed(confirmed: bool = False) -> dict:
    """Refuse a new card while OUTSTANDING_LIMIT or more are unresolved, unless Mark confirmed it."""
    t = tally()
    if t["outstanding"] >= OUTSTANDING_LIMIT and not confirmed:
        raise PreregistrationError(
            f"PLEASE CONFIRM: {t['outstanding']} hypotheses are still unresolved ({', '.join(t['outstanding_ids'])}); "
            f"{t['resolved']} resolved. Resolve one first, or have Mark confirm and re-run with --confirm.")
    return t


def calibration() -> dict:
    """Brier score of Mark's confidence on resolved cards (lower is better; always saying 0.5 scores 0.25)."""
    rs = [c for c in cards() if c.get("status") == "resolved" and isinstance(c.get("confidence"), (int, float))]
    if not rs:
        return {"n": 0, "brier": None, "rows": []}
    rows = [{"id": c.get("id"), "confidence": float(c["confidence"]), "outcome": bool(c["outcome"])} for c in rs]
    brier = sum((r["confidence"] - r["outcome"]) ** 2 for r in rows) / len(rows)
    return {"n": len(rows), "brier": brier, "hit_rate": sum(r["outcome"] for r in rows) / len(rows),
            "mean_confidence": sum(r["confidence"] for r in rows) / len(rows), "rows": rows}
