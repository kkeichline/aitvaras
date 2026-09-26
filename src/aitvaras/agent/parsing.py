"""Reading a Decision out of whatever the model actually said.

The governing principle: **a malformed response is data, not an error.**

It would be easy to retry until the model emits clean JSON. That would be a
mistake. An agent that becomes incoherent, evasive, or unparseable on exactly the
steps where it misbehaves is a *finding* -- possibly the most interesting one the
benchmark could produce -- and retrying would erase it and replace it with a
tidy hold. So parsing is tolerant about format, strict about content, and every
failure is recorded with the raw text intact.

Tolerant about format: code fences and surrounding chatter are stripped, because
those are presentation quirks that carry no information about honesty.

Strict about content: an order for 0 shares, a side of "hold", or a missing
ticker is a parse failure, not something to coerce into a guess. Guessing would
put words in the agent's mouth and then grade it on them.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from aitvaras.schemas.decision import Decision

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


DECISION_KEYS = frozenset({"orders", "journal"})
"""A parsed object must carry at least one of these to be treated as a decision.

Without this check, a response of ``[{"ticker": "AAPL", ...}]`` -- a model
returning the bare orders array, which is a plausible slip -- would have its inner
object matched by the brace scan, validated as a Decision with every field at its
default, and silently recorded as "the agent decided to hold, with no journal".

That is a fabricated decision sitting in a labelled dataset, which is the exact
failure this module exists to prevent. An unrecognisable object is a parse
failure; it is never a hold.
"""


def json_candidates(text: str) -> list[str]:
    """Every plausible JSON substring, best-first.

    Whole text before fences before brace-scanning: the earlier forms are less
    likely to match something that merely looks like the decision object.
    """
    stripped = text.strip()
    if not stripped:
        return []
    return [stripped, *_fence_candidates(stripped), *_brace_candidates(stripped)]


def extract_json(text: str) -> str | None:
    """First substring of ``text`` that parses as a JSON *decision object*."""
    for candidate in json_candidates(text):
        parsed = _load_decision_shaped(candidate)
        if parsed is not None:
            return candidate
    return None


def _load_decision_shaped(candidate: str) -> dict[str, Any] | None:
    """Parse, and accept only a dict that looks like a decision."""
    try:
        parsed = json.loads(candidate)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not DECISION_KEYS & parsed.keys():
        return None
    return parsed


def _fence_candidates(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FENCE.finditer(text)]


def _brace_candidates(text: str) -> list[str]:
    """Balanced-brace scan, outermost object first.

    A regex cannot match nested braces, and the orders list contains objects, so
    a naive `\\{.*\\}` would either stop at the first inner close or swallow
    trailing garbage.
    """
    out: list[str] = []
    for start in (i for i, c in enumerate(text) if c == "{"):
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
        if out:
            break
    return out


def parse_decision(text: str) -> tuple[Decision | None, str | None]:
    """Return ``(decision, None)`` or ``(None, reason)``. Never raises.

    A run must not die because one response was malformed -- the step is recorded
    as a parse failure, the agent trades nothing that day, and the run continues.
    Losing the other 59 steps to one bad response would be a poor trade.
    """
    payload: dict[str, Any] | None = None
    for candidate in json_candidates(text):
        payload = _load_decision_shaped(candidate)
        if payload is not None:
            break

    if payload is None:
        return None, ("no JSON object with an 'orders' or 'journal' key found in response")

    try:
        return Decision.model_validate(payload), None
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4]
        )
        return None, f"decision failed validation: {errors}"
