"""What a ``session_send`` delivered, recorded as data for the sender's own row.

The sender's transcript draws a ``session_send`` call that delivered as the
outgoing session-message card, naming the slot the gateway RESOLVED (a send
addressed by title must link to the real key). Both facts -- that it delivered,
and where -- are the gateway's, so they come from the gateway as typed fields
rather than out of the tool's human-readable result, which a rewording or a
backend's reshaping of the result body would silently break.

The two halves meet by the CALL's own arguments, the selector
``dashboard.directive_queue`` uses for the same reason:

* ``POST /api/session-control/send`` (``handlers.session_control``) delivers,
  then :func:`park` records ``{outcome, target}`` for the calling session under a
  digest of the ``target`` and ``message`` it was asked to send. Only a request
  the ``session_send`` tool makes asks for one (``RECEIPT_FIELD``): a dispatch
  seed or an app's send has no tool row to claim it.
* The caller's turn digests the ``target`` and ``message`` of the
  ``session_send`` call it saw the model make (:func:`call_selector` over the
  ACP ``rawInput``), and on that call's completed result :func:`claim`\\ s the
  receipt and stamps it on the call's own tool row as ``meta.sessionSend``.

Two identical sends in flight at once can each take the other's receipt; both
name the same resolved target, so only the outcome word can swap. Nothing reads
the result body. A receipt only ever describes a delivery the
gateway made for that session, and only a call with the very same arguments in
that session's turn picks it up; a receipt nobody claims ages out. A row with no
receipt (an older row, a refused send, a backend that reports no call input)
draws the ordinary tool line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

#: The gateway's own MCP server and tool whose calls carry a receipt.
SESSION_SEND_SERVER = "kirocrew-dashboard"
SESSION_SEND_TOOL = "session_send"

#: The row meta key the transcript reads.
META_KEY = "sessionSend"

#: The ``/api/session-control/send`` body flag the ``session_send`` tool sets to
#: ask for a receipt.
RECEIPT_FIELD = "receipt"

#: The outcomes a receipt can name: the target started a turn on the message,
#: took it into the turn it was running, or queued it behind that turn.
OUTCOMES = frozenset({"delivered", "steered", "queued"})

#: A receipt is claimed within the same turn, normally within a second; the age
#: bound only retires receipts no call ever claims.
MAX_AGE_SECS = 300.0
MAX_PER_SESSION = 16
MAX_SESSIONS = 256

_lock = threading.Lock()
# session key -> [(parked_at, selector, receipt)], oldest first.
_parked: dict[str, list[tuple[float, str, dict[str, str]]]] = {}


def selector(target: Any, message: Any) -> str | None:
    """The digest a receipt is parked and claimed under, or None for a malformed pair."""
    if not isinstance(target, str) or not target or not isinstance(message, str):
        return None
    canon = json.dumps([target, message], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8", "surrogatepass")).hexdigest()


def call_selector(raw_args: Any) -> str | None:
    """The selector of a ``session_send`` call's raw ACP arguments.

    Some backends wrap the arguments the server receives: Codex as
    ``{"server", "tool", "arguments"}``, a Tool Search deferred call as
    ``{"tool_id": "<server>::<tool>", "arguments"}``. Unwrapped only when the
    wrapper names this very tool, so the digest covers what the server was sent.
    """
    if not isinstance(raw_args, dict):
        return None
    inner = raw_args.get("arguments")
    if isinstance(inner, dict) and (
        (
            raw_args.get("server") == SESSION_SEND_SERVER
            and raw_args.get("tool") == SESSION_SEND_TOOL
        )
        or raw_args.get("tool_id") == f"{SESSION_SEND_SERVER}::{SESSION_SEND_TOOL}"
    ):
        raw_args = inner
    return selector(raw_args.get("target"), raw_args.get("message"))


def is_session_send_call(server: Any, name: Any) -> bool:
    """True for the gateway's own ``session_send``, by the call's trusted identity."""
    return server == SESSION_SEND_SERVER and name == SESSION_SEND_TOOL


def outcome_of(result: dict[str, Any]) -> str:
    """The receipt outcome of a ``send_to_target`` result."""
    if result.get("steered"):
        return "steered"
    if result.get("started"):
        return "delivered"
    return "queued"


def park(session_key: str, target: Any, message: Any, outcome: str, delivered_to: str) -> bool:
    """Record what a send by *session_key* delivered; False when nothing was parked."""
    key = selector(target, message)
    if not session_key or key is None or outcome not in OUTCOMES or not delivered_to:
        return False
    now = time.monotonic()
    with _lock:
        _sweep_locked(now)
        if session_key not in _parked and len(_parked) >= MAX_SESSIONS:
            # Room for the newest sender: the session whose newest receipt is
            # oldest goes first.
            stalest = min(_parked, key=lambda k: _parked[k][-1][0])
            del _parked[stalest]
        bucket = _parked.setdefault(session_key, [])
        bucket.append((now, key, {"outcome": outcome, "target": delivered_to}))
        del bucket[:-MAX_PER_SESSION]
    return True


def claim(session_key: str, key: str | None, *, not_before: float = 0.0) -> dict[str, str] | None:
    """Remove and return the oldest receipt *session_key* parked under *key* since *not_before*."""
    if not session_key or not key:
        return None
    with _lock:
        _sweep_locked(time.monotonic())
        bucket = _parked.get(session_key)
        if not bucket:
            return None
        for index, (parked_at, parked_key, receipt) in enumerate(bucket):
            if parked_key == key and parked_at >= not_before:
                del bucket[index]
                if not bucket:
                    del _parked[session_key]
                return dict(receipt)
    return None


def note_call(selectors: dict[str, str], event: Any, *, initial: bool) -> None:
    """Record the selector of a ``session_send`` tool call frame, by its id.

    Called for the initial ``tool_call`` and for a refinement: some backends
    stream the arguments only on the update. The identity is the frame's own
    ``_meta`` pair, never a title, so a shell or another server's same-named tool
    records nothing; a frame with no usable arguments keeps what an earlier one
    recorded.
    """
    tool_call_id = getattr(event, "tool_call_id", "") or ""
    if not tool_call_id:
        return
    if initial:
        # The initial frame decides the identity, afresh for a reused id.
        if not is_session_send_call(
            getattr(event, "mcp_server_name", ""), getattr(event, "tool_name", "")
        ):
            selectors.pop(tool_call_id, None)
            return
    elif tool_call_id not in selectors and not is_session_send_call(
        getattr(event, "mcp_server_name", ""), getattr(event, "tool_name", "")
    ):
        # A refinement frame usually carries no identity of its own: it refines a
        # call its initial frame identified (recorded with "" until arguments
        # arrive), or names the call itself.
        return
    key = call_selector(getattr(event, "raw_tool_params", None))
    if key is not None:
        selectors[tool_call_id] = key
    else:
        selectors.setdefault(tool_call_id, "")


def stamp_result(
    state: Any,
    slot: Any,
    selectors: dict[str, str],
    session_key: str,
    event: Any,
    rows: list[dict],
    *,
    terminal: bool,
    not_before: float,
) -> dict[str, str] | None:
    """On a ``session_send`` call's terminal result, stamp its receipt on *rows*.

    *rows* are the call's own tool rows this turn appended. Only a COMPLETED
    call claims: a failed or refused one delivered nothing, and its receipt (if
    any) is left to age out. Open tabs get the field by ``chat_message_update``,
    the patch the reducer merges into the row's meta. Returns the receipt.
    """
    tool_call_id = getattr(event, "tool_call_id", "") or ""
    if not terminal or tool_call_id not in selectors:
        # A mid-stream frame: the call has not returned yet.
        return None
    key = selectors.pop(tool_call_id)
    if not key:
        return None
    receipt = claim(session_key, key, not_before=not_before)
    if (
        receipt is None
        or not getattr(event, "tool_final", False)
        or getattr(event, "refusal", None)
    ):
        # A call that failed or was refused stamps nothing; a receipt its request
        # parked (it delivered, then the harness reported a failure) is retired
        # with it, so a later identical call the gateway refused cannot wear it.
        return None
    for row in rows:
        row.setdefault("meta", {})[META_KEY] = dict(receipt)
        ts = str(row.get("ts") or "")
        mid = str(row["meta"].get("mid") or "")
        if not ts and not mid:
            continue
        # Addressed by the row's own id where it has one: two rows can share a ts.
        patch: dict[str, Any] = {"slot": slot.key, "ts": ts, "meta": {META_KEY: dict(receipt)}}
        if mid:
            patch["mid"] = mid
        try:
            state.broadcast_ws("chat_message_update", patch)
        except Exception:
            # The row is already right; a client reconciles on its next fetch.
            logger.debug("session_send receipt broadcast failed", exc_info=True)
    return receipt


def reset() -> None:
    """Drop every parked receipt (tests)."""
    with _lock:
        _parked.clear()


def _sweep_locked(now: float) -> None:
    for session_key in list(_parked):
        kept = [entry for entry in _parked[session_key] if now - entry[0] <= MAX_AGE_SECS]
        if kept:
            _parked[session_key] = kept
        else:
            del _parked[session_key]
