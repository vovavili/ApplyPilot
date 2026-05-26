"""Structured JSONL event logging for pipeline postmortems."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from applypilot.config import LOG_DIR

_LOCK = threading.Lock()
_RUN_ID: str | None = None
_MAX_STRING_CHARS = 1000
_MAX_LIST_ITEMS = 50
_REDACT_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "captcha",
    "cookie",
    "password",
    "private_minimum",
    "secret",
    "token",
)


def event_log_path() -> Path:
    """Return the JSONL event log path.

    Tests and power users can override this with APPLYPILOT_EVENT_LOG.
    """
    override = os.environ.get("APPLYPILOT_EVENT_LOG")
    return Path(override) if override else LOG_DIR / "events.jsonl"


def start_run(run_id: str | None = None) -> str:
    """Create a new run id for grouping events from one pipeline invocation."""
    global _RUN_ID
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _RUN_ID = run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
    return _RUN_ID


def current_run_id() -> str:
    """Return the active run id, creating one lazily for non-pipeline entry points."""
    global _RUN_ID
    if _RUN_ID is None:
        start_run()
    return str(_RUN_ID)


def emit_event(
    event: str,
    *,
    level: str = "info",
    stage: str | None = None,
    status: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Append one structured event to the JSONL log.

    Logging must never break job runs, so file write failures are swallowed.
    The returned record is useful in unit tests.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": current_run_id(),
        "level": level,
        "event": event,
    }
    if stage is not None:
        record["stage"] = stage
    if status is not None:
        record["status"] = status
    record.update({str(key): _clean_value(str(key), value) for key, value in fields.items()})

    try:
        path = event_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass

    return record


def emit_error(event: str, exc: BaseException, *, stage: str | None = None, **fields: Any) -> dict[str, Any]:
    """Append an error event with exception type and message."""
    return emit_event(
        event,
        level="error",
        stage=stage,
        error_type=type(exc).__name__,
        error=str(exc),
        **fields,
    )


def _clean_value(key: str, value: Any) -> Any:
    if _should_redact(key):
        return "[redacted]"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _truncate(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _clean_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        cleaned = [_clean_value(key, item) for item in items[:_MAX_LIST_ITEMS]]
        if len(items) > _MAX_LIST_ITEMS:
            cleaned.append(f"... truncated {len(items) - _MAX_LIST_ITEMS} item(s)")
        return cleaned
    return _truncate(str(value))


def _should_redact(key: str) -> bool:
    folded = key.casefold()
    return any(part in folded for part in _REDACT_KEY_PARTS)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_STRING_CHARS:
        return text
    return f"{text[:_MAX_STRING_CHARS]}... [truncated {len(text) - _MAX_STRING_CHARS} chars]"
