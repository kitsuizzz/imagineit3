"""Plaintext job parameters that must never reach a datastore.

The client sends parameters over TLS so the GPU can render them, but the
database only keeps the sealed envelope.  The dispatcher therefore reads the
live copy from this process-local map.  Losing the map (restart, crash) makes
in-flight jobs unretryable, so those rows are failed at startup instead of
being silently dispatched without a prompt.
"""

from __future__ import annotations

from typing import Any

_params: dict[str, dict[str, Any]] = {}


def put(promise_id: str, params: dict[str, Any]) -> None:
    _params[promise_id] = dict(params)


def get(promise_id: str) -> dict[str, Any] | None:
    value = _params.get(promise_id)
    return dict(value) if value is not None else None


def pop(promise_id: str) -> None:
    _params.pop(promise_id, None)


def clear() -> None:
    _params.clear()
