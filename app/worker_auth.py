"""Per-worker authentication and encryption keys.

The app never stores plaintext keys in the database.  Keys live in a JSON file
(or the ``GPU_WORKERS`` environment variable) with restrictive permissions.

Accepted file shape::

    {
      "workers": [
        {"worker_id": "gpu-01", "key_id": "k1", "key": "<base64 32 bytes>"}
      ]
    }

Multiple entries for the same ``worker_id`` are allowed so a key can be rotated
with a short overlap period.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from shared.protocol import KEY_SIZE, ProtocolError, b64d

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerKey:
    worker_id: str
    key_id: str
    key: bytes
    allow_instances: bool = False


class WorkerKeyStore:
    def __init__(self, keys: list[WorkerKey] | None = None):
        self._keys: dict[tuple[str, str], bytes] = {}
        self._instance_keys: list[WorkerKey] = []
        for item in keys or []:
            if len(item.key) != KEY_SIZE:
                raise ValueError(
                    f"worker {item.worker_id!r} key {item.key_id!r} must be 32 bytes"
                )
            self._keys[(item.worker_id, item.key_id)] = item.key
            if item.allow_instances:
                self._instance_keys.append(item)

    @property
    def worker_ids(self) -> set[str]:
        return {worker_id for worker_id, _ in self._keys}

    def get(self, worker_id: str, key_id: str) -> bytes | None:
        exact = self._keys.get((worker_id, key_id))
        if exact is not None:
            return exact

        # A multi-GPU launcher derives ``gpu-01-1``, ``gpu-01-2``, ... from
        # the base worker ``gpu-01`` while reusing the base key.  Instance
        # matching is opt-in so an ordinary key cannot be used by arbitrary
        # worker IDs.
        for item in self._instance_keys:
            if item.key_id != key_id:
                continue
            prefix = item.worker_id + "-"
            if worker_id.startswith(prefix) and worker_id[len(prefix) :].isdigit():
                return item.key
        return None

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> WorkerKeyStore:
        key_path = Path(path)
        if os.name == "posix":
            mode = key_path.stat().st_mode & 0o777
            if mode & 0o077:
                logger.warning(
                    "Worker key file %s is accessible by group/others (mode %o); chmod 600 it",
                    key_path,
                    mode,
                )
        raw = json.loads(key_path.read_text())
        return cls(cls._parse(raw))

    @classmethod
    def from_env_or_file(cls) -> WorkerKeyStore:
        raw_env = os.getenv("GPU_WORKERS")
        if raw_env:
            return cls(cls._parse(json.loads(raw_env)))

        path = os.getenv("GPU_WORKERS_FILE", "gpu_workers.json")
        if not Path(path).exists():
            logger.warning(
                "No GPU worker key file found at %s; GPU workers cannot connect",
                path,
            )
            return cls()
        return cls.from_file(path)

    @staticmethod
    def _parse(raw: object) -> list[WorkerKey]:
        if isinstance(raw, dict):
            raw = raw.get("workers", raw)
        if isinstance(raw, dict):
            # Convenience form: {"gpu-01": "<base64 key>"}
            result: list[WorkerKey] = []
            for worker_id, value in raw.items():
                if not isinstance(value, str):
                    raise ValueError("worker key must be a base64 string")
                result.append(WorkerKey(worker_id, "default", b64d(value)))
            return result
        if not isinstance(raw, list):
            raise ValueError("worker key file must contain a list or object")

        result = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("each worker key entry must be an object")
            worker_id = entry.get("worker_id")
            key_id = entry.get("key_id", "default")
            key = entry.get("key")
            allow_instances = entry.get("allow_instances", False) is True
            if not isinstance(worker_id, str) or not worker_id:
                raise ValueError("worker_id is required")
            if not isinstance(key_id, str) or not key_id:
                raise ValueError("key_id must be a string")
            if not isinstance(key, str):
                raise ValueError("key must be a base64 string")
            try:
                decoded = b64d(key)
            except ProtocolError as exc:
                raise ValueError(f"invalid key for worker {worker_id!r}") from exc
            result.append(WorkerKey(worker_id, key_id, decoded, allow_instances))
        return result
