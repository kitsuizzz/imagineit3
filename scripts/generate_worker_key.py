#!/usr/bin/env python3
"""Generate a per-worker pre-shared key for the app and GPU host.

Example:

    python scripts/generate_worker_key.py \
        --worker-id gpu-01 \
        --app-file gpu_workers.json \
        --worker-file gpu-01.worker.json

The app file contains every worker public entry (base64 key); the worker file
contains only that worker's key.  Keep both files out of git and chmod 600.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
from pathlib import Path


def _write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    if os.name == "posix":
        path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--key-id", default="k1")
    parser.add_argument("--app-file", default="gpu_workers.json")
    parser.add_argument("--worker-file", default=None)
    args = parser.parse_args()

    key = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    app_path = Path(args.app_file)
    if app_path.exists():
        raw = json.loads(app_path.read_text())
        if not isinstance(raw, dict) or not isinstance(raw.get("workers"), list):
            raise SystemExit("app file must contain a {'workers': [...]} object")
        workers = raw["workers"]
    else:
        workers = []
    workers = [
        entry
        for entry in workers
        if not (
            isinstance(entry, dict)
            and entry.get("worker_id") == args.worker_id
            and entry.get("key_id") == args.key_id
        )
    ]
    workers.append(
        {
            "worker_id": args.worker_id,
            "key_id": args.key_id,
            "key": key,
            "allow_instances": True,
        }
    )
    _write_private(app_path, {"workers": workers})

    worker_path = Path(args.worker_file or f"{args.worker_id}.worker.json")
    _write_private(
        worker_path,
        {"worker_id": args.worker_id, "key_id": args.key_id, "key": key},
    )
    print(f"Wrote app key entry to {app_path}")
    print(f"Wrote worker key file to {worker_path}")
    print(f"Run on the GPU host: python gpu_worker.py --key-file {worker_path}")
    print(
        "Multi-GPU host: add --gpus auto "
        "(the key entry allows numbered instance worker IDs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
