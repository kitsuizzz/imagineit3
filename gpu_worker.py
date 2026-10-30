#!/usr/bin/env python3
"""Convenience entrypoint for the outbound-only GPU worker."""

import logging
import sys

try:
    from gpu.worker import main
except ModuleNotFoundError as exc:
    if exc.name in {"websockets", "cryptography"}:
        print(
            "GPU dependencies are missing. Run: pip install -e '.[gpu]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    raise

if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        logging.getLogger("gpu_worker").info("Interrupted")
        raise SystemExit(130)
