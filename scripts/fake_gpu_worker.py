#!/usr/bin/env python3
"""Development-only fake GPU worker.

It exercises the real WebSocket, authentication, encryption and job protocol
without loading a model.  Useful for testing the app on a laptop:

    python scripts/fake_gpu_worker.py --app-url ws://localhost:4600/internal/v1/workers/ws \
        --key-file gpu-test.worker.json
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
from pathlib import Path

from gpu.errors import InferenceCancelled
from gpu.worker import GpuWorker, WorkerConfig, _read_key_file

logger = logging.getLogger("fake_gpu_worker")

# Fallback if Pillow is unavailable; the real GPU extra includes Pillow.
FAKE_JPEG = b"\xff\xd8\xff\xd9"


def _placeholder_jpeg(prompt: str, width: int, height: int, seed: int) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return FAKE_JPEG

    width = max(64, min(width, 1024))
    height = max(64, min(height, 1024))
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    phase = seed % 256
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 255) // width,
                (y * 255) // height,
                (phase + x // 4 + y // 4) % 256,
            )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 42), fill=(0, 0, 0))
    draw.text((12, 12), f"FAKE GPU: {prompt[:70]}", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class FakeBackend:
    async def infer(
        self,
        promise_id: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
        scheduler: str,
        on_step=None,
    ) -> bytes:
        logger.info("Fake inference for %s: %s steps", promise_id, steps)
        for step in range(1, max(steps, 1) + 1):
            await asyncio.sleep(0.05)
            if on_step is not None and not await on_step(step, steps):
                raise InferenceCancelled()
        return _placeholder_jpeg(prompt, width, height, seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app-url", default="ws://localhost:4600/internal/v1/workers/ws"
    )
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--model", default="fake-model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    worker_id, key_id, key = _read_key_file(Path(args.key_file), None, None)
    config = WorkerConfig(
        app_url=args.app_url,
        worker_id=worker_id,
        key_id=key_id,
        key=key,
        model=args.model,
        name="fake-gpu",
    )
    worker = GpuWorker(config, FakeBackend())

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        for sig in (2, 15):  # SIGINT, SIGTERM
            try:
                loop.add_signal_handler(sig, worker._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        await worker.run()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
