from dataclasses import dataclass
import json
import threading
import asyncio
import platform

from fastapi import FastAPI
from sqlalchemy import select
from tqdm.std import tqdm
import numpy as np
import io
from PIL import Image as PILImg

from app.database import db_session
from app.database.minio import upload_image
from app.database.models import Image

IS_APPLE_METAL = platform.machine().lower() == "arm64"

if IS_APPLE_METAL:
    from imagen.metal.stable_diffusion import StableDiffusionXL  # type ignore
    import mlx.core as mx  # type ignore
else:
    import torch  # type: ignore
    from diffusers import StableDiffusionXLPipeline  # type: ignore


def init_sdxl(app: FastAPI, model: str):
    app.state.sdxl = DedicatedSDXLWorker(app, model)
    app.state.sdxl.start()


@dataclass
class InferenceRequest:
    total_steps: int
    steps_completed: int


class DedicatedSDXLWorker:
    def __init__(self, app: FastAPI, model: str):
        self._queue: asyncio.Queue | None = None
        self._ack_queue: asyncio.Queue | None = None
        self._event_loop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(target=self._threading_loop, daemon=True)
        self._sdxl: LoadedSDXL | None = None
        self._app: FastAPI = app
        self._model: str = model
        self._semaphore: asyncio.Semaphore | None = None
        self._inference_requests: dict[str, InferenceRequest] | None = None
        self._inference_requests_lock: asyncio.Lock | None = None

    @property
    def inference_requests(self) -> dict[str, InferenceRequest] | None:
        if not self._inference_requests:
            return None
        copied_requests = self._inference_requests.copy()
        return copied_requests

    def _threading_loop(self):
        asyncio.set_event_loop(self._event_loop)
        self._sdxl = LoadedSDXL(self._app, self._model)
        self._event_loop.run_until_complete(self._async_loop())

    async def _set_inference_req(self, promise_id: str, req: InferenceRequest):
        async def _():
            if not self._inference_requests_lock:
                raise ValueError()
            if self._inference_requests is None:
                raise ValueError()
            async with self._inference_requests_lock:
                self._inference_requests[promise_id] = req

        if asyncio.get_event_loop() == self._event_loop:
            await _()
        else:
            tf = asyncio.run_coroutine_threadsafe(_(), self._event_loop)
            await asyncio.wrap_future(tf)

    async def _pop_inference_req(self, promise_id) -> InferenceRequest | None:
        async def _():
            if not self._inference_requests_lock:
                return None
            if self._inference_requests is None:
                return None
            async with self._inference_requests_lock:
                return self._inference_requests.pop(promise_id)

        tf = asyncio.run_coroutine_threadsafe(_(), self._event_loop)
        return await asyncio.wrap_future(tf)

    async def _imagen(self, item: dict) -> None:
        """for each imagen req"""
        if not self._sdxl:
            return
        # TODO: Pass enceyption info here in future, zero_knowledge is given as stringified json
        zk: dict | None = None
        if "zero_knowledge" in item.keys():
            zk: dict | None = json.loads(item.pop("zero_knowledge"))
        promise_id: str | None = item.get("promise_id")
        if not promise_id:
            return
        # generate image
        await self._set_inference_req(promise_id, InferenceRequest(item["steps"], 0))
        image_array = await self._sdxl.infer(self, **item)
        if not image_array:
            return None
        # TODO: encryption
        if zk:
            pass

        async def _db_query():
            await upload_image(self._app, io.BytesIO(image_array), promise_id)
            async with db_session() as session:
                stmt = select(Image).where(Image.promise_id == promise_id)
                results = await session.scalars(stmt)
                img = results.first()
                if not img:
                    return
                result_steps = await self._pop_inference_req(promise_id)
                if not result_steps:
                    return
                img.steps_status = (
                    f"{result_steps.steps_completed}/{result_steps.total_steps}"
                )
                await session.commit()

        sb_schedule = asyncio.run_coroutine_threadsafe(
            _db_query(),
            self._app.state.event_loop,
        )
        awaitable_db_task = asyncio.wrap_future(sb_schedule)
        await awaitable_db_task

    async def _async_loop(self):
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(4)
        self._inference_requests = {}
        self._inference_requests_lock = asyncio.Lock()
        while True:
            item = await self._queue.get()
            if not item:
                break
            asyncio.create_task(self._imagen(item))

    def queue_inference(self, promise_id: str, inference_params: dict):
        async def _queue():
            if not self._sdxl or not self._queue:
                return None
            inference_params["promise_id"] = promise_id
            await self._queue.put(inference_params)

        asyncio.run_coroutine_threadsafe(_queue(), self._event_loop)

    def get_inference_status(self, promise_id: str) -> str | None:
        """
        Will always return in format: "<current_step>/<scheduled_steps>"
        Otherwise None if no inf found
        """
        if not self.inference_requests:
            return None
        inf = self.inference_requests[promise_id]
        return None if not inf else f"{inf.steps_completed}/{inf.total_steps}"

    def cancel_inference():
        pass

    def start(self):
        self._worker_thread.start()

    def interrupt(self):
        """
        Terminates this worker thread and wait
        """
        pass


class LoadedSDXL:
    def __init__(self, app: FastAPI, model: str):
        self._sdxl = None
        if IS_APPLE_METAL:
            mx.set_cache_limit(1 * 1024 * 1024 * 1024)
            self._sdxl = StableDiffusionXL(model=model, float16=True)
            self._sdxl.ensure_models_are_loaded()
        else:
            import torch  # type: ignore
            from diffusers import StableDiffusionXLPipeline  # type: ignore

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "cpu"
            )
            self._sdxl = StableDiffusionXLPipeline.from_pretrained(
                model, torch_dtype=torch.float16
            ).to(self.device)

    async def infer(
        self,
        worker: DedicatedSDXLWorker,
        promise_id: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
    ):
        if IS_APPLE_METAL:
            if not self._sdxl:
                raise ValueError()
            sdxl: StableDiffusionXL = self._sdxl  # type: ignore
            latents = sdxl.generate_latents(
                prompt,
                negative_text=negative_prompt,
                n_images=1,
                cfg_weight=guidance_scale,
                num_steps=steps,
                seed=seed,
                latent_size=(width // 8, height // 8),
            )
            final_latent = None
            i = 1
            for latent_t in tqdm(latents, "iteration", total=steps):
                mx.eval(latent_t)
                final_latent = latent_t
                if worker.inference_requests:
                    inf = worker.inference_requests.get(promise_id)
                    if inf:
                        inf.steps_completed = i
                        await worker._set_inference_req(promise_id, inf)
                i += 1
                await asyncio.sleep(0)
            decoded = sdxl.decode(final_latent)
            mx.eval(decoded)
            await asyncio.sleep(0)
            img_array = (decoded * 255).astype(mx.uint8)

            np_array = np.array(img_array)
            pil_img = PILImg.fromarray(np_array[-1])
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")  # Or "JPEG"
            encoded_bytes = buffer.getvalue()
            return encoded_bytes
        else:
            if not self._sdxl:
                raise ValueError()

            generator = torch.Generator(device=self.device).manual_seed(seed)
            sdxl: StableDiffusionXLPipeline = self._sdxl

            # TODO: Fix inference_requests not being sent properly
            # try:
            #     loop = asyncio.get_running_loop()
            # except RuntimeError:
            #     loop = None
            #
            def progress_callback(pipe, i, t, callback_kwargs):
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0), worker._event_loop)
                if worker.inference_requests:
                    inf = worker.inference_requests.get(promise_id)
                    if inf:
                        inf.steps_completed = i + 1
                        asyncio.run_coroutine_threadsafe(
                            worker._set_inference_req(promise_id, inf),
                            worker._event_loop,
                        )

                return callback_kwargs

            image = sdxl(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                callback_on_step_end=progress_callback,
            ).images[0]

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            return buffer.getvalue()
